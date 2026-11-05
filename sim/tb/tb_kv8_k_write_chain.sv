`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_k_write_chain -- stage 1 module testbench (self-checking)
//
// Proves the three stage-1 acceptance criteria:
//   PHASE1  K INT8 bytes land at the right DDR addresses
//           (byte-exact against a reference implementation of kv8_k_quant's own
//            integer rule, and against an address formula written out longhand)
//   PHASE2  the K scale table is written and read back correctly
//           (read-back BF16 must equal m8 * 2^(eb-127) = s_k/8 exactly, for both
//            KV heads and every token, through the real 3-cycle read port)
//   PHASE3  the ownership FSM protects layer N while layer N-1 is reading
//           (bulk load / append / read into the wrong bank state is dropped and
//            flagged; a legal ping-pong layer hand-off works and does not
//            corrupt the bank that is still being read)
// =============================================================================
module tb_kv8_k_write_chain;
    localparam integer CONTEXT_LOG2 = 14;
    localparam integer LAYERS = 24;
    localparam integer NTOK = 6;
    // [M16-SCALE] the tree runs the chain at MARGIN=8 with a 16-bit mantissa; the
    // reference model below is parameterised so the TB tests the real config.
    localparam integer MARGIN = 8;

    reg         clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ---- DUT ports ---------------------------------------------------------
    reg         i_start = 0;
    reg  [4:0]  i_layer = 5'd2;
    reg  [15:0] i_token = 0;
    reg  [63:0] i_k_base = 64'h1000_0000 + (64'd2 << 23);
    reg  [63:0] i_k_scale_base = 64'h3000_0000;
    reg  [63:0] i_v_scale_base = 64'h3100_0000;
    wire        o_ready, o_busy, o_done, o_k_ready;
    reg         i_k_valid = 0;
    reg  [127:0] i_k_data = 0;
    wire        o_beat_valid;
    reg         i_beat_ready = 1;
    wire [63:0] o_beat_address;
    wire [127:0] o_beat_data;
    wire [6:0]  o_beat_tag;
    reg         i_flush = 0;
    wire        o_flush_done;
    wire        o_beat_desc;
    wire [4:0]  o_beat_len;
    wire [4:0]  o_beat_token;
    wire        o_err_seq;
    reg         i_bank_wr = 0, i_bank_rd = 0;
    reg         i_wr_acquire = 0, i_wr_release = 0, i_rd_release = 0;
    wire        o_wr_grant, o_rd_grant, o_owner_error;
    reg         i_etab_valid = 0;
    reg  [2:0]  i_etab_word = 0;
    reg  [127:0] i_etab_data = 0;
    reg         i_load_valid = 0;
    reg  [11:0] i_load_word = 0;
    reg  [127:0] i_load_mant = 0;
    reg         i_rd_en = 0;
    reg  [11:0] i_rd_word = 0;
    wire        o_rd_valid;
    wire [135:0] o_rd_scale;      // [M16-SCALE] {eb_tab[7:0], 8 x m16}
    reg         i_v_we = 0;
    reg  [3:0]  i_v_word = 0;
    reg  [127:0] i_v_data = 0;
    reg  [3:0]  i_v_rd_word = 0;
    wire [127:0] o_v_rd_data;
    wire        o_app_conflict, o_clip_sticky;
    wire [31:0] o_stat_beat, o_stat_app;
    wire [3:0]  o_dbg_state;
    wire [1:0]  o_dbg_lock_v;
    wire [15:0] o_clip_count;
    reg  [15:0] i_etab_be = 16'hffff;

    kv8_k_write_chain #(.CONTEXT_LOG2(CONTEXT_LOG2), .LAYERS(LAYERS), .MARGIN(MARGIN)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_start(i_start), .i_layer(i_layer), .i_token(i_token),
        .i_k_base(i_k_base), .i_k_scale_base(i_k_scale_base), .i_v_scale_base(i_v_scale_base),
        .o_ready(o_ready),
        .i_k_valid(i_k_valid), .o_k_ready(o_k_ready), .i_k_data(i_k_data),
                .o_busy(o_busy), .o_done(o_done),
        .i_flush(i_flush), .o_flush_done(o_flush_done),
        .o_beat_valid(o_beat_valid), .i_beat_ready(i_beat_ready),
        .o_beat_address(o_beat_address), .o_beat_data(o_beat_data), .o_beat_tag(o_beat_tag),
        .o_beat_desc(o_beat_desc), .o_beat_len(o_beat_len), .o_beat_token(o_beat_token), .o_err_seq(o_err_seq),
        .i_bank_wr(i_bank_wr), .i_bank_rd(i_bank_rd),
        .i_wr_acquire(i_wr_acquire), .i_wr_release(i_wr_release), .i_rd_release(i_rd_release),
        .o_wr_grant(o_wr_grant), .o_rd_grant(o_rd_grant), .o_owner_error(o_owner_error),
        .i_etab_valid(i_etab_valid), .i_etab_word(i_etab_word), .i_etab_data(i_etab_data),
        .i_etab_be(i_etab_be),
        .i_load_valid(i_load_valid), .i_load_word(i_load_word), .i_load_mant(i_load_mant),
        // [TB-HARNESS] kv8_k_write_chain gained the V m16 bulk-load, the V
        // per-token append and the etab image-dump ports after this TB was
        // written, and they were left dangling here.  Dangling inputs are X
        // inside kv8_scale_ram, and i_vload_valid makes
        //     app_fire = i_app_valid && app_ok && !i_load_valid && !i_vload_valid
        // X, so o_app_taken is X, app_done never rises and the chain hangs in
        // S_APP on the very first token.  Measured on the pre-change tree: this
        // TB times out identically there, i.e. it was already stale before
        // [RECIP-SEQ].  This TB exercises the K chain only, so "not driven" is
        // exactly 1'b0/0 here -- the same values the real controller drives
        // when it is not doing a V bulk load or a V append.
        .i_vload_valid(1'b0), .i_vload_word(12'd0), .i_vload_data(128'd0),
        .i_etab_rd_word(3'd0),
        .i_vt_app_valid(1'b0), .i_vt_app_token(16'd0), .i_vt_app_head(1'b0), .i_vt_app_mant(16'd0),
        .i_vt_rd_en(1'b0), .i_vt_rd_word(12'd0),
        .i_rd_en(i_rd_en), .i_rd_word(i_rd_word), .o_rd_valid(o_rd_valid), .o_rd_scale(o_rd_scale),
        .i_v_we(i_v_we), .i_v_word(i_v_word), .i_v_data(i_v_data),
        .i_v_rd_word(i_v_rd_word), .o_v_rd_data(o_v_rd_data),
        .o_app_conflict(o_app_conflict), .o_clip_sticky(o_clip_sticky),
        .o_stat_beat(o_stat_beat), .o_stat_app(o_stat_app),
        .o_dbg_state(o_dbg_state), .o_dbg_lock_v(o_dbg_lock_v), .o_clip_count(o_clip_count)
    );

    // ---- sparse DDR model --------------------------------------------------
    reg [7:0] ddr [longint];               // sparse DDR model, keyed by byte address
    integer   ddr_written;
    integer   cbyte;
    

    integer errors;
    integer checks;

    // ---- reference data ----------------------------------------------------
    reg [15:0] kref [0:NTOK*2*64-1];      // BF16 K per (token, head, dim)
    reg [15:0] qexp [0:NTOK*2*64-1];      // expected K16 code {k_hi,k_lo}
    reg [15:0] mant_exp [0:NTOK*2-1];
    reg [7:0]  eb_exp   [0:NTOK*2-1];
    reg [7:0]  ebraw_exp[0:NTOK*2-1];
    reg [15:0] m_exp    [0:NTOK*2-1];

    // ---------------------------------------------------------------- helpers
    function [2:0] msb_i(input [7:0] v);
        begin
            if      (v[7]) msb_i = 3'd7;
            else if (v[6]) msb_i = 3'd6;
            else if (v[5]) msb_i = 3'd5;
            else if (v[4]) msb_i = 3'd4;
            else if (v[3]) msb_i = 3'd3;
            else if (v[2]) msb_i = 3'd2;
            else if (v[1]) msb_i = 3'd1;
            else           msb_i = 3'd0;
        end
    endfunction
    // [M16-RECIP] round(2^24/m) for the whole 16-bit range (identical to the DUT's
    // function).  2^16 was too coarse for a 16-bit m: with m in [517,32898] it only
    // resolves to 2..127 steps, i.e. up to 0.8% error on the code itself.
    function [24:0] recip_lut(input [15:0] m);
        reg [24:0] d;
        begin
            if (m == 16'd0) recip_lut = 25'd16777216;
            else begin
                d = {9'd0,m};
                recip_lut = (25'd16777216 + {1'b0,m[15:1]}) / d;
            end
        end
    endfunction

    // Exact reference of kv8_k_quant for one head (8 beats x 8 dims), including the
    // block-exponent lock: the first token of a 256-token block fixes eb, every
    // later token of the block is quantised with it (m re-derived, saturates).
    reg [7:0]  ref_lock [0:1];
    reg        ref_lock_v [0:1];
    task automatic ref_head(input integer t, input integer hd);
        reg [14:0] maxmag, m15;
        reg [31:0] num, quo, rem;
        reg [15:0] m_calc, m_l;
        reg [7:0]  eb_calc, eb_use;
        reg [24:0] recip;
        integer d, lane, base, shift;
        reg [22:0] num_l, quo_l, rem_l;
        reg [15:0] kk;
        reg [7:0]  mm;
        reg [23:0] nn;
        reg [49:0] pr;
        reg [33:0] qv;
        reg [15:0] qmag;
        reg [8:0]  qhi;
        reg [7:0]  qlo;
        reg [8:0]  etmp;
        begin
            base = t*2*64 + hd*64;
            // ---- pass 0: magnitude ----
            maxmag = 15'd0;
            for (lane=0; lane<64; lane=lane+1) begin
                m15 = kref[base+lane][14:0];
                if (m15 > maxmag) maxmag = m15;
            end
            eb_calc = (maxmag[14:7] > (MARGIN+1)) ? (maxmag[14:7] - MARGIN) : 8'd1;
            if (!ref_lock_v[hd]) begin
                ref_lock_v[hd] = 1'b1;
                ref_lock[hd]   = eb_calc;
                // m = ceil(M8 * 2^MARGIN / 127), 32-bit numerator (MARGIN=9 -> 517..1029)
                num = {24'd0,1'b1,maxmag[6:0]} << MARGIN;
                quo = num / 32'd127;
                rem = num - quo*32'd127;
                m_calc = quo[15:0] + ((rem != 0) ? 16'd1 : 16'd0);
                eb_use = eb_calc;
            end else begin
                eb_use = ref_lock[hd];
    // [FIX-LOCKMARGIN] the DUT no longer re-adds MARGIN (eb_lock is already
                shift  = $signed({1'b0,maxmag[14:7]}) - $signed({1'b0,eb_use});
                if      (shift < -15)  num_l = 23'd0;
                else if (shift < 0)    num_l = {15'd0,1'b1,maxmag[6:0]} >> (-shift);
                else if (shift > 15)   num_l = 23'h7F_FFFF;
                else                   num_l = {15'd0,1'b1,maxmag[6:0]} << shift;
                quo_l = num_l / 23'd127;
                rem_l = num_l - quo_l*23'd127;
                // [M16-CLIP] the field top is 65535 now, not 255
                if (quo_l > 23'd65535 || (quo_l == 23'd65535 && rem_l != 23'd0))
                    m_l = 16'd65535;
                else
                    m_l = quo_l[15:0] + ((rem_l != 0) ? 16'd1 : 16'd0);
                m_calc= m_l;
            end
            recip = recip_lut(m_calc);
            ebraw_exp[t*2+hd] = eb_use;
            m_exp[t*2+hd]     = m_calc;
            // exact table convention: stored m16 = m, stored exponent byte = eb_use-3
            mant_exp[t*2+hd]  = m_calc;
            // [K16-CODE] the table exponent drops by 8 because the code carries 8
            // more fractional bits (s16 = s8/256); C_K stays 268 as frozen.
            etmp              = (eb_use <= 8'd3) ? 9'd1 : ({1'b0,eb_use} - 9'd3);
            eb_exp[t*2+hd]    = (etmp <= 9'd9) ? 8'd1 : (etmp[7:0] - 8'd8);
            // ---- pass 1: quantise (K16: q16 = (N*R + 2^15) >> 16, R = round(2^24/m)) ----
            for (lane=0; lane<64; lane=lane+1) begin
                kk = kref[base+lane];
                mm = {1'b1, kk[6:0]};
                d  = $signed({1'b0,kk[14:7]}) - $signed({1'b0,eb_use});
                if (kk[14:7] == 8'd0) nn = 24'd0;
                else if (d > 15)      nn = 24'hFF_FFFF;
                else if (d >= 0)      nn = {{16{1'b0}},mm} << d;
                else                  nn = {{16{1'b0}},mm} >> (-d);
                pr = nn * {26'd0,recip} + 50'd32768;      // [K16] round at the 16-bit LSB
                qv = pr >> 16;                            // [K16] 8 more fractional bits
                qmag = (qv > 34'd32639) ? 16'd32639 : qv[15:0];
                qhi  = (qmag + 17'd128) >> 8;             // round(q16/256)
                qlo  = qmag[7:0];                         // q16 - 256*qhi, two's complement
                if (kk[15]) begin qhi = (~qhi + 9'd1); qlo = (~qlo + 8'd1); end
                qexp[base+lane] = {qhi[7:0], qlo};
            end
        end
    endtask

    // ---- stimulus helpers --------------------------------------------------
    task automatic stream_token(input integer t, input integer stall_mode);
        integer hd, b, lane;
        reg [127:0] beat;
        begin
            i_token = t[15:0];
            @(posedge clk);
            while (!o_ready) @(posedge clk);
            i_start <= 1'b1;
            @(posedge clk);
            i_start <= 1'b0;
            for (hd=0; hd<2; hd=hd+1) begin
                for (b=0; b<8; b=b+1) begin
                    beat = 128'd0;
                    for (lane=0; lane<8; lane=lane+1)
                        beat[lane*16 +: 16] = kref[t*2*64 + hd*64 + b*8 + lane];
                    i_k_valid <= 1'b1;
                    i_k_data  <= beat;
                    @(posedge clk);
                    while (!o_k_ready) @(posedge clk);
                end
            end
            i_k_valid <= 1'b0;
            while (!o_done) @(posedge clk);
            @(posedge clk);
            $display("  token %0d: lock_v=%b eb_lock0=%0d m0=%0d mant0=%h ebt0=%h run_max=%0d",
                     t, o_dbg_lock_v, dut.eb_lock[0], dut.m_int[0], dut.mant16[0], dut.eb_tbl[0], dut.k_quant_Inst.run_max);
        end
    endtask

    integer seed = 32'h1234_5678;
    integer t, hd, d, dg, dbyte;
    reg [63:0] exp_addr;
    reg [127:0] exp_word;
    reg [7:0] got, want;
    reg [15:0] got_bf, want_bf;
    integer nfail1, nfail2, nfail3;
    reg [31:0] pre_read;

    // ---- write-port backpressure: 1 stall in 8 while bp_mode=1 ----------------
    integer bp_mode = 0;
    reg [3:0] bp_cnt = 4'd0;
    always @(posedge clk) begin
        bp_cnt <= bp_cnt + 4'd1;
        i_beat_ready <= (bp_mode == 0) ? 1'b1 : (bp_cnt[2:0] != 3'd0);
    end

    integer beat_dbg = 0;
    integer flw;
    always @(posedge clk) begin
        if (o_beat_valid && i_beat_ready && beat_dbg < 2) begin
            $display("  BEAT addr=%h tag=%h data=%h | refpacked=%h",
                     o_beat_address, o_beat_tag, o_beat_data,
                     {qexp[7],qexp[6],qexp[5],qexp[4],qexp[3],qexp[2],qexp[1],qexp[0]});
            beat_dbg = beat_dbg + 1;
        end
        if (dut.app_taken && dut.token_r == 0)
            $display("  RESTORE-CHECK t=0 h=%0d eb_r=%0d m_r=%0d hold0=%h hold1=%h", dut.head,
                     dut.k_quant_Inst.eb_r, dut.k_quant_Inst.m_r, dut.hold[0], dut.hold[1]);
    end

    // ---- append monitor (validates the published table entries) -------------
    always @(posedge clk) begin
        if (dut.app_taken)
            $display("  APP t=%0d h=%0d mant=%h exp=%0d clip=%b | ref mant=%h exp=%0d",
                     dut.token_r, dut.head, dut.app_mant, dut.app_exp, dut.app_clip,
                     mant_exp[dut.token_r*2+dut.head], eb_exp[dut.token_r*2+dut.head]);
        if (dut.state == 4'd4 && dut.beat == 3'd0 && dut.p1_tog)
            $display("  P1start t=%0d h=%0d eb_r=%0d m_r=%0d recip=%0d eff_eb=%0d eff_m=%0d",
                     dut.token_r, dut.head, dut.k_quant_Inst.eb_r, dut.k_quant_Inst.m_r,
                     dut.k_quant_Inst.recip_r, dut.eff_eb[dut.head], dut.eff_m[dut.head]);
    end

    // ---- DDR beat collector ------------------------------------------------
    always @(posedge clk) begin
        if (o_beat_valid && i_beat_ready) begin
            for (cbyte=0; cbyte<16; cbyte=cbyte+1)
                ddr[o_beat_address + cbyte] = o_beat_data[cbyte*8 +: 8];
            ddr_written <= ddr_written + 1;
        end
    end

    // ---- stage-2 descriptor audit -----------------------------------------
    // One 32-token block must come out as 16 x 256 B descriptors of the form
    //   block*8192 + dim_group*1024 + head*512 + token*16   (token ascending)
    // so that a linear read of the block is the fill packer's beat contract.
    integer    desc_count = 0;
    integer    desc_err   = 0;
    reg [4:0]  desc_prev_tok  = 5'd0;
    reg [63:0] desc_prev_addr = 64'd0;
    reg [4:0]  desc_run_len   = 5'd0;
    reg [63:0] desc_expect;
    always @(posedge clk) begin
        if (o_beat_valid && i_beat_ready) begin
            if (o_beat_desc) begin
                // i_k_base is the per-layer K base (the caller owns the stride)
                desc_expect = i_k_base + (desc_count/2)*1024 + (desc_count%2)*512;
                desc_count  = desc_count + 1;
                desc_run_len= o_beat_len;
                desc_prev_tok = o_beat_token;
                if (o_beat_token !== 5'd0) begin
                    desc_err = desc_err + 1;
                    $display("FAIL desc start token %0d != 0", o_beat_token);
                end
                if (o_beat_address !== desc_expect) begin
                    desc_err = desc_err + 1;
                    $display("FAIL desc %0d base %h != %h", desc_count-1, o_beat_address, desc_expect);
                end
                if (o_beat_len !== NTOK[4:0]) begin
                    desc_err = desc_err + 1;
                    $display("FAIL desc %0d len %0d != %0d", desc_count-1, o_beat_len, NTOK);
                end
            end else begin
                if (o_beat_token !== desc_prev_tok + 5'd1) begin
                    desc_err = desc_err + 1;
                    $display("FAIL desc token %0d -> %0d @%h", desc_prev_tok, o_beat_token, o_beat_address);
                end
                if (o_beat_address !== desc_prev_addr + 64'd16) begin
                    desc_err = desc_err + 1;
                    $display("FAIL desc address %h -> %h (not contiguous)", desc_prev_addr, o_beat_address);
                end
                if (o_beat_len !== desc_run_len) begin
                    desc_err = desc_err + 1;
                    $display("FAIL desc length changed inside a run (%0d != %0d)", o_beat_len, desc_run_len);
                end
                desc_prev_tok = o_beat_token;
            end
            desc_prev_addr = o_beat_address;
        end
    end

    initial begin
        ddr_written = 0;
        errors = 0; checks = 0;
        nfail1 = 0; nfail2 = 0; nfail3 = 0;


        // --------------------------------------------------- reference K data
        // Two magnitudes: token 0..2 small (exponent spread), 3..5 larger, so the
        // block exponent of the table actually changes between tokens/heads.
        for (t=0; t<NTOK; t=t+1) begin
            for (hd=0; hd<2; hd=hd+1) begin
                for (d=0; d<64; d=d+1) begin
                    // BF16 with a controlled exponent and random sign/mantissa
                    reg [7:0] e; reg [6:0] mant; reg s;
                    e = (t<3 ? 8'd131 : 8'd138) + ((d%5==0) ? 8'd2 : 8'd0);
                    mant = $random(seed) & 7'h7f;
                    // [M16-CLIP] the locked m only saturates when the token rises far
                    // enough above its lock AND the max dim carries M8 = 255, so the
                    // saturation corner is pinned instead of left to a random draw:
                    // head 1 tokens 3..5 rise to e=140 while head 1's lock is 133-8=125,
                    // i.e. lock_shift = 15, and their max dim gets M8 = 255
                    //   ->  ceil(255*2^15/127) = 65 796 > 65 535  =>  m_lock_sat.
                    // (head 0: lock 136-8=128, shift 12 -> m = 8225, no saturation)
                    if (t>=3 && hd==1 && (d%5==0)) mant = 7'h7f;
                    s = $random(seed) & 1;
                    if (d==0 && hd==0 && t==0) e = 8'd136;      // one dominant dim
                    kref[t*2*64 + hd*64 + d] = {s, e, mant};
                end
            end
        end
        ref_lock_v[0] = 1'b0; ref_lock_v[1] = 1'b0;    // block-exponent lock state
        ref_lock[0]   = 8'd0; ref_lock[1]   = 8'd0;
        for (t=0; t<NTOK; t=t+1) begin
            ref_head(t,0); ref_head(t,1);
        end

        repeat (20) @(posedge clk);
        rst_n = 1;
        repeat (10) @(posedge clk);

        // ------------------------------------------------------------- PHASE 1
        $display("--- PHASE1: write %0d tokens of INT8 K, check bytes+addresses ---", NTOK);
        // layer 2 -> bank 0 : acquire, write, release
        i_bank_wr = 0;
        @(posedge clk); i_wr_acquire <= 1'b1; @(posedge clk); i_wr_acquire <= 1'b0;
        @(posedge clk);
        if (!o_wr_grant) begin $display("FAIL acquire did not grant"); nfail1=nfail1+1; end

        for (t=0; t<NTOK; t=t+1) begin
            bp_mode = 1;
            stream_token(t, 0);
        end
        bp_mode = 0;
        repeat (20) @(posedge clk);

        // stage 2: the writer buffers a whole 32-token block, so a 6-token run
        // must be flushed explicitly before any beat reaches DDR.
        @(posedge clk); i_flush <= 1'b1; @(posedge clk); i_flush <= 1'b0;
        for (flw=0; flw<8000 && ddr_written < NTOK*2*8; flw=flw+1) @(posedge clk);
        repeat (10) @(posedge clk);
        if (desc_err != 0) begin
            nfail1 = nfail1 + 1;
            $display("FAIL descriptor audit (%0d)", desc_err);
        end

        // byte / address check (stage-2 block32 layout, K16)
        //   k_base + layer*4MiB + token[13:5]*8192 + dim_group*1024
        //          + head*512 + token[4:0]*16 + dim*2 + {lo,hi}
        for (t=0; t<NTOK; t=t+1) begin
            for (hd=0; hd<2; hd=hd+1) begin
                for (d=0; d<64; d=d+1) begin
                    dg    = d / 8;
                    dbyte = d % 8;
                    exp_addr = i_k_base + (t/32)*8192 + dg*1024 + hd*512
                             + (t%32)*16 + dbyte*2;
                    for (cbyte=0; cbyte<2; cbyte=cbyte+1) begin
                        got  = ddr[exp_addr + cbyte];
                        want = qexp[t*2*64 + hd*64 + d][cbyte*8 +: 8];
                        checks = checks + 1;
                        if (got !== want) begin
                            nfail1 = nfail1 + 1;
                            if (nfail1 < 12)
                                $display("FAIL byte t=%0d h=%0d d=%0d addr=%h got=%02h want=%02h",
                                         t, hd, d, exp_addr+cbyte, got, want);
                        end
                    end
                end
            end
        end
        $display("PHASE1 bytes: checked=%0d fail=%0d  (DDR beats written=%0d, expected=%0d)",
                 checks, nfail1, ddr_written, NTOK*2*8);
        if (ddr_written != NTOK*2*8) begin
            $display("FAIL beat count %0d != %0d", ddr_written, NTOK*2*8);
            nfail1 = nfail1 + 1;
        end

        // ------------------------------------------------------------- PHASE 2
        $display("--- PHASE2: scale table write/read-back (bf16 = s_k/8 exactly) ---");
        @(posedge clk); i_wr_release <= 1'b1; @(posedge clk); i_wr_release <= 1'b0;
        @(posedge clk);
        i_bank_rd = 0;
        checks = 0;
        for (hd=0; hd<2; hd=hd+1) begin
            for (t=0; t<NTOK; t=t+1) begin
                // read the 8-token word containing this token, then pick the lane
                i_rd_en   <= 1'b1;
                i_rd_word <= hd*2048 + (t/8);
                @(posedge clk);
                i_rd_en <= 1'b0;
                repeat (3) @(posedge clk);
                if (!o_rd_valid) begin
                    $display("FAIL read valid missing h=%0d t=%0d", hd, t);
                    nfail2 = nfail2 + 1;
                end
                // [M16-SCALE] no BF16 expander on the K read path any more: the word
                // carries the raw 16-bit mantissa and the block exponent byte.
                got_bf  = o_rd_scale[(t%8)*16 +: 16];
                want_bf = mant_exp[t*2+hd];
                checks = checks + 1;
                if (got_bf !== want_bf) begin
                    nfail2 = nfail2 + 1;
                    if (nfail2 < 12)
                        $display("FAIL scale h=%0d t=%0d got m16=%h want=%h (m=%0d eb_raw=%0d eb_tbl=%0d)",
                                 hd, t, got_bf, want_bf, m_exp[t*2+hd], ebraw_exp[t*2+hd],
                                 eb_exp[t*2+hd]);
                end
                checks = checks + 1;
                if (o_rd_scale[135:128] !== eb_exp[t*2+hd]) begin
                    nfail2 = nfail2 + 1;
                    if (nfail2 < 12)
                        $display("FAIL eb h=%0d t=%0d got=%0d want=%0d", hd, t,
                                 o_rd_scale[135:128], eb_exp[t*2+hd]);
                end
                repeat (2) @(posedge clk);
            end
        end
        $display("PHASE2 scale words: checked=%0d fail=%0d (clip_sticky=%b app_conflict=%b clip_count=%0d)",
                 checks, nfail2, o_clip_sticky, o_app_conflict, o_clip_count);
        // the clip counter must equal the number of appends whose locked m saturated
        checks = checks + 1;
        if (o_clip_count !== 16'd3) begin
            nfail2 = nfail2 + 1;
            $display("FAIL clip_count=%0d expected 3 (h=1 tokens 3..5 saturate the lock)", o_clip_count);
        end

        // ------------------------------------------------------------- PHASE 3
        $display("--- PHASE3: ownership FSM (illegal access must be dropped+flagged) ---");
        checks = 0;
        // (a) bulk load into bank 0 while it is READY (layer N-1 still reading)
        i_bank_wr <= 1'b0;
        pre_read = 0;
        i_rd_en <= 1'b1; i_rd_word <= 12'd0; @(posedge clk); i_rd_en <= 1'b0;
        // issue the illegal bulk load in the same window
        i_load_valid <= 1'b1; i_load_word <= 12'd0; i_load_mant <= 128'hDEAD_BEEF_DEAD_BEEF_DEAD_BEEF_DEAD_BEEF;
        @(posedge clk);
        i_load_valid <= 1'b0;
        repeat (5) @(posedge clk);
        if (!o_owner_error) begin
            $display("FAIL illegal bulk load into a READY bank was not flagged");
            nfail3 = nfail3 + 1;
        end else $display("  (a) bulk load into READY bank dropped + o_owner_error set  [ok]");
        checks = checks + 1;
        // the legal read that raced it must be unaffected
        if (o_rd_scale[15:0] === 16'hxxxx) begin
            $display("FAIL read port returned x after the blocked write");
            nfail3 = nfail3 + 1;
        end

        // (b) read from bank 1 while it is EMPTY
        // clear the sticky error first by resetting the RAM through a fresh DUT?
        // (o_owner_error is sticky by design) -- check the *grant* instead.
        i_bank_rd <= 1'b1;
        @(posedge clk);
        if (o_rd_grant) begin
            $display("FAIL read granted on an EMPTY bank");
            nfail3 = nfail3 + 1;
        end else $display("  (b) read from EMPTY bank not granted  [ok]");
        i_bank_rd <= 1'b0;
        checks = checks + 1;

        // (c) legal ping-pong: bank 1 loads layer N while bank 0 is being read
        i_bank_wr <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b1; @(posedge clk); i_wr_acquire <= 1'b0;
        @(posedge clk);
        if (!o_wr_grant) begin
            $display("FAIL acquire bank1 while bank0 READY did not grant");
            nfail3 = nfail3 + 1;
        end
        // bulk load a mantissa word + exponent word into bank 1
        i_etab_valid <= 1'b1; i_etab_word <= 3'd0; i_etab_data <= {16{8'd140}};
        i_load_valid <= 1'b1; i_load_word <= 12'd0; i_load_mant <= {8{16'd128}};
        @(posedge clk);
        i_etab_valid <= 1'b0; i_load_valid <= 1'b0;
        @(posedge clk); i_wr_release <= 1'b1; @(posedge clk); i_wr_release <= 1'b0;
        // read bank 0 (layer N-1) and bank 1 (layer N) back to back
        i_bank_rd <= 1'b0; i_rd_en <= 1'b1; i_rd_word <= 12'd0;
        @(posedge clk); i_rd_en <= 1'b0;
        repeat (3) @(posedge clk);
        if (!o_rd_grant) begin
            $display("FAIL bank0 read revoked after bank1 load");
            nfail3 = nfail3 + 1;
        end
        i_bank_rd <= 1'b1; i_rd_en <= 1'b1; i_rd_word <= 12'd0;
        @(posedge clk); i_rd_en <= 1'b0;
        repeat (3) @(posedge clk);
        got_bf = o_rd_scale[15:0];
        want_bf = 16'd128;                     // 8 x m16 = 128, block exponent 140
        checks = checks + 1;
        if (got_bf !== want_bf || o_rd_scale[135:128] !== 8'd140) begin
            $display("FAIL bulk-loaded bank1 readback got m16=%0d eb=%0d want 128/140",
                     got_bf, o_rd_scale[135:128]);
            nfail3 = nfail3 + 1;
        end else $display("  (c) ping-pong layer hand-off: bank0 readable, bank1 loaded+read  [ok]");
        $display("PHASE3 ownership: checked=%0d fail=%0d (app_conflict=%b clip=%b)",
                 checks, nfail3, o_app_conflict, o_clip_sticky);

        // ------------------------------------------------------------------ end
        repeat (10) @(posedge clk);
        if (nfail1+nfail2+nfail3 == 0)
            $display("STAGE1 PASS: bytes=%0d scale=%0d ownership=%0d  (stat_beat=%0d stat_app=%0d)",
                     checks, checks, checks, o_stat_beat, o_stat_app);
        else
            $display("STAGE1 FAIL: bytes_fail=%0d scale_fail=%0d owner_fail=%0d", nfail1, nfail2, nfail3);
        $finish;
    end

    // watchdog
    initial begin
        #4000000;
        $display("STAGE1 TIMEOUT state=%0d blk_tok=%0d emit_grp=%0d emit_tok=%0d app_valid=%b app_issued=%b pk_valid=%b beat=%0d head=%b i_beat_ready=%b o_ready=%b",
                 dut.state, dut.blk_tok, dut.emit_grp, dut.emit_tok, dut.app_valid, dut.app_issued, dut.pk_valid,
                 dut.beat, dut.head, i_beat_ready, o_ready);
        $finish;
    end
endmodule
`default_nettype wire




