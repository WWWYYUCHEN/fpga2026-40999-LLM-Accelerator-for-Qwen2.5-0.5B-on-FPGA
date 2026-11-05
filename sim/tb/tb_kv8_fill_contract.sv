`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_fill_contract -- stage 2 WRITER -> PACKER contract testbench
//
// What this proves (self-checking, no golden files):
//   PHASE1  the writer's DDR image is the 32-token-block layout: every beat sits
//           at block*4096 + dim_group*1024 + head*512 + token[4:0]*16, and every
//           (block, dim_group, head, token) that the writer reports is present.
//   PHASE2  a LINEAR read of one 32-token block (ascending address, one 16 B beat
//           per step) fed to kv8_fill_packer MODE_KV=0 -- filtered to one KV head,
//           with the tail of a partial block zero-padded exactly like
//           qwen25_attention.v does -- produces no protocol error
//           (o_err_seq/o_err_head = 0) and fills the real operand bank
//           (qwen25_fixed_weight, unmodified) so that
//              bank[row = dim][column = token] == DDR byte of (token, head, dim)
//           for all 64 dims x 64 tokens (2 blocks).
//   PHASE3  every one of the 64 rows x 2 lanes of the tile is written once.
//   The comparator reads the bank through the real 1024-bit read port, i.e. the
//   same bytes the INT8 array consumes.
//
// The packer's own bijection is NOT re-derived here: work_pack\tb_kv8_fill_packer
// covers that (unchanged, re-run in this session).  This TB only tests the thing
// that changed -- the writer's address order against the packer's beat contract.
// =============================================================================
module tb_kv8_fill_contract;
    localparam integer NTOK     = 40;              // 32 (full block) + 8 (flush)
    localparam integer NBLOCK   = 2;
    localparam integer TILE_TOK = NBLOCK*32;       // 64 tokens in the operand bank
    localparam integer LAYER    = 11;
    localparam integer NTOK1    = NTOK - 32;       // tokens in the partial block

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ---------------------------------------------------------------- DUT: writer
    reg         i_start = 0;
    reg  [15:0] i_token = 0;
    // per-layer K base (the caller owns the layer stride): LAYER*8 MiB + 0x40000000
    reg  [63:0] i_k_base = 64'h0000_0000_4000_0000 + (LAYER*64'h80_0000);
    reg  [63:0] i_k_scale_base = 64'h0000_0000_5000_0000;
    reg  [63:0] i_v_scale_base = 64'h0000_0000_5100_0000;
    wire        w_ready, w_busy, w_done, w_k_ready;
    reg         i_k_valid = 0;
    reg  [127:0] i_k_data = 0;
    wire        w_beat_valid;
    wire [63:0] w_beat_address;
    wire [127:0] w_beat_data;
    wire [6:0]  w_beat_tag;
    wire        w_beat_desc;
    wire [4:0]  w_beat_len, w_beat_token;
    wire        w_err_seq;
    reg         i_flush = 0;
    wire        w_flush_done;
    reg         i_beat_ready = 1;
    reg         i_bank_wr = 0, i_bank_rd = 0;
    reg         i_wr_acquire = 0, i_wr_release = 0, i_rd_release = 0;
    wire        w_wr_grant, w_rd_grant, w_owner_error;
    reg         i_etab_valid = 0, i_load_valid = 0, i_rd_en = 0;
    reg  [2:0]  i_etab_word = 0;
    reg  [127:0] i_etab_data = 0;
    reg  [15:0] i_etab_be = 16'hffff;
    reg  [11:0] i_load_word = 0, i_rd_word = 0;
    reg  [63:0] i_load_mant = 0;
    wire        w_rd_valid;
    wire [127:0] w_rd_bf16;
    reg         i_v_we = 0;
    reg  [3:0]  i_v_word = 0, i_v_rd_word = 0;
    reg  [127:0] i_v_data = 0;
    wire [127:0] w_v_rd_data;
    wire        w_app_conflict, w_clip_sticky;
    wire [31:0] w_stat_beat, w_stat_app;
    wire [3:0]  w_dbg_state;
    wire [1:0]  w_dbg_lock_v;
    wire [15:0] w_clip_count;

    kv8_k_write_chain #(.CONTEXT_LOG2(14), .LAYERS(24)) writer (
        .clk(clk), .rst_n(rst_n),
        .i_start(i_start), .i_layer(LAYER[4:0]), .i_token(i_token),
        .i_k_base(i_k_base), .i_k_scale_base(i_k_scale_base), .i_v_scale_base(i_v_scale_base),
        .o_ready(w_ready),
        .i_k_valid(i_k_valid), .o_k_ready(w_k_ready), .i_k_data(i_k_data),
        .o_busy(w_busy), .o_done(w_done),
        .i_flush(i_flush), .o_flush_done(w_flush_done),
        .o_beat_valid(w_beat_valid), .i_beat_ready(i_beat_ready),
        .o_beat_address(w_beat_address), .o_beat_data(w_beat_data), .o_beat_tag(w_beat_tag),
        .o_beat_desc(w_beat_desc), .o_beat_len(w_beat_len), .o_beat_token(w_beat_token),
        .o_err_seq(w_err_seq),
        .i_bank_wr(i_bank_wr), .i_bank_rd(i_bank_rd),
        .i_wr_acquire(i_wr_acquire), .i_wr_release(i_wr_release), .i_rd_release(i_rd_release),
        .o_wr_grant(w_wr_grant), .o_rd_grant(w_rd_grant), .o_owner_error(w_owner_error),
        .i_etab_valid(i_etab_valid), .i_etab_word(i_etab_word), .i_etab_data(i_etab_data),
        .i_etab_be(i_etab_be),
        .i_load_valid(i_load_valid), .i_load_word(i_load_word), .i_load_mant(i_load_mant),
        .i_rd_en(i_rd_en), .i_rd_word(i_rd_word), .o_rd_valid(w_rd_valid), .o_rd_bf16(w_rd_bf16),
        .i_v_we(i_v_we), .i_v_word(i_v_word), .i_v_data(i_v_data),
        .i_v_rd_word(i_v_rd_word), .o_v_rd_data(w_v_rd_data),
        .o_app_conflict(w_app_conflict), .o_clip_sticky(w_clip_sticky),
        .o_stat_beat(w_stat_beat), .o_stat_app(w_stat_app),
        .o_dbg_state(w_dbg_state), .o_dbg_lock_v(w_dbg_lock_v), .o_clip_count(w_clip_count)
    );

    // ------------------------------------------------- DUT: packer + real bank
    // One channel per KV head: the K^T bank holds a single KV head at a time
    // (row = dim, no head bit) -- exactly how qwen25_attention.v uses it.
    reg  [1:0]   p_beat_valid;
    wire  [1:0]  p_beat_ready;
    reg  [16:0]  p_beat_index [0:1];
    reg  [127:0] p_beat_data  [0:1];
    wire  [1:0]  p_wr_valid;
    wire [9:0]   p_wr_word  [0:1];
    wire [255:0] p_wr_data  [0:1];
    wire  [1:0]  p_err_head, p_err_seq, p_busy;
    wire  [1:0]  b_load_ready;
    wire  [1:0]  b_valid;
    wire [1023:0] b_q [0:1];
    reg         b_rd_en = 0;
    reg  [7:0]  b_rd_word = 0;
    reg         b_fill_done = 0;

    genvar ch;
    generate
        for (ch=0; ch<2; ch=ch+1) begin : gen_ch
            localparam integer CH_ID = ch;
            kv8_fill_packer #(.MODE_KV(0)) pack (
                .i_clk(clk), .i_rstn(rst_n),
                .i_beat_valid(p_beat_valid[CH_ID]), .o_beat_ready(p_beat_ready[CH_ID]),
                .i_beat_index(p_beat_index[CH_ID]), .i_beat_data(p_beat_data[CH_ID]),
                .i_head_sel(ch[0]), .i_bank(1'b0),
                .o_wr_valid(p_wr_valid[CH_ID]), .i_wr_ready(b_load_ready[CH_ID]),
                .o_wr_bank(), .o_wr_word(p_wr_word[CH_ID]), .o_wr_data(p_wr_data[CH_ID]),
                .o_err_head(p_err_head[CH_ID]), .o_err_seq(p_err_seq[CH_ID]), .o_busy(p_busy[CH_ID])
            );
            qwen25_fixed_weight bank (
                .clk(clk), .rst_n(rst_n),
                .fill_done(b_fill_done && (CH_ID==0)), .fill_bank(1'b0),
                .release_bank(1'b0), .release_en(1'b0), .full(),
                .load_valid(p_wr_valid[CH_ID]), .load_ready(b_load_ready[CH_ID]), .load_bank(1'b0),
                .load_word(p_wr_word[CH_ID]), .load_data(p_wr_data[CH_ID]),
                .direct_valid(4'd0), .direct_ready(), .direct_bank(1'b0),
                .direct_word(32'd0), .direct_data(1024'd0),
                .gemm_rd_en(b_rd_en), .gemm_bank(1'b0), .gemm_word(b_rd_word),
                .gemm_valid(b_valid[CH_ID]), .gemm_q(b_q[CH_ID]),
                .gemm_rd_en_pair(1'b0), .gemm_bank_pair(1'b0), .gemm_word_pair(8'd0),
                .gemm_valid_pair(), .gemm_q_pair()
            );
        end
    endgenerate

    // ------------------------------------------------------------ DDR model
    reg [7:0] ddr [longint];
    integer   n_beats = 0;
    reg [63:0] desc_exp;
    integer   nfail1, nfail2, nfail3;
    integer   n_desc  = 0;
    always @(posedge clk) begin
        if (w_beat_valid && i_beat_ready) begin
            ddr[w_beat_address + 0]  = w_beat_data[7:0];
            ddr[w_beat_address + 1]  = w_beat_data[15:8];
            ddr[w_beat_address + 2]  = w_beat_data[23:16];
            ddr[w_beat_address + 3]  = w_beat_data[31:24];
            ddr[w_beat_address + 4]  = w_beat_data[39:32];
            ddr[w_beat_address + 5]  = w_beat_data[47:40];
            ddr[w_beat_address + 6]  = w_beat_data[55:48];
            ddr[w_beat_address + 7]  = w_beat_data[63:56];
            ddr[w_beat_address + 8]  = w_beat_data[71:64];
            ddr[w_beat_address + 9]  = w_beat_data[79:72];
            ddr[w_beat_address + 10] = w_beat_data[87:80];
            ddr[w_beat_address + 11] = w_beat_data[95:88];
            ddr[w_beat_address + 12] = w_beat_data[103:96];
            ddr[w_beat_address + 13] = w_beat_data[111:104];
            ddr[w_beat_address + 14] = w_beat_data[119:112];
            ddr[w_beat_address + 15] = w_beat_data[127:120];
            n_beats = n_beats + 1;
            // descriptor audit: every 256 B descriptor starts at
            //   layer_base + block*4096 + dg*1024 + head*512 + token*16
            // with token in {0,16} and length = min(16, tokens_left_in_run)
            if (w_beat_desc) begin
                n_desc = n_desc + 1;
                desc_exp = f_layer_base(LAYER)
                         + ((w_beat_address - f_layer_base(LAYER)) >> 12)*4096
                         + {9'd0, w_beat_tag[1:0], 10'd0}      // dim_group*1024
                         + {9'd0, w_beat_tag[2],    9'd0}      // head*512
                         + {9'd0, w_beat_token,     4'd0};     // token*16
                if (w_beat_token !== 5'd0 && w_beat_token !== 5'd16) begin
                    nfail1 = nfail1 + 1;
                    $display("FAIL desc token %0d not on a 16-token boundary", w_beat_token);
                end
                if (w_beat_address !== desc_exp) begin
                    nfail1 = nfail1 + 1;
                    $display("FAIL desc addr %h != %h (tag %b token %0d)", w_beat_address, desc_exp, w_beat_tag, w_beat_token);
                end
            end
        end
    end

    // ------------------------------------------------------------ bookkeeping
    integer errors = 0;
    integer checks = 0;
    integer seed = 32'h9E37_79B9;
    reg [15:0] kref [0:NTOK*2*64-1];
    reg [7:0]  wcount [0:2*64*2-1];      // [head][row*4+lane] bank write coverage

    function [63:0] f_layer_base;
        input integer lay;
        begin
            f_layer_base = i_k_base;
        end
    endfunction

    task automatic wait_cycles(input integer n);
        integer j;
        begin
            for (j=0;j<n;j=j+1) @(posedge clk);
        end
    endtask

    // feed one 16 B beat into packer channel ch
    task automatic feed(input integer chn, input [6:0] tile_token, input [1:0] dg, input [127:0] data);
        begin
            while (!p_beat_ready[chn]) @(negedge clk);
            p_beat_valid[chn] <= 1'b1;
            p_beat_index[chn] <= {7'd0, tile_token, chn[0], dg};
            p_beat_data[chn]  <= data;
            @(posedge clk);
            @(negedge clk);
            p_beat_valid[chn] <= 1'b0;
        end
    endtask

    integer  r, c, d, h, b, t, e, w;
    reg [127:0] beat;
    reg [1023:0] q;
    reg [7:0]  got, want;
    reg [63:0] addr;
    integer    cover_bad, ntok_of_block;

    initial begin
        nfail1 = 0; nfail2 = 0; nfail3 = 0;
        n_beats = 0; n_desc = 0;
        for (r=0;r<2*64*2;r=r+1) wcount[r] = 8'd0;
        for (e=0;e<2;e=e+1) begin
            p_beat_valid[e] = 1'b0; p_beat_index[e] = 17'd0; p_beat_data[e] = 128'd0;
        end

        // ---- stimulus: two magnitude classes per token, so the block exponent
        // lock (and the clip path on the larger tokens) is exercised ----
        for (t=0;t<NTOK;t=t+1) begin
            for (h=0;h<2;h=h+1) begin
                for (d=0;d<64;d=d+1) begin
                    begin : gen_val
                        reg [7:0] ex; reg [6:0] mant; reg s;
                        ex   = (t%3==0) ? 8'd131 : 8'd133;
                        if (d%7==0) ex = ex + 8'd2;
                        mant = $random(seed) & 7'h7f;
                        s    = $random(seed) & 1;
                        kref[t*2*64 + h*64 + d] = {s,ex,mant};
                    end
                end
            end
        end

        repeat (10) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);

        // scale table: bank 0 LOADING -> READY (appends need one of the two)
        i_bank_wr = 0;
        @(posedge clk); i_wr_acquire <= 1'b1; @(posedge clk); i_wr_acquire <= 1'b0;
        @(posedge clk); i_wr_release <= 1'b1; @(posedge clk); i_wr_release <= 1'b0;
        wait_cycles(4);

        // ---------------------------------------------------------- PHASE 0
        $display("--- PHASE0: stream %0d tokens (16 BF16 beats each) into the writer ---", NTOK);
        for (t=0;t<NTOK;t=t+1) begin
            i_token = t[15:0];
            @(posedge clk);
            while (!w_ready) @(posedge clk);
            i_start <= 1'b1;
            @(posedge clk);
            i_start <= 1'b0;
            for (h=0;h<2;h=h+1) begin
                for (b=0;b<8;b=b+1) begin
                    beat = 128'd0;
                    for (c=0;c<8;c=c+1) beat[c*16 +: 16] = kref[t*2*64 + h*64 + b*8 + c];
                    i_k_valid <= 1'b1;
                    i_k_data  <= beat;
                    @(posedge clk);
                    while (!w_k_ready) @(posedge clk);
                end
            end
            i_k_valid <= 1'b0;
            while (!w_done) @(posedge clk);
            @(posedge clk);
        end
        // the partial block (8 tokens) must be flushed explicitly
        @(posedge clk); i_flush <= 1'b1; @(posedge clk); i_flush <= 1'b0;
        for (w=0; w<4000 && n_beats < NTOK*8; w=w+1) @(posedge clk);
        wait_cycles(20);
        $display("  writer: clip_count=%0d app_conflict=%b beats=%0d (expect %0d) descriptors=%0d (expect 24) err_seq=%b stat_beat=%0d",
                 w_clip_count, w_app_conflict, n_beats, NTOK*8, n_desc, w_err_seq, w_stat_beat);
        if (n_beats != NTOK*8) begin nfail1=nfail1+1; $display("FAIL beat count"); end
        // block 0 (32 tokens) = 16 x 256 B descriptors, block 1 (8 tokens) = 8
        if (n_desc  != 24)     begin nfail1=nfail1+1; $display("FAIL descriptor count"); end
        if (w_err_seq)         begin nfail1=nfail1+1; $display("FAIL writer err_seq"); end

        // ---------------------------------------------------------- PHASE 1
        // write completeness of the block32 layout
        $display("--- PHASE1: block32 layout write completeness ---");
        checks = 0;
        for (b=0;b<NBLOCK;b=b+1) begin
            ntok_of_block = (b==0) ? 32 : NTOK1;
            for (d=0;d<4;d=d+1) begin
                for (h=0;h<2;h=h+1) begin
                    for (t=0;t<ntok_of_block;t=t+1) begin
                        addr = f_layer_base(LAYER) + b*4096 + d*1024 + h*512 + t*16;
                        for (c=0;c<16;c=c+1) begin
                            checks = checks + 1;
                            if (ddr[addr+c] === 8'hxx) begin
                                nfail1 = nfail1 + 1;
                                if (nfail1 < 10)
                                    $display("FAIL unwritten byte block=%0d dg=%0d h=%0d t=%0d s=%0d",
                                             b, d, h, t, c);
                            end
                        end
                    end
                end
            end
        end
        $display("PHASE1 block bytes: checked=%0d fail=%0d", checks, nfail1);

        // ---------------------------------------------------------- PHASE 2
        $display("--- PHASE2: packer fill from that linear read + bank read-back ---");
        for (b=0;b<NBLOCK;b=b+1) begin
            for (d=0; d<4; d=d+1) begin        // dim_group
                for (h=0; h<2; h=h+1) begin    // KV head
                    for (t=0; t<32; t=t+1) begin
                        // 32-token alignment: the tail of a partial block is
                        // zero-padded (what the attention FSM does past context)
                        if (b==0 || t < NTOK1) begin
                            addr = f_layer_base(LAYER) + b*4096 + d*1024 + h*512 + t*16;
                            beat = 128'd0;
                            for (c=0;c<16;c=c+1) beat[c*8 +: 8] = ddr[addr + c];
                        end else begin
                            beat = 128'd0;
                        end
                        feed(h, {b[2:0],t[4:0]}, d[1:0], beat);
                    end
                end
            end
        end
        for (w=0; w<200 && (p_busy[0]||p_busy[1]); w=w+1) @(posedge clk);
        wait_cycles(10);
        checks = checks + 2;
        if (p_err_seq[0] || p_err_seq[1]) begin
            nfail2 = nfail2 + 1; $display("FAIL packer o_err_seq (beat order contract violated)");
        end
        if (p_err_head[0] || p_err_head[1]) begin
            nfail2 = nfail2 + 1; $display("FAIL packer o_err_head");
        end

        // ---- bank read-back: row = dim, column = token ----
        checks = 0;
        for (h=0; h<2; h=h+1) begin
            for (d=0; d<64; d=d+1) begin
                @(negedge clk);
                b_rd_en   <= 1'b1;
                b_rd_word <= {1'b0, d[5:0], 1'b0};      // {row, half=0}
                @(posedge clk);
                @(negedge clk);
                b_rd_en <= 1'b0;
                @(posedge clk);
                q = b_q[h];
                for (t=0; t<TILE_TOK; t=t+1) begin
                    if (t < 32 || (t-32) < NTOK1) begin
                        addr = f_layer_base(LAYER) + (t/32)*4096 + (d/16)*1024 + h*512
                             + (t%32)*16 + (d%16);
                        want = ddr[addr];
                    end else begin
                        want = 8'h00;               // zero-padded tail of block 1
                    end
                    got = q[t*8 +: 8];
                    checks = checks + 1;
                    if (got !== want) begin
                        nfail2 = nfail2 + 1;
                        if (nfail2 < 12)
                            $display("FAIL bank h=%0d dim=%0d tok=%0d got=%02h want=%02h",
                                     h, d, t, got, want);
                    end
                end
            end
        end
        $display("PHASE2 bank bytes: checked=%0d fail=%0d (packer err_seq=%b err_head=%b)",
                 checks, nfail2, p_err_seq[0]|p_err_seq[1], p_err_head[0]|p_err_head[1]);

        // ---------------------------------------------------------- PHASE 3
        $display("--- PHASE3: write bijection (64 rows x 2 lanes) ---");
        cover_bad = 0;
        // one bijection per KV head channel: index = head*128 + row*4 + lane
        for (h=0;h<2;h=h+1) begin
            for (e=0;e<64;e=e+1) begin
                for (c=0;c<2;c=c+1) begin
                    r = h*128 + e*2 + c;
                    if (wcount[r] !== 8'd1) begin
                        cover_bad = cover_bad + 1;
                        if (cover_bad < 300)
                            $display("COV head %0d row %0d lane %0d count %0d", h, e, c, wcount[r]);
                    end
                end
            end
        end
        checks = checks + 1;
        if (cover_bad != 0) nfail3 = nfail3 + 1;
        $display("PHASE3 coverage: bad words=%0d", cover_bad);
        for (e=0;e<64;e=e+1)
            $display("  MATRIX row %0d : lane0=%0d lane1=%0d", e,
                     wcount[0*128+e*2+0], wcount[0*128+e*2+1]);

        wait_cycles(10);
        if (nfail1+nfail2+nfail3 == 0)
            $display("CONTRACT PASS: block-bytes=%0d bank-bytes=%0d coverage=128/128", NBLOCK*512, checks);
        else
            $display("CONTRACT FAIL: phase1=%0d phase2=%0d phase3=%0d", nfail1, nfail2, nfail3);
        $finish;
    end

    // (row,lane) write coverage, decoded from the packer's write port
    integer dbg_n = 0;
    always @(posedge clk) begin
        for (e=0;e<2;e=e+1) begin
            if (p_wr_valid[e] && b_load_ready[e]) begin
                dbg_n = dbg_n + 1;
                if (p_wr_word[e][9] !== 1'b0) begin
                    nfail3 = nfail3 + 1; $display("FAIL load_word[9] set");
                end
                if (p_wr_word[e][2] !== 1'b0) begin
                    nfail3 = nfail3 + 1; $display("FAIL half != 0");
                end
                wcount[e*128 + {p_wr_word[e][8:3], 1'b0} + p_wr_word[e][1:0]] =
                    wcount[e*128 + {p_wr_word[e][8:3], 1'b0} + p_wr_word[e][1:0]] + 8'd1;
            end
        end
    end

    // watchdog
    initial begin
        #3000000;
        $display("CONTRACT TIMEOUT writer_state=%0d busy=%b n_beats=%0d p_busy=%b%b n_desc=%0d",
                 w_dbg_state, w_busy, n_beats, p_busy[1], p_busy[0], n_desc);
        $finish;
    end
endmodule
`default_nettype wire
