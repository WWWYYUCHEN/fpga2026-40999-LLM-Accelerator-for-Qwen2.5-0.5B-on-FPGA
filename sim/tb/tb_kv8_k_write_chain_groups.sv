`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_k_write_chain_groups -- the flush-at-row-group-boundary contract
//
// The E2E prefill feeds the K producer one row group at a time (8 tokens) and the
// attention of that group runs before the next group is projected.  The producer
// must therefore copy out what it has on every flush, at ABSOLUTE token offsets
// inside the 32-token block, and keep filling the same block afterwards.
//
//   RUN 0 (reference) : 30 tokens, one flush at the end
//   RUN 1 (E2E shape) : 30 tokens in groups 8/8/8/6, one flush after each group
//
// Proves:
//   * every beat of RUN 1 lands on the address RUN 0 produced (same bytes, same
//     place), so the group flush re-uses the block instead of restarting it
//   * after each group flush every token of that group is readable in DDR (the
//     attention of that group can run) -- no token is left in the buffer
//   * the descriptor stream stays 256 B / (dim_group, head, 16-token) runs with the
//     right length, and the beat/descriptor counts match the flush schedule
//   * o_err_seq stays 0 (the token stream still ascends across the flushes)
// =============================================================================
module tb_kv8_k_write_chain_groups;
    localparam integer NTOK = 30;
    localparam integer GROUPS = 4;
    localparam integer glen [0:3] = '{8, 16, 24, 30};   // cumulative tokens per flush

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

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
    wire [4:0]  o_blk_tok;
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
    wire [135:0] o_rd_scale;   // [M16-SCALE] {eb_tab, 8 x m16}
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

    kv8_k_write_chain #(.CONTEXT_LOG2(14), .LAYERS(24)) dut (
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
        // [TB-HARNESS] see tb_kv8_k_write_chain.sv: the V bulk-load / V append /
        // etab-dump ports postdate this TB and were dangling (X inside
        // kv8_scale_ram -> o_app_taken X -> hang in S_APP).  Not driven == 0.
        .i_vload_valid(1'b0), .i_vload_word(12'd0), .i_vload_data(128'd0),
        .i_etab_rd_word(3'd0),
        .i_vt_app_valid(1'b0), .i_vt_app_token(16'd0), .i_vt_app_head(1'b0), .i_vt_app_mant(16'd0),
        .i_vt_rd_en(1'b0), .i_vt_rd_word(12'd0),
        .i_rd_en(i_rd_en), .i_rd_word(i_rd_word), .o_rd_valid(o_rd_valid), .o_rd_scale(o_rd_scale),
        .i_v_we(i_v_we), .i_v_word(i_v_word), .i_v_data(i_v_data),
        .i_v_rd_word(i_v_rd_word), .o_v_rd_data(o_v_rd_data),
        .o_app_conflict(o_app_conflict), .o_clip_sticky(o_clip_sticky),
        .o_stat_beat(o_stat_beat), .o_stat_app(o_stat_app),
        .o_dbg_state(o_dbg_state), .o_dbg_lock_v(o_dbg_lock_v), .o_clip_count(o_clip_count),
        .o_blk_tok(o_blk_tok)
    );

    // ---- stimulus data: BF16 K per (token, head, dim) ----------------------
    reg [15:0] kref [0:NTOK*2*64-1];
    integer seed = 32'h0BAD_F00D;
    integer t, hd, d;
    reg [7:0] e_tmp; reg [6:0] m_tmp; reg s_tmp;

    // ---- capture / check state --------------------------------------------
    reg [7:0]  img0 [longint];      // run 0 image
    reg [7:0]  img1 [longint];      // run 1 image
    integer    beats0, beats1, descs0, descs1, checks, errors;
    integer    pdesc, plen, ptok, perrs, gerr;
    reg [63:0] paddr;
    integer    nflush;              // flushes seen

    reg        present [0:NTOK*2*8-1];   // (token,head,dg) has been written at least once
    integer    cov_err;

    task automatic chk(input integer cond, input [255:0] msg);
        begin
            checks = checks + 1;
            if (!cond) begin
                errors = errors + 1;
                if (errors < 16) $display("FAIL %0s", msg);
            end
        end
    endtask

    // ---- beat capture + descriptor audit ----------------------------------
    // Expectation for descriptor j of a flush that carries N tokens:
    //   N<=16 : j=0..7  -> grp=j, tok0=0,      len=N
    //   N> 16 : j=0..15 -> grp=j/2, tok0=(j%2)*16, len=min(16, N-tok0)
    integer flush_n, dj, grp, dg, hd_e, tok0, len_e;
    reg [63:0] addr_e;
    reg        run1;
    integer    ndbg;      // print guard of the missing-coverage diagnostic
    always @(posedge clk) begin
        if (rst_n && o_beat_valid && i_beat_ready) begin
            if (run1) begin
                beats1 = beats1 + 1;
                img1[o_beat_address] = o_beat_data[7:0];
                for (integer b=0;b<16;b=b+1) img1[o_beat_address + b] = o_beat_data[b*8 +: 8];
                if (o_beat_desc) begin
                    descs1 = descs1 + 1;
                    // a run of >16 tokens is split at token 16 (256 B per descriptor)
                    if ((flush_n > 16) ? (o_beat_token !== 5'd0 && o_beat_token !== 5'd16)
                                       : (o_beat_token !== 5'd0)) begin
                        gerr=gerr+1; $display("FAIL desc tok0=%0d", o_beat_token);
                    end
                end else if (o_beat_address !== paddr + 64'd16) begin
                    gerr = gerr + 1; $display("FAIL beat not contiguous %h -> %h", paddr, o_beat_address);
                end
                paddr = o_beat_address;
                // coverage: which (token, head, dim_group) has been written
                for (integer tt=0;tt<NTOK;tt=tt+1)
                    for (integer hh=0;hh<2;hh=hh+1)
                        for (integer gg=0;gg<8;gg=gg+1)
                            if (o_beat_address == i_k_base + (tt/32)*8192 + gg*1024 + hh*512 + (tt%32)*16)
                                present[(tt*2+hh)*8+gg] = 1'b1;
            end else begin
                beats0 = beats0 + 1;
                if (o_beat_desc) descs0 = descs0 + 1;
                for (integer b=0;b<16;b=b+1) img0[o_beat_address + b] = o_beat_data[b*8 +: 8];
            end
        end
    end

    // ---- helpers -----------------------------------------------------------
    // one beat is offered only when the producer is already ready, and it is
    // withdrawn immediately after the accepting edge: no stale beat can leak into
    // the next head / the next token.
    task automatic stream_token(input integer tt);
        integer h, b, lane;
        reg [127:0] beat;
        begin
            i_token = tt[15:0];
            @(negedge clk);
            while (!o_ready) @(negedge clk);
            i_start <= 1'b1;
            @(negedge clk);
            i_start <= 1'b0;
            for (h=0; h<2; h=h+1) begin
                for (b=0; b<8; b=b+1) begin
                    beat = 128'd0;
                    for (lane=0; lane<8; lane=lane+1)
                        beat[lane*16 +: 16] = kref[(tt*2+h)*64 + b*8 + lane];
                    i_k_data <= beat;
                    @(negedge clk);
                    while (!o_k_ready) @(negedge clk);
                    i_k_valid <= 1'b1;
                    @(posedge clk);              // transfer
                    @(negedge clk);
                    i_k_valid <= 1'b0;
                end
            end
            i_k_valid <= 1'b0;
            while (!o_done) @(negedge clk);
            @(negedge clk);
        end
    endtask

    reg flush_seen;
    always @(posedge clk) if (o_flush_done) flush_seen <= 1'b1;
    task automatic do_flush(input integer ntok);
        begin
            flush_seen = 1'b0;
            flush_n = ntok;
            @(negedge clk);
            while (!o_ready) @(negedge clk);
            i_flush <= 1'b1;
            @(negedge clk);
            i_flush <= 1'b0;
            while (!flush_seen) @(posedge clk);
            repeat (6) @(posedge clk);
        end
    endtask

    integer n, i0, g, k, nfail_cov;
    initial begin
        errors = 0; checks = 0; beats0=0; beats1=0; descs0=0; descs1=0; gerr=0;
        for (i0=0;i0<NTOK*2*8;i0=i0+1) present[i0] = 1'b0;

        // ---- stimulus ------------------------------------------------------
        for (t=0; t<NTOK; t=t+1)
            for (hd=0; hd<2; hd=hd+1)
                for (d=0; d<64; d=d+1) begin
                    e_tmp = 8'd130 + ((t/10) % 3) + ((d%7==0) ? 8'd1 : 8'd0);
                    m_tmp = $random(seed) & 7'h7f;
                    s_tmp = $random(seed) & 1;
                    kref[(t*2+hd)*64 + d] = {s_tmp, e_tmp, m_tmp};
                end

        // ===================== RUN 0: one flush at the end ==================
        run1 = 1'b0;
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);
        i_bank_wr = 0;
        @(negedge clk); i_wr_acquire <= 1'b1; @(negedge clk); i_wr_acquire <= 1'b0;
        @(negedge clk);
        chk(o_wr_grant === 1'b1, "bank acquire grants the writer");
        for (t=0; t<NTOK; t=t+1) stream_token(t);
        do_flush(NTOK);
        chk(o_err_seq === 1'b0, "RUN0: ascending token stream");
        $display("RUN0 (single flush): beats=%0d descs=%0d image_bytes=%0d",
                 beats0, descs0, beats0*16);
        chk(beats0 == NTOK*16, "RUN0: 16 beats per token");
        chk(descs0 == 32, "RUN0: 32 descriptors for a 30-token block");

        // ===================== RUN 1: flush per group =======================
        // fresh reset (the E2E starts a new layer the same way)
        run1 = 1'b1;
        rst_n = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (5) @(posedge clk);
        @(negedge clk); i_wr_acquire <= 1'b1; @(negedge clk); i_wr_acquire <= 1'b0;
        @(negedge clk);
        for (g=0; g<GROUPS; g=g+1) begin
            for (t = (g==0 ? 0 : glen[g-1]); t < glen[g]; t=t+1) stream_token(t);
            // the flush must make every token of this group readable
            do_flush(glen[g]);
            nfail_cov = 0;
            for (k=0; k<glen[g]; k=k+1)
                for (hd=0; hd<2; hd=hd+1)
                    for (d=0; d<8; d=d+1)
                        if (!present[(k*2+hd)*8+d]) nfail_cov = nfail_cov + 1;
            if (nfail_cov != 0) begin
                ndbg = 0;
                for (k=0; k<glen[g]; k=k+1)
                    for (hd=0; hd<2; hd=hd+1)
                        for (d=0; d<8; d=d+1)
                            if (!present[(k*2+hd)*8+d] && ndbg < 4) begin
                                $display("MISSING k=%0d hd=%0d dg=%0d expect %h",
                                         k, hd, d,
                                         i_k_base + (k/32)*8192 + d*1024 + hd*512 + (k%32)*16);
                                ndbg = ndbg + 1;
                            end
            end
            chk(nfail_cov == 0, "after this group's flush all its tokens are in DDR");
            $display("  group %0d: tokens 0..%0d flushed, beats so far=%0d descs=%0d",
                     g, glen[g]-1, beats1, descs1);
        end
        chk(o_err_seq === 1'b0, "RUN1: ascending token stream across flushes");
        chk(gerr == 0, "RUN1: descriptor shape (256 B runs, contiguous beats)");

        // ---- the group run must reproduce the single-flush image exactly ---
        // 8+16+24+30 = 78 token copies x 16 beats
        chk(beats1 == (8+16+24+30)*16, "RUN1: one copy-out per flush, no token lost");
        chk(descs1 == 16+16+32+32, "RUN1: descriptor count follows the flush schedule");
        nfail_cov = 0;
        for (t=0; t<NTOK; t=t+1)
            for (hd=0; hd<2; hd=hd+1)
                for (d=0; d<64; d=d+1) begin
                    integer dg2; integer db2; reg [63:0] ea; integer by;
                    dg2 = d/8; db2 = d%8;              // K16: 8 values of 2 B per beat
                    ea = i_k_base + (t/32)*8192 + dg2*1024 + hd*512 + (t%32)*16 + db2*2;
                    for (by=0; by<2; by=by+1)
                        if (img1[ea+by] !== img0[ea+by]) begin
                            nfail_cov = nfail_cov + 1;
                            if (nfail_cov < 8) $display("FAIL img t=%0d h=%0d d=%0d addr=%h g=%02h r=%02h",
                                                        t, hd, d, ea+by, img1[ea+by], img0[ea+by]);
                        end
                end
        chk(nfail_cov == 0, "RUN1 byte image == RUN0 byte image (absolute offsets)");
        $display("RUN1 (per-group flush): beats=%0d descs=%0d image_bytes=%0d",
                 beats1, descs1, beats1*16);

        if (errors == 0) $display("GROUPS PASS: checks=%0d", checks);
        else             $display("GROUPS FAIL: checks=%0d errors=%0d", checks, errors);
        $finish;
    end

    initial begin
        #8000000; $display("GROUPS TIMEOUT"); $finish;
    end
endmodule
`default_nettype wire
