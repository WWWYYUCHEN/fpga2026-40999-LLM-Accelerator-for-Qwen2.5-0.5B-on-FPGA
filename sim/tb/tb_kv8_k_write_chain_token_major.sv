`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_k_write_chain_token_major -- the emission contract of kv8_k_write_chain
// in TOKEN_MAJOR=1 mode, which is the mode the E2E runs and which NO other TB
// instantiates (every other K-chain TB takes TOKEN_MAJOR's default of 0).
//
// Why this exists
//   The E2E deadlocks with an orphan descriptor: a 128 B write command accepted,
//   awout=1, and 0 of its 8 W beats ever sent (wleft=8).  If this chain ever
//   emits 7 or 9 beats for a token, puts o_beat_desc on the wrong beat, or drops
//   a beat while the downstream packer back-pressures, the FSM's beat count and
//   the command's byte count disagree and the shared write engine is wedged
//   exactly that way.  So this TB counts beats itself, per token, off the
//   valid/ready handshake, and never trusts the DUT's own counters.
//
// Contract asserted (per 256 B token run, TOKEN_MAJOR=1, frozen K16 format):
//   * exactly 16 accepted beats per token, every o_beat_len == 16
//   * o_beat_desc on the offset-0 beat only, never mid-token
//   * address == k_base + token*256 + head*128 + dg*16, i.e. 16 distinct
//     offsets 0,16,...,240 inside the token's 256 B run, ascending
//   * o_beat_token == o_beat_tag == offset/16 (the beat index within the token)
//   * tokens ascend, no beat carries the next token's base before the previous
//     token's 16 beats are complete
//   * all of the above still holds with i_beat_ready back-pressuring, across the
//     E2E's 8/8/8/6 row-group boundaries, and with i_flush asserted at a
//     boundary (TOKEN_MAJOR=1 defines flush as a no-op; assert it, require no
//     disturbance).
// =============================================================================
module tb_kv8_k_write_chain_token_major;

    localparam integer NTOK     = 30;          // 8 + 8 + 8 + 6, the E2E's prefill
    localparam integer CONTEXT_LOG2 = 14;
    localparam integer LAYERS   = 24;
    localparam [63:0]  K_BASE   = 64'h0000_0000_1000_0000;
    localparam [15:0]  BF16_ONE = 16'h3F80;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------------ DUT ports
    reg         i_start = 0;
    reg  [4:0]  i_layer = 0;
    reg  [15:0] i_token = 0;
    reg  [63:0] i_k_base = K_BASE;
    reg  [63:0] i_k_scale_base = 64'h0000_0000_1000_8000;
    reg  [63:0] i_v_scale_base = 64'h0000_0000_1000_8000;
    wire        o_ready;
    reg         i_k_valid = 0;
    wire        o_k_ready;
    reg  [127:0] i_k_data = 0;
    wire        o_busy, o_done;
    reg         i_flush = 0;
    wire        o_flush_done;
    wire        o_beat_valid;
    reg         i_beat_ready = 1;
    wire [63:0] o_beat_address;
    wire [127:0] o_beat_data;
    wire [6:0]  o_beat_tag;
    wire        o_beat_desc;
    wire [4:0]  o_beat_len;
    wire [4:0]  o_beat_token;
    wire        o_err_seq;
    reg         i_bank_wr = 1, i_bank_rd = 0;
    reg         i_wr_acquire = 0, i_wr_release = 0, i_rd_release = 0;
    wire        o_wr_grant, o_rd_grant, o_owner_error;
    reg         i_etab_valid = 0;
    reg  [2:0]  i_etab_word = 0;
    reg  [127:0] i_etab_data = 0;
    reg  [15:0] i_etab_be = 16'hffff;
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
    wire [4:0]  o_blk_tok;

    kv8_k_write_chain #(.CONTEXT_LOG2(CONTEXT_LOG2), .LAYERS(LAYERS), .MARGIN(2),
                        .BLOCK_LOG2(8), .TOKEN_MAJOR(1))
    dut (
        .clk(clk), .rst_n(rst_n),
        .i_start(i_start), .i_layer(i_layer), .i_token(i_token),
        .i_k_base(i_k_base), .i_k_scale_base(i_k_scale_base), .i_v_scale_base(i_v_scale_base),
        .o_ready(o_ready),
        .i_k_valid(i_k_valid), .o_k_ready(o_k_ready), .i_k_data(i_k_data),
        .o_busy(o_busy), .o_done(o_done),
        .i_flush(i_flush), .o_flush_done(o_flush_done),
        .o_beat_valid(o_beat_valid), .i_beat_ready(i_beat_ready),
        .o_beat_address(o_beat_address), .o_beat_data(o_beat_data), .o_beat_tag(o_beat_tag),
        .o_beat_desc(o_beat_desc), .o_beat_len(o_beat_len), .o_beat_token(o_beat_token),
        .o_err_seq(o_err_seq),
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
        .o_dbg_state(o_dbg_state), .o_dbg_lock_v(o_dbg_lock_v),
        .o_clip_count(o_clip_count), .o_blk_tok(o_blk_tok)
    );

    // ------------------------------------------------------- back-pressure on the
    // beat stream: stall 3 out of every 7 cycles (deterministic, so a failure is
    // reproducible), which is the condition under which the E2E live-stalls.
    reg  [31:0] cycle_count = 0;
    reg         bp_enable = 0;
    always @(posedge clk) cycle_count <= cycle_count + 32'd1;
    always @(*) i_beat_ready = !bp_enable || (cycle_count % 32'd7) > 32'd2;

    // ------------------------------------------------------------- beat capture
    localparam integer MAXB = 4096;
    reg [63:0]  cap_addr [0:MAXB-1];
    reg [127:0] cap_data [0:MAXB-1];
    reg [4:0]   cap_len  [0:MAXB-1];
    reg [4:0]   cap_tokn [0:MAXB-1];
    reg [6:0]   cap_tag  [0:MAXB-1];
    reg         cap_desc [0:MAXB-1];
    integer     ncap = 0;

    always @(posedge clk) begin
        if (rst_n && o_beat_valid && i_beat_ready) begin
            if (ncap < MAXB) begin
                cap_addr[ncap] = o_beat_address;
                cap_data[ncap] = o_beat_data;
                cap_len [ncap] = o_beat_len;
                cap_tokn[ncap] = o_beat_token;
                cap_tag [ncap] = o_beat_tag;
                cap_desc[ncap] = o_beat_desc;
            end
            ncap = ncap + 1;
        end
    end

    // ---------------------------------------------------------------- driver
    integer i, b, errors = 0;
    reg [15:0] beat_data;

    // If 1, wait until the chain AND the packer are empty before starting the next
    // token.  The E2E does NOT do this (it only waits for o_ready = S_IDLE), so
    // running with 0 is the E2E's actual contract; 1 isolates the mechanism.
    reg drain_before_start = 1;

    // ------------------------------------------------ windowed internal trace
    // Only tokens 6 and 7: the last good token and the first bad one.
    reg trace_en = 1'b1;
    always @(posedge clk) begin
        if (rst_n && trace_en &&
            ((dut.token_r == 16'd0) || (dut.token_r == 16'd1)) &&
            (dut.state != 4'd0)) begin
            $display("TRK t=%0d st=%0d head=%b beat=%0d tog=%b qv=%b ql=%b p1dg=%0d | qqv=%b pkv=%b pkr=%b ov=%b addr=%h",
                     dut.token_r, dut.state, dut.head, dut.beat, dut.p1_tog,
                     dut.q_valid, dut.q_last, dut.p1_dg,
                     dut.q_qv, dut.pk_valid, dut.pk_ready,
                     dut.pack2_Inst.out_valid,
                     o_beat_address);
        end
    end

    task automatic do_token(input [15:0] tok);
        integer t;
        begin
            // wait for the chain to be idle, then pulse i_start for exactly 1 cycle
            t = 0;
            while (!o_ready && t < 20000) begin @(negedge clk); t = t + 1; end
            if (drain_before_start) begin
                t = 0;
                while ((o_busy || o_beat_valid) && t < 20000) begin @(negedge clk); t = t + 1; end
            end            if (t >= 20000) begin
                $display("FAIL: chain never became ready before token %0d (dbg_state=%0d busy=%b done=%b ncap=%0d blk_tok=%0d tok_line=%0d)",
                         tok, o_dbg_state, o_busy, o_done, ncap, o_blk_tok, o_beat_token);
                $finish;
            end
            @(negedge clk);
            i_token = tok; i_start = 1'b1;
            @(negedge clk);
            i_start = 1'b0;
            // 16 BF16 beats = 2 heads x 8 beats of 8 dims, every dim = 1.0
            beat_data = {8{BF16_ONE}};
            for (b = 0; b < 16; b = b + 1) begin
                @(negedge clk);
                i_k_valid = 1'b1;
                i_k_data  = {8{beat_data}};
                @(posedge clk);
                t = 0;
                while (!o_k_ready && t < 10000) begin @(posedge clk); t = t + 1; end
                if (t >= 10000) begin $display("FAIL: o_k_ready never came, token %0d beat %0d", tok, b); $finish; end
                @(negedge clk);
                i_k_valid = 1'b0;
            end
            $display("  [tok] %0d ncap=%0d stat=%0d", tok, ncap, o_stat_beat);
        end
    endtask

    // ------------------------------------------------------------- test body
    integer k, n, t_start;
    integer beats_of_token [0:NTOK-1];
    integer offs_seen  [0:NTOK-1][0:15];  // 1 = that 16 B offset was seen
    integer base_of_token [0:NTOK-1];
    reg     first_off_zero [0:NTOK-1];
    reg     desc_ok        [0:NTOK-1];
    reg     len_ok         [0:NTOK-1];
    reg     tokfld_ok      [0:NTOK-1];
    integer tokidx;
    integer off;

    initial begin
        $display("=== tb_kv8_k_write_chain_token_major : TOKEN_MAJOR=1 emission contract ===");
        i_start = 0; i_k_valid = 0; i_k_data = 0; i_flush = 0;
        repeat (8) @(negedge clk);
        rst_n = 1'b1;
        repeat (4) @(negedge clk);

        // take write ownership of the scale-RAM bank (EMPTY -> LOADING), otherwise
        // every append is refused and the chain parks in S_APP
        i_bank_wr = 1'b1;
        @(negedge clk); i_wr_acquire = 1'b1;
        @(negedge clk); i_wr_acquire = 1'b0;
        repeat (2) @(negedge clk);
        if (!o_wr_grant) begin
            $display("FAIL: scale-RAM write acquire was not granted (owner_error=%b)", o_owner_error);
            $finish;
        end

        // ---- phase 1: no back-pressure, 8 tokens (a full row group) ----
        bp_enable = 1;
        ncap = 0;
        for (k = 0; k < 8; k = k + 1) do_token(k);
        // ---- phase 2: back-pressure, next 8 tokens ----
        bp_enable = 1;
        for (k = 8; k < 16; k = k + 1) do_token(k);
        // ---- phase 3: flush at the group boundary, then 8 more ----
        @(negedge clk); i_flush = 1'b1;
        @(negedge clk); i_flush = 1'b0;
        repeat (10) @(negedge clk);
        for (k = 16; k < 24; k = k + 1) do_token(k);
        // ---- phase 4: the short tail group (6 tokens), back-pressure still on ----
        @(negedge clk); i_flush = 1'b1;
        @(negedge clk); i_flush = 1'b0;
        for (k = 24; k < NTOK; k = k + 1) do_token(k);
        // let the last beats drain
        // [RECIP-SEQ] 200 -> 2000 cycles: do_token() returns as soon as it has
        // pushed its 16 beats, so this window has to cover the chain finishing
        // the LAST token (two magnitude ceilings, one block-lock ceiling and two
        // restores now run on a 25-cycle iterative divider instead of
        // combinational logic: ~300 cycles per token instead of ~40).  With 200
        // the analysis ran while the last token's second head was still in
        // S_REST and reported its 8 beats as "never emitted".  The checks
        // themselves are unchanged.
        repeat (2000) @(negedge clk);

        // ------------------------------------------------------------ analysis
        for (k = 0; k < NTOK; k = k + 1) begin
            beats_of_token[k] = 0;
            desc_ok[k] = 1'b1; len_ok[k] = 1'b1; tokfld_ok[k] = 1'b1;
            first_off_zero[k] = 1'b0;
            for (n = 0; n < 16; n = n + 1) offs_seen[k][n] = 0;
        end
        $display("captured beats = %0d (expected %0d)", ncap, NTOK*16);
        if (ncap != NTOK*16) begin
            $display("FAIL: beat count %0d != %0d", ncap, NTOK*16);
            errors = errors + 1;
        end
        for (i = 0; i < ncap; i = i + 1) begin
            if (cap_addr[i] < K_BASE) begin
                $display("FAIL beat %0d: address %h below k_base %h", i, cap_addr[i], K_BASE);
                errors = errors + 1;
            end else begin
                tokidx = (cap_addr[i] - K_BASE) / 256;
                off    = (cap_addr[i] - K_BASE) % 256;
                if (tokidx >= NTOK) begin
                    $display("FAIL beat %0d: address %h maps to token %0d >= %0d", i, cap_addr[i], tokidx, NTOK);
                    errors = errors + 1;
                end else begin
                    beats_of_token[tokidx] = beats_of_token[tokidx] + 1;
                    if ((off % 16) != 0) begin
                        $display("FAIL beat %0d: offset %0d inside the token is not a 16 B step", i, off);
                        errors = errors + 1;
                    end else begin
                        if (offs_seen[tokidx][off/16] != 0) begin
                            $display("FAIL beat %0d: token %0d offset %0d seen twice", i, tokidx, off);
                            errors = errors + 1;
                        end
                        offs_seen[tokidx][off/16] = 1;
                        if (off == 0) first_off_zero[tokidx] = 1'b1;
                        if (cap_desc[i] != (off == 0)) begin
                            $display("FAIL beat %0d: token %0d offset %0d o_beat_desc=%b (contract: desc iff offset 0)",
                                     i, tokidx, off, cap_desc[i]);
                            desc_ok[tokidx] = 1'b0;
                            errors = errors + 1;
                        end
                        if (cap_tokn[i] != (off/16)) begin
                            $display("FAIL beat %0d: token %0d offset %0d o_beat_token=%0d (expect %0d)",
                                     i, tokidx, off, cap_tokn[i], off/16);
                            tokfld_ok[tokidx] = 1'b0;
                            errors = errors + 1;
                        end
                        if (cap_tag[i] != {1'b0, (off/16)}) begin
                            $display("FAIL beat %0d: token %0d offset %0d o_beat_tag=%h (expect %0d)",
                                     i, tokidx, off, cap_tag[i], off/16);
                            errors = errors + 1;
                        end
                    end
                end
            end
            if (cap_len[i] != 5'd16) begin
                $display("FAIL beat %0d: o_beat_len=%0d (contract: 16)", i, cap_len[i]);
                len_ok[i/16] = 1'b0;
                errors = errors + 1;
            end
            // data sanity: every 16-bit value of a beat equal and nonzero
            // (constant 1.0 input, so all 8 values of the dim_group quantise the same)
            for (n = 0; n < 8; n = n + 1)
                if (cap_data[i][n*16 +: 16] != cap_data[i][0 +: 16]) begin
                    $display("FAIL beat %0d: value %0d = %04h differs from value 0 = %04h",
                             i, n, cap_data[i][n*16 +: 16], cap_data[i][0 +: 16]);
                    errors = errors + 1;
                    n = 8;
                end
            if (cap_data[i][15:0] == 16'h0000) begin
                $display("FAIL beat %0d: all-zero K16 code for a 1.0 input", i);
                errors = errors + 1;
            end
        end
        // per token: exactly 8 beats, all 8 offsets present, first beat offset 0
        for (k = 0; k < NTOK; k = k + 1) begin
            if (beats_of_token[k] != 16) begin
                $display("FAIL token %0d: %0d beats (contract: exactly 16)", k, beats_of_token[k]);
                errors = errors + 1;
            end
            for (n = 0; n < 16; n = n + 1)
                if (offs_seen[k][n] == 0) begin
                    $display("FAIL token %0d: offset %0d (beat %0d) never emitted", k, n*16, n);
                    errors = errors + 1;
                end
            if (!first_off_zero[k]) begin
                $display("FAIL token %0d: no beat with offset 0 (desc would be missing)", k);
                errors = errors + 1;
            end
        end
        // ordering: tokens ascending, and a token's 8 beats are contiguous
        t_start = -1;
        for (i = 0; i < ncap; i = i + 1) begin
            if (cap_addr[i] >= K_BASE) begin
                tokidx = (cap_addr[i] - K_BASE) / 256;
                if (tokidx != t_start) begin
                    if (tokidx < t_start) begin
                        $display("FAIL beat %0d: token index went backwards (%0d after %0d)", i, tokidx, t_start);
                        errors = errors + 1;
                    end
                    if ((tokidx > t_start + 1) && (t_start >= 0)) begin
                        $display("FAIL: token %0d follows %0d (a token was skipped)", tokidx, t_start);
                        errors = errors + 1;
                    end
                    t_start = tokidx;
                end
            end
        end
        $display("tokens=%0d beats/token=%0d clip=%0d err_seq=%b app_conflict=%b packer_err_pair=%b",
                 NTOK, ncap/NTOK, o_clip_count, o_err_seq, o_app_conflict, dut.pk_err);
        if (o_err_seq) begin
            $display("FAIL: o_err_seq sticky set (token stream not ascending)"); errors = errors + 1;
        end
        if (errors == 0) $display("TOKEN-MAJOR PASS: %0d tokens x 16 beats, desc/len/token/address/tag all exact", NTOK);
        else             $display("TOKEN-MAJOR FAIL: %0d error(s)", errors);
        $display("TOKEN_MAJOR_PROBE_DONE");
        $finish;
    end

    initial begin
        #(400*1000*1000);
        $display("FAIL: global timeout (ncap=%0d)", ncap);
        $finish;
    end
endmodule
`default_nettype wire

