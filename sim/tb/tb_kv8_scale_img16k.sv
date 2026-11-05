`timescale 1ns/1ps
// =============================================================================
// tb_kv8_scale_img16k -- [16K-ONE-LAYER] 16K scale-image load / read-back proof
//
// What it proves (the acceptance evidence for the 16K restore):
//   1. A full 16384-token image for ONE layer (K m16 64 KiB + V m16 64 KiB +
//      V 2^ 256 B + K etab 256 B) can be bulk-loaded through the four ports the
//      controller drives and every single entry reads back bit-exactly:
//        32768 K m16 entries + 32768 V m16 entries (token 16383 included)
//        128 V 2^ entries, 128 K block exponents (block boundary 63 included)
//   2. The per-token append still lands in the right 16-bit lane of the right
//      word (token 16383 = the LAST lane of the LAST word), on top of a loaded
//      image, without disturbing the other 7 lanes.
//   3. The block-exponent read that the attention sees follows the 16K block
//      index (256-token blocks, 64 of them per head).
//   4. No ownership violation (o_owner_error stays 0).
//   5. Two words per region per head survive a controller-timed dump/load
//      with a stuck-word-pointer negative control (section 8).
//
// Run: powershell -NoProfile -ExecutionPolicy Bypass -File .\run_xsim.ps1 -Tb tb_kv8_scale_img16k
// =============================================================================
module tb_kv8_scale_img16k;
    localparam integer CL2    = 14;
    localparam integer HEADS  = 2;
    localparam integer NW     = (1<<(CL2-3));      // 2048 words per head
    localparam integer KW     = HEADS*NW;          // 4096 words
    localparam integer BLOCKS = 64;                // 256-token blocks per head
    localparam integer ETW    = 16;                // 16 image words (8 blocks each)

    reg clk = 1'b0;
    always #2 clk = ~clk;
    reg rst_n = 1'b0;

    // ---- model: every entry is a distinct function of (head, token) ---------
    function [15:0] k_m(input integer head, input integer tok);
        begin k_m = {2'b01, head[0], tok[12:0]}; end
    endfunction
    function [15:0] v_m(input integer head, input integer tok);
        begin v_m = {2'b10, head[0], tok[12:0]}; end
    endfunction
    function [7:0] e_b(input integer head, input integer blk);
        begin e_b = (8'h40 + blk[5:0] + (head ? 8'h20 : 8'h00)); end
    endfunction
    function [15:0] v2_w(input integer w);
        begin v2_w = 16'h3C00 + w[3:0]*16'h0011; end
    endfunction

    // ---- DUT ports ----------------------------------------------------------
    reg         i_bank_wr = 1'b0, i_bank_rd = 1'b0;
    reg  [4:0]  i_layer_wr = 5'd0, i_layer_rd = 5'd0;
    reg         i_wr_acquire = 1'b0, i_wr_release = 1'b0, i_rd_release = 1'b0;
    wire        o_wr_grant, o_rd_grant, o_owner_error;
    reg         i_etab_valid = 1'b0;
    reg  [2:0]  i_etab_word = 3'd0;
    reg  [127:0] i_etab_data = 128'd0;
    reg  [15:0] i_etab_be = 16'hffff;
    reg  [2:0]  i_etab_rd_word = 3'd0;
    wire [127:0] o_etab_rd_data;
    reg         i_load_valid = 1'b0;
    reg  [11:0] i_load_word = 12'd0;
    reg  [127:0] i_load_mant = 128'd0;
    reg         i_vload_valid = 1'b0;
    reg  [11:0] i_vload_word = 12'd0;
    reg  [127:0] i_vload_data = 128'd0;
    reg         i_app_valid = 1'b0;
    wire        o_app_ready;
    reg  [15:0] i_app_token = 16'd0;
    reg         i_app_head = 1'b0;
    reg  [15:0] i_app_mant = 16'd0;
    reg  [7:0]  i_app_exp = 8'd0;
    reg         i_app_clip = 1'b0;
    wire        o_app_taken, o_app_conflict;
    reg         i_rd_en = 1'b0;
    reg  [11:0] i_rd_word = 12'd0;
    wire        o_rd_valid;
    wire [127:0] o_rd_m16;
    wire [7:0]  o_rd_eb;
    reg         i_v_we = 1'b0;
    reg  [3:0]  i_v_word = 4'd0;
    reg  [127:0] i_v_data = 128'd0;
    reg  [3:0]  i_v_rd_word = 4'd0;
    wire [127:0] o_v_rd_data;
    reg         i_vt_app_valid = 1'b0;
    wire        o_vt_app_ready;
    reg  [15:0] i_vt_app_token = 16'd0;
    reg         i_vt_app_head = 1'b0;
    reg  [15:0] i_vt_app_mant = 16'd0;
    wire        o_vt_app_taken;
    reg         i_vt_rd_en = 1'b0;
    reg  [11:0] i_vt_rd_word = 12'd0;
    wire [127:0] o_vt_rd_mant;
    wire        o_vt_rd_valid;
    wire        o_clip_sticky;
    wire [31:0] o_stat_load, o_stat_app, o_stat_rd;

    kv8_scale_ram #(.CONTEXT_LOG2(CL2), .BANKS(2), .V_DIM(64), .V_HEADS(HEADS),
                    .BLOCK_LOG2(8), .LAYERS(24), .TBL_TOK_LOG2(10))
    dut (
        .clk(clk), .rst_n(rst_n),
        .i_bank_wr(i_bank_wr), .i_bank_rd(i_bank_rd),
        .i_layer_wr(i_layer_wr), .i_layer_rd(i_layer_rd),
        .i_wr_acquire(i_wr_acquire), .i_wr_release(i_wr_release), .i_rd_release(i_rd_release),
        .o_wr_grant(o_wr_grant), .o_rd_grant(o_rd_grant), .o_owner_error(o_owner_error),
        .i_etab_valid(i_etab_valid), .o_etab_ready(), .i_etab_word(i_etab_word),
        .i_etab_data(i_etab_data), .i_etab_be(i_etab_be),
        .i_etab_rd_word(i_etab_rd_word), .o_etab_rd_data(o_etab_rd_data),
        .i_load_valid(i_load_valid), .o_load_ready(), .i_load_word(i_load_word), .i_load_mant(i_load_mant),
        .i_vload_valid(i_vload_valid), .o_vload_ready(), .i_vload_word(i_vload_word), .i_vload_data(i_vload_data),
        .i_app_valid(i_app_valid), .o_app_ready(o_app_ready), .i_app_token(i_app_token),
        .i_app_head(i_app_head), .i_app_mant(i_app_mant), .i_app_exp(i_app_exp), .i_app_clip(i_app_clip),
        .o_app_taken(o_app_taken), .o_app_conflict(o_app_conflict),
        .i_rd_en(i_rd_en), .i_rd_word(i_rd_word), .o_rd_valid(o_rd_valid),
        .o_rd_m16(o_rd_m16), .o_rd_eb(o_rd_eb),
        .i_v_we(i_v_we), .i_v_word(i_v_word), .i_v_data(i_v_data),
        .i_v_rd_word(i_v_rd_word), .o_v_rd_data(o_v_rd_data),
        .i_vt_app_valid(i_vt_app_valid), .o_vt_app_ready(o_vt_app_ready),
        .i_vt_app_token(i_vt_app_token), .i_vt_app_head(i_vt_app_head),
        .i_vt_app_mant(i_vt_app_mant), .o_vt_app_taken(o_vt_app_taken),
        .i_vt_rd_en(i_vt_rd_en), .i_vt_rd_word(i_vt_rd_word),
        .o_vt_rd_mant(o_vt_rd_mant), .o_vt_rd_valid(o_vt_rd_valid),
        .o_clip_sticky(o_clip_sticky), .o_stat_load(o_stat_load),
        .o_stat_app(o_stat_app), .o_stat_rd(o_stat_rd)
    );

    integer errors = 0;
    integer k_checked = 0, v_checked = 0, v2_checked = 0, e_checked = 0;
    integer th_checked = 0;
    reg app_phase = 1'b0;

    // ---- [16K-FIX] two-heads-differ case 7 constants ------------------------
    // The real numbers the 16K E2E failure produced at decode layer 0 token 0:
    // head 0 0x0107, head 1 0x01f2 (K m16), V head 0 0x0008, head 1 0x0005.
    // The two heads must never be interchangeable: a dump/load round trip that
    // mixes them is exactly the bug this case exists to catch.
    localparam [127:0] K_H0 = 128'h0000_0000_0000_0000_0000_0000_0000_0107;
    localparam [127:0] K_H1 = 128'h0000_0000_0000_0000_0000_0000_0000_01f2;
    localparam [127:0] V_H0 = 128'h0000_0000_0000_0000_0000_0000_0000_0008;
    localparam [127:0] V_H1 = 128'h0000_0000_0000_0000_0000_0000_0000_0005;
    localparam [127:0] JUNK = 128'hDEAD_BEEF_DEAD_BEEF_DEAD_BEEF_DEAD_BEEF;
    reg [127:0] ddr_img [0:3];      // dump image: K h0, K h1, V h0, V h1
    // ---- case 8: TWO words per region per head -----------------------------
    localparam [127:0] JUNK2 = 128'h0BAD_0BAD_0BAD_0BAD_0BAD_0BAD_0BAD_0BAD;
    reg [127:0] mw_exp [0:3][0:1];  // [item][word], item = K h0, K h1, V h0, V h1
    reg [127:0] mw_dump[0:3][0:1];  // dump captured with the controller's timing
    reg [127:0] mw_chk [0:3][0:1];  // load read-back
    integer     mw_checked = 0;
    reg [127:0] chk_img [0:3];      // load read-back

    // ---- image word builders ------------------------------------------------
    function [127:0] k_img(input integer head, input integer w);
        integer j;
        begin
            for (j=0;j<8;j=j+1) k_img[j*16 +: 16] = k_m(head, w*8+j);
        end
    endfunction
    function [127:0] v_img(input integer head, input integer w);
        integer j;
        begin
            for (j=0;j<8;j=j+1) v_img[j*16 +: 16] = v_m(head, w*8+j);
        end
    endfunction
    // one 128-bit etab image word = 8 x 16-bit entries {8'h00, eb}
    function [127:0] e_img(input integer head, input integer iw);
        integer j;
        begin
            for (j=0;j<8;j=j+1) e_img[j*16 +: 16] = {8'h00, e_b(head, iw*8+j)};
        end
    endfunction
    // the on-chip etab write for image word iw: 8 bytes into half iw[0]
    function [127:0] e_chip(input integer head, input integer iw);
        integer j;
        reg [63:0] pk;
        begin
            for (j=0;j<8;j=j+1) pk[j*8 +: 8] = e_b(head, iw*8+j);
            e_chip = {2{pk}};
        end
    endfunction

    // ---- read-back pipeline bookkeeping ------------------------------------
    // K read port: address in cycle c, o_rd_valid in c+3.  V per-token port: c+1.
    reg [11:0] k_idx_1, k_idx_2, k_idx_3;
    reg        k_act_1, k_act_2, k_act_3;
    reg [11:0] v_idx_1;
    reg        v_act_1;
    reg [11:0] k_iss, v_iss;
    reg [15:0] sweep_left;

    integer h, w, i, j, blk;
    reg [127:0] got;
    reg [15:0]  lane_exp, lane_got;

    task check_k_word(input integer head, input integer wd, input [127:0] g);
        integer jj;
        begin
            for (jj=0;jj<8;jj=jj+1) begin
                lane_exp = k_m(head, wd*8+jj);
                lane_got = g[jj*16 +: 16];
                k_checked = k_checked + 1;
                if (lane_got !== lane_exp) begin
                    errors = errors + 1;
                    if (errors < 12)
                        $display("FAIL K  h=%0d tok=%0d exp=%04h got=%04h", head, wd*8+jj, lane_exp, lane_got);
                end
            end
        end
    endtask
    task check_v_word(input integer head, input integer wd, input [127:0] g);
        integer jj;
        begin
            for (jj=0;jj<8;jj=jj+1) begin
                lane_exp = v_m(head, wd*8+jj);
                lane_got = g[jj*16 +: 16];
                v_checked = v_checked + 1;
                if (lane_got !== lane_exp) begin
                    errors = errors + 1;
                    if (errors < 12)
                        $display("FAIL V  h=%0d tok=%0d exp=%04h got=%04h", head, wd*8+jj, lane_exp, lane_got);
                end
            end
        end
    endtask
    task check_k_eb(input integer head, input integer wd, input [7:0] got_eb);
        begin
            e_checked = e_checked + 1;
            if (got_eb !== e_b(head, wd>>5)) begin
                errors = errors + 1;
                if (errors < 12)
                    $display("FAIL EB h=%0d tokword=%0d blk=%0d exp=%02h got=%02h",
                             head, wd, wd>>5, e_b(head, wd>>5), got_eb);
            end
        end
    endtask

    // ---- read-back monitors ------------------------------------------------
    always @(posedge clk) begin
        if (rst_n) begin
            if (!app_phase && o_rd_valid && k_act_3) check_k_word(k_idx_3[11], k_idx_3[10:0], o_rd_m16);
            if (!app_phase && o_rd_valid && k_act_3) check_k_eb(k_idx_3[11], k_idx_3[10:0], o_rd_eb);
            if (!app_phase && o_vt_rd_valid && v_act_1) check_v_word(v_idx_1[11], v_idx_1[10:0], o_vt_rd_mant);
            k_idx_3 <= k_idx_2; k_act_3 <= k_act_2;
            k_idx_2 <= k_idx_1; k_act_2 <= k_act_1;
            k_idx_1 <= i_rd_word; k_act_1 <= i_rd_en;
            v_idx_1 <= i_vt_rd_word; v_act_1 <= i_vt_rd_en;
        end
    end

    // =========================================================================
    // stimulus
    // =========================================================================
    initial begin : stim
        errors = 0;
        // ---------------- reset ------------------------------------------
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n <= 1'b1;
        repeat (2) @(posedge clk);

        $display("=== tb_kv8_scale_img16k : 16384-token scale image, one layer ===");

        // ---------------- 1. bulk load the whole image -------------------
        // bank EMPTY -> LOADING (the controller does this at each layer start)
        @(posedge clk); i_wr_acquire <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b0;
        repeat (2) @(posedge clk);
        if (!o_wr_grant) begin errors=errors+1; $display("FAIL ownership: no wr grant in LOADING"); end

        // K m16: 2 heads x 2048 words, 1 beat/cycle
        for (h=0;h<HEADS;h=h+1)
            for (w=0;w<NW;w=w+1) begin
                @(posedge clk);
                i_load_valid <= 1'b1; i_load_word <= {h[0], w[10:0]}; i_load_mant <= k_img(h,w);
            end
        @(posedge clk); i_load_valid <= 1'b0;
        // V m16
        for (h=0;h<HEADS;h=h+1)
            for (w=0;w<NW;w=w+1) begin
                @(posedge clk);
                i_vload_valid <= 1'b1; i_vload_word <= {h[0], w[10:0]}; i_vload_data <= v_img(h,w);
            end
        @(posedge clk); i_vload_valid <= 1'b0;
        // V 2^: 16 words
        for (w=0;w<16;w=w+1) begin
            @(posedge clk);
            i_v_we <= 1'b1; i_v_word <= w[3:0]; i_v_data <= {8{v2_w(w)}};
        end
        @(posedge clk); i_v_we <= 1'b0;
        // K etab image: 16 words of 8 x 16-bit entries -> 8 on-chip words
        for (h=0;h<HEADS;h=h+1)
            for (i=0;i<8;i=i+1) begin
                @(posedge clk);
                i_etab_valid <= 1'b1;
                i_etab_word  <= {h[0], i[2:1]};
                i_etab_data  <= e_chip(h,i);
                i_etab_be    <= i[0] ? 16'hFF00 : 16'h00FF;
            end
        @(posedge clk); i_etab_valid <= 1'b0;
        repeat (4) @(posedge clk);
        $display("info: load beats = %0d (K %0d + V %0d; V2^/etab use their own ports)",
                 o_stat_load, HEADS*NW, HEADS*NW);
        if (o_stat_load !== HEADS*NW*2) begin
            errors=errors+1; $display("FAIL load count %0d", o_stat_load);
        end

        // LOADING -> READY
        i_wr_release <= 1'b1; @(posedge clk); i_wr_release <= 1'b0;
        repeat (2) @(posedge clk);
        if (!o_rd_grant) begin errors=errors+1; $display("FAIL ownership: no rd grant in READY"); end

        // ---------------- 2. read back every entry ----------------------
        // pipelined: one K address + one V address per cycle
        k_iss = 0; v_iss = 0;
        for (i=0;i<KW;i=i+1) begin
            @(posedge clk);
            i_rd_en    <= 1'b1; i_rd_word    <= i[11:0];
            i_vt_rd_en <= 1'b1; i_vt_rd_word <= i[11:0];
        end
        @(posedge clk); i_rd_en <= 1'b0; i_vt_rd_en <= 1'b0;
        repeat (8) @(posedge clk);
        $display("info: K m16 entries checked=%0d  V m16 entries checked=%0d  eb checked=%0d",
                 k_checked, v_checked, e_checked);
        if (k_checked !== KW*8) begin errors=errors+1; $display("FAIL k_checked=%0d", k_checked); end
        if (v_checked !== KW*8) begin errors=errors+1; $display("FAIL v_checked=%0d", v_checked); end
        if (e_checked !== KW)   begin errors=errors+1; $display("FAIL e_checked=%0d", e_checked); end

        // ---------------- 3. V 2^ table ---------------------------------
        for (i=0;i<16;i=i+1) begin
            @(posedge clk); i_v_rd_word <= i[3:0];
            @(posedge clk); #1;
            got = o_v_rd_data;
            for (j=0;j<8;j=j+1) begin
                v2_checked = v2_checked + 1;
                if (got[j*16 +: 16] !== v2_w(i)) begin
                    errors=errors+1;
                    if (errors<12) $display("FAIL V2^ word=%0d lane=%0d exp=%04h got=%04h", i, j, v2_w(i), got[j*16 +: 16]);
                end
            end
        end
        // ---------------- 4. etab whole-word read port ------------------
        for (i=0;i<8;i=i+1) begin
            @(posedge clk); i_etab_rd_word <= i[2:0];
            @(posedge clk); #1;
            got = o_etab_rd_data;
            for (j=0;j<16;j=j+1) begin
                e_checked = e_checked + 1;
                // word i = {head=i[2], block[5:4]=i[1:0]} -> block = i[1:0]*16 + j
                if (got[j*8 +: 8] !== e_b(i[2], i[1:0]*16 + j)) begin
                    errors=errors+1;
                    if (errors<12) $display("FAIL ETAB word=%0d byte=%0d exp=%02h got=%02h",
                                            i, j, e_b(i[2], i[1:0]*16+j), got[j*8 +: 8]);
                end
            end
        end
        $display("info: V2^ entries checked=%0d  etab bytes checked=%0d", v2_checked, e_checked);
        if (v2_checked !== 128) begin errors=errors+1; $display("FAIL v2_checked=%0d", v2_checked); end
        if (e_checked !== KW + 128) begin errors=errors+1; $display("FAIL e_checked=%0d", e_checked); end

        // ---------------- 5. append on top of the loaded image ----------
        // Token 16383 = the last lane of the last word, both tables, both heads.
        // Requires the bank to accept appends (READY is allowed).
        app_phase = 1'b1;
        for (h=0;h<HEADS;h=h+1) begin
            @(posedge clk);
            i_app_valid <= 1'b1; i_app_token <= 16'd16383; i_app_head <= h[0];
            i_app_mant <= 16'hA5A0 + h[0]; i_app_exp <= e_b(h[0],63); i_app_clip <= 1'b0;
            @(posedge clk); i_app_valid <= 1'b0;
            repeat (5) @(posedge clk);
            @(posedge clk);
            i_vt_app_valid <= 1'b1; i_vt_app_token <= 16'd16383; i_vt_app_head <= h[0];
            i_vt_app_mant <= 16'hB6B0 + h[0];
            @(posedge clk); i_vt_app_valid <= 1'b0;
            repeat (5) @(posedge clk);
        end
        if (o_app_conflict) begin errors=errors+1; $display("FAIL append conflict flagged"); end
        // read the last words back (static: after the pipeline drains, the last
        // read word stays on the port, so a #1 sample is unambiguous).
        // app_phase stays 1: the sweep monitors must not judge these words.
        for (h=0;h<HEADS;h=h+1) begin
            @(posedge clk); i_rd_word <= {h[0], 11'd2047}; i_rd_en <= 1'b1;
            i_vt_rd_word <= {h[0], 11'd2047}; i_vt_rd_en <= 1'b1;
            @(posedge clk); i_rd_en <= 1'b0; i_vt_rd_en <= 1'b0;
            repeat (5) @(posedge clk); #1;
            got = o_rd_m16;
            $display("info: append readback head=%0d K=%032h V=%032h", h, o_rd_m16, o_vt_rd_mant);
            for (j=0;j<7;j=j+1) begin
                if (got[j*16 +: 16] !== k_m(h, 16376+j)) begin
                    errors=errors+1; $display("FAIL append clobbered K h=%0d lane %0d: %04h", h, j, got[j*16 +: 16]);
                end
                if (o_vt_rd_mant[j*16 +: 16] !== v_m(h, 16376+j)) begin
                    errors=errors+1; $display("FAIL append clobbered V h=%0d lane %0d: %04h", h, j, o_vt_rd_mant[j*16 +: 16]);
                end
                end
            if (got[7*16 +: 16] !== (16'hA5A0 + h[0])) begin
                errors=errors+1; $display("FAIL append K h=%0d lane7 exp=%04h got=%04h", h, 16'hA5A0+h[0], got[7*16 +: 16]);
            end
            if (o_vt_rd_mant[7*16 +: 16] !== (16'hB6B0 + h[0])) begin
                errors=errors+1; $display("FAIL append V h=%0d lane7 exp=%04h got=%04h", h, 16'hB6B0+h[0], o_vt_rd_mant[7*16 +: 16]);
            end
        end

        // ---------------- 6. ownership ----------------------------------
        if (o_owner_error) begin errors=errors+1; $display("FAIL o_owner_error set"); end
        i_rd_release <= 1'b1; @(posedge clk); i_rd_release <= 1'b0;
        repeat (2) @(posedge clk);
        if (o_owner_error) begin errors=errors+1; $display("FAIL o_owner_error set after release"); end

        // =====================================================================
        // 7. two heads hold DIFFERENT values at the same token index:
        //    dump (controller timing) -> DDR image -> load -> read back.
        //
        //    This is the case step 2 cannot see: the image sweep reads every word
        //    densely and never round-trips, so a dump port that silently returns
        //    the OTHER head's word still reads back correctly.  Here the pass
        //    condition is that head 0 and head 1 stay distinct end to end.
        //    The dump timing is the controller's: a ONE-CYCLE read enable with the
        //    address held for the 3-cycle latency and the beat taken 4 cycles
        //    after the enable (kv8_ls_state LS_D_CMD -> LS_D_WAIT x3 -> LS_D_BEAT).
        // =====================================================================
        app_phase = 1'b1;

        // (a) put the distinctive pair on chip (LOADING window, full-word writes)
        @(posedge clk); i_wr_acquire <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_load_valid <= 1'b1;  i_load_word  <= 12'd0;    i_load_mant <= K_H0;
        @(posedge clk);                        i_load_word  <= 12'd2048; i_load_mant <= K_H1;
        @(posedge clk); i_load_valid <= 1'b0;
        @(posedge clk); i_vload_valid <= 1'b1; i_vload_word <= 12'd0;    i_vload_data <= V_H0;
        @(posedge clk);                        i_vload_word <= 12'd2048; i_vload_data <= V_H1;
        @(posedge clk); i_vload_valid <= 1'b0;
        @(posedge clk); i_wr_release <= 1'b1;
        @(posedge clk); i_wr_release <= 1'b0;
        repeat (2) @(posedge clk);

        // (b) DUMP: items 0..3 = K h0, K h1, V h0, V h1 at word 0 of each head
        for (i=0;i<4;i=i+1) begin
            @(posedge clk);
            i_rd_word    <= (i[0] ? 12'h800 : 12'h000);
            i_vt_rd_word <= (i[0] ? 12'h800 : 12'h000);
            i_rd_en      <= (i[1] == 1'b0);
            i_vt_rd_en   <= (i[1] == 1'b1);
            @(posedge clk);
            i_rd_en <= 1'b0; i_vt_rd_en <= 1'b0;       // address stays selected
            repeat (4) @(posedge clk);                 // reach the beat cycle
            #1 ddr_img[i] = i[1] ? o_vt_rd_mant : o_rd_m16;
        end
        $display("info: dump image  K h0=%032h  K h1=%032h  V h0=%032h  V h1=%032h",
                 ddr_img[0], ddr_img[1], ddr_img[2], ddr_img[3]);
        for (i=0;i<4;i=i+1) begin
            th_checked = th_checked + 1;
            case (i)
                0: got = K_H0;
                1: got = K_H1;
                2: got = V_H0;
                default: got = V_H1;
            endcase
            if (ddr_img[i] !== got) begin
                errors=errors+1;
                $display("FAIL dump item=%0d exp=%032h got=%032h", i, got, ddr_img[i]);
            end
        end
        if (ddr_img[0] === ddr_img[1]) begin
            errors=errors+1; $display("FAIL dump: the two K heads came back identical");
        end
        if (ddr_img[2] === ddr_img[3]) begin
            errors=errors+1; $display("FAIL dump: the two V heads came back identical");
        end

        // (c) clobber both heads, then load the dumped image back over them, so a
        //     load that silently does nothing cannot pass.
        @(posedge clk); i_rd_release <= 1'b1;
        @(posedge clk); i_rd_release <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_wr_acquire <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_load_valid <= 1'b1;  i_load_word  <= 12'd0;    i_load_mant <= JUNK;
        @(posedge clk);                        i_load_word  <= 12'd2048; i_load_mant <= JUNK;
        @(posedge clk); i_load_valid <= 1'b0;
        @(posedge clk); i_vload_valid <= 1'b1; i_vload_word <= 12'd0;    i_vload_data <= JUNK;
        @(posedge clk);                        i_vload_word <= 12'd2048; i_vload_data <= JUNK;
        @(posedge clk); i_vload_valid <= 1'b0;
        @(posedge clk); i_load_valid <= 1'b1;  i_load_word  <= 12'd0;    i_load_mant <= ddr_img[0];
        @(posedge clk);                        i_load_word  <= 12'd2048; i_load_mant <= ddr_img[1];
        @(posedge clk); i_load_valid <= 1'b0;
        @(posedge clk); i_vload_valid <= 1'b1; i_vload_word <= 12'd0;    i_vload_data <= ddr_img[2];
        @(posedge clk);                        i_vload_word <= 12'd2048; i_vload_data <= ddr_img[3];
        @(posedge clk); i_vload_valid <= 1'b0;
        @(posedge clk); i_wr_release <= 1'b1;
        @(posedge clk); i_wr_release <= 1'b0;
        repeat (2) @(posedge clk);

        // (d) load read-back: head 0 must be head 0, head 1 must be head 1
        for (i=0;i<4;i=i+1) begin
            @(posedge clk);
            i_rd_word    <= (i[0] ? 12'h800 : 12'h000);
            i_vt_rd_word <= (i[0] ? 12'h800 : 12'h000);
            i_rd_en      <= (i[1] == 1'b0);
            i_vt_rd_en   <= (i[1] == 1'b1);
            @(posedge clk);
            i_rd_en <= 1'b0; i_vt_rd_en <= 1'b0;
            repeat (4) @(posedge clk);
            #1 chk_img[i] = i[1] ? o_vt_rd_mant : o_rd_m16;
        end
        $display("info: load image  K h0=%032h  K h1=%032h  V h0=%032h  V h1=%032h",
                 chk_img[0], chk_img[1], chk_img[2], chk_img[3]);
        for (i=0;i<4;i=i+1) begin
            th_checked = th_checked + 1;
            if (chk_img[i] !== ddr_img[i]) begin
                errors=errors+1;
                $display("FAIL load item=%0d exp=%032h got=%032h", i, ddr_img[i], chk_img[i]);
            end
        end
        if (chk_img[0] !== K_H0) begin errors=errors+1; $display("FAIL load K h0 got=%032h", chk_img[0]); end
        if (chk_img[1] !== K_H1) begin errors=errors+1; $display("FAIL load K h1 got=%032h", chk_img[1]); end
        if (chk_img[2] !== V_H0) begin errors=errors+1; $display("FAIL load V h0 got=%032h", chk_img[2]); end
        if (chk_img[3] !== V_H1) begin errors=errors+1; $display("FAIL load V h1 got=%032h", chk_img[3]); end

        // (e) negative control: the same one-cycle read, but the address is
        //     DROPPED after the enable cycle (what the 16K dump did before the
        //     fix: the port fell back to the attention's last address = head 1's
        //     word).  The data port follows the address every cycle, so the beat
        //     sees head 1's word while head 0 was asked for.  This asserts the
        //     mechanism is still observable, i.e. that this case really exercises
        //     it, and it documents the read-port contract: HOLD THE ADDRESS.
        @(posedge clk);
        i_rd_word <= 12'h000; i_rd_en <= 1'b1;
        @(posedge clk);
        i_rd_en <= 1'b0; i_rd_word <= 12'h800;      // address reverts (head 1)
        repeat (4) @(posedge clk); #1 got = o_rd_m16;
        $display("info: negative control (address not held): asked word=000, port shows %032h", got);
        if (got === K_H0) begin
            errors=errors+1;
            $display("FAIL negative control: the port held word 0 anyway -- the dump failure mechanism is not being reproduced");
        end else if (got !== K_H1) begin
            errors=errors+1;
            $display("FAIL negative control: expected the reverted address's word %032h got %032h", K_H1, got);
        end

        // =====================================================================
        // 8. TWO WORDS per region per head -- the dimension cases 1..7 cannot
        //    see.  Case 7 separates the two HEADS on one word; this case
        //    separates word 0 from word 1 as well, with every (item, word) a
        //    different value.  Dump uses the controller timing (one-cycle read
        //    enable, address held through the 3-cycle latency, beat at +4); the
        //    load drives the controller's ramp {head,cnt} = 0,1 / 0x800,0x801.
        // =====================================================================
        app_phase = 1'b1;
        for (i=0;i<4;i=i+1)
            for (w=0;w<2;w=w+1) begin
                lane_exp = 16'hA000 + i*16'h0100 + w*16'h0010;
                mw_exp[i][w] = {8{lane_exp}};
            end

        // (a) READY -> EMPTY -> LOADING, put all eight words on chip
        @(posedge clk); i_rd_release <= 1'b1;
        @(posedge clk); i_rd_release <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_wr_acquire <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b0;
        repeat (2) @(posedge clk);
        if (!o_wr_grant) begin errors=errors+1; $display("FAIL case8: no wr grant"); end
        for (i=0;i<4;i=i+1) begin
            if (i[1] == 1'b0) begin
                for (w=0;w<2;w=w+1) begin
                    @(posedge clk); i_load_valid <= 1'b1;
                    i_load_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                    i_load_mant <= mw_exp[i][w];
                end
                @(posedge clk); i_load_valid <= 1'b0;
            end else begin
                for (w=0;w<2;w=w+1) begin
                    @(posedge clk); i_vload_valid <= 1'b1;
                    i_vload_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                    i_vload_data <= mw_exp[i][w];
                end
                @(posedge clk); i_vload_valid <= 1'b0;
            end
        end
        repeat (4) @(posedge clk);
        @(posedge clk); i_wr_release <= 1'b1;
        @(posedge clk); i_wr_release <= 1'b0;
        repeat (2) @(posedge clk);

        // (b) DUMP 4 items x 2 words with the controller's timing
        for (i=0;i<4;i=i+1)
            for (w=0;w<2;w=w+1) begin
                @(posedge clk);
                i_rd_word    <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                i_vt_rd_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                i_rd_en      <= (i[1] == 1'b0);
                i_vt_rd_en   <= (i[1] == 1'b1);
                @(posedge clk);
                i_rd_en <= 1'b0; i_vt_rd_en <= 1'b0;
                repeat (4) @(posedge clk);
                #1 mw_dump[i][w] = i[1] ? o_vt_rd_mant : o_rd_m16;
            end
        for (i=0;i<4;i=i+1) begin
            for (w=0;w<2;w=w+1) begin
                mw_checked = mw_checked + 1;
                if (mw_dump[i][w] !== mw_exp[i][w]) begin
                    errors=errors+1;
                    $display("FAIL case8 dump item=%0d word=%0d exp=%032h got=%032h",
                             i, w, mw_exp[i][w], mw_dump[i][w]);
                end
            end
            if (mw_dump[i][0] === mw_dump[i][1]) begin
                errors=errors+1; $display("FAIL case8 dump item=%0d words identical", i);
            end
        end
        $display("info: case8 dump  K h0 w0=%032h w1=%032h  K h1 w0=%032h w1=%032h",
                 mw_dump[0][0], mw_dump[0][1], mw_dump[1][0], mw_dump[1][1]);
        $display("info: case8 dump  V h0 w0=%032h w1=%032h  V h1 w0=%032h w1=%032h",
                 mw_dump[2][0], mw_dump[2][1], mw_dump[3][0], mw_dump[3][1]);

        // (c) clobber all eight words, then load the dumped image back over them
        @(posedge clk); i_rd_release <= 1'b1;
        @(posedge clk); i_rd_release <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_wr_acquire <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b0;
        repeat (2) @(posedge clk);
        for (i=0;i<4;i=i+1) begin
            if (i[1] == 1'b0) begin
                for (w=0;w<2;w=w+1) begin
                    @(posedge clk); i_load_valid <= 1'b1;
                    i_load_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                    i_load_mant <= JUNK;
                end
                @(posedge clk); i_load_valid <= 1'b0;
            end else begin
                for (w=0;w<2;w=w+1) begin
                    @(posedge clk); i_vload_valid <= 1'b1;
                    i_vload_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                    i_vload_data <= JUNK;
                end
                @(posedge clk); i_vload_valid <= 1'b0;
            end
        end
        for (i=0;i<4;i=i+1) begin
            if (i[1] == 1'b0) begin
                for (w=0;w<2;w=w+1) begin
                    @(posedge clk); i_load_valid <= 1'b1;
                    i_load_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                    i_load_mant <= mw_dump[i][w];
                end
                @(posedge clk); i_load_valid <= 1'b0;
            end else begin
                for (w=0;w<2;w=w+1) begin
                    @(posedge clk); i_vload_valid <= 1'b1;
                    i_vload_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                    i_vload_data <= mw_dump[i][w];
                end
                @(posedge clk); i_vload_valid <= 1'b0;
            end
        end
        repeat (4) @(posedge clk);
        @(posedge clk); i_wr_release <= 1'b1;
        @(posedge clk); i_wr_release <= 1'b0;
        repeat (2) @(posedge clk);

        // (d) read every word back: word 0 must not be word 1, head 0 not head 1
        for (i=0;i<4;i=i+1)
            for (w=0;w<2;w=w+1) begin
                @(posedge clk);
                i_rd_word    <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                i_vt_rd_word <= (i[0] ? 12'h800 : 12'h000) + w[11:0];
                i_rd_en      <= (i[1] == 1'b0);
                i_vt_rd_en   <= (i[1] == 1'b1);
                @(posedge clk);
                i_rd_en <= 1'b0; i_vt_rd_en <= 1'b0;
                repeat (4) @(posedge clk);
                #1 mw_chk[i][w] = i[1] ? o_vt_rd_mant : o_rd_m16;
            end
        for (i=0;i<4;i=i+1) begin
            for (w=0;w<2;w=w+1) begin
                mw_checked = mw_checked + 1;
                if (mw_chk[i][w] !== mw_exp[i][w]) begin
                    errors=errors+1;
                    $display("FAIL case8 load item=%0d word=%0d exp=%032h got=%032h",
                             i, w, mw_exp[i][w], mw_chk[i][w]);
                end
            end
        end
        $display("info: case8 load  K h0 w0=%032h w1=%032h  K h1 w0=%032h w1=%032h",
                 mw_chk[0][0], mw_chk[0][1], mw_chk[1][0], mw_chk[1][1]);

        // (e) negative control: a STUCK word pointer (both beats written to word
        //     0) must be detectable by exactly the check above.  The assertion is
        //     that the read-back then differs: word 0 holds the SECOND beat and
        //     word 1 still holds the clobber value, so (d) would report two FAILs.
        @(posedge clk); i_rd_release <= 1'b1;
        @(posedge clk); i_rd_release <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_wr_acquire <= 1'b1;
        @(posedge clk); i_wr_acquire <= 1'b0;
        repeat (2) @(posedge clk);
        @(posedge clk); i_load_valid <= 1'b1; i_load_word <= 12'h000; i_load_mant <= JUNK2;
        @(posedge clk);                       i_load_word <= 12'h001; i_load_mant <= JUNK2;
        @(posedge clk); i_load_valid <= 1'b0;
        @(posedge clk); i_load_valid <= 1'b1; i_load_word <= 12'h000; i_load_mant <= mw_exp[0][0];
        @(posedge clk);                       i_load_word <= 12'h000; i_load_mant <= mw_exp[0][1];
        @(posedge clk); i_load_valid <= 1'b0;
        repeat (4) @(posedge clk);
        @(posedge clk); i_wr_release <= 1'b1;
        @(posedge clk); i_wr_release <= 1'b0;
        repeat (2) @(posedge clk);
        for (w=0;w<2;w=w+1) begin
            @(posedge clk); i_rd_word <= 12'h000 + w[11:0]; i_rd_en <= 1'b1;
            @(posedge clk); i_rd_en <= 1'b0;
            repeat (4) @(posedge clk); #1 mw_chk[0][w] = o_rd_m16;
        end
        $display("info: case8 negative control (stuck word index): w0=%032h w1=%032h",
                 mw_chk[0][0], mw_chk[0][1]);
        if (mw_chk[0][0] === mw_exp[0][0]) begin
            errors=errors+1;
            $display("FAIL case8 negative control: word 0 still holds the first beat -- the stuck-pointer failure is invisible");
        end
        if (mw_chk[0][0] !== mw_exp[0][1]) begin
            errors=errors+1;
            $display("FAIL case8 negative control: expected the second beat in word 0, got %032h", mw_chk[0][0]);
        end
        if (mw_chk[0][1] !== JUNK2) begin
            errors=errors+1;
            $display("FAIL case8 negative control: word 1 should keep the clobber value, got %032h", mw_chk[0][1]);
        end

        // ---------------- 9. ownership ----------------------------------
        if (o_owner_error) begin errors=errors+1; $display("FAIL o_owner_error set (case 7)"); end
        i_rd_release <= 1'b1; @(posedge clk); i_rd_release <= 1'b0;
        repeat (2) @(posedge clk);
        if (o_owner_error) begin errors=errors+1; $display("FAIL o_owner_error set after release (case 7)"); end

        // ---------------- verdict ---------------------------------------
        if (errors == 0)
            $display("RESULT: PASS  (16K image: %0d K + %0d V m16 entries, %0d V2^, %0d etab bytes; two-head dump/load: %0d; two-word dump/load: %0d)",
                     k_checked, v_checked, v2_checked, e_checked, th_checked, mw_checked);
        else
            $display("RESULT: FAIL  errors=%0d", errors);
        $finish;
    end

    initial begin
        #2000000;
        $display("RESULT: FAIL (timeout)");
        $finish;
    end
endmodule
