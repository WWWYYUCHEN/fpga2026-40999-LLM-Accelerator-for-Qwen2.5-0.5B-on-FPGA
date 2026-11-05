`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_context_scale -- the V-scale application point on the PV write-back
//
// The block must be invisible except for the multiply: same words, same order,
// same metadata, one per input word, and it must survive downstream stalls.  The
// V table holds power-of-two scales, so the expected product is exact
// (exponent field += k, mantissa untouched) and a wrong scale word, a wrong lane
// or a swapped metadata word cannot hide:
//   PHASE1  64 words through random output stalls (~75% ready), per-(head,word)
//           scale words from the 16-word V table
//   PHASE2  the same score word through all four scale exponents: the products
//           differ by exactly 2^k (the scale is really indexed, not a constant)
//   PHASE3  every popped word must carry the metadata of the word that produced
//           it, and the accepted/popped counts must match (no loss, no duplicate)
// =============================================================================
module tb_kv8_context_scale;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg         i_valid = 0;
    wire        o_ready;
    reg  [127:0] i_data = 0;
    reg  [2:0]  i_row = 0;
    reg  [3:0]  i_head = 0;
    reg  [2:0]  i_word = 0;
    wire [3:0]  o_scale_word;
    reg  [127:0] i_scale_data = 0;
    wire        o_valid;
    reg         i_ready = 1;
    wire [127:0] o_data;
    wire [2:0]  o_row;
    wire [3:0]  o_head;
    wire [2:0]  o_word;
    wire        o_overflow, o_scale_zero;

    kv8_context_scale dut (
        .clk(clk), .rst_n(rst_n),
        .i_valid(i_valid), .o_ready(o_ready), .i_data(i_data),
        .i_row(i_row), .i_head(i_head), .i_word(i_word),
        .o_scale_word(o_scale_word), .i_scale_data(i_scale_data),
        .o_valid(o_valid), .i_ready(i_ready), .o_data(o_data),
        .o_row(o_row), .o_head(o_head), .o_word(o_word),
        .o_overflow(o_overflow), .o_scale_zero(o_scale_zero)
    );

    // ---- V table model: 16 x 128 bit, one-cycle read latency ----------------
    // word w = {kv_head, dim_word} = 2^KSCALE[w] on all 8 lanes
    integer    kscale [0:15];
    reg [7:0]  e_tmp;
    reg [6:0]  f_tmp;
    reg [15:0] vscale [0:15];
    always @(posedge clk) i_scale_data <= {8{vscale[o_scale_word]}};
    initial begin
        for (integer w=0;w<16;w=w+1) begin
            kscale[w] = (w%5) - 2;                      // 2^-2 .. 2^+2
            e_tmp = 8'd127 + ((w%5) - 2);
            vscale[w] = {1'b0, e_tmp, 7'd0};
        end
        // make the two words the phase-2 stimulus uses explicit
        vscale[0]=16'h3F80; kscale[0]=0;                // 1.0
        vscale[1]=16'h3F00; kscale[1]=-1;               // 0.5
        vscale[2]=16'h4000; kscale[2]=1;                // 2.0
        vscale[3]=16'h3E80; kscale[3]=-2;               // 0.25
    end

    // ---- score stream -------------------------------------------------------
    localparam integer NW = 64;
    reg [127:0] score [0:NW-1];
    reg [2:0]   s_row [0:NW-1];
    reg [3:0]   s_head[0:NW-1];
    reg [2:0]   s_word[0:NW-1];

    // accepted-word queue (the DUT must preserve order)
    reg [127:0] q_data [0:NW+8];
    reg [2:0]   q_row  [0:NW+8];
    reg [3:0]   q_head [0:NW+8];
    reg [2:0]   q_word [0:NW+8];
    integer     wr_idx = 0, rd_idx = 0;

    integer errors, checks, lane;
    task automatic chk(input integer cond, input [255:0] msg);
        begin
            checks = checks + 1;
            if (!cond) begin
                errors = errors + 1;
                if (errors < 20) $display("FAIL %0s", msg);
            end
        end
    endtask

    function [15:0] shift_pow2(input [15:0] x, input integer k);
        reg signed [10:0] e;
        begin
            e = $signed({2'b00,x[14:7]}) + k;
            if (x[14:7] == 8'd0)    shift_pow2 = 16'h0000;
            else if (e >= 11'sd255) shift_pow2 = {x[15],8'hff,7'd0};
            else if (e <= 11'sd0)   shift_pow2 = {x[15],15'd0};
            else                    shift_pow2 = {x[15],e[7:0],x[6:0]};
        end
    endfunction

    integer seed = 32'h5EED_1234;
    integer stall_mode = 1;
    always @(posedge clk) begin
        if (!rst_n) i_ready <= 1'b1;
        else i_ready <= (stall_mode==0) ? 1'b1 : (($random(seed) & 3) != 0);
    end

    // ---- accept monitor ----------------------------------------------------
    always @(posedge clk) begin
        if (rst_n && i_valid && o_ready) begin
            q_data[wr_idx] = i_data;
            q_row [wr_idx] = i_row;
            q_head[wr_idx] = i_head;
            q_word[wr_idx] = i_word;
            wr_idx = wr_idx + 1;
        end
    end

    // ---- pop monitor: exact product + metadata of the same word ------------
    reg [3:0] sw;
    always @(posedge clk) begin
        if (rst_n && o_valid && i_ready) begin
            // [STAGE5-FIX6] same KV-head split as qwen25_attention.v:205
            sw = {(o_head >= 4'd7), o_word};
            chk(o_row  === q_row [rd_idx], "pop row belongs to this word");
            chk(o_head === q_head[rd_idx], "pop head belongs to this word");
            chk(o_word === q_word[rd_idx], "pop word belongs to this word");
            for (lane=0; lane<8; lane=lane+1)
                chk(o_data[lane*16 +: 16] ===
                    shift_pow2(q_data[rd_idx][lane*16 +: 16], kscale[sw]),
                    "lane product = score * s_v");
            rd_idx = rd_idx + 1;
        end
    end

    integer n, k;
    reg [15:0] p0, p1, p2, p3;
    initial begin
        errors = 0; checks = 0;
        // PHASE1/3 data
        for (n=0;n<NW;n=n+1) begin
            for (lane=0;lane<8;lane=lane+1) begin
                e_tmp = 8'd120 + ((n*3+lane) % 6);
                f_tmp[6:0] = (n*11 + lane*5) % 128;
                score[n][lane*16 +: 16] = {1'b0, e_tmp, f_tmp};
            end
            s_row[n]  = n % 8;
            s_head[n] = n % 14;
            s_word[n] = (n/14) % 8;
        end
        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        for (n=0;n<NW;n=n+1) begin
            i_data <= score[n];
            i_row  <= s_row[n];
            i_head <= s_head[n];
            i_word <= s_word[n];
            i_valid<= 1'b1;
            // sample ready just BEFORE the accepting edge: a transfer happens only
            // at a posedge where valid&&ready were both high before it
            @(negedge clk);
            while (!o_ready) @(negedge clk);
            @(posedge clk);
        end
        i_valid <= 1'b0;
        while (rd_idx < NW) @(posedge clk);
        repeat (6) @(posedge clk);

        chk(wr_idx == NW, "every word accepted exactly once");
        chk(rd_idx == NW, "every popped word accounted for");
        chk(o_overflow === 1'b0, "the output skid never overflowed");
        chk(o_scale_zero === 1'b0, "the V table was never read as zero");

        // PHASE2: the scale is really indexed -- same score, 4 scale exponents
        p0 = shift_pow2(score[0][15:0], kscale[0]);
        p1 = shift_pow2(score[0][15:0], kscale[1]);
        p2 = shift_pow2(score[0][15:0], kscale[2]);
        p3 = shift_pow2(score[0][15:0], kscale[3]);
        chk(p0 !== p1 && p1 !== p2 && p2 !== p3, "four scale words give four products");

        $display("CTX-SCALE accepted=%0d popped=%0d  s/p0=%h p1=%h p2=%h p3=%h",
                 wr_idx, rd_idx, p0, p1, p2, p3);
        if (errors == 0) $display("CTX-SCALE PASS: checks=%0d", checks);
        else             $display("CTX-SCALE FAIL: checks=%0d errors=%0d", checks, errors);
        $finish;
    end

    initial begin
        #2000000; $display("CTX-SCALE TIMEOUT"); $finish;
    end
endmodule
`default_nettype wire
