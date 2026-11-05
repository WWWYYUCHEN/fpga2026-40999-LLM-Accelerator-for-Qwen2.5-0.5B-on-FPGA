`timescale 1ns/1ps
`default_nettype none

// ============================================================================
// tb_gemm_array_col -- equivalence of the 256-PE COL=32 array against the
//                      certified COL=16 one, column by column.
//
// The claim being tested is the whole justification of the [ARRAY-256] change:
// a COL=32 beat computes the SAME per-column INT24 and the SAME block exponent
// as the COL=16 beat that used to carry that column, for arbitrary weights and
// arbitrary block-floating-point activations.  If that holds, no downstream
// number can move, which is what the diag's 15/15 identical dumps then confirm
// end to end.
//
// Column map (why slots pair up): the 128-slot INT8 weight store is read as
// (128/COL) slots x COL columns, so COL=32 slot s covers exactly the columns
// that COL=16 slots 2s and 2s+1 covered:
//     COL32(s, row, c)  ==  COL16(2s + c/16, row, c%16)
//
// Stimulus: 8 different activation rows (random BF16 with a spread of
// exponents, plus zero lanes to exercise align_lane's e==0 / gap==15 path) and
// random INT8 weights in [-128,127] for all 128 slots.
//
// All stimulus is driven on the NEGEDGE and sampled after the following
// POSEDGE: driving at the posedge races the DUT's own always_ff (that bug
// silently dropped 7 of the 8 weight-load beats the first time this ran, and
// the resulting all-X datapath was the symptom).
// ============================================================================
module tb_gemm_array_col;
    localparam integer LANE    = 8;
    localparam integer NSLOT16 = 8;
    localparam integer NSLOT32 = 4;
    localparam integer LAT     = LANE + 1;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    reg [LANE*16-1:0] act [0:7];
    reg signed [7:0]  w [0:127][0:LANE-1];

    integer i, j, k, n, seed, tmp;

    // ------------------------------------------------------------- COL=16 DUT
    reg  a16_valid = 1'b0, a16_adv = 1'b0, a16_wl_valid = 1'b0;
    reg  [LANE*16-1:0] a16_act = {LANE*16{1'b0}};
    reg  [2047:0] a16_wdata = 2048'd0;
    reg  [2:0]  a16_wrow = 3'd0;
    reg  [159:0] a16_meta = 160'd0;
    wire a16_ready, a16_wl_ready, a16_mvalid, a16_busy;
    wire [16*32-1:0] a16_data;
    wire [167:0] a16_mmeta;

    qwen_gemm_array_i #(.DISTRIBUTED_COMPUTE_BANK(0), .COL(16), .LANE(LANE), .SLOT(8)) dut16 (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .s_valid(a16_valid), .s_ready(a16_ready), .s_activation(a16_act),
        .s_weight_advance(a16_adv), .s_meta(a16_meta),
        .s_weight_load_valid(a16_wl_valid), .s_weight_load_ready(a16_wl_ready),
        .s_weight_load_data(a16_wdata), .s_weight_load_row(a16_wrow),
        .s_weight_load_bank(1'b0), .s_compute_bank(1'b0),
        .s_compute_bank_rows(8'd0), .s_compute_release(1'b0),
        .m_valid(a16_mvalid), .m_ready(1'b1), .m_data(a16_data), .m_meta(a16_mmeta),
        .busy(a16_busy), .protocol_error()
    );

    // ------------------------------------------------------------- COL=32 DUT
    reg  a32_valid = 1'b0, a32_adv = 1'b0, a32_wl_valid = 1'b0;
    reg  [LANE*16-1:0] a32_act = {LANE*16{1'b0}};
    reg  [2047:0] a32_wdata = 2048'd0;
    reg  [2:0]  a32_wrow = 3'd0;
    reg  [159:0] a32_meta = 160'd0;
    wire a32_ready, a32_wl_ready, a32_mvalid, a32_busy;
    wire [32*32-1:0] a32_data;
    wire [167:0] a32_mmeta;

    qwen_gemm_array_i #(.DISTRIBUTED_COMPUTE_BANK(0), .COL(32), .LANE(LANE), .SLOT(8)) dut32 (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .s_valid(a32_valid), .s_ready(a32_ready), .s_activation(a32_act),
        .s_weight_advance(a32_adv), .s_meta(a32_meta),
        .s_weight_load_valid(a32_wl_valid), .s_weight_load_ready(a32_wl_ready),
        .s_weight_load_data(a32_wdata), .s_weight_load_row(a32_wrow),
        .s_weight_load_bank(1'b0), .s_compute_bank(1'b0),
        .s_compute_bank_rows(8'd0), .s_compute_release(1'b0),
        .m_valid(a32_mvalid), .m_ready(1'b1), .m_data(a32_data), .m_meta(a32_mmeta),
        .busy(a32_busy), .protocol_error()
    );

    // ---------------------------------------------------------- capture banks
    reg signed [23:0] o16 [0:NSLOT16-1][0:7][0:15];
    reg signed [23:0] o32 [0:NSLOT32-1][0:7][0:31];
    reg [7:0] e16 [0:NSLOT16-1][0:7];
    reg [7:0] e32 [0:NSLOT32-1][0:7];

    always @(posedge clk) if (rst_n) begin
        if (a16_mvalid) begin
            for (n = 0; n < 16; n = n + 1)
                o16[a16_mmeta[133:131]][a16_mmeta[130:128]][n] <= a16_data[n*24 +: 24];
            e16[a16_mmeta[133:131]][a16_mmeta[130:128]] <= a16_mmeta[167:160];
        end
        if (a32_mvalid) begin
            for (n = 0; n < 32; n = n + 1)
                o32[a32_mmeta[133:131]][a32_mmeta[130:128]][n] <= a32_data[n*24 +: 24];
            e32[a32_mmeta[133:131]][a32_mmeta[130:128]] <= a32_mmeta[167:160];
        end
    end

    integer checks, fails;

    initial begin
        checks = 0; fails = 0;
        seed = 12345;
        for (i = 0; i < 8; i = i + 1) begin
            act[i] = 128'd0;
            for (j = 0; j < 8; j = j + 1) begin
                if (((i * 8 + j) % 5) == 3)
                    act[i][j*16 +: 16] = 16'd0;
                else begin
                    tmp = ((($unsigned($random(seed)) % 2) << 15)
                         | ((118 + $unsigned($random(seed)) % 17) << 7)
                         | ($unsigned($random(seed)) % 128));
                    act[i][j*16 +: 16] = tmp[15:0];
                end
            end
        end
        for (n = 0; n < 128; n = n + 1)
            for (j = 0; j < LANE; j = j + 1)
                w[n][j] = $random(seed) % 256;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ---------------------------------------------- weight load, 8 lanes
        for (k = 0; k < LANE; k = k + 1) begin
            @(negedge clk);
            for (n = 0; n < 128; n = n + 1)
                a16_wdata[n*16 +: 16] = {8'h00, w[n][k]};
            a32_wdata = a16_wdata;
            a16_wrow = k[2:0]; a32_wrow = k[2:0];
            a16_wl_valid = 1'b1; a32_wl_valid = 1'b1;
            while (!(a16_wl_ready && a32_wl_ready)) @(negedge clk);
            @(posedge clk);
            @(negedge clk);
            a16_wl_valid = 1'b0; a32_wl_valid = 1'b0;
        end
        repeat (4) @(posedge clk);

        // -------------------------------------------- COL=16: 8 slots x 8 rows
        for (k = 0; k < NSLOT16; k = k + 1)
            for (j = 0; j < 8; j = j + 1) begin
                @(negedge clk);
                a16_act = act[j]; a16_adv = (j == 7); a16_valid = 1'b1;
                a16_meta = {1'b0, 25'd0, k[2:0], j[2:0], 128'd0};
                while (!a16_ready) @(negedge clk);
                @(posedge clk);
                @(negedge clk);
                a16_valid = 1'b0; a16_adv = 1'b0;
            end
        @(posedge clk);
        while (a16_busy) @(posedge clk);
        repeat (4) @(posedge clk);

        // -------------------------------------------- COL=32: 4 slots x 8 rows
        for (k = 0; k < NSLOT32; k = k + 1)
            for (j = 0; j < 8; j = j + 1) begin
                @(negedge clk);
                a32_act = act[j]; a32_adv = (j == 7); a32_valid = 1'b1;
                a32_meta = {1'b0, 25'd0, k[2:0], j[2:0], 128'd0};
                while (!a32_ready) @(negedge clk);
                @(posedge clk);
                @(negedge clk);
                a32_valid = 1'b0; a32_adv = 1'b0;
            end
        @(posedge clk);
        while (a32_busy) @(posedge clk);
        repeat (4) @(posedge clk);

        // ------------------------------------------------------------ compare
        for (k = 0; k < NSLOT32; k = k + 1)
            for (j = 0; j < 8; j = j + 1)
                for (n = 0; n < 32; n = n + 1) begin
                    checks = checks + 1;
                    if ((^{o32[k][j][n], o16[2*k + n/16][j][n%16]}) === 1'bx) begin
                        fails = fails + 1;
                        if (fails < 8)
                            $display("X VALUE slot32=%0d row=%0d col=%0d 32=%0d 16=%0d",
                                     k, j, n, o32[k][j][n], o16[2*k + n/16][j][n%16]);
                    end else if (o32[k][j][n] !== o16[2*k + n/16][j][n%16]) begin
                        fails = fails + 1;
                        if (fails < 8)
                            $display("MISMATCH slot32=%0d row=%0d col=%0d  32=%0d  16(slot %0d,col %0d)=%0d",
                                     k, j, n, o32[k][j][n], 2*k + n/16, n%16, o16[2*k + n/16][j][n%16]);
                    end
                end
        for (k = 0; k < NSLOT32; k = k + 1)
            for (j = 0; j < 8; j = j + 1) begin
                checks = checks + 1;
                if ((^{e32[k][j], e16[2*k][j], e16[2*k+1][j]}) === 1'bx) begin
                    fails = fails + 1;
                    $display("X EXP slot32=%0d row=%0d 32=%0d 16a=%0d 16b=%0d",
                             k, j, e32[k][j], e16[2*k][j], e16[2*k+1][j]);
                end else if (e32[k][j] !== e16[2*k][j] || e32[k][j] !== e16[2*k+1][j]) begin
                    fails = fails + 1;
                    $display("EXP MISMATCH slot32=%0d row=%0d  32=%0d  16a=%0d 16b=%0d",
                             k, j, e32[k][j], e16[2*k][j], e16[2*k+1][j]);
                end
            end

        $display("ARRAY_COL checks=%0d fails=%0d  (LAT=%0d, COL16=%0d DSP, COL32=%0d DSP)",
                 checks, fails, LAT, 16*LANE, 32*LANE);
        if (fails == 0) $display("ARRAY_COL PASS");
        else            $display("ARRAY_COL FAIL");
        $finish;
    end

    initial begin
        #4000000;
        $display("ARRAY_COL TIMEOUT");
        $finish;
    end
endmodule
`default_nettype wire
