`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_fixpoint_probe -- settles the ONE unverified numeric assumption of the
// INT8 KV cache: what does the INT8 array think the weight byte {8'h00,code} is
// worth?
//
// Background (REPORT_STAGE5.md ?7, "??????????? item 1):
//   the K scale table stores  s_k = m * 2^(eb-130)
//   the V scale table stores  s_v = m * 2^(eb-127)
//   Those two are only mutually consistent if the datapath that multiplies a
//   code by its table entry has a known, fixed gain.  The report assumed the
//   array treats the byte as a 1.7 fraction (code/128) but never measured it.
//   If the gain is 1 (integer code) instead, every K/V value is off by a
//   constant 2^k ==> the A/B logits would break completely, not "slightly",
//   and the next person would chase quantisation noise that is not there.
//
// What is measured here (directly, on the real RTL, no reference model):
//   qwen_gemm_array_i + qwen_gemm_int24_bridge driven with
//     * a single K-block of 8 lanes (128-bit activation = 8 x bf16)
//     * all 128 weight slots = {high_byte, code}
//   and the fp32 column that comes out of the bridge is printed.
//
//   With one active lane holding bf16 1.0 and code = 64 the three candidate
//   conventions predict exactly:
//        integer code        -> 64.0
//        code/128 (1.7)      ->  0.5
//        code/256            ->  0.25
//   so the printout is the answer, with no model in the loop.
//
// Everything is an exact power of two, so a correct datapath returns exact
// values and any small deviation would itself be a finding.
// =============================================================================
module tb_kv8_fixpoint_probe;

    reg clk = 1'b0;
    reg rst_n = 1'b0;
    always #5 clk = ~clk;

    // ---------------------------------------------------------------- DUT wiring
    reg  [2047:0] w_load_data;
    reg  [2:0]    w_load_row;
    reg           w_load_bank;
    reg           w_load_valid;
    wire          w_load_ready;

    reg  [127:0]  a_act;
    reg  [159:0]  a_meta;
    reg           a_advance;
    reg           a_valid;
    wire          a_ready;

    wire          m_valid;
    wire [511:0]  m_data;
    wire [167:0]  m_meta;
    wire          a_busy;
    wire          a_proto_err;
    wire          m_ready = 1'b1;

    wire [511:0]  b_fp32;
    wire          b_valid;
    wire          b_inflight;

    // weight word: 128 slots x 16 bit, the array only looks at the low byte
    reg [7:0] w_code;
    reg [7:0] w_high;
    integer wn;
    always @(*) begin
        w_load_data = 2048'd0;
        for (wn = 0; wn < 128; wn = wn + 1) begin
            w_load_data[wn*16 +: 8] = w_code;
            w_load_data[wn*16 + 8 +: 8] = w_high;
        end
    end

    qwen_gemm_array_i #(.DISTRIBUTED_COMPUTE_BANK(1)) array_Inst (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .s_weight_load_valid(w_load_valid), .s_weight_load_ready(w_load_ready),
        .s_weight_load_data(w_load_data), .s_weight_load_row(w_load_row),
        .s_weight_load_bank(w_load_bank),
        .s_compute_bank(1'b0), .s_compute_bank_rows(8'h00), .s_compute_release(1'b0),
        .s_valid(a_valid), .s_ready(a_ready), .s_activation(a_act),
        .s_weight_advance(a_advance), .s_meta(a_meta),
        .m_valid(m_valid), .m_ready(m_ready), .m_data(m_data), .m_meta(m_meta),
        .busy(a_busy), .protocol_error(a_proto_err)
    );

    qwen_gemm_int24_bridge bridge_Inst (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .in_valid(m_valid && m_ready), .int24_flat(m_data[383:0]),
        .block_exp(m_meta[167:160]),
        .fp32_flat(b_fp32), .out_valid(b_valid), .inflight(b_inflight)
    );

    // ------------------------------------------------------------- captures
    reg [23:0] cap_int24;
    reg [7:0]  cap_exp;
    reg [31:0] cap_fp32;
    integer    m_count;
    integer    b_count;

    always @(posedge clk) begin
        if (!rst_n) begin
            cap_int24 <= 24'd0; cap_exp <= 8'd0; m_count <= 0;
        end else if (m_valid && m_ready) begin
            cap_int24 <= m_data[23:0];
            cap_exp   <= m_meta[167:160];
            m_count   <= m_count + 1;
        end
    end
    always @(posedge clk) begin
        if (!rst_n) begin
            cap_fp32 <= 32'd0; b_count <= 0;
        end else if (b_valid) begin
            cap_fp32 <= b_fp32[31:0];
            b_count  <= b_count + 1;
        end
    end

    // ------------------------------------------------------------- tasks
    task automatic load_weights(input [7:0] code, input [7:0] high);
        integer r;
        begin
            w_code = code; w_high = high;
            @(negedge clk);
            for (r = 0; r < 8; r = r + 1) begin
                w_load_row = r[2:0]; w_load_bank = 1'b0; w_load_valid = 1'b1;
                @(posedge clk);
                while (!w_load_ready) @(posedge clk);
                @(negedge clk);
            end
            w_load_valid = 1'b0;
            repeat (3) @(negedge clk);
        end
    endtask

    task automatic issue_block(input [127:0] act);
        integer guard;
        begin
            @(negedge clk);
            a_act = act; a_meta = 160'd0; a_advance = 1'b1; a_valid = 1'b1;
            guard = 0;
            @(posedge clk);
            while (!a_ready && guard < 1000) begin guard = guard + 1; @(posedge clk); end
            if (guard >= 1000) begin $display("FAIL: array never became ready"); $finish; end
            @(negedge clk);
            a_valid = 1'b0;
        end
    endtask

    // one full experiment: weights + one block, returns through cap_* / b_count
    task automatic run_case(input [7:0] code, input [7:0] high, input [127:0] act,
                            input [255:0] name);
        integer m0, b0, guard;
        begin
            m0 = m_count; b0 = b_count;
            load_weights(code, high);
            issue_block(act);
            guard = 0;
            while ((m_count == m0 || b_count == b0) && guard < 2000) begin
                @(negedge clk); guard = guard + 1;
            end
            if (guard >= 2000) begin $display("FAIL: no output for %0s", name); $finish; end
            $display("CASE %0s  code=%0d high=0x%02h act=0x%04h_...  int24=0x%06h (%0d) block_exp=%0d  fp32=0x%08h = %f",
                     name, $signed(code), high, act[15:0], cap_int24, $signed(cap_int24),
                     cap_exp, cap_fp32, $bitstoshortreal(cap_fp32));
        end
    endtask

    // bf16 helpers: bf16 = {sign, exp[7:0], mant[6:0]}
    function [15:0] bf16_of(input integer e, input [6:0] m);
        bf16_of = {1'b0, e[7:0], m};
    endfunction

    // lane 0 of the array is s_activation[15:0] (gen_align index gi=0)
    localparam [127:0] ACT_ALL_ONE   = {8{16'h3F80}};   // 8 lanes of 1.0
    localparam [127:0] ACT_ONE_L0    = 128'h0000_0000_0000_0000_0000_0000_0000_3F80;
    localparam [127:0] ACT_TWO_L0    = 128'h0000_0000_0000_0000_0000_0000_0000_4000;

    integer i;
    initial begin
        $display("=== tb_kv8_fixpoint_probe : what is {8'h00,code} worth? ===");
        w_load_valid = 0; w_load_row = 0; w_load_bank = 0;
        w_code = 8'd0; w_high = 8'd0;
        a_valid = 0; a_act = 128'd0; a_meta = 160'd0; a_advance = 1'b1;

        rst_n = 0;
        repeat (8) @(negedge clk);
        rst_n = 1;
        repeat (4) @(negedge clk);

        // ---- A: the exact question asked. 8 lanes of 1.0, every code = 64.
        //         integer code -> 512.0 ; code/128 -> 4.0 ; code/256 -> 2.0
        run_case(8'd64, 8'h00, ACT_ALL_ONE, "A_8lane_act1_code64");

        // ---- B: ONE lane only -> output is a single code x 1.0, i.e. the byte's
        //         own value with no summing ambiguity.
        run_case(8'd64, 8'h00, ACT_ONE_L0,  "B_1lane_act1_code64");

        // ---- C: sign. 8-bit two's complement -64.
        run_case(8'hC0, 8'h00, ACT_ONE_L0,  "C_1lane_act1_code-64");

        // ---- D: the pack8 convention. If the high byte leaks into the product
        //         this changes; qwen25_weight_pack8 always writes 8'h00.
        run_case(8'd64, 8'hAB, ACT_ONE_L0,  "D_1lane_act1_code64_hiAB");

        // ---- E: linearity in the activation (2.0 instead of 1.0).
        run_case(8'd64, 8'h00, ACT_TWO_L0,  "E_1lane_act2_code64");

        // ---- F/G: full-scale codes.
        run_case(8'd127, 8'h00, ACT_ONE_L0, "F_1lane_act1_code127");
        run_case(8'h80, 8'h00, ACT_ONE_L0,  "G_1lane_act1_code-128");
        run_case(8'd0,  8'h00, ACT_ONE_L0,  "H_1lane_act1_code0");

        $display("PROBE_DONE outputs=%0d", b_count);
        $finish;
    end

    initial begin
        #(200*1000*1000);
        $display("FAIL: global timeout");
        $finish;
    end
endmodule
`default_nettype wire
