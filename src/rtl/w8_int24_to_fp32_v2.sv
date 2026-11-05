`timescale 1ns/1ps
`default_nettype none

// INT24(+8 bit 块指数) -> FP32，压缩版。
//
// 与 v1 的区别：
//   * 前导零检测改成 4 bit 分组（6 组）+ 组内 4 bit，替代 24 次 for 循环
//   * 条件取负合并进尾数通路（符号位单独走）
//   * 桶形移位只做 24 bit、5 级，最后一级用整字移位
// 数值与 v1 完全一致（都是精确转换：|S| < 2^24 一定装得进 binary32 的 24 bit 有效位）。
//
// 延迟 2。
module w8_int24_to_fp32_v2 (
    input  wire                clk,
    input  wire                rst_n,
    input  wire                in_valid,
    input  wire signed [23:0]  sum24,
    input  wire [7:0]          block_exp,
    output logic [31:0]        fp32,
    output logic               out_valid
);
    // ---------------------------------------------- 条件取负（符号位单独走）
    wire [23:0] mag = sum24[23] ? (~sum24 + 24'd1) : sum24;

    // ---------------------------------------------- 4 bit 分组前导零
    wire g5 = |mag[23:20], g4 = |mag[19:16], g3 = |mag[15:12];
    wire g2 = |mag[11:8], g1 = |mag[7:4];

    reg [2:0] gi;
    always @(*) begin
        if      (g5) gi = 3'd5;
        else if (g4) gi = 3'd4;
        else if (g3) gi = 3'd3;
        else if (g2) gi = 3'd2;
        else if (g1) gi = 3'd1;
        else         gi = 3'd0;
    end

    reg [3:0] gs;
    always @(*) begin
        case (gi)
            3'd5:    gs = mag[23:20];
            3'd4:    gs = mag[19:16];
            3'd3:    gs = mag[15:12];
            3'd2:    gs = mag[11:8];
            3'd1:    gs = mag[7:4];
            default: gs = mag[3:0];
        endcase
    end

    wire [1:0] lo     = gs[3] ? 2'd0 : gs[2] ? 2'd1 : gs[1] ? 2'd2 : 2'd3;
    // [INT 修复] 原型为 {gi,2'b00}+lo = 4*gi+lo，把"最高非零半字节下标"当成前导零个数，
    // 方向反了：lz 应为 (5-gi)*4+lo。后果是 exp_field 与桶形移位同时错，
    // INT24->FP32 输出全错（INT24 输出本身是对的，所以原型 TB 没抓到）。
    // 定向核对：S=1,E=138 原式给 1048577，正确值应为 1。
    wire [4:0] lz_tot = ({3'd0, (3'd5 - gi)} << 2) + {3'd0, lo};
    wire [4:0] lead   = 5'd23 - lz_tot;

    // ---------------------------------------------- 24 bit 桶形左移
    wire [23:0] s1 = lz_tot[0] ? {mag[22:0], 1'b0} : mag;
    wire [23:0] s2 = lz_tot[1] ? {s1[21:0], 2'b00} : s1;
    wire [23:0] s3 = lz_tot[2] ? {s2[19:0], 4'b0000} : s2;
    wire [23:0] s4 = lz_tot[3] ? {s3[15:0], 8'b0} : s3;
    wire [23:0] s5 = lz_tot[4] ? {s4[7:0], 16'b0} : s4;

    // ---------------------------------------------- 指数重基
    wire signed [9:0] exp_field = $signed({2'b00, block_exp}) - 10'sd11 + $signed({5'b0, lead});

    // ---------------------------------------------- 流水
    logic        s_r, zero_r, ovf_r, v_r;
    logic [22:0] frac_r;
    logic [7:0]  exp_r;

    always_ff @(posedge clk) begin
        if (!rst_n) v_r <= 1'b0;
        else        v_r <= in_valid;
        s_r    <= sum24[23];
        zero_r <= (mag == 24'd0);
        frac_r <= s5[22:0];
        ovf_r  <= (exp_field >= 10'sd255) || (exp_field <= 10'sd0);
        exp_r  <= exp_field[7:0];
    end

    always_ff @(posedge clk) begin
        if (zero_r || ovf_r) fp32 <= 32'h00000000;
        else                 fp32 <= {s_r, exp_r, frac_r};
        out_valid <= v_r;
    end
endmodule
`default_nettype wire
