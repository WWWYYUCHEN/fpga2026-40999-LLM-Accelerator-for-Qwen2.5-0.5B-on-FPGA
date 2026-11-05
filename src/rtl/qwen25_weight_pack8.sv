`timescale 1ns/1ps
`default_nettype none

// ============================================================================
// qwen25_weight_pack8 —— qwen25_weight_dequant 的 INT8 打包变体（步骤 3 的低风险部分）
//
// 语义：与 qwen25_weight_dequant 完全相同的握手/寄存器/地址通路，只是把
//       "INT8 -> BF16 精确转换"（qwen_int8_to_bf16，16 bit 用满）换成
//       "INT8 码放进 16 bit 槽的低字节"（高字节写 0）。
//
// 为什么这样是安全的：
//   * 权重 buffer 的投影权重只被 GEMM 读（write 口来自本模块；read 口 weight_q
//     供 GEMM 与 attention 的 K/V 共用，而 K/V 是 attention 自己写进去的 BF16，
//     不经过本模块）；
//   * 阵列 qwen_gemm_array_i 只吃每个 16 bit 槽的低 8 bit（见其装载逻辑），
//     高字节被忽略，所以"格式"从 bf16(int8) 换成 {8'h00,int8} 对阵列等价；
//   * 位宽/地址/深度/bank 全部不变，所以 buffer、K^T 转置、V 通路都不用动。
//   * 省掉 16 个 qwen_int8_to_bf16（实测 1800 LUT / 800 FF，见 MMU面积分析.md:57,96）。
//
// 注意：本模块**不是**把 buffer 改成"每槽 8 bit"（那需要改读口宽度、K^T 和 V 通路，
// 是 REPORT 里的 step 3 剩余工作）。
// ============================================================================
module qwen25_weight_pack8 (
    input  wire        i_clk,
    input  wire        i_rstn,
    input  wire        i_valid,
    output wire        o_ready,
    input  wire [127:0] i_data,
    input  wire [9:0]  i_word,
    input  wire [2047:0] i_scales,
    output wire        o_valid,
    input  wire        i_ready,
    output wire [255:0] o_data,
    output wire [15:0] o_word,
    output wire        o_busy
);
    reg        valid_reg;
    reg [255:0] data_reg;
    reg [9:0]  word_reg;
    wire [255:0] pk_word;
    genvar lane;
    assign o_ready = !valid_reg || i_ready;
    assign o_valid = valid_reg;
    assign o_data  = data_reg;
    assign o_word  = {6'd0, word_reg};
    assign o_busy  = valid_reg;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : gen_pack
            assign pk_word[lane*16 +: 8]     = i_data[lane*8 +: 8];
            assign pk_word[lane*16 + 8 +: 8] = 8'h00;
        end
    endgenerate

    always @(posedge i_clk) begin
        if (!i_rstn) valid_reg <= 1'b0;
        else if (o_ready) begin
            valid_reg <= i_valid;
            if (i_valid) begin
                data_reg <= pk_word;
                word_reg <= i_word;
            end
        end
    end
endmodule
`default_nettype wire
