`timescale 1ns / 1ps
`default_nettype none

// ============================================================================
// qwen_gemm_int24_bridge —— 替换 qwen_fp32_sum8 的 16 列 INT24 -> FP32 桥接组
//
// 位置：阵列输出 FIFO 之后、qwen_gemm_partial 之前（即"第二级"入口）。
// 块指数由阵列的身份 FIFO 一起送过来（m_meta[167:160]），所以这里拿到的
// 一定是产生这 384 bit 段和的那一块的指数。
//
// 延迟 2 拍；out_valid 与 in_valid 对齐；inflight 覆盖这 2 拍在途事务，
// 供上层把 busy 延长，避免"阵列 credit 已清零但桥接还在飞"导致提前读 partial。
// ============================================================================
module qwen_gemm_int24_bridge #(parameter integer COL = 16) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,
    input  wire         in_valid,
    input  wire [COL*24-1:0] int24_flat,
    input  wire [7:0]   block_exp,
    output wire [COL*32-1:0] fp32_flat,
    output wire         out_valid,
    output wire         inflight
);
    genvar gc;
    generate
        for (gc = 0; gc < COL; gc = gc + 1) begin : gen_conv
            w8_int24_to_fp32_v2 conv_Inst (
                .clk(clk), .rst_n(rst_n), .in_valid(in_valid),
                .sum24(int24_flat[gc*24 +: 24]), .block_exp(block_exp),
                .fp32(fp32_flat[gc*32 +: 32]), .out_valid()
            );
        end
    endgenerate

    // 转换器固定 2 拍（见 w8_int24_to_fp32_v2），valid 与在途计数对齐
    logic [1:0] vld;
    logic [1:0] busy_pipe;
    always_ff @(posedge clk) begin
        if (!rst_n ) begin
            vld <= 2'd0;
            busy_pipe <= 2'd0;
        end else begin
            vld <= {vld[0], in_valid};
            busy_pipe <= {busy_pipe[0], in_valid};
        end
    end
    assign out_valid = vld[1];
    assign inflight  = busy_pipe[0] | busy_pipe[1];
endmodule
`default_nettype wire
