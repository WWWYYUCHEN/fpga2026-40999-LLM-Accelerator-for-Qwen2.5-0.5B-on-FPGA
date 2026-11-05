`timescale 1ns / 1ps
`default_nettype none

// 精确BF16乘积到FP32接线；operator_valid逐级驱动归约，reset只取消旧token。
module qwen_bf16_mul_valid (
    input wire clk,
    input wire rst_n,
    input wire clear,
    input wire input_valid,
    input wire [15:0] a,
    input wire [15:0] b,
    output wire [31:0] result,
    output wire output_valid
);
    logic [2:0] reset_token;
    wire [24:0] exact_product;
    wire operator_valid;
    bf16_mul operator_Inst (
        .clk(clk), .input_valid(input_valid),
        .a(a), .b(b), .result(exact_product), .output_valid(operator_valid)
    );
    assign result = {exact_product[24:16], exact_product[14:0], 8'b0};
    assign output_valid = operator_valid && reset_token[2];
    always_ff @(posedge clk) begin
        if (!rst_n ) reset_token <= '0;
        else reset_token <= {reset_token[1:0], input_valid};
    end
endmodule
`default_nettype wire
