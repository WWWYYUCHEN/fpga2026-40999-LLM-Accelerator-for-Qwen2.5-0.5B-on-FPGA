`timescale 1ns / 1ps
`default_nettype none

// 带系统复位有效性隔离的FP32算子。计算就绪只由底层output_valid报告。
// reset_token仅过滤复位前已发事务，不用它替代算子完成信号。
module qwen_fp32_add_valid (
    input wire clk,
    input wire rst_n,
    input wire clear,
    input wire input_valid,
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] result,
    output wire output_valid
);
    logic [6:0] reset_token;
    wire operator_valid;
    fp32_add operator_Inst (
        .clk(clk), .input_valid(input_valid),
        .a(a), .b(b), .result(result), .output_valid(operator_valid)
    );
    assign output_valid = operator_valid && reset_token[6];
    always_ff @(posedge clk) begin
        if (!rst_n || clear) reset_token <= '0;
        else reset_token <= {reset_token[5:0], input_valid};
    end
endmodule
`default_nettype wire
