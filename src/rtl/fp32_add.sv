`timescale 1ns / 1ps
`default_nettype none

// GEMM uses the user's Floating-Point 7.1 ADD IP: FP32, NonBlocking, latency 7.
module fp32_add (
    input  wire        clk,
    input  wire        input_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result,
    output wire        output_valid
);
    floating_point_0 floating_add_Inst (
        .aclk(clk),
        .s_axis_a_tvalid(input_valid),
        .s_axis_a_tdata(a),
        .s_axis_b_tvalid(input_valid),
        .s_axis_b_tdata(b),
        .m_axis_result_tvalid(output_valid),
        .m_axis_result_tdata(result)
    );
endmodule

`default_nettype wire
