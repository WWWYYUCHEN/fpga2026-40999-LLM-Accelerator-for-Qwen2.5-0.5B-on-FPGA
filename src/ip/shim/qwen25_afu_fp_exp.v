`timescale 1ns / 1ps
`default_nettype none
// ===========================================================================
// qwen25_afu_fp_exp -- hardware implementation of the FP32 exponential core.
//
// Replaces the behavioural model of the same name.  Port list identical.
// Backing IP : xilinx.com:ip:floating_point:7.1  (ip_fp_exp)
//   Operation_Type Exponential / Single / NonBlocking / no result tready
//   C_Latency = 16  (matches qwen_operator_valid_guard #(.C_LATENCY(16)) at
//                    rtl\afu\qwen25_afu_lane.v:54)
// ===========================================================================
module qwen25_afu_fp_exp (
    input  wire        aclk,
    input  wire        s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    output wire        m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    ip_fp_exp core (
        .aclk                (aclk),
        .s_axis_a_tvalid     (s_axis_a_tvalid),
        .s_axis_a_tdata      (s_axis_a_tdata),
        .m_axis_result_tvalid(m_axis_result_tvalid),
        .m_axis_result_tdata (m_axis_result_tdata)
    );
endmodule
`default_nettype wire


