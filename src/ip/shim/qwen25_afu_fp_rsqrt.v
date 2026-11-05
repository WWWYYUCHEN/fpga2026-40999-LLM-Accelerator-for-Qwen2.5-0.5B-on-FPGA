`timescale 1ns / 1ps
`default_nettype none
// ===========================================================================
// qwen25_afu_fp_rsqrt -- hardware implementation of the FP32 1/sqrt core.
//
// Replaces the behavioural model of the same name.  Port list identical.
// Backing IP : xilinx.com:ip:floating_point:7.1  (ip_fp_rsqrt)
//   Operation_Type Rec_Square_Root / Single / NonBlocking / no result tready
//   C_Latency = 19  (matches qwen_operator_valid_guard #(.C_LATENCY(19)) at
//                    rtl\afu\qwen25_afu.v:273)
// ===========================================================================
module qwen25_afu_fp_rsqrt (
    input  wire        aclk,
    input  wire        s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    output wire        m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    ip_fp_rsqrt core (
        .aclk                (aclk),
        .s_axis_a_tvalid     (s_axis_a_tvalid),
        .s_axis_a_tdata      (s_axis_a_tdata),
        .m_axis_result_tvalid(m_axis_result_tvalid),
        .m_axis_result_tdata (m_axis_result_tdata)
    );
endmodule
`default_nettype wire


