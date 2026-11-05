`timescale 1ns / 1ps
`default_nettype none
// ===========================================================================
// qwen25_afu_fp_div -- hardware implementation of the FP32 divide core.
//
// Replaces the behavioural model of the same name.  Port list identical,
// including aresetn and aclken, which the vendor core really has
// (Has_ARESETn / Has_ACLKEN were enabled for this instance).
//
// Backing IP : xilinx.com:ip:floating_point:7.1  (ip_fp_div)
//   Operation_Type Divide / Single / NonBlocking / no result tready
//   C_Latency = 28  (the model's LATENCY; the maximum the core offers)
//
// NOTE: no instantiation of qwen25_afu_fp_div exists anywhere in the current
// tree -- it is implemented only so the seven-core set is complete.
// ===========================================================================
module qwen25_afu_fp_div (
    input  wire        aclk, input wire aresetn, input wire aclken,
    input  wire        s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    input  wire        s_axis_b_tvalid, input wire [31:0] s_axis_b_tdata,
    output wire        m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    ip_fp_div core (
        .aclk                (aclk),
        .aclken              (aclken),
        .aresetn             (aresetn),
        .s_axis_a_tvalid     (s_axis_a_tvalid),
        .s_axis_a_tdata      (s_axis_a_tdata),
        .s_axis_b_tvalid     (s_axis_b_tvalid),
        .s_axis_b_tdata      (s_axis_b_tdata),
        .m_axis_result_tvalid(m_axis_result_tvalid),
        .m_axis_result_tdata (m_axis_result_tdata)
    );
endmodule
`default_nettype wire
