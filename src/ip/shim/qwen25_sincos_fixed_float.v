`timescale 1ns / 1ps
`default_nettype none
// ===========================================================================
// qwen25_sincos_fixed_float -- hardware implementation of the Q2.46 -> FP32
// converter used by the RoPE sin/cos path.
//
// Replaces  models\qwen25_sincos_verilator.sv :: qwen25_sincos_fixed_float
//          (newdma\work_afu\sim_ip_models.sv :: qwen25_sincos_fixed_float)
// Port list identical.
//
// Backing IP : xilinx.com:ip:floating_point:7.1  (ip_fixed_to_float)
//   Operation_Type Fixed_to_float
//   A_Precision_Type Custom, C_A_Exponent_Width 2, C_A_Fraction_Width 46
//                                 -> signed 48-bit Q2.46, i.e. value/2^46
//   Result_Precision_Type Single
//   C_Latency = 4
//
// The latency MUST be 4: rtl\rope\qwen25_rope_sincos.v:83 gates the result with
// qwen_operator_valid_guard #(.C_LATENCY(4)).  The behavioural model's own
// comment claims the generated core had C_LATENCY=7; 7 is not reachable for
// this configuration (the legal range is 1..6) and would break that guard, so
// the core is built at 4 like the RTL and the model demand.
// ===========================================================================
module qwen25_sincos_fixed_float (
    input  wire        aclk,
    input  wire        s_axis_a_tvalid,
    input  wire [47:0] s_axis_a_tdata,
    output wire        m_axis_result_tvalid,
    output wire [31:0] m_axis_result_tdata
);
    ip_fixed_to_float core (
        .aclk                (aclk),
        .s_axis_a_tvalid     (s_axis_a_tvalid),
        .s_axis_a_tdata      (s_axis_a_tdata),
        .m_axis_result_tvalid(m_axis_result_tvalid),
        .m_axis_result_tdata (m_axis_result_tdata)
    );
endmodule
`default_nettype wire


