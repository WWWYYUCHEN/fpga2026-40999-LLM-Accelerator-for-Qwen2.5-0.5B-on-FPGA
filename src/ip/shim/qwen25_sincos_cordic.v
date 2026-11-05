`timescale 1ns / 1ps
`default_nettype none
// ===========================================================================
// qwen25_sincos_cordic -- hardware implementation of the sin/cos core.
//
// Replaces  models\qwen25_sincos_verilator.sv :: qwen25_sincos_cordic
//          (newdma\work_afu\sim_ip_models.sv :: qwen25_sincos_cordic)
// Port list identical, including s_axis_phase_tready.
//
// Backing IP : xilinx.com:ip:cordic:6.0  (ip_cordic_sincos)
//   Architectural_Configuration Word_Serial
//   Functional_Selection        Sin_and_Cos
//   Phase_Format                Scaled_Radians   (pi radians per unit)
//   Data_Format                 SignedFraction
//   Input_Width 48, Output_Width 48   -> Q3.45 phase in, two Q2.46 outputs
//   Coarse_Rotation true, Round_Mode Nearest_Even, NonBlocking
//
// PARAMETERS: C_LATENCY / C_INITIATION_INTERVAL are kept only so that existing
// instantiations keep elaborating.  A vendor core's latency and initiation
// interval are fixed by its configuration and cannot be reprogrammed from RTL,
// so these parameters are DOCUMENTATION ONLY.  The latency this core actually
// produces is measured in sim/ (see REPORT_IPGEN.md); the surrounding
// rtl\rope\qwen25_rope_sincos.v does not hard-code it -- it uses
// s_axis_phase_tready / m_axis_dout_tvalid only -- so a different latency
// changes throughput, not function.
// ===========================================================================
module qwen25_sincos_cordic #(
    parameter integer C_LATENCY = 52,
    parameter integer C_INITIATION_INTERVAL = 46
) (
    input  wire        aclk,
    input  wire        s_axis_phase_tvalid,
    output wire        s_axis_phase_tready,
    input  wire [47:0] s_axis_phase_tdata,
    output wire        m_axis_dout_tvalid,
    output wire [95:0] m_axis_dout_tdata
);
    ip_cordic_sincos core (
        .aclk               (aclk),
        .s_axis_phase_tvalid(s_axis_phase_tvalid),
        .s_axis_phase_tready(s_axis_phase_tready),
        .s_axis_phase_tdata (s_axis_phase_tdata),
        .m_axis_dout_tvalid (m_axis_dout_tvalid),
        .m_axis_dout_tdata  (m_axis_dout_tdata)
    );
endmodule
`default_nettype wire
