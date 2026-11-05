`timescale 1ns / 1ps
`default_nettype none

// GEMM uses the user's Floating-Point 7.1 ADD IP: FP32, NonBlocking, latency 7.
//
// [OPT-3167] The IP has NINE ports; the original wrapper connected seven and left
//   s_axis_operation_tvalid / s_axis_operation_tdata floating.  Synthesis only
//   warns ("[Opt 31-155] Driverless net ... s_axis_operation_tdata[0] is driving
//   LUT input pin I0"), but opt_design then aborts 49 times with
//       ERROR: [Opt 31-67] A LUT2 cell ... is missing a connection on input pin I0
//   (one per fp32_add instance; full text in work_synth200\logs\impl200_vivado.log).
//   The operation field is tied to ADD (tdata = 8'd0 == add, tvalid = 1'b1), which
//   is the only operation this wrapper performs and what the ADD_SUBTRACT IP needs
//   to see on every beat.  Numerically identical to the previous behaviour.
//
// Why the `ifdef VERILATOR around it: the behavioural stand-in for this IP used by
//   the Verilator E2E build (models\qwen25_scalar_math_verilator.sv, shared with the
//   A baseline) declares `floating_point_0' with the seven data ports only and
//   hard-codes OPERATION=0 (add).  Verilator rejects the two extra pins outright
//   (%Error-PINNOTFOUND), so the reduced instantiation is kept for that model only;
//   every synthesis/simulation tool that has the real IP sees the full port list.
//   The two forms are functionally identical (both select ADD).
module fp32_add (
    input  wire        clk,
    input  wire        input_valid,
    input  wire [31:0] a,
    input  wire [31:0] b,
    output wire [31:0] result,
    output wire        output_valid
);
`ifdef VERILATOR
    // models\qwen25_scalar_math_verilator.sv: 7 ports, operation fixed to ADD
    floating_point_0 floating_add_Inst (
        .aclk(clk),
        .s_axis_a_tvalid(input_valid),
        .s_axis_a_tdata(a),
        .s_axis_b_tvalid(input_valid),
        .s_axis_b_tdata(b),
        .m_axis_result_tvalid(output_valid),
        .m_axis_result_tdata(result)
    );
`else
    // real Floating-Point 7.1 IP: operation must be driven (ADD = 8'd0)
    floating_point_0 floating_add_Inst (
        .aclk(clk),
        .s_axis_a_tvalid(input_valid),
        .s_axis_a_tdata(a),
        .s_axis_b_tvalid(input_valid),
        .s_axis_b_tdata(b),
        .s_axis_operation_tvalid(1'b1),
        .s_axis_operation_tdata(8'd0),
        .m_axis_result_tvalid(output_valid),
        .m_axis_result_tdata(result)
    );
`endif
endmodule

`default_nettype wire
