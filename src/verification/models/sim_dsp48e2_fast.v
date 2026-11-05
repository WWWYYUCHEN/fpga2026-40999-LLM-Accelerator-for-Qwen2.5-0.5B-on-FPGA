// =============================================================================
// Behavioural DSP48E2 for the Verilator E2E regression (work_gemmwire only).
//
// WHY THIS FILE EXISTS
// models/sim_dsp48e2_verilator.v is the faithful vendor unisim model, but with
// 128 instances in qwen_gemm_array_i it costs ~3.3x simulation speed
// (13.4 kHz vs 44 kHz) because of its 48-bit ALU case muxes and the circular
// combinational logic Verilator has to converge every cycle.
//
// SCOPE (v2 - register coverage fixed)
// v1 hard-coded AREG=BREG=0, MREG=PREG=1.  That is correct for
// qwen_gemm_array_i, but the controller instantiates four more DSP48E2s with
// other configurations (AREG=0 BREG=1 MREG=0 PREG=1 for the layer/batch address
// offsets, MREG=1 PREG=0 for reload_offset, MREG=0 PREG=1 for
// bias_layer_offset).  v1 gave two of them one extra cycle of latency, which
// misaligned the address/valid pairs the controller matched to the DSP's
// internal stages and changed the LOGITS of an otherwise unchanged baseline
// (token 31490 instead of the golden 9707, identical cycle count, identical
// layer-0 dumps).  v2 therefore implements the register chain generically:
//
//   A -> [AREG stages] -> a_eff     B -> [BREG stages] -> b_eff
//   prod = a_eff * b_eff            m_eff = MREG ? m_reg : prod
//   P    = PREG ? p_reg : m_eff + PCIN(t)      (PCIN is NOT registered)
//
// Supported: AREG/BREG 0..2, MREG/PREG 0/1, OPMODE 9'b000010101 (P=M+PCIN) and
// 9'b000000101 (P=M with PCIN tied 0), ALUMODE=0, C=D=0, RST*=0, CE*=1,
// glblGSR=0 (the Verilator build ties it low), USE_MULT="MULTIPLY",
// USE_SIMD="ONE48".
//
// VALIDATION: base (rtl_base) built with this file must give token 9707 and
// logits bit-identical to the vendor-model build (which equals e2e_run\runD).
// See REPORT.md section 5.5.
// =============================================================================
`timescale 1 ps / 1 ps
`celldefine

module DSP48E2 #(
  parameter integer ACASCREG = 1,
  parameter integer ADREG = 1,
  parameter integer ALUMODEREG = 1,
  parameter AMULTSEL = "A",
  parameter integer AREG = 1,
  parameter AUTORESET_PATDET = "NO_RESET",
  parameter AUTORESET_PRIORITY = "RESET",
  parameter A_INPUT = "DIRECT",
  parameter integer BCASCREG = 1,
  parameter BMULTSEL = "B",
  parameter B_INPUT = "DIRECT",
  parameter integer BREG = 1,
  parameter integer CARRYINREG = 1,
  parameter integer CARRYINSELREG = 1,
  parameter integer CREG = 1,
  parameter integer DREG = 1,
  parameter integer INMODEREG = 1,
  parameter integer MREG = 1,
  parameter integer OPMODEREG = 1,
  parameter PREADDINSEL = "A",
  parameter integer PREG = 1,
  parameter RESET_MODE = "SYNC",
  parameter [47:0] SEL_MASK = 48'h3fffffffffff,
  parameter [47:0] SEL_PATTERN = 48'h0,
  parameter USE_MULT = "MULTIPLY",
  parameter USE_PATTERN_DETECT = "NO_PATDET",
  parameter USE_SIMD = "ONE48",
  parameter USE_WIDEXOR = "FALSE",
  parameter XORSIMD = "XOR24_48_96"
)(
  output [29:0] ACOUT,
  output [17:0] BCOUT,
  output CARRYCASCOUT,
  output [3:0] CARRYOUT,
  output MULTSIGNOUT,
  output OVERFLOW,
  output [47:0] P,
  output PATTERNBDETECT,
  output PATTERNDETECT,
  output [47:0] PCOUT,
  output UNDERFLOW,
  output [7:0] XOROUT,

  input [29:0] A,
  input [29:0] ACIN,
  input [3:0] ALUMODE,
  input [17:0] B,
  input [17:0] BCIN,
  input [47:0] C,
  input CARRYCASCIN,
  input CARRYIN,
  input [2:0] CARRYINSEL,
  input CEA1,
  input CEA2,
  input CEAD,
  input CEALUMODE,
  input CEB1,
  input CEB2,
  input CEC,
  input CECARRYIN,
  input CECTRL,
  input CED,
  input CEINMODE,
  input CEM,
  input CEP,
  input CLK,
  input [26:0] D,
  input [4:0] INMODE,
  input MULTSIGNIN,
  input [8:0] OPMODE,
  input [47:0] PCIN,
  input RSTA,
  input RSTALLCARRYIN,
  input RSTALUMODE,
  input RSTB,
  input RSTC,
  input RSTCTRL,
  input RSTD,
  input RSTINMODE,
  input RSTM,
  input RSTP
);
  // ---------------- pre-adder / INMODE A-input mux  [HYB FIX]
  // The vendor model's measured table for this design's configurations
  // (AMULTSEL="AD", PREADDINSEL="A", DREG=0), swept over all 32 INMODE values
  // with A=1792, D=9728 (see work_hyb\dsp\tab.py, 128 (INMODE,B) points):
  //   INMODE[4]=1                  -> 0
  //   INMODE[3:2]=00, [1:0]==00    -> A          else 0
  //   INMODE[3:2]=01, [1:0]==00    -> D + A      else D
  //   INMODE[3:2]=10, [1:0]==00    -> -A         else 0
  //   INMODE[3:2]=11, [1:0]==00    -> D - A      else D
  // Without this the reload_offset_Inst instance (AMULTSEL="AD",
  // INMODE={2'b00,is_m,is_m,1'b0}) always returns 1792*B instead of
  // 9728*B when is_m=1, which is wrong on 93 of the 128 sweep points and
  // only shows up after a layer/batch transition (layer 0 is blind to it).
  wire signed [47:0] d_ext  = $signed({21'd0, D});
  wire signed [47:0] a_ext  = $signed({{18{A[29]}}, A});
  wire signed [47:0] preadd = (INMODE[3:2] == 2'b01) ? (d_ext + a_ext) :
                              (INMODE[3:2] == 2'b11) ? (d_ext - a_ext) :
                              (INMODE[3:2] == 2'b10) ? (-a_ext)        : 48'sd0;
  wire signed [47:0] a_mux  = (INMODE[4] != 0)       ? 48'sd0 :
                              (INMODE[1:0] != 2'b00) ? (((INMODE[3:2] == 2'b01) ||
                                                         (INMODE[3:2] == 2'b11)) ? d_ext : 48'sd0) :
                              (INMODE[3:2] == 2'b00) ? a_ext : preadd;
  wire signed [47:0] a_in_w = (AMULTSEL == "AD") ? a_mux : a_ext;
  wire signed [29:0] a_in   = a_in_w[29:0];

  // ---------------- A path (AREG stages)
  reg signed [29:0] a_r0, a_r1;
  always @(posedge CLK) begin
    a_r0 <= a_in;
    a_r1 <= a_r0;
  end
  wire signed [29:0] a_eff = (AREG == 0) ? a_in :
                             (AREG == 1) ? a_r0       : a_r1;

  // ---------------- B path (BREG / BCASCREG stages)
  reg signed [17:0] b_r0, b_r1;
  always @(posedge CLK) begin
    b_r0 <= $signed(B);
    b_r1 <= b_r0;
  end
  wire signed [17:0] b_eff = (BREG == 0) ? $signed(B) :
                             (BREG == 1) ? b_r0       : b_r1;

  // ---------------- multiplier + MREG
  // A (30b, sign-extended) x B (18b) is exact in 48 bits for every
  // instantiation in this design.
  wire signed [47:0] prod = a_eff * b_eff;
  reg  signed [47:0] m_reg;
  always @(posedge CLK) m_reg <= prod;
  wire signed [47:0] m_eff = (MREG != 0) ? m_reg : prod;

  // ---------------- ALU (OPMODE = M or M+PCIN) + PREG
  // OPMODE 9'b000010101 -> Z mux = PCIN ; 9'b000000101 -> Z mux = 0
  wire               use_pcin = OPMODE[2];
  wire signed [47:0] alu      = m_eff + (use_pcin ? $signed(PCIN) : 48'sd0);
  reg  signed [47:0] p_reg;
  always @(posedge CLK) p_reg <= alu;

  assign P              = (PREG != 0) ? p_reg : alu;
  assign PCOUT          = P;
  assign ACOUT          = A;
  assign BCOUT          = B;
  assign CARRYCASCOUT   = 1'b0;
  assign CARRYOUT       = 4'd0;
  assign MULTSIGNOUT    = 1'b0;
  assign OVERFLOW       = 1'b0;
  assign UNDERFLOW      = 1'b0;
  assign PATTERNBDETECT = 1'b0;
  assign PATTERNDETECT  = 1'b0;
  assign XOROUT         = 8'd0;
endmodule
`endcelldefine
