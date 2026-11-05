`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_div_seq -- one-bit-per-cycle restoring divider for the INT8 KV quantisers.
//
// [RECIP-SEQ] WHY THIS EXISTS (measured, not reasoned):
//   kv8_k_quant / kv8_v_quant used to compute  R = round(2^RSHIFT/m)  and
//   m = ceil(num/127) with *combinational* `/' operators.  Synthesis expanded
//   them into restoring dividers and they became the worst path in the whole
//   accelerator (Vivado 2026.1, xck26-sfvc784-2LV-c, 5.000 ns OOC):
//       q_data_reg[97] -> k_quant_Inst/recip_r_reg[0]   142 levels  34.874 ns
//                                                        slack -29.877 ns
//       q_data_reg[97] -> k_quant_Inst/m_r_reg[4]        44 levels  14.545 ns
//                                                        slack  -9.545 ns
//   The divider alone was ~98 levels / ~20.3 ns of that 34.874 ns.
//
//   Latency is free here: one reciprocal is needed per (head, token) and a
//   token costs ~13 M cycles (134 082 442 cycles for the 1-token prefill), so
//   16..25 iterations are invisible.  The consumers (kv8_k_write_chain /
//   kv8_v_write_chain) already wait for the quantiser's o_done / o_lock_done /
//   o_restore_done pulses, so the extra latency is absorbed by the handshake.
//
// ALGORITHM (plain restoring division, one quotient bit per cycle):
//     rem' = (rem<<1) | num[MSB]
//     if (rem' >= den) begin rem <= rem' - den; q <= (q<<1)|1; end
//       else           begin rem <= rem';       q <= (q<<1)|0; end
//   After NUMW cycles:  q = floor(num/den),  rem = num mod den   (den >= 1)
// This is *exactly* what Verilog's `/` and `%` mean for unsigned operands, so
// the numeric result is bit-identical to the combinational version -- proven
// exhaustively (every reachable operand pair) by tb_kv8_quant_equiv.sv.
//
// Widths are chosen by the instantiation:
//   K: NUMW=25 (2^24 + m/2 needs 25 bits), DIVW=16
//   V: NUMW=24 (lock numerator <= 2^23),   DIVW=16
// Cost per instance, measured post-synth: ~40 LUT + ~90 FF (see
// REPORT_TIMING_CHEAP.md §4), replacing ~600 LUT of combinational dividers.
// -----------------------------------------------------------------------------
module kv8_div_seq #(
    parameter integer NUMW = 25,        // dividend / quotient width
    parameter integer DIVW = 16         // divisor width; remainder needs DIVW+1
) (
    input  wire             clk,
    input  wire             rst_n,
    input  wire             i_start,    // 1-cycle pulse, wins over a running divide
    input  wire [NUMW-1:0]  i_num,      // dividend (unsigned)
    input  wire [DIVW-1:0]  i_den,      // divisor  (unsigned, must be >= 1)
    output wire             o_busy,
    output wire             o_done,     // 1-cycle pulse: o_quo/o_rem valid now
    output wire [NUMW-1:0]  o_quo,      // floor(i_num/i_den)
    output wire [DIVW-1:0]  o_rem       // i_num mod i_den
);
    // ------------------------------------------------------------------ widths
    // $clog2(NUMW+1) holds NUMW itself, so the counter can start at NUMW.
    localparam integer CW = $clog2(NUMW+1);
    localparam [CW-1:0] NCYC = NUMW;

    reg [NUMW-1:0] num_s;      // denominator-side shift register (dividend)
    reg [NUMW-1:0] quo_s;      // quotient, shifted in from the LSB
    reg [DIVW:0]   rem_s;      // partial remainder < den, so DIVW+1 bits
    reg [DIVW-1:0] den_r;
    reg [CW-1:0]   cnt;
    reg            busy_r;
    reg            done_r;

    // next partial remainder = (rem<<1) | next dividend bit
    wire [DIVW:0] rem_sh  = {rem_s[DIVW-1:0], num_s[NUMW-1]};
    wire [DIVW:0] den_zx  = {1'b0, den_r};
    wire          ge      = (rem_sh >= den_zx);
    wire [DIVW:0] rem_sub = rem_sh - den_zx;

    always @(posedge clk) begin
        if (!rst_n) begin
            num_s <= {NUMW{1'b0}}; quo_s <= {NUMW{1'b0}}; rem_s <= {(DIVW+1){1'b0}};
            den_r <= {DIVW{1'b0}}; cnt   <= {CW{1'b0}};
            busy_r <= 1'b0; done_r <= 1'b0;
        end else begin
            done_r <= 1'b0;
            if (i_start) begin
                num_s  <= i_num;
                den_r  <= i_den;
                quo_s  <= {NUMW{1'b0}};
                rem_s  <= {(DIVW+1){1'b0}};
                cnt    <= NCYC;
                busy_r <= 1'b1;
            end else if (busy_r) begin
                num_s <= {num_s[NUMW-2:0], 1'b0};
                rem_s <= ge ? rem_sub : rem_sh;
                quo_s <= {quo_s[NUMW-2:0], ge};
                cnt   <= cnt - 1'b1;
                if (cnt == {{(CW-1){1'b0}}, 1'b1}) begin
                    busy_r <= 1'b0;
                    done_r <= 1'b1;
                end
            end
        end
    end

    assign o_busy = busy_r;
    assign o_done = done_r;
    assign o_quo  = quo_s;
    assign o_rem  = rem_s[DIVW-1:0];
endmodule
`default_nettype wire
