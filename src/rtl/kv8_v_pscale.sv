`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// kv8_v_pscale -- fold the per-token V scale into the PV probabilities.
//
// [V-PERTOK-M] A per-token V scale cannot be factored out of the PV reduction
// (the reduction runs over tokens), so it has to enter the reduction itself: the
// attention multiplies the probability of token t by that token's V scale
// mantissa before the accumulate,
//
//     p'[t] = p[t] * m[t]                      (BF16 x 8-bit integer -> BF16, RNE)
//
// and the layer's common power of two 2^(eb_lock-134) stays in kv8_context_scale
// on the context write-back, so
//
//     context = sum_t p[t]*code[t]*m[t]*2^(eb_lock-134)
//             = sum_t p'[t]*code[t]*2^(eb_lock-134)
//
// which is the software model's arithmetic (code * m * 2^(eb-134)): the model
// rounds code*s to BF16, this rounds p*m to BF16, and neither rounds the other.
// m is an exact BF16 integer (1..255), so only the 8x8 significand product needs
// rounding.
//
// [V-MARGIN8-REVERTED] A 16-bit-m version of this module was measured this round
// (V MARGIN 2->8): it cost 15 067 LUT instead of 3 267 for 64 lanes because a
// 24-bit leading-one normalise is a barrel shifter, while the corr gain was
// +0.000054 (V is limited by its 8-bit CODE -- floor 0.8635 % -- not by the
// scale).  So the 8-bit-m datapath is the right cost point and MARGIN stays 2.
// If V's code is ever widened to 16 bits, THIS module must be widened with it
// (see the 16-bit version in the round's history / REPORT_STAGE6_K16.md §10).
//
// Cost target (owner's decision, 2026-11-02): the significand product is forced
// into a DSP48E2 (`use_dsp`), because a 27x18 DSP per lane costs no LUT while the
// LUT-mapped version measured 8 320 LUT for 64 lanes (8.7 % of the KV260).  This
// is why the module is CLOCKED now: the product register is what maps to the DSP's
// M register, and the normalize/round after it is one conditional shift + RNE.
// The attention loads the scaled activation one state later (S_G_BEGIN instead of
// S_P_WAIT), so the extra register costs no cycle.
//
// Scaling p BEFORE the array (not the aligned mantissa after it) keeps the INT24
// accumulator bounded: the array normalises each K-block of 8 tokens to the
// largest p*m in that block, so sum <= 2048*127*(sum p*m)/(p*m)max, i.e. the same
// order as before the change.  Scaling after the alignment would inflate the sum
// by up to 255x and clamp.
//
// BF16 conventions follow the tree's bf16_mul.sv: an exponent-zero probability is
// zero (DAZ) and a result above the BF16 range saturates.
// =============================================================================
module kv8_v_pscale #(parameter integer LANES = 8) (
    input  wire                clk,
    input  wire [LANES*16-1:0] i_p,       // LANES x BF16 probability
    input  wire [LANES*16-1:0] i_m,       // LANES x 16-bit mantissa (low 8 bits used)
    output wire [LANES*16-1:0] o_p        // LANES x BF16 p*m, valid 1 cycle after i_*
);
    function [2:0] msb8(input [7:0] m);
        begin
            if      (m[7]) msb8 = 3'd7;
            else if (m[6]) msb8 = 3'd6;
            else if (m[5]) msb8 = 3'd5;
            else if (m[4]) msb8 = 3'd4;
            else if (m[3]) msb8 = 3'd3;
            else if (m[2]) msb8 = 3'd2;
            else if (m[1]) msb8 = 3'd1;
            else           msb8 = 3'd0;
        end
    endfunction

    genvar gi;
    generate
        for (gi = 0; gi < LANES; gi = gi + 1) begin : gen_lane
            wire [15:0] p  = i_p[gi*16 +: 16];
            wire [7:0]  m8 = i_m[gi*16 +: 8];
            wire [2:0]  k  = msb8(m8);
            // m -> 1.xxxxxxx * 2^k : the significand of an exact BF16 integer
            wire [7:0]  mm = (m8 == 8'd0) ? 8'h00 : (m8 << (3'd7 - k));

            // 9x9 significand product in a DSP (8x8 is the real size; the top bit
            // of each operand is the implicit 1, so the product is in [2^14,2^16))
            (* use_dsp = "yes" *) reg [15:0] pr;
            reg [7:0] ep_r;
            reg [2:0] k_r;
            reg       sg_r;
            reg       vl_r;
            always @(posedge clk) begin
                pr   <= {1'b1,p[6:0]} * mm;
                ep_r <= p[14:7];
                k_r  <= k;
                sg_r <= p[15];
                vl_r <= (m8 != 8'd0) && (p[14:7] != 8'd0);
            end

            // one conditional normalize + RNE (both significands are in [1,2),
            // so the product is in [1,4): at most one bit of normalization)
            wire        hi   = pr[15];
            wire [7:0]  mant = hi ? pr[15:8] : pr[14:7];
            wire        rnd  = hi ? pr[7]    : pr[6];
            wire        stk  = hi ? (|pr[6:0]) : (|pr[5:0]);
            wire        rup  = rnd && (stk || mant[0]);
            wire [8:0]  mant9 = {1'b0,mant} + {8'd0,rup};
            wire [8:0]  ex    = {1'b0,ep_r} + {6'd0,k_r} + {8'd0,hi};
            wire [8:0]  ex2   = mant9[8] ? (ex + 9'd1) : ex;
            wire [7:0]  mantf = mant9[8] ? 8'd0 : mant9[7:0];

            assign o_p[gi*16 +: 16] =
                (!vl_r)               ? 16'h0000 :
                (ex2 == 9'd0)         ? 16'h0000 :
                (ex2 > 9'd254)        ? {sg_r, 8'hff, 7'h00} :
                                        {sg_r, ex2[7:0], mantf[6:0]};
        end
    endgenerate
endmodule
`default_nettype wire
