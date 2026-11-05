`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_score_scale8 -- 8-lane score scaling for the QK^T and PV paths.
//
// MANT_BITS=8 (DEFAULT, the shipped behaviour, bit-for-bit):
//   y[lane] = x[lane] * s[lane]   (8 x BF16 multiply + RNE)
//   This is the module that replaced the fixed 1/8 exponent decrement of
//   rtl/attention/qwen25_score_ingress.v:90-94.  The 1/8 is folded into the
//   stored table (kv8_scale_ram stored s_k/8), so the block is exactly one BF16
//   multiply per lane.  Latency 2, one word (8 lanes) per cycle, no backpressure.
//   inf/nan pass through; zero inputs flush to zero (matches the score_ingress
//   flush for masked columns).
//
// MANT_BITS=16 (the m16 / split-INT8 scale path):
//   i_scale = 8 x 16-bit INTEGER mantissa, i_eb = the block/table exponent.
//       y = a * m16 * 2^(eb_table - CEX)          per lane
//   with a = 1.f * 2^(Ea-127) (BF16) and (from the write chain)
//       s_k/8 = m16 * 2^(eb_table - 127)
//   so  y = a * s_k/8 * 2^(134-CEX+127-... ) -- see the derivation below.
//
//   [CONSTANT] CEX is the scale constant, not a fudge factor:
//       CEX = 268 for K : 268 = 134 + 134,   134 = BF16 bias + significand width
//                                            (127+7), the second 134 = bias + the
//                                            retired 2^-7 (127+7).  With 268 the
//                                            multiply ALREADY contains the 2^-7,
//                                            so kv8_score_scale_fix must be
//                                            deleted on the K path or every score
//                                            comes out 128x too small.
//       CEX = 261 for V : 261 = 134 + 127.   CEX_K - CEX_V = 7 IS the 1/sqrt(d)
//                                            (V's table byte is eb_r-7, K's is
//                                            eb_r-3, so the ratio 2^7 that the K
//                                            table carries is the 1/sqrt(64)).
//                                            The 1/sqrt(d) must appear exactly
//                                            ONCE, on K.
//   Derivation of the implemented form (see check_scale16.py for the independent
//   exact-Fraction reference):  y = Sa * m16 * 2^(Ea + eb - CEX) with
//   Sa = {1'b1,a[6:0]}.  With p = Sa*m16 = nrm * 2^-n (nrm normalised to bit 23)
//       Ef = Ea + eb - (CEX - 150) - n + round_carry
//   which is -118 in the K instance and -111 in the V instance.
//
//   The product is Sa(8b) x m16(16b) = one DSP48E2 per lane (27x18), tagged
//   (* use_dsp = "yes" *) on the product register; the OOC synthesis, not the
//   behavioural model, is what proves the DSP count (Verilator never sees one).
//   One rounding only, at the BF16 conversion (RNE, ties to even).
// -----------------------------------------------------------------------------
module kv8_score_scale8 #(
    parameter integer MANT_BITS = 8,      // 8 = BF16 x BF16 (shipped), 16 = m16
    parameter integer CEX       = 268     // scale constant: 268 = K, 261 = V
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         i_valid,
    input  wire [127:0] i_score,      // 8 x BF16
    input  wire [127:0] i_scale,      // MANT_BITS=8: 8 x BF16;  16: 8 x m16
    input  wire [7:0]   i_eb,         // MANT_BITS=16 only: table exponent
    output wire         o_valid,
    output wire [127:0] o_score
);
    reg  [1:0] valid_pipe;
    reg        valid_s1;          // stage-1 valid: stage 2 must not be gated on i_valid
    assign o_valid = valid_pipe[1];

    // CEX - 150: 118 for K (268), 111 for V (261).  All-signed arithmetic below.
    localparam signed [11:0] CEX_OFF = CEX - 150;

    genvar lane;
    generate
    if (MANT_BITS == 8) begin : g_legacy
        for (lane=0;lane<8;lane=lane+1) begin : gen_lane
            wire [15:0] a = i_score[lane*16 +: 16];
            wire [15:0] b = i_scale[lane*16 +: 16];

            // ---- stage 1: significand product, biased exponent sum ----
            (* use_dsp = "yes" *) reg [15:0] prod;
            reg [8:0] esum;
            reg       sgn;
            reg [15:0] a_r,b_r;
            always @(posedge clk) begin
                if (i_valid) begin
                    prod <= {1'b1,a[6:0]} * {1'b1,b[6:0]};   // 0x4000 .. 0xFE01
                    esum <= {1'b0,a[14:7]} + {1'b0,b[14:7]};
                    sgn  <= a[15] ^ b[15];
                    a_r  <= a;
                    b_r  <= b;
                end
            end

            // ---- stage 2: normalise, round-to-nearest-even, debias ----
            // Sa,Sb are 1.f in 8 bits; the raw product is 2^14-scaled, so one
            // left shift (s=1) makes it 1.f x 2^15 and costs one exponent.
            //   Ef = Ea + Eb - 126 - s + round_carry
            wire [15:0] nrm  = prod[15] ? prod : {prod[14:0],1'b0};
            wire        s    = ~prod[15];
            wire [7:0]  rb   = nrm[7:0];                             // bits under the 7 LSBs
            wire        up   = (rb > 8'h80) || (rb == 8'h80 && nrm[8]);
            wire [8:0]  msum = {1'b0,nrm[15:8]} + {8'd0,up};
            wire signed [10:0] ef = $signed({2'b0,esum}) - 11'sd126
                                    - $signed({10'd0,s}) + $signed({10'd0,msum[8]});
            reg  [15:0] y;
            always @(posedge clk) begin
                if (valid_s1) begin
                    if (a_r[14:7]==8'hff || b_r[14:7]==8'hff)
                        y <= (a_r[14:7]==8'hff) ? a_r : b_r;     // inf/nan pass through
                    else if (a_r[14:7]==8'd0 || b_r[14:7]==8'd0)
                        y <= {sgn,15'd0};                        // flush zero
                    else if (ef >= 11'sd255)
                        y <= {sgn,8'hff,7'd0};                   // overflow -> inf
                    else if (ef <= 11'sd0)
                        y <= {sgn,15'd0};                        // underflow -> zero
                    else
                        y <= {sgn,ef[7:0],msum[6:0]};
                end
            end
            assign o_score[lane*16 +: 16] = y;
        end
    end else begin : g_m16
        // y = Sa * m16 * 2^(Ea + eb - CEX);  Ef = Ea + eb - CEX_OFF - n
        for (lane=0;lane<8;lane=lane+1) begin : gen_lane16
            wire [15:0] a  = i_score[lane*16 +: 16];
            wire [15:0] m16 = i_scale[lane*16 +: 16];

            (* use_dsp = "yes" *) reg [23:0] p;    // 8b x 16b -> one DSP48E2
            reg [9:0]  ebase;
            reg        sgn;
            reg [15:0] a_r;
            always @(posedge clk) begin
                if (i_valid) begin
                    p     <= {1'b1,a[6:0]} * m16;    // one DSP/lane, 8b x 16b
                    ebase <= {2'b0,a[14:7]} + {2'b0,i_eb};   // Ea + eb_table
                    sgn   <= a[15];
                    a_r   <= a;
                end
            end

            // normalise: shift p left until bit 23 is set (p >= 128 always, so
            // 16 shifts cover every mantissa down to m16 = 1)
            integer t;
            reg [23:0] nrm;
            reg [4:0]  n;
            always @(*) begin
                nrm = p;
                n = 5'd0;
                for (t=0;t<16;t=t+1)
                    if (!nrm[23]) begin
                        nrm = nrm << 1;
                        n   = n + 5'd1;
                    end
            end
            wire        up   = (nrm[15:0] > 16'h8000) ||
                               (nrm[15:0] == 16'h8000 && nrm[16]);   // RNE
            wire [8:0]  msum = {1'b0,nrm[22:16]} + {8'd0,up};
            // [RNE-CARRY] the kept field is 7 bits (nrm[22:16], with nrm[23] the
            // implicit leading 1 and NOT part of the sum), so the rounding carry
            // leaves msum at bit 7 - bit 8 is unreachable.  Taking msum[8] here
            // dropped the carry and made those scores exactly 2x too small
            // (measured: 4 of 160 words, all at Sa=131, m=1000).
            wire signed [11:0] ef = $signed({4'b0,ebase}) - CEX_OFF
                                    - $signed({7'd0,n}) + $signed({11'd0,msum[7]});
            reg  [15:0] y;
            always @(posedge clk) begin
                if (valid_s1) begin
                    if (a_r[14:7]==8'hff)
                        y <= a_r;                                // inf/nan pass through
                    else if (a_r[14:7]==8'd0 || p == 24'd0)
                        y <= {sgn,15'd0};                        // flush zero
                    else if (ef >= 12'sd255)
                        y <= {sgn,8'hff,7'd0};                   // overflow -> inf
                    else if (ef <= 12'sd0)
                        y <= {sgn,15'd0};                        // underflow -> zero
                    else
                        y <= {sgn,ef[7:0],msum[6:0]};
                end
            end
            assign o_score[lane*16 +: 16] = y;
        end
    end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin valid_pipe <= 2'd0; valid_s1 <= 1'b0; end
        else begin
            valid_pipe <= {valid_pipe[0],i_valid};
            valid_s1   <= i_valid;
        end
    end
endmodule
`default_nettype wire
