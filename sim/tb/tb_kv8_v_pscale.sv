`timescale 1ns/1ps
`default_nettype none
// tb_kv8_v_pscale -- value check of the PV probability scaler (p * m -> BF16).
//
// [V-PERTOK-M] This is the only NEW arithmetic the V per-token scale adds: the
// attention multiplies the probability of token t by that token's 8-bit V
// mantissa before the PV reduction.  The mantissa is the quantiser's exact m
// (1..255), so the significand product is real and needs RNE.
//
// The module is clocked (its product register is what maps to the DSP's M
// register), so each case is driven, one clock waited, then sampled.  The sweep
// is deterministic and walks the corners that decide the rounding:
//   * m = 0 (must give 0, never a denormal) and m = 1 (identity);
//   * every binade crossing of m (1,2,3,4,7,8,...,255) because m's own
//     normalisation shifts its significand;
//   * p mantissa 0x7F with odd m (RNE must carry into the exponent:
//     1.9921875 * 3 = 5.9765625 -> 6.0);
//   * p exponent 0 (zero/denormal -> 0) and the top of the p range (saturation).
// check_v_pscale.py recomputes every line in float64 and rounds to BF16 by RNE;
// it is the oracle, so the DUT is never compared against itself.
module tb_kv8_v_pscale;
    reg clk = 0;
    always #5 clk = ~clk;

    reg  [1023:0] i_p = 0;
    reg  [1023:0] i_m = 0;
    wire [1023:0] o_p;

    kv8_v_pscale #(.LANES(64)) dut (.clk(clk), .i_p(i_p), .i_m(i_m), .o_p(o_p));

    integer pe [0:9];
    integer pm [0:7];
    integer ms [0:17];
    integer i, j, k, l;
    reg [15:0] pw;
    reg [1023:0] cap;

    initial begin
        pe[0]=0;   pe[1]=1;   pe[2]=100; pe[3]=118; pe[4]=125;
        pe[5]=126; pe[6]=127; pe[7]=128; pe[8]=140; pe[9]=127;
        pm[0]=0;   pm[1]=1;   pm[2]=16;  pm[3]=32;
        pm[4]=63;  pm[5]=64;  pm[6]=127; pm[7]=126;
        ms[0]=0;   ms[1]=1;   ms[2]=2;   ms[3]=3;   ms[4]=4;   ms[5]=5;
        ms[6]=7;   ms[7]=8;   ms[8]=9;   ms[9]=15;  ms[10]=16; ms[11]=17;
        ms[12]=31; ms[13]=32; ms[14]=63; ms[15]=64; ms[16]=127; ms[17]=255;

        for (i=0;i<10;i=i+1) begin
            for (j=0;j<8;j=j+1) begin
                for (k=0;k<18;k=k+1) begin
                    for (l=0;l<64;l=l+1) begin
                        pw = ({1'b0, pe[(i+l)%10][7:0]} << 7) | pm[(j+l*3)%8][6:0];
                        i_p[l*16 +: 16] = pw;
                        i_m[l*16 +: 16] = {8'd0, ms[(k+l*5)%18][7:0]};
                    end
                    @(negedge clk);          // operands stable for this edge
                    @(posedge clk);          // product registers here
                    @(negedge clk);
                    cap = o_p;               // valid one cycle after the operands
                    for (l=0;l<64;l=l+1)
                        $display("PS p=%h m=%0d y=%h",
                                 i_p[l*16 +: 16], i_m[l*16 +: 16],
                                 cap[l*16 +: 16]);
                end
            end
        end
        $display("PS DONE cases=%0d", 10*8*18*64);
        $finish;
    end
endmodule
`default_nettype wire
