`timescale 1ns/1ps
`default_nettype none
// tb_kv8_score_scale16 -- exact-value check of kv8_score_scale8 in MANT_BITS=16.
//
// Sweep is deterministic, not random: at MARGIN=9 the reachable mantissa is
// m in [517,1029], so the m list below walks the binade crossings (1022..1025)
// and the RNE-carry corner (lane 5: Sa=131, m=1000 -> nrm[22:16]=0x7F, up=1,
// carry out of bit 7 -> y must be 0x3C80, not 0x3C00).
// Lane 0 is the absolute-scale case: a = 1.0 (Sa=128, Ea=127), m16=1024, eb=124
// must give exactly y = 1.0 = 0x3F80 -- this is what pins C = 134+127+7 and
// catches a score that is 128x off (double-counted 2^-7).
// check_scale16.py compares every word against an exact-Fraction reference.
module tb_kv8_score_scale16;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg          i_valid = 0;
    reg  [127:0] i_score = 0;
    reg  [127:0] i_scale = 0;
    reg  [7:0]   i_eb    = 0;
    wire         o_valid;
    wire [127:0] o_score;

    // K instance: CEX = 268 (contains the retired 2^-7)
    kv8_score_scale8 #(.MANT_BITS(16), .CEX(268)) dut (
        .clk(clk), .rst_n(rst_n), .i_valid(i_valid),
        .i_score(i_score), .i_scale(i_scale), .i_eb(i_eb),
        .o_valid(o_valid), .o_score(o_score));

    // [V-SIDE] the V instance of the same module must use CEX = 261, because
    // CEX_K - CEX_V = 7 and the 1/sqrt(d) must appear exactly once, on K.
    //
    // The two tables store their byte on different conventions: K's is eb_r-3
    // (the /8 is folded in) and V's is eb_r-7, i.e. for the SAME physical scale
    //     eb_V = eb_K - 4.
    // With that byte the V result must be exactly 8x the K result, bit for bit:
    //     y_V = a*m*2^(eb-4-127) = 8 * a*m*2^(eb-134) = 8 * y_K
    // That 2^3 IS the 1/sqrt(d).  (Driving V with K's own byte instead would give
    // 2^7 = 128x here and would be comparing two different physical scales.)
    wire [127:0] o_score_v;
    wire [7:0]   i_eb_v = i_eb - 8'd4;
    kv8_score_scale8 #(.MANT_BITS(16), .CEX(261)) dut_v (
        .clk(clk), .rst_n(rst_n), .i_valid(i_valid),
        .i_score(i_score), .i_scale(i_scale), .i_eb(i_eb_v),
        .o_valid(), .o_score(o_score_v));

    integer ms [0:8];
    integer ebs[0:3];
    integer ea [0:7];
    integer sa [0:7];
    integer i, j, l;
    reg [127:0] sc;
    reg [15:0]  aw;

    task drive(input integer mm, input integer ee);
        begin
            sc = 128'd0;
            for (l=0;l<8;l=l+1) begin
                aw = ({1'b0, ea[l][7:0]} << 7) | (sa[l]-128);
                sc = sc | ({112'd0, aw} << (l*16));
            end
            @(negedge clk);
            i_score = sc;
            i_scale = {8{mm[15:0]}};
            i_eb    = ee[7:0];
            i_valid = 1'b1;
            @(posedge clk);
            @(negedge clk);
            i_valid = 1'b0;
            @(posedge clk);
            #1;
            for (l=0;l<8;l=l+1)
                $display("S16 m=%0d eb=%0d lane=%0d ea=%0d sa=%0d y=%h yv=%h",
                         mm, ee, l, ea[l], sa[l], o_score[l*16 +: 16],
                         o_score_v[l*16 +: 16]);
        end
    endtask

    initial begin
        ms[0]=517; ms[1]=518; ms[2]=767; ms[3]=1000; ms[4]=1022;
        ms[5]=1023; ms[6]=1024; ms[7]=1025; ms[8]=1029;
        ebs[0]=120; ebs[1]=124; ebs[2]=128; ebs[3]=134;
        ea[0]=127; ea[1]=127; ea[2]=127; ea[3]=127;
        ea[4]=127; ea[5]=125; ea[6]=120; ea[7]=140;
        sa[0]=128; sa[1]=129; sa[2]=191; sa[3]=254;
        sa[4]=255; sa[5]=131; sa[6]=200; sa[7]=250;

        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(posedge clk);
        for (i=0;i<9;i=i+1)
            for (j=0;j<4;j=j+1)
                drive(ms[i], ebs[j]);
        $display("S16 DONE cases=%0d", 9*4*8);
        $finish;
    end
endmodule
`default_nettype wire
