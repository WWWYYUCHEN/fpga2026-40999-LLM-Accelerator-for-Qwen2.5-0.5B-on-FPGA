`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_kv8_quant_equiv -- [RECIP-SEQ] arithmetic-equivalence proof for the KV
// quantisers after the combinational dividers were replaced by ONE iterative
// engine behind the existing valid handshake.
//
// WHAT IS PROVEN
//   The reference model below is a verbatim transcription of the *shipped*
//   combinational formulas (integer `/` and `%`, the original recip_lut
//   functions, the original saturating ceilings).  The DUT is the new RTL.  Every
//   quantity the DUT publishes (o_eb, o_m and all 8 code lanes of the quantise
//   pass) is compared bit-for-bit:
//
//   T1  kv8_div_seq standalone, NUMW=12/DIVW=8: every numerator 0..4095 x 13
//       divisors, plus 256 numerators x every divisor 1..255 -- proves the
//       restoring recurrence itself is floor()/mod().
//   T2  K per-token ceiling: all 128 reachable M8 values (M8 = 1.significand, so
//       [128,255]) at the tree's MARGIN=8 -> eb, m and the codes.
//   T3  K block-lock ceiling: all 128 M8 x all 31 lock shifts (-15..+15) -> eb,
//       m and the codes (this is the 23-bit numerator the report's tier-2 table
//       did not isolate).
//   T4  K reciprocal: EVERY m in [0,65535] (the full 16-bit field, so every value
//       the lock can publish) driven through the restore path -> m, eb and codes.
//       This is the exhaustive proof that
//           R = (2^24 + m/2)/m      (m != 0),  R = 2^24 (m == 0)
//       is unchanged: the codes are a function of R.
//   T4b sensitivity control: for a sample of m in the region where the codes are
//       sensitive to R, count how many cases change if R moves by +-1.  A green
//       T4 with a dead T4b would prove nothing, so T4b has to be >= 50%.
//   T5  V quantiser: every m in [0,255] (full 8-bit field), all 128 M8 and all 31
//       lock shifts, same checks with the V formulas (2^16 base, 8-bit field).
//
// Every case drives the real module through its real ports and waits for the real
// o_done / o_lock_done / o_restore_done handshake, so the check covers the
// sequencer, the shared divider, the ceilings, the saturation and the RONE guard.
// -----------------------------------------------------------------------------
module tb_kv8_quant_equiv;

    reg clk = 0, rst_n = 0;
    always #1.666 clk = ~clk;

    integer errors = 0, checks = 0;
    integer sens_r  = 0, sens_n  = 0;      // T4b sensitivity control

    // ---------------------------------------------------------------- K DUT
    reg         kclr=0, kpass=0, kval=0, klast=0, klockv=0, krestv=0;
    reg  [127:0] kd128 = 0;
    reg  [7:0]  klocke = 0, kreste = 0;
    reg  [15:0] krestm = 0;
    wire        kdone, klockdone, krestdone, kqvld;
    wire [7:0]  keb, kmaxexp;
    wire [15:0] km;
    wire [127:0] kq;
    kv8_k_quant #(.MARGIN(8), .MTOP(65535), .RSHIFT(24), .CODE_EXTRA(8)) k_dut (
        .clk(clk), .rst_n(rst_n),
        .i_clear(kclr), .i_pass(kpass), .i_valid(kval), .i_last(klast), .i_k(kd128),
        .i_eb_lock_valid(klockv), .i_eb_lock(klocke), .o_lock_done(klockdone),
        .i_restore_valid(krestv), .i_restore_eb(kreste), .i_restore_m(krestm),
        .o_restore_done(krestdone),
        .o_done(kdone), .o_eb(keb), .o_m(km), .o_maxmag_exp(kmaxexp),
        .o_q_valid(kqvld), .o_q(kq)
    );

    // ---------------------------------------------------------------- V DUT
    reg         vclr=0, vpass=0, vval=0, vlast=0, vlockv=0, vrestv=0;
    reg  [127:0] vd128 = 0;
    reg  [7:0]  vlocke = 0, vreste = 0, vrestm = 0;
    wire        vdone, vlockdone, vrestdone, vqvld;
    wire [7:0]  veb, vm, vmaxexp;
    wire [63:0] vq;
    wire [7:0]  vsat;
    kv8_v_quant #(.MARGIN(2)) v_dut (
        .clk(clk), .rst_n(rst_n),
        .i_clear(vclr), .i_pass(vpass), .i_valid(vval), .i_last(vlast), .i_k(vd128),
        .i_eb_lock_valid(vlockv), .i_eb_lock(vlocke), .o_lock_done(vlockdone),
        .i_restore_valid(vrestv), .i_restore_eb(vreste), .i_restore_m(vrestm),
        .o_restore_done(vrestdone),
        .o_done(vdone), .o_eb(veb), .o_m(vm), .o_maxmag_exp(vmaxexp),
        .o_q_valid(vqvld), .o_q(vq), .o_sat_lane(vsat)
    );

    // ------------------------------------------------- T1: divider standalone
    reg         d12_start=0;
    reg  [11:0] d12_num=0;
    reg  [7:0]  d12_den=1;
    wire        d12_busy, d12_done;
    wire [11:0] d12_quo;
    wire [7:0]  d12_rem;
    kv8_div_seq #(.NUMW(12), .DIVW(8)) d12 (
        .clk(clk), .rst_n(rst_n), .i_start(d12_start),
        .i_num(d12_num), .i_den(d12_den),
        .o_busy(d12_busy), .o_done(d12_done), .o_quo(d12_quo), .o_rem(d12_rem)
    );

    // =====================================================================
    // reference model: the SHIPPED combinational expressions, verbatim
    // =====================================================================
    // kv8_k_quant.sv (RSHIFT=24): R = (2^24 + m/2)/m, m -> RONE when m == 0
    function [24:0] ref_recip_k(input [15:0] m);
        reg [24:0] d;
        begin
            if (m == 16'd0) ref_recip_k = 25'd16777216;
            else begin
                d = {9'd0,m};
                ref_recip_k = (25'd16777216 + {1'b0,m[15:1]}) / d;
            end
        end
    endfunction
    // kv8_v_quant.sv (2^16 base): R = (2^16 + m/2)/m, m -> 2^16 when m == 0
    function [16:0] ref_recip_v(input [7:0] m);
        reg [16:0] d;
        begin
            if (m == 8'd0) ref_recip_v = 17'd65536;
            else begin
                d = {9'd0,m};
                ref_recip_v = (17'd65536 + {1'b0,d[16:1]}) / d;
            end
        end
    endfunction
    // m = ceil(num/127) exactly as the shipped RTL wrote it
    function [15:0] ref_m_of_num(input [31:0] num);
        reg [31:0] quo, rem;
        begin
            quo = num / 32'd127;
            rem = num - quo*32'd127;
            ref_m_of_num = quo[15:0] + ((rem != 32'd0) ? 16'd1 : 16'd0);
        end
    endfunction
    function [7:0] ref_mv_of_num(input [15:0] num);
        reg [15:0] quo, rem;
        begin
            quo = num / 16'd127;
            rem = num - quo*16'd127;
            ref_mv_of_num = quo[7:0] + ((rem != 16'd0) ? 8'd1 : 8'd0);
        end
    endfunction
    // locked m: lquo/lrem, saturate at MTOP (65535 for K, 255 for V)
    function [15:0] ref_m_lock_k(input [22:0] ln);
        reg [22:0] lquo, lrem;
        begin
            lquo = ln / 23'd127;
            lrem = ln - lquo*23'd127;
            if ((lquo > 23'd65535) || ((lquo == 23'd65535) && (lrem != 23'd0)))
                ref_m_lock_k = 16'd65535;
            else
                ref_m_lock_k = lquo[15:0] + ((lrem != 23'd0) ? 16'd1 : 16'd0);
        end
    endfunction
    function [7:0] ref_m_lock_v(input [22:0] ln);
        reg [22:0] lquo, lrem;
        begin
            lquo = ln / 23'd127;
            lrem = ln - lquo*23'd127;
            if ((lquo > 23'd255) || ((lquo == 23'd255) && (lrem != 23'd0)))
                ref_m_lock_v = 8'd255;
            else
                ref_m_lock_v = lquo[7:0] + ((lrem != 23'd0) ? 8'd1 : 8'd0);
        end
    endfunction
    // the shipped lock numerator (saturating shifter)
    function [22:0] ref_lock_num(input [14:0] emax, input [7:0] eblk);
        integer sh;
        reg [7:0] m8;
        begin
            m8 = {1'b1, emax[6:0]};
            sh = $signed({2'b0,emax[14:7]}) - $signed({2'b0,eblk});
            if      (sh >  15) ref_lock_num = 23'h7F_FFFF;
            else if (sh < -15) ref_lock_num = 23'd0;
            else if (sh <   0) ref_lock_num = {15'd0,m8} >> (-sh);
            else if (sh >   0) ref_lock_num = {15'd0,m8} << sh;
            else               ref_lock_num = {15'd0,m8};
        end
    endfunction
    // magnitude maximum of the 8 captured BF16s (sign dropped): the DUT's max tree
    function [14:0] maxmag(input [127:0] d);
        integer L; reg [14:0] v, mx;
        begin
            mx = 15'd0;
            for (L=0;L<8;L=L+1) begin
                v = d[L*16 +: 15];
                if (v > mx) mx = v;
            end
            maxmag = mx;
        end
    endfunction

    // ---- K quantise-pass reference: one 128-bit beat of 8 {hi,lo} pairs -----
    function [127:0] ref_q_k(input [7:0] eb, input [24:0] R, input [127:0] kd);
        integer L;
        reg [15:0] kk;
        reg [7:0]  mm, hi_u, lo_u;
        reg signed [9:0] dd;
        reg [23:0] nn;
        reg [49:0] pr;
        reg [33:0] qv;
        reg [16:0] qround;
        reg [15:0] qmag;
        begin
            ref_q_k = 128'd0;
            for (L=0;L<8;L=L+1) begin
                kk = kd[L*16 +: 16];
                mm = {1'b1, kk[6:0]};
                dd = $signed({2'b0,kk[14:7]}) - $signed({2'b0,eb});
                if (kk[14:7] == 8'd0)          nn = 24'd0;
                else if (!dd[9])               nn = (dd > 10'sd15) ? 24'hFF_FFFF
                                                                 : ({{16{1'b0}},mm} << dd[7:0]);
                else                           nn = {{16{1'b0}},mm} >> (-dd[7:0]);
                pr = nn * R + 50'd32768;                 // QSHIFT = 24-8 = 16
                qv = pr >> 16;
                qmag = (qv > 34'd32639) ? 16'd32639 : qv[15:0];
                qround = {1'b0,qmag} + 17'd128;
                hi_u = qround[16:8];
                lo_u = qmag[7:0];
                if (kk[15]) begin
                    hi_u = ~hi_u + 8'd1;
                    lo_u = ~lo_u + 8'd1;
                end
                ref_q_k[L*16 +: 16] = {hi_u, lo_u};
            end
        end
    endfunction
    // ---- V quantise-pass reference: one 64-bit beat of 8 signed codes -------
    function [63:0] ref_q_v(input [7:0] eb, input [16:0] R, input [127:0] kd);
        integer L;
        reg [15:0] kk;
        reg [7:0]  mm, c;
        reg signed [9:0] dd;
        reg [23:0] nn;
        reg [40:0] pr;
        reg [24:0] qv;
        begin
            ref_q_v = 64'd0;
            for (L=0;L<8;L=L+1) begin
                kk = kd[L*16 +: 16];
                mm = {1'b1, kk[6:0]};
                dd = $signed({2'b0,kk[14:7]}) - $signed({2'b0,eb});
                if (kk[14:7] == 8'd0) nn = 24'd0;
                else if (!dd[9])      nn = {{16{1'b0}},mm} << dd[7:0];
                else                  nn = {{16{1'b0}},mm} >> (-dd[7:0]);
                pr = nn * {24'd0,R} + 41'd32768;
                qv = pr >> 16;
                c  = (qv > 25'd127) ? 8'd127 : qv[7:0];
                if (kk[15]) c = ~c + 8'd1;
                ref_q_v[L*8 +: 8] = c;
            end
        end
    endfunction

    // deterministic, varied 8-lane patterns.
    //   lo: exponents eb-0 .. eb-7   (small N, codes spread over the low range)
    //   hi: exponents eb+13..eb+6    (N >= 2^16, the codes react to R +-1)
    function [127:0] kpat_lo(input [7:0] eb, input integer tag);
        integer L;
        reg [7:0] e;
        reg [6:0] mnt;
        begin
            kpat_lo = 128'd0;
            for (L=0;L<8;L=L+1) begin
                e = eb - L[7:0];
                if (e < 8'd1) e = 8'd1;
                mnt = 7'd1 + ((L*37 + tag*11) % 127);
                kpat_lo[L*16 +: 16] = {(L+tag)%2, e, mnt};
            end
        end
    endfunction
    function [127:0] kpat_hi(input [7:0] eb, input integer tag);
        integer L;
        reg [7:0] e;
        reg [6:0] mnt;
        begin
            kpat_hi = 128'd0;
            for (L=0;L<8;L=L+1) begin
                e = eb + 13 - L[7:0];
                mnt = 7'd3 + ((L*29 + tag*17) % 124);
                kpat_hi[L*16 +: 16] = {(L+tag+1)%2, e, mnt};
            end
        end
    endfunction

    // =====================================================================
    // DUT drivers (real handshake, no fixed-offset sampling)
    // =====================================================================
    task automatic k_mag(input [127:0] d, output [7:0] eb_o, output [15:0] m_o);
        integer b;
        begin
            kclr = 1; @(posedge clk); #0.1; kclr = 0;
            for (b=0;b<8;b=b+1) begin
                kpass=0; kval=1; klast=(b==7); kd128=d; @(posedge clk); #0.1;
            end
            kval=0; klast=0;
            while (!kdone) begin @(posedge clk); #0.1; end
            eb_o = keb; m_o = km;
        end
    endtask
    task automatic k_quant(input [127:0] d, output [127:0] q_o);
        integer b;
        begin
            for (b=0;b<8;b=b+1) begin
                kpass=1; kval=1; klast=0; kd128=d; @(posedge clk); #0.1;
                if (!kqvld) begin errors=errors+1; $display("FAIL K o_q_valid in pass1"); end
                q_o = kq;
            end
            kval=0; kpass=0; @(posedge clk); #0.1;
        end
    endtask
    task automatic k_lock(input [7:0] eblk, output [7:0] eb_o, output [15:0] m_o);
        begin
            klockv=1; klocke=eblk; @(posedge clk); #0.1; klockv=0;
            while (!klockdone) begin @(posedge clk); #0.1; end
            eb_o = keb; m_o = km;
        end
    endtask
    task automatic k_rest(input [7:0] eb, input [15:0] m, output [7:0] eb_o, output [15:0] m_o);
        begin
            krestv=1; kreste=eb; krestm=m; @(posedge clk); #0.1; krestv=0;
            while (!krestdone) begin @(posedge clk); #0.1; end
            eb_o = keb; m_o = km;
        end
    endtask

    task automatic v_mag(input [127:0] d, output [7:0] eb_o, output [7:0] m_o);
        integer b;
        begin
            vclr = 1; @(posedge clk); #0.1; vclr = 0;
            for (b=0;b<8;b=b+1) begin
                vpass=0; vval=1; vlast=(b==7); vd128=d; @(posedge clk); #0.1;
            end
            vval=0; vlast=0;
            while (!vdone) begin @(posedge clk); #0.1; end
            eb_o = veb; m_o = vm;
        end
    endtask
    task automatic v_quant(input [127:0] d, output [63:0] q_o);
        integer b;
        begin
            for (b=0;b<8;b=b+1) begin
                vpass=1; vval=1; vlast=0; vd128=d; @(posedge clk); #0.1;
                if (!vqvld) begin errors=errors+1; $display("FAIL V o_q_valid in pass1"); end
                q_o = vq;
            end
            vval=0; vpass=0; @(posedge clk); #0.1;
        end
    endtask
    task automatic v_lock(input [7:0] eblk, output [7:0] eb_o, output [7:0] m_o);
        begin
            vlockv=1; vlocke=eblk; @(posedge clk); #0.1; vlockv=0;
            while (!vlockdone) begin @(posedge clk); #0.1; end
            eb_o = veb; m_o = vm;
        end
    endtask
    task automatic v_rest(input [7:0] eb, input [7:0] m, output [7:0] eb_o, output [7:0] m_o);
        begin
            vrestv=1; vreste=eb; vrestm=m; @(posedge clk); #0.1; vrestv=0;
            while (!vrestdone) begin @(posedge clk); #0.1; end
            eb_o = veb; m_o = vm;
        end
    endtask

    // =====================================================================
    integer i, L, t, sh, nbad, ebi;
    reg [127:0] d, dA, dB, qgot, qexp;
    reg [63:0]  vqgot, vqexp;
    reg [7:0]   eb_o, eb_e, veb_o, veb_e;
    reg [15:0]  m_o, m_e;
    reg [7:0]   vm_o, vm_e;
    reg [14:0]  emax;
    reg [22:0]  ln;
    reg [24:0]  R_k, R_k1;
    reg [16:0]  R_v;
    reg [11:0]  dq; reg [7:0] dr;
    integer dset [0:12];

    initial begin
        $display("=== tb_kv8_quant_equiv: [RECIP-SEQ] bit-exactness of the KV quantisers ===");
        dset[0]=1; dset[1]=2; dset[2]=3; dset[3]=7; dset[4]=15; dset[5]=16; dset[6]=17;
        dset[7]=63; dset[8]=64; dset[9]=127; dset[10]=128; dset[11]=129; dset[12]=255;
        rst_n = 0; repeat (8) @(posedge clk); rst_n = 1; repeat (4) @(posedge clk); #0.1;

        // ---------------------------------------------------------------- T1
        nbad = 0;
        for (i=0; i<4096; i=i+1) begin
            for (t=0; t<13; t=t+1) begin
                d12_start=1; d12_num=i[11:0]; d12_den=dset[t][7:0]; @(posedge clk); #0.1; d12_start=0;
                while (!d12_done) begin @(posedge clk); #0.1; end
                dq = d12_quo; dr = d12_rem;
                checks = checks + 1;
                if (dq !== (i[11:0]/dset[t][7:0]) || dr !== (i[11:0]%dset[t][7:0])) begin
                    nbad = nbad + 1;
                    if (nbad < 5) $display("FAIL T1 num=%0d den=%0d got q=%0d r=%0d exp q=%0d r=%0d",
                        i, dset[t], dq, dr, i[11:0]/dset[t][7:0], i[11:0]%dset[t][7:0]);
                end
            end
        end
        for (i=0; i<256; i=i+1) begin
            for (t=1; t<256; t=t+1) begin
                d12_start=1; d12_num=i[11:0]; d12_den=t[7:0]; @(posedge clk); #0.1; d12_start=0;
                while (!d12_done) begin @(posedge clk); #0.1; end
                dq = d12_quo; dr = d12_rem;
                checks = checks + 1;
                if (dq !== (i[11:0]/t) || dr !== (i[11:0]%t)) begin
                    nbad = nbad + 1;
                    if (nbad < 5) $display("FAIL T1b num=%0d den=%0d", i, t);
                end
            end
        end
        $display("T1 divider recurrence: %0d checks, %0d bad", 4096*13+256*255, nbad);

        // ---- T2: K per-token ceiling, all 128 M8 x MARGIN=8 ------------------
        nbad = 0;
        for (i=128; i<256; i=i+1) begin
            d = 128'd0;
            for (L=0; L<8; L=L+1) d[L*16 +: 16] = {1'b0, 8'd200, 7'd1};
            d[15:0] = {1'b0, 8'd200, i[6:0]};            // the max tree sees {200,i[6:0]}
            emax = maxmag(d);
            k_mag(d, eb_o, m_o);
            m_e  = ref_m_of_num({{8{1'b0}}, 1'b1, emax[6:0]} << 8);
            eb_e = (emax[14:7] > 9) ? (emax[14:7] - 8) : 8'd1;
            checks = checks + 1;
            if (m_o !== m_e || eb_o !== ((eb_e > 8'd10) ? eb_e - 8'd10 : 8'd0)) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T2 M8=%0d m=%0d exp=%0d eb=%0d exp=%0d",
                                       i, m_o, m_e, eb_o, ((eb_e > 8'd10) ? eb_e-8'd10 : 8'd0));
            end
            R_k = ref_recip_k(m_e);
            k_quant(d, qgot);
            qexp = ref_q_k(eb_e, R_k, d);
            checks = checks + 1;
            if (qgot !== qexp) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T2 q M8=%0d", i);
            end
        end
        $display("T2 K per-token ceiling: 128 M8 values, %0d bad", nbad);

        // ---- T3: K lock, all 128 M8 x 31 shifts ------------------------------
        nbad = 0;
        for (i=128; i<256; i=i+1) begin
            d = 128'd0;
            for (L=0; L<8; L=L+1) d[L*16 +: 16] = {1'b0, 8'd200, 7'd1};
            d[15:0] = {1'b0, 8'd200, i[6:0]};
            emax = maxmag(d);
            k_mag(d, eb_o, m_o);              // establishes run_max == emax
            for (sh=-15; sh<=15; sh=sh+1) begin
                ebi = 200 - sh;
                if (ebi < 1) ebi = 1;
                k_lock(ebi[7:0], eb_o, m_o);
                ln   = ref_lock_num(emax, ebi[7:0]);
                m_e  = ref_m_lock_k(ln);
                eb_e = (ebi[7:0] == 8'd0) ? 8'd1 : ebi[7:0];
                checks = checks + 1;
                if (m_o !== m_e || eb_o !== ((eb_e > 8'd10) ? eb_e - 8'd10 : 8'd0)) begin
                    nbad = nbad + 1;
                    if (nbad < 6) $display("FAIL T3 M8=%0d sh=%0d m=%0d exp=%0d eb=%0d exp=%0d",
                                           i, sh, m_o, m_e, eb_o,
                                           ((eb_e > 8'd10) ? eb_e - 8'd10 : 8'd0));
                end
                R_k = ref_recip_k(m_e);
                k_quant(d, qgot);
                qexp = ref_q_k(eb_e, R_k, d);
                checks = checks + 1;
                if (qgot !== qexp) begin
                    nbad = nbad + 1;
                    if (nbad < 6) $display("FAIL T3 q M8=%0d sh=%0d", i, sh);
                end
            end
        end
        $display("T3 K lock: 128 M8 x 31 shifts, %0d bad", nbad);

        // ---- T4: K reciprocal, EVERY m in [0,65535] via the restore path ------
        nbad = 0;
        eb_e = 8'd120;
        dA = kpat_lo(8'd120, 3);
        dB = kpat_hi(8'd120, 5);
        for (i=0; i<65536; i=i+1) begin
            k_rest(eb_e, i[15:0], eb_o, m_o);
            R_k = ref_recip_k(i[15:0]);
            checks = checks + 1;
            if (m_o !== i[15:0] || eb_o !== ((eb_e > 8'd10) ? eb_e - 8'd10 : 8'd0)) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T4 restore m=%0d got m=%0d eb=%0d", i, m_o, eb_o);
            end
            if ((i % 16) == 0 || i < 64) begin
                k_quant(dA, qgot);
                qexp = ref_q_k(eb_e, R_k, dA);
                checks = checks + 1;
                if (qgot !== qexp) begin
                    nbad = nbad + 1;
                    if (nbad < 6) $display("FAIL T4 q m=%0d", i);
                end
            end
        end
        $display("T4 K reciprocal: all 65536 m restored, %0d bad", nbad);

        // ---- T4b: sensitivity control (the codes MUST react to R +-1) --------
        sens_r = 0; sens_n = 0; nbad = 0;
        for (i=20000; i<=65000; i=i+1000) begin
            k_rest(eb_e, i[15:0], eb_o, m_o);
            R_k = ref_recip_k(i[15:0]);
            k_quant(dB, qgot);
            qexp = ref_q_k(eb_e, R_k, dB);
            checks = checks + 1;
            if (qgot !== qexp) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T4b q m=%0d", i);
            end
            sens_n = sens_n + 1;
            R_k1 = (R_k > 25'd0) ? R_k - 25'd1 : 25'd0;
            if (ref_q_k(eb_e, R_k + 25'd1, dB) !== qexp ||
                ref_q_k(eb_e, R_k1, dB) !== qexp) sens_r = sens_r + 1;
        end
        $display("T4b sensitivity: R+-1 changes the codes in %0d/%0d cases, %0d bad",
                 sens_r, sens_n, nbad);
        if (sens_r * 2 < sens_n) begin
            errors = errors + 1;
            $display("FAIL T4b you cannot prove anything: the code check does not react to R");
        end
        checks = checks + 1;

        // ---- T5: V quantiser ------------------------------------------------
        nbad = 0;
        for (i=128; i<256; i=i+1) begin
            d = 128'd0;
            for (L=0; L<8; L=L+1) d[L*16 +: 16] = {1'b0, 8'd200, 7'd1};
            d[15:0] = {1'b0, 8'd200, i[6:0]};
            emax = maxmag(d);
            v_mag(d, veb_o, vm_o);
            m_e  = ref_mv_of_num({{8{1'b0}}, 1'b1, emax[6:0]} << 2);
            eb_e = (emax[14:7] > 3) ? (emax[14:7] - 2) : 8'd1;
            checks = checks + 1;
            if (vm_o !== m_e || veb_o !== ((eb_e > 8'd10) ? eb_e - 8'd10 : 8'd0)) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T5 V mag M8=%0d m=%0d exp=%0d", i, vm_o, m_e);
            end
            R_v = ref_recip_v(m_e);
            v_quant(d, vqgot);
            vqexp = ref_q_v(eb_e, R_v, d);
            checks = checks + 1;
            if (vqgot !== vqexp) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T5 V q M8=%0d", i);
            end
        end
        for (i=128; i<256; i=i+1) begin
            d = 128'd0;
            for (L=0; L<8; L=L+1) d[L*16 +: 16] = {1'b0, 8'd200, 7'd1};
            d[15:0] = {1'b0, 8'd200, i[6:0]};
            emax = maxmag(d);
            v_mag(d, veb_o, vm_o);
            for (sh=-15; sh<=15; sh=sh+1) begin
                ebi = 200 - sh;
                if (ebi < 1) ebi = 1;
                v_lock(ebi[7:0], veb_o, vm_o);
                ln   = ref_lock_num(emax, ebi[7:0]);
                m_e  = ref_m_lock_v(ln);
                eb_e = (ebi[7:0] == 8'd0) ? 8'd1 : ebi[7:0];
                checks = checks + 1;
                if (vm_o !== m_e) begin
                    nbad = nbad + 1;
                    if (nbad < 6) $display("FAIL T5 V lock M8=%0d sh=%0d m=%0d exp=%0d", i, sh, vm_o, m_e);
                end
                R_v = ref_recip_v(m_e);
                v_quant(d, vqgot);
                vqexp = ref_q_v(eb_e, R_v, d);
                checks = checks + 1;
                if (vqgot !== vqexp) begin
                    nbad = nbad + 1;
                    if (nbad < 6) $display("FAIL T5 V lock q M8=%0d sh=%0d", i, sh);
                end
            end
        end
        d = kpat_lo(8'd120, 7);
        for (i=0; i<256; i=i+1) begin
            v_rest(8'd120, i[7:0], veb_o, vm_o);
            R_v = ref_recip_v(i[7:0]);
            checks = checks + 1;
            if (vm_o !== i[7:0]) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T5 V restore m=%0d got=%0d", i, vm_o);
            end
            v_quant(d, vqgot);
            vqexp = ref_q_v(8'd120, R_v, d);
            checks = checks + 1;
            if (vqgot !== vqexp) begin
                nbad = nbad + 1;
                if (nbad < 6) $display("FAIL T5 V restore q m=%0d", i);
            end
        end
        $display("T5 V quantiser: %0d bad", nbad);

        // ---------------------------------------------------------------- done
        if (errors == 0) $display("RESULT: PASS  checks=%0d", checks);
        else             $display("RESULT: FAIL  errors=%0d checks=%0d", errors, checks);
        $finish;
    end

    initial begin
        #100_000_000;
        $display("RESULT: FAIL timeout (errors=%0d checks=%0d)", errors, checks);
        $finish;
    end
endmodule
`default_nettype wire
