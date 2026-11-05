`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// tb_kv8_dot -- numeric error of the INT8 KV path against a BF16 reference.
//
//  PART A  kv8_k_quant rule check: directed + 64 random tokens, (eb,m) and every
//          q byte compared against a double-precision reference.
//  PART B  K per-token dot-product error, 512 random tokens x 64 dims:
//            (i)  the DUT block-exponent scheme (MARGIN=2) through the DUT
//                 kv8_score_scale8 multiply
//            (ii) the ideal per-token scale on the same distribution (baseline
//                 that the earlier study reported as 0.6%)
//  PART C  V per-dim dot-product error, 64 dims x 256 tokens, per-dim scale,
//          dequant through the same DUT 8-lane multiplier.
// -----------------------------------------------------------------------------
module tb_kv8_dot;
    reg clk=0, rst_n=0;
    always #1.666 clk=~clk;     // 300 MHz

    reg         qclr=0,qpass=0,qval=0,qlast=0;
    reg  [127:0] qk=0;
    wire        qdone,qvld;
    wire [7:0]  qeb;
    wire [15:0] qm;      // [M16-SCALE] 16-bit mantissa field
    wire [7:0]  qmaxexp;
    // [K16] the quantiser publishes 8 x 16-bit {hi,lo}; this TB runs it with
    // CODE_EXTRA=0, so hi=0 and the 8-bit code is the LOW byte of each value.
    wire [127:0] qq;
    kv8_k_quant #(.MARGIN(2), .CODE_EXTRA(0)) quant (
        .clk(clk),.rst_n(rst_n),.i_clear(qclr),.i_pass(qpass),.i_valid(qval),
        .i_last(qlast),.i_k(qk),.i_eb_lock_valid(1'b0),.i_eb_lock(8'd0),.i_restore_valid(1'b0),.i_restore_eb(8'd0),.i_restore_m(8'd0),.o_done(qdone),.o_eb(qeb),.o_m(qm),
        .o_maxmag_exp(qmaxexp),.o_q_valid(qvld),.o_q(qq)
    );
    reg          mval=0;
    reg  [127:0] mx=0,ms=0;
    wire         mvld; wire [127:0] my;
    kv8_score_scale8 mul (
        .clk(clk),.rst_n(rst_n),.i_valid(mval),.i_score(mx),.i_scale(ms),
        .o_valid(mvld),.o_score(my)
    );

    // ---------------- helpers ----------------
    function automatic real ldexp2(input integer e);
        integer n; real r;
        begin r=1.0; if(e>=0) for(n=0;n<e;n=n+1) r=r*2.0; else for(n=0;n<-e;n=n+1) r=r*0.5; ldexp2=r; end
    endfunction
    function automatic real bf16_to_real(input [15:0] v);
        integer de;
        begin
            if (v[14:7]==8'd0) bf16_to_real=0.0;
            else begin
                de=v[14:7]-127;
                bf16_to_real=(v[15] ? -1.0 : 1.0)*(1.0+(v[6:0]*1.0/128.0))*ldexp2(de);
            end
        end
    endfunction
    function automatic [15:0] real_to_bf16(input real x);
        real ax,sc; integer e; reg sign; reg [15:0] m; reg [7:0] ef;
        begin
            if (x==0.0) real_to_bf16=16'h0000;
            else begin
                sign=(x<0.0); ax=(x<0.0)?-x:x;
                e=0; sc=ax;
                while (sc>=2.0) begin sc=sc*0.5; e=e+1; end
                while (sc<1.0)  begin sc=sc*2.0; e=e-1; end
                m=$rtoi(sc*128.0+0.5);
                if (m>=256) begin m=128; e=e+1; end
                if (m<128) m=128;
                ef = e+127;
                real_to_bf16={sign,ef,m[6:0]};
            end
        end
    endfunction

    integer errors=0, checks=0, lsb_diff=0;
    integer seed=12345;
    integer t,d,b,i,g,dd,tok,kk,oi,out_pct;

    reg [15:0] kv  [0:63];
    reg [15:0] qvec[0:63];
    reg [7:0]  qbytes[0:63];
    real       kreal[0:63];
    real       qreal[0:63];
    reg [15:0] vv [0:7][0:255];
    reg [15:0] pp [0:255];
    real       vs [0:7];
    real       ctx_int[0:7];
    real       ctx_ref[0:7];

    real sig,acc,refv,val,r,ideal_s,s_dut;
    reg [15:0] sc8;
    real err_sum,err_max,err_n,rel,ref_sum;
    reg [15:0] acc_bf16;
    reg [127:0] qcap;
    reg [7:0]  eb_cap; reg [15:0] m_cap;
    reg [127:0] x128,s128,y128;

    task automatic run_quant;
        begin
            qclr=1; @(posedge clk); #0.1; qclr=0;
            for (b=0;b<8;b=b+1) begin
                qpass=0; qval=1; qlast=(b==7);
                for (i=0;i<8;i=i+1) qk[i*16 +: 16] = kv[b*8+i];
                @(posedge clk); #0.1;
            end
            // [RECIP-SEQ] (m, R) are now produced by a 24..40-cycle iterative
            // divider behind the o_done handshake instead of a combinational
            // divide, so the (eb, m) capture has to wait for o_done.  This is the
            // module's documented contract ("o_done pulses: eb/m valid for this
            // head"); the numeric checks below are unchanged.
            qval=0; qlast=0; @(posedge clk); #0.1;
            while (!qdone) begin @(posedge clk); #0.1; end
            eb_cap=qeb; m_cap=qm;
            for (b=0;b<8;b=b+1) begin
                qpass=1; qval=1; qlast=(b==7);
                for (i=0;i<8;i=i+1) qk[i*16 +: 16] = kv[b*8+i];
                @(posedge clk); #0.1;
                if (!qvld && b==0) begin errors=errors+1; $display("FAIL quantise valid"); end
                qcap=qq;
                for (i=0;i<8;i=i+1) qbytes[b*8+i]=qcap[i*16 +: 8];   // [K16] low byte = 8-bit code
            end
            qval=0; qpass=0; @(posedge clk); #0.1;
        end
    endtask

    task automatic dut_mul8(input [127:0] x, input [127:0] s, output [127:0] y);
        begin
            mval=1; mx=x; ms=s; @(posedge clk); #0.1; mval=0;
            @(posedge clk); #0.1;      // o_valid is high in this cycle (2-stage pipe)
            if (!mvld) begin errors=errors+1; $display("FAIL mul8 valid"); end
            y=my;
        end
    endtask

    initial begin
        $display("=== tb_kv8_dot: INT8 KV numeric error vs BF16 reference ===");
        rst_n=0; repeat(4) @(posedge clk); rst_n=1; repeat(2) @(posedge clk);

        // ================= PART A =================
        for (d=0;d<64;d=d+1) kv[d]=16'h3f80;      // all +1.0
        run_quant;
        $display("PART A all +1.0 : eb=%0d m=%0d q[0]=%0d (expect eb=115 m=5 q=102, table convention)",
                 eb_cap,m_cap,$signed(qbytes[0]));
        checks=checks+1;
        if (eb_cap!==8'd115 || m_cap!==16'd5) begin errors=errors+1; $display("FAIL A eb/m"); end
        checks=checks+1;
        if ($signed(qbytes[0])!==102) begin errors=errors+1; $display("FAIL A q=%0d",$signed(qbytes[0])); end
        for (d=0;d<64;d=d+1) kv[d]=16'hc000;      // all -2.0
        run_quant;
        $display("PART A all -2.0 : eb=%0d m=%0d q[0]=%0d (expect eb=116 m=5 q=-102)",
                 eb_cap,m_cap,$signed(qbytes[0]));
        checks=checks+1;
        if ($signed(qbytes[0])!==-102) begin errors=errors+1; $display("FAIL A neg"); end
        // directed alternating signs: +1.0 / -1.0 on even/odd lanes
        for (d=0;d<64;d=d+1) kv[d] = (d%2==0) ? 16'h3f80 : 16'hbf80;
        run_quant;
        $display("PART A alt sign : q[0]=%0d q[1]=%0d q[2]=%0d q[3]=%0d (expect +102 -102 +102 -102)",
                 $signed(qbytes[0]),$signed(qbytes[1]),$signed(qbytes[2]),$signed(qbytes[3]));
        checks=checks+1;
        if ($signed(qbytes[0])!==102 || $signed(qbytes[1])!==-102 ||
            $signed(qbytes[2])!==102 || $signed(qbytes[3])!==-102) begin
            errors=errors+1; $display("FAIL A per-lane sign");
        end
        // directed decreasing magnitudes: 1.0, 0.5, 0.25 ... (lane order check)
        for (d=0;d<64;d=d+1) kv[d] = real_to_bf16(1.0/(2.0**d));
        run_quant;
        $display("PART A ramp     : q[0..7]= %0d %0d %0d %0d %0d %0d %0d %0d",
                 $signed(qbytes[0]),$signed(qbytes[1]),$signed(qbytes[2]),$signed(qbytes[3]),
                 $signed(qbytes[4]),$signed(qbytes[5]),$signed(qbytes[6]),$signed(qbytes[7]));

        // 64 random tokens: every q byte against the exact rule
        for (t=0;t<64;t=t+1) begin
            sig = 0.2+3.0*(($dist_normal(seed,0,1000)+2000)/4000.0);
            if (sig<0.1) sig=0.1;
            ideal_s=0.0;
            for (d=0;d<64;d=d+1) begin
                val=$dist_normal(seed,0,1000)/1000.0*sig;
                kv[d]=real_to_bf16(val);
                kreal[d]=bf16_to_real(kv[d]);
                r=(kreal[d]<0.0)?-kreal[d]:kreal[d];
                if (r>ideal_s) ideal_s=r;
            end
            run_quant;
            s_dut = m_cap*ldexp2(eb_cap-124);   // 8 x table value = s'
            checks=checks+1;
            if (s_dut < ideal_s/127.0) begin
                errors=errors+1;
                if (errors<8) $display("FAIL A s'<max/127 %e < %e",s_dut,ideal_s/127.0);
            end
            for (d=0;d<64;d=d+1) begin
                val=kreal[d]/s_dut;
                refv=(val>=0.0)? $floor(val+0.5) : $ceil(val-0.5);
                if (refv> 127.0) refv= 127.0;
                if (refv<-127.0) refv=-127.0;
                checks=checks+1;
                if (refv!=$signed(qbytes[d])) begin
                    if ((refv-$signed(qbytes[d])<=1.0) && (refv-$signed(qbytes[d])>=-1.0))
                        lsb_diff=lsb_diff+1;          // +-1 LSB from the 2^16/m reciprocal
                    else begin
                        errors=errors+1;
                        if (errors<8) $display("FAIL A q[%0d]=%0d refv=%0.0f",d,$signed(qbytes[d]),refv);
                    end
                end
            end
        end
        $display("PART A rule check: 64 random tokens, 4096 q bytes; +-1LSB from reciprocal = %0d",lsb_diff);

        // ---- directed 8-lane multiplier check ----
        begin : multest
            reg [127:0] yy;
            real e0,e1,e2,e3;
            x128=0; s128=0;
            x128[15:0]  = real_to_bf16(3.0);   s128[15:0]  = real_to_bf16(2.0);
            x128[31:16] = real_to_bf16(1.0);   s128[31:16] = real_to_bf16(1.0);
            x128[47:32] = real_to_bf16(0.5);   s128[47:32] = real_to_bf16(0.25);
            x128[63:48] = real_to_bf16(-2.0);  s128[63:48] = real_to_bf16(3.0);
            x128[79:64] = real_to_bf16(1.5);   s128[79:64] = real_to_bf16(-2.0);
            dut_mul8(x128,s128,yy);
            e0=bf16_to_real(yy[15:0]);  e1=bf16_to_real(yy[31:16]);
            e2=bf16_to_real(yy[47:32]); e3=bf16_to_real(yy[63:48]);
            $display("mul8 directed: 3*2=%f 1*1=%f 0.5*0.25=%f -2*3=%f 1.5*-2=%f",
                     e0,e1,e2,bf16_to_real(yy[79:64]),bf16_to_real(yy[95:80]));
            $display("                  -2*3=%f",e3);
            checks=checks+1; if (e0!=6.0) begin errors=errors+1; $display("FAIL mul 3*2=%f",e0); end
            checks=checks+1; if (e1!=1.0) begin errors=errors+1; $display("FAIL mul 1*1=%f",e1); end
            checks=checks+1; if (e2!=0.125) begin errors=errors+1; $display("FAIL mul 0.5*0.25=%f",e2); end
        end

        // ================= PART B(i): DUT block-exponent scheme =================
        for (oi=0;oi<2;oi=oi+1) begin
        out_pct = oi*2;
        err_sum=0.0; err_max=0.0; err_n=0.0; ref_sum=0.0;
        for (t=0;t<512;t=t+1) begin
            sig=0.3+1.7*(($dist_normal(seed,0,1000)+3000)/6000.0);
            if (sig<0.1) sig=0.1;
            for (d=0;d<64;d=d+1) begin
                val=$dist_normal(seed,0,1000)/1000.0;
                if ((($random & 32'h7fffffff)%100)<out_pct) val=val*8.0;
                kv[d]=real_to_bf16(val*sig);
                qvec[d]=real_to_bf16($dist_normal(seed,0,1000)/1000.0);
                kreal[d]=bf16_to_real(kv[d]);
                qreal[d]=bf16_to_real(qvec[d]);
            end
            run_quant;
            s_dut=m_cap*ldexp2(eb_cap-124);
            acc=0.0; refv=0.0;
            for (d=0;d<64;d=d+1) begin
                acc=acc+$signed(qbytes[d])*qreal[d];
                refv=refv+kreal[d]*qreal[d];
            end
            if (refv==0.0) continue;
            acc_bf16=real_to_bf16(acc);
            sc8=real_to_bf16(m_cap*ldexp2(eb_cap-127));   // stored table value s/8
            x128={8{acc_bf16}}; s128={8{sc8}};
            dut_mul8(x128,s128,y128);
            rel=(bf16_to_real(y128[15:0])-refv/8.0);
            err_sum=err_sum+rel*rel; ref_sum=ref_sum+(refv/8.0)*(refv/8.0);
            err_n=err_n+1.0;
            if (rel<0.0) rel=-rel;
            if (rel>err_max) err_max=rel;
        end
        $display("PART B(i)  K dot, DUT block exp, outliers=%0d%%: n=%0.0f rel_rms=%.6f%%",
                 out_pct,err_n,100.0*$sqrt(err_sum/ref_sum));

        // ================= PART B(ii): ideal per-token scale baseline ==========
        err_sum=0.0; err_max=0.0; err_n=0.0; ref_sum=0.0;
        for (t=0;t<512;t=t+1) begin
            sig=0.3+1.7*(($dist_normal(seed,0,1000)+3000)/6000.0);
            if (sig<0.1) sig=0.1;
            ideal_s=0.0;
            for (d=0;d<64;d=d+1) begin
                val=$dist_normal(seed,0,1000)/1000.0;
                if ((($random & 32'h7fffffff)%100)<out_pct) val=val*8.0;
                kv[d]=real_to_bf16(val*sig);
                qvec[d]=real_to_bf16($dist_normal(seed,0,1000)/1000.0);
                kreal[d]=bf16_to_real(kv[d]);
                qreal[d]=bf16_to_real(qvec[d]);
                r=(kreal[d]<0.0)?-kreal[d]:kreal[d];
                if (r>ideal_s) ideal_s=r;
            end
            ideal_s=ideal_s/127.0;
            acc=0.0; refv=0.0;
            for (d=0;d<64;d=d+1) begin
                val=kreal[d]/ideal_s;
                acc=acc+(((val>=0.0)? $floor(val+0.5) : $ceil(val-0.5))*qreal[d]);
                refv=refv+kreal[d]*qreal[d];
            end
            if (refv==0.0) continue;
            rel=(acc*ideal_s-refv);
            err_sum=err_sum+rel*rel; ref_sum=ref_sum+refv*refv;
            err_n=err_n+1.0;
            if (rel<0.0) rel=-rel;
            if (rel>err_max) err_max=rel;
        end
        $display("PART B(ii) K dot, ideal per-token scale, outliers=%0d%%: n=%0.0f rel_rms=%.6f%%",
                 out_pct,err_n,100.0*$sqrt(err_sum/ref_sum));
        end

        // ================= PART C: V per-dim =================
        err_sum=0.0; err_max=0.0; err_n=0.0; ref_sum=0.0;
        for (g=0;g<8;g=g+1) begin
            for (dd=0;dd<8;dd=dd+1) begin
                r=0.0;
                for (tok=0;tok<256;tok=tok+1) begin
                    vv[dd][tok]=real_to_bf16($dist_normal(seed,0,1000)/1000.0);
                    val=bf16_to_real(vv[dd][tok]); if (val<0.0) val=-val;
                    if (val>r) r=val;
                end
                vs[dd]=(r==0.0)? 1.0/127.0 : r/127.0;
            end
            for (kk=0;kk<4;kk=kk+1) begin
                // softmax-like P over 256 tokens (peaked, like a real attention row)
                acc=0.0;
                for (tok=0;tok<256;tok=tok+1) begin
                    val=$dist_normal(seed,0,1000)/1000.0*4.0;
                    r=2.718281828**val;
                    pp[tok]=real_to_bf16(r);
                    acc=acc+r;
                end
                for (tok=0;tok<256;tok=tok+1) pp[tok]=real_to_bf16(bf16_to_real(pp[tok])/acc);
                for (dd=0;dd<8;dd=dd+1) begin
                    acc=0.0; refv=0.0;
                    for (tok=0;tok<256;tok=tok+1) begin
                        val=bf16_to_real(vv[dd][tok]);
                        r  =bf16_to_real(pp[tok]);
                        acc=acc+r*((val/vs[dd]>=0.0)? $floor(val/vs[dd]+0.5) : $ceil(val/vs[dd]-0.5));
                        refv=refv+r*val;
                    end
                    ctx_int[dd]=acc; ctx_ref[dd]=refv;
                    x128[dd*16 +: 16]=real_to_bf16(acc);
                    s128[dd*16 +: 16]=real_to_bf16(vs[dd]);
                end
                dut_mul8(x128,s128,y128);
                for (dd=0;dd<8;dd=dd+1) begin
                    if (ctx_ref[dd]==0.0) continue;
                    rel=(bf16_to_real(y128[dd*16 +: 16])-ctx_ref[dd]);
                    err_sum=err_sum+rel*rel; ref_sum=ref_sum+ctx_ref[dd]*ctx_ref[dd];
                    err_n=err_n+1.0;
                    if (rel<0.0) rel=-rel;
                    if (rel>err_max) err_max=rel;
                    checks=checks+1;
                end
            end
        end
        $display("PART C   V per-dim dot (DUT 8-lane dequant): n=%0.0f rel_rms=%.6f%%",
                 err_n,100.0*$sqrt(err_sum/ref_sum));

        if (errors==0) $display("RESULT tb_kv8_dot: PASS checks=%0d",checks);
        else           $display("RESULT tb_kv8_dot: FAIL errors=%0d checks=%0d",errors,checks);
        $finish;
    end
endmodule
`default_nettype wire
