`timescale 1ns/1ps
`default_nettype none
// Self-checking testbench for kv8_scale_ram.
//  PHASE1 protocol: load before acquire rejected, read before release rejected
//  PHASE2 full-layer bulk load (4096 words) + 128 B exponent table
//  PHASE3 read sweep of the whole layer, 1 word/cycle, exact BF16 comparison
//  PHASE4 ping-pong: bank1 bulk load overlaps bank0 attention read
//  PHASE5 single-token appends (8-bit lane writes) + read-back, conflict/clip
//  PHASE6 V scale table 16 words
//  PHASE7 rd_release -> reads rejected, re-acquire -> layer N+2 overwrite
// The expected scale is computed in double (m * 2^(eb-127)); the DUT BF16 output
// is converted back to double and compared exactly, so this TB does not
// duplicate the RTL bit manipulation.
module tb_kv8_scale;
    localparam integer CLK_PS = 3333;   // 300 MHz
    reg clk=0, rst_n=0;
    always #(CLK_PS/1000.0/2) clk = ~clk;

    reg        wr_bank=0, rd_bank=0;
    reg        wr_acq=0,wr_rel=0,rd_rel=0;
    wire       wr_grant,rd_grant,owner_err;
    reg        etab_v=0; wire etab_r;
    reg  [2:0] etab_w=0; reg [127:0] etab_d=0;
    reg        load_v=0; wire load_r;
    reg [11:0] load_w=0; reg [127:0] load_m=0;      // [M16-SCALE] 8 tokens x m16
    reg        app_v=0; wire app_r,app_taken,app_conf,clip_sticky;
    reg [15:0] app_tok=0; reg app_hd=0; reg [15:0] app_m=0; reg [7:0] app_e=0; reg app_clip=0;
    reg        rd_en=0; reg [11:0] rd_word=0;
    wire       rd_valid;
    wire [127:0] rd_m16;      // [M16-SCALE] 8 x m16, raw (no BF16 expander)
    wire [7:0]   rd_eb;
    wire [135:0] rd_scale = {rd_eb, rd_m16};
    reg        v_we=0; reg [3:0] v_word=0; reg [127:0] v_data=0;
    reg  [3:0] v_rw=0; wire [127:0] v_rdata;
    wire [31:0] stat_load,stat_app,stat_rd;

    kv8_scale_ram #(.BANKS(2)) dut (
        .clk(clk),.rst_n(rst_n),
        .i_bank_wr(wr_bank),.i_bank_rd(rd_bank),
        .i_wr_acquire(wr_acq),.i_wr_release(wr_rel),.i_rd_release(rd_rel),
        .o_wr_grant(wr_grant),.o_rd_grant(rd_grant),.o_owner_error(owner_err),
        .i_etab_valid(etab_v),.o_etab_ready(etab_r),.i_etab_word(etab_w),.i_etab_data(etab_d),.i_etab_be(16'hffff),
        .i_load_valid(load_v),.o_load_ready(load_r),.i_load_word(load_w),.i_load_mant(load_m),
        .i_app_valid(app_v),.o_app_ready(app_r),.i_app_token(app_tok),.i_app_head(app_hd),
        .i_app_mant(app_m),.i_app_exp(app_e),.i_app_clip(app_clip),
        .o_app_taken(app_taken),.o_app_conflict(app_conf),
        .i_rd_en(rd_en),.i_rd_word(rd_word),.o_rd_valid(rd_valid),.o_rd_m16(rd_m16),.o_rd_eb(rd_eb),
        .i_v_we(v_we),.i_v_word(v_word),.i_v_data(v_data),.i_v_rd_word(v_rw),.o_v_rd_data(v_rdata),
        .o_clip_sticky(clip_sticky),.o_stat_load(stat_load),.o_stat_app(stat_app),.o_stat_rd(stat_rd)
    );

    integer errors=0, checks=0;

    // ---------------------------------------------------------------- model
    // [M16-SCALE] the model now spans the whole 16-bit mantissa range: an 8-bit
    // pattern would not prove that two bytes per token are stored and returned.
    function automatic [15:0] f_m(input integer word,input integer lane);
        integer v;
        begin
            v = (word*7 + lane*13) % 65535;
            f_m = v[15:0] + 16'd1;
        end
    endfunction
    function automatic [7:0] f_e(input integer head,input integer block);
        f_e = 8'd112 + ((head*17 + block*5) % 12);
    endfunction
    integer ov_n=0;
    integer ov_word[0:15];
    integer ov_lane[0:15];
    reg [15:0] ov_m[0:15];
    reg [7:0] ov_e[0:15];
    function automatic [15:0] model_m(input integer word,input integer lane);
        integer i;
        begin
            model_m = f_m(word,lane);
            for (i=0;i<ov_n;i=i+1)
                if (ov_word[i]==word && ov_lane[i]==lane) model_m = ov_m[i];
        end
    endfunction
    function automatic [7:0] model_e(input integer head,input integer block);
        integer i;
        begin
            model_e = f_e(head,block);
            for (i=0;i<ov_n;i=i+1)
                if ((ov_word[i]/2048)==head && ((ov_word[i]%2048)/32)==block) model_e = ov_e[i];
        end
    endfunction
    // exact power of two (Verilog "**" cannot take the unsigned 8-bit fields
    // used here without sign surprises in some simulators)
    function automatic real ldexp2(input integer e);
        integer n;
        real r;
        begin
            r = 1.0;
            if (e>=0) for (n=0;n<e;n=n+1) r = r*2.0;
            else      for (n=0;n<-e;n=n+1) r = r*0.5;
            ldexp2 = r;
        end
    endfunction
    function automatic real bf16_to_real(input [15:0] v);
        real m;
        integer de;
        begin
            if (v[14:7]==8'd0) bf16_to_real = 0.0;
            else begin
                m  = 1.0 + (v[6:0] * 1.0 / 128.0);
                de = v[14:7] - 127;
                bf16_to_real = m * ldexp2(de);
            end
        end
    endfunction
    // [M16-SCALE] no BF16 expander any more: the read word is the raw m16 word
    // plus the block exponent, so this compares integers (exact, no float).
    task automatic chk_word(input integer word,input [135:0] got);
        integer ln,head,block;
        reg [15:0] mm;
        reg [7:0]  ee;
        begin
            head  = word/2048;
            block = (word%2048)/32;
            ee    = model_e(head,block);
            checks = checks + 1;
            if (got[135:128] !== ee) begin
                errors = errors + 1;
                if (errors<10) $display("FAIL word=%0d eb got=%0d want=%0d",word,got[135:128],ee);
            end
            for (ln=0;ln<8;ln=ln+1) begin
                mm = model_m(word,ln);
                checks = checks + 1;
                if (got[ln*16 +: 16] !== mm) begin
                    errors = errors + 1;
                    if (errors<10)
                        $display("FAIL word=%0d lane=%0d m16 got=%0d want=%0d",
                                 word,ln,got[ln*16 +: 16],mm);
                end
            end
        end
    endtask

    // ---------------------------------------------------------------- helpers
    task automatic acquire_wr(input integer b);
        begin
            wr_bank=b[0]; wr_acq=1; @(posedge clk); #1; wr_acq=0; @(posedge clk); #1;
            if (!wr_grant) begin errors=errors+1; $display("FAIL no write grant after acquire bank %0d",b); end
        end
    endtask
    task automatic release_wr_t; begin wr_rel=1; @(posedge clk); #1; wr_rel=0; @(posedge clk); #1; end endtask
    task automatic release_rd_t; begin rd_rel=1; @(posedge clk); #1; rd_rel=0; @(posedge clk); #1; end endtask

    task automatic load_etab(input integer b);
        integer w,k,h,blk;
        begin
            for (w=0;w<8;w=w+1) begin
                etab_w=w[2:0];
                for (k=0;k<16;k=k+1) begin
                    h=(w*16+k)/64; blk=(w*16+k)%64;
                    etab_d[k*8 +: 8] = f_e(h,blk);
                end
                etab_v=1; @(posedge clk); #1; etab_v=0;
            end
            @(posedge clk); #1;
        end
    endtask
    task automatic load_mant(input integer b);
        integer w,k;
        begin
            for (w=0;w<4096;w=w+1) begin
                load_w=w[11:0];
                for (k=0;k<8;k=k+1) load_m[k*16 +: 16] = f_m(w,k);
                load_v=1; @(posedge clk); #1;
            end
            load_v=0; @(posedge clk); #1; @(posedge clk); #1;
        end
    endtask

    task automatic load_vtable;
        integer n;
        begin
            for (n=0;n<16;n=n+1) begin
                v_word=n[3:0]; v_data={120'd0,n[3:0],12'd0} | {124'd0,n[3:0]};
                v_we=1; @(posedge clk); #1;
            end
            v_we=0; @(posedge clk); #1;
        end
    endtask

    integer i,w,ln,head,block,cyc,gotn;
    reg [127:0] cap;

    initial begin
        $display("=== tb_kv8_scale (kv8_scale_ram, BANKS=2) ===");
        // ---------------- PHASE1 protocol ----------------
        rst_n=0; repeat(4) @(posedge clk); rst_n=1; repeat(2) @(posedge clk); #1;
        wr_bank=0; rd_bank=0;
        load_v=1; load_w=0; load_m=128'h00010002000300040005000600070008;
        @(posedge clk); #1; load_v=0; @(posedge clk); #1;
        checks=checks+1;
        if (!owner_err) begin errors=errors+1; $display("FAIL load before acquire did not raise owner_error"); end
        else $display("PHASE1 load-before-acquire rejected (owner_error=1)");
        rd_en=1; rd_word=0; @(posedge clk); #1; rd_en=0; repeat(3) @(posedge clk); #1;
        checks=checks+1;
        if (rd_valid) begin errors=errors+1; $display("FAIL read before release returned valid"); end
        else $display("PHASE1 read-before-release rejected");

        // ---------------- PHASE2 load bank 0 ----------------
        acquire_wr(0);
        load_etab(0);
        load_mant(0);
        release_wr_t;
        $display("PHASE2 load: stat_load=%0d (expect 4096)",stat_load);
        checks=checks+1;
        if (stat_load!==4096) begin errors=errors+1; $display("FAIL stat_load %0d",stat_load); end

        // ---------------- PHASE3 read sweep ----------------
        rd_bank=0;
        checks=checks+1;
        if (!rd_grant) begin errors=errors+1; $display("FAIL no read grant after release"); end
        gotn=0;
        for (w=0;w<4096+6;w=w+1) begin
            rd_en   = (w<4096);
            rd_word = w[11:0];
            @(posedge clk); #1;
            if (rd_valid) begin
                chk_word(gotn,rd_scale);
                gotn=gotn+1;
            end
        end
        rd_en=0;
        $display("PHASE3 read sweep: %0d words checked, stat_rd=%0d",gotn,stat_rd);
        checks=checks+1;
        if (gotn!==4096) begin errors=errors+1; $display("FAIL read count %0d",gotn); end

        // ---------------- latency ----------------
        @(posedge clk); #1;
        rd_en=1; rd_word=12'd5; @(posedge clk); #1; rd_en=0;
        cyc=1;
        while (!rd_valid && cyc<10) begin @(posedge clk); #1; cyc=cyc+1; end
        $display("PHASE3 read latency = %0d cycles (i_rd_en -> o_rd_valid)",cyc);
        chk_word(5,rd_scale);

        // ---------------- PHASE4 ping-pong overlap ----------------
        fork
            begin : loader1
                wr_bank=1;
                acquire_wr(1);
                load_etab(1);
                load_vtable;
                load_mant(1);
                release_wr_t;
            end
            begin : reader0
                integer g;
                g=0;
                for (i=0;i<256+6;i=i+1) begin
                    rd_bank=0;
                    rd_en   = (i<256);
                    rd_word = i[11:0];
                    @(posedge clk); #1;
                    if (rd_valid) begin chk_word(g,rd_scale); g=g+1; end
                end
                rd_en=0;
                $display("PHASE4 overlapped: bank0 read %0d words while bank1 loaded",g);
                checks=checks+1;
                if (g!==256) begin errors=errors+1; $display("FAIL overlap read count %0d",g); end
            end
        join
        $display("PHASE4 ping-pong bank1 load+release done, stat_load=%0d",stat_load);

        // ---------------- PHASE5 appends into bank 1 (READY) ----------------
        wr_bank=1; rd_bank=1;
        ov_n=0;
        for (i=0;i<8;i=i+1) begin
            app_tok = 3000 + i*7;
            app_hd  = 1;
            app_m   = 16'd40000 + i[15:0];   // exercises the high byte of the 2-byte mantissa
            app_e   = f_e(1,(3000+i*7)/256);
            app_clip= 0;
            app_v=1; @(posedge clk); #1;
            while (!app_r) begin @(posedge clk); #1; end
            app_v=0;
            ov_word[ov_n] = 2048 + (3000+i*7)/8;
            ov_lane[ov_n] = ((3000+i*7)%8);
            ov_m[ov_n]    = app_m;
            ov_e[ov_n]    = app_e;
            ov_n=ov_n+1;
            @(posedge clk); #1;
        end
        repeat(3) @(posedge clk); #1;
        $display("PHASE5 append: stat_app=%0d (expect 8)",stat_app);
        checks=checks+1;
        if (stat_app!==8) begin errors=errors+1; $display("FAIL stat_app %0d",stat_app); end
        for (i=0;i<8;i=i+1) begin
            rd_en=1; rd_word=ov_word[i][11:0]; @(posedge clk); #1; rd_en=0;
            repeat(3) @(posedge clk); #1;
            chk_word(ov_word[i],rd_scale);
        end
        checks=checks+1;
        if (app_conf) begin errors=errors+1; $display("FAIL unexpected exponent conflict"); end
        // deliberate conflict + clip on the last token of the layer
        app_tok=16383; app_hd=1; app_m=16'd65535; app_e=f_e(1,63)+8'd1; app_clip=1;
        app_v=1; @(posedge clk); #1; app_v=0; repeat(3) @(posedge clk); #1;
        checks=checks+1;
        if (!app_conf) begin errors=errors+1; $display("FAIL exponent conflict not reported"); end
        else $display("PHASE5 exponent conflict reported, clip_sticky=%0d",clip_sticky);
        checks=checks+1;
        if (!clip_sticky) begin errors=errors+1; $display("FAIL clip sticky not set"); end

        // ---------------- PHASE6 V table (loaded during bank1 LOADING) --------
        wr_bank=1; rd_bank=1;
        for (i=0;i<16;i=i+1) begin
            v_rw=i[3:0]; @(posedge clk); #1;
            checks=checks+1;
            if (v_rdata !== ({120'd0,i[3:0],12'd0} | {124'd0,i[3:0]})) begin
                errors=errors+1;
                if (errors<15) $display("FAIL vtable %0d got %h",i,v_rdata);
            end
        end
        $display("PHASE6 V scale table 16 words checked");

        // ---------------- PHASE7 release / reuse ----------------
        rd_bank=1; release_rd_t;
        rd_en=1; rd_word=0; @(posedge clk); #1; rd_en=0; repeat(3) @(posedge clk); #1;
        checks=checks+1;
        if (rd_valid) begin errors=errors+1; $display("FAIL read after rd_release returned data"); end
        else $display("PHASE7 read after rd_release rejected");
        wr_bank=1;
        acquire_wr(1);
        checks=checks+1;
        if (!wr_grant) begin errors=errors+1; $display("FAIL bank1 not writable for layer N+2"); end
        load_v=1; load_w=12'd0; load_m=128'h00010002000300040005000600070008; @(posedge clk); #1; load_v=0;
        @(posedge clk); #1; @(posedge clk); #1;
        release_wr_t;
        $display("PHASE7 layer N+2 overwrite accepted (owner_error=%0d)",owner_err);

        // ---------------- summary ----------------
        $display("STATS load=%0d app=%0d rd=%0d owner_error=%0d conflict=%0d clip=%0d",
                 stat_load,stat_app,stat_rd,owner_err,app_conf,clip_sticky);
        if (errors==0) $display("RESULT tb_kv8_scale: PASS checks=%0d",checks);
        else           $display("RESULT tb_kv8_scale: FAIL errors=%0d checks=%0d",errors,checks);
        $finish;
    end
endmodule
`default_nettype wire
