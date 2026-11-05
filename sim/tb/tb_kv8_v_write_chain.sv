`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_v_write_chain -- INT8 V producer (write side of the V half)
//
//   PHASE1  4 tokens x 2 heads x 8 beats: every INT8 byte at the token-major
//           address v_base + token*128 + head*64 + dg*16 + byte, byte-exact
//           against a reference of the quantiser's own integer rule.  The first
//           token of the layer locks ONLY the exponent; every token re-derives
//           its own mantissa (K's rule), so its codes use their full range.
//   PHASE2  the 256 B per-layer V scale image now carries the layer's power of
//           two 2^(eb_lock-134) (i.e. m = 1) because the per-token mantissa
//           travels on o_m_app_* instead.
//   PHASE3  per-token mantissa appends: the (token, head) -> m sequence must
//           match the reference exactly, o_m_max must be its maximum, and only
//           a token whose m saturates at MTOP may be counted as a clip.  The
//           saturated-CODE count (o_clip_codes) is checked against the number of
//           lanes the reference sees over range.
// =============================================================================
module tb_kv8_v_write_chain;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    localparam integer NTOK = 4;

    reg         i_start = 0;
    reg  [4:0]  i_layer = 5'd1;
    reg  [15:0] i_token = 0;
    reg  [63:0] i_v_base = 64'h2000_0000 + (64'd1 << 23);
    wire        o_ready, o_busy, o_done, o_v_ready;
    reg         i_v_valid = 0;
    reg  [127:0] i_v_data = 0;
    wire        o_beat_valid, o_beat_last;
    reg         i_beat_ready = 1;
    wire [63:0] o_beat_address;
    wire [127:0] o_beat_data;
    wire        o_scale_valid;
    wire [31:0] o_scale_bf16;
    reg         i_scale_clear = 0;
    wire        o_clip_sticky;
    wire [15:0] o_clip_count;
    wire [31:0] o_clip_codes;
    wire [7:0]  o_m_max;
    // [V-PERTOK-M] per-token mantissa append
    wire        o_m_app_valid;
    reg         i_m_app_ready = 1;
    wire [15:0] o_m_app_token;
    wire        o_m_app_head;
    wire [15:0] o_m_app_mant;

    kv8_v_write_chain #(.MARGIN(2)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_start(i_start), .i_layer(i_layer), .i_token(i_token), .i_v_base(i_v_base),
        .o_ready(o_ready),
        .i_v_valid(i_v_valid), .o_v_ready(o_v_ready), .i_v_data(i_v_data),
        .o_busy(o_busy), .o_done(o_done),
        .o_beat_valid(o_beat_valid), .i_beat_ready(i_beat_ready),
        .o_beat_address(o_beat_address), .o_beat_data(o_beat_data), .o_beat_last(o_beat_last),
        .o_scale_valid(o_scale_valid), .o_scale_bf16(o_scale_bf16), .i_scale_clear(i_scale_clear),
        .o_v_we(), .o_v_word(), .o_v_data(),
        .o_m_app_valid(o_m_app_valid), .i_m_app_ready(i_m_app_ready),
        .o_m_app_token(o_m_app_token), .o_m_app_head(o_m_app_head),
        .o_m_app_mant(o_m_app_mant),
        .o_clip_sticky(o_clip_sticky), .o_clip_count(o_clip_count),
        .o_clip_codes(o_clip_codes), .o_m_max(o_m_max),
        .o_token_beats(), .o_dbg_state()
    );

    // ---- stimulus data -----------------------------------------------------
    reg [15:0] vref [0:NTOK*2*64-1];       // BF16 V per (token, head, dim)
    reg [7:0]  qexp [0:NTOK*2*64-1];       // expected INT8 code
    reg [15:0] scale_exp [0:1];            // expected BF16 layer scale per head
    reg [7:0]  mexp [0:NTOK*2-1];          // expected mantissa per (token,head)
    integer   clip_exp;                    // tokens whose m hit MTOP (or 0)
    integer   codes_exp;                   // saturated codes over the whole layer
    integer   mmax_exp;
    reg       clip_any;      // any head of this token saturated its mantissa

    function [2:0] msb_i(input [7:0] v);
        begin
            if      (v[7]) msb_i = 3'd7;
            else if (v[6]) msb_i = 3'd6;
            else if (v[5]) msb_i = 3'd5;
            else if (v[4]) msb_i = 3'd4;
            else if (v[3]) msb_i = 3'd3;
            else if (v[2]) msb_i = 3'd2;
            else if (v[1]) msb_i = 3'd1;
            else           msb_i = 3'd0;
        end
    endfunction
    function [16:0] recip_lut(input [7:0] m);
        reg [16:0] d;
        begin
            if (m == 8'd0) recip_lut = 17'd65536;
            else begin
                d = {9'd0,m};
                recip_lut = (17'd65536 + {1'b0,d[16:1]}) / d;
            end
        end
    endfunction
    // [V-PERTOK-M] round the mantissa UP to a power of two, clamped to 128
    // (kept as the reference for the rejected power-of-two variant's history; the
    // tree publishes the exact mantissa now, so this is unused)
    function [7:0] pow2_ceil8(input [7:0] m);
        reg [7:0] p2;
        begin
            if (m == 8'd0) pow2_ceil8 = 8'd0;
            else begin
                p2 = 8'd1;
                while (p2 < m && p2 != 8'd128) p2 = p2 << 1;
                pow2_ceil8 = p2;
            end
        end
    endfunction
    // m * 2^(eb-127) -> BF16 (identical to kv8_scale_ram's expander)
    function [15:0] to_bf16(input [7:0] m, input [7:0] eb);
        reg [2:0] k;
        reg [8:0] sh;
        reg [7:0] fe;
        begin
            if (m == 8'd0) to_bf16 = 16'h0000;
            else begin
                k = msb_i(m);
                sh = {1'b0,m} << (3'd7-k);
                fe = eb + {5'd0,k};
                if (fe == 8'd0) to_bf16 = 16'h0000;
                else            to_bf16 = {1'b0,fe,sh[6:0]};
            end
        end
    endfunction

    // ---- reference model of the V chain ------------------------------------
    reg [7:0] lock_eb [0:1];
    reg       lock_v  [0:1];
    task automatic ref_head(input integer t, input integer hd);
        reg [14:0] maxmag, m15;
        reg [15:0] num, quo, rem;
        reg [22:0] lock_num, lquo, lrem;
        reg [8:0]  m9;
        reg [7:0]  m_eff, eb_eff, m_calc;
        reg [16:0] recip;
        integer d, lane, base, shift;
        reg [15:0] kk;
        reg [7:0]  mm, q;
        reg [23:0] nn;
        reg [40:0] pr;
        reg [24:0] qv;
        begin
            base = t*2*64 + hd*64;
            maxmag = 15'd0;
            for (lane=0; lane<64; lane=lane+1) begin
                m15 = vref[base+lane][14:0];
                if (m15 > maxmag) maxmag = m15;
            end
            if (!lock_v[hd]) begin
                // first token of the layer: only its EXPONENT becomes the lock
                lock_v[hd] = 1'b1;
                eb_eff = (maxmag[14:7] > 8'd3) ? (maxmag[14:7] - 8'd2) : 8'd1;
                lock_eb[hd] = eb_eff;
                num = {1'b1,maxmag[6:0]} << 2;
                quo = num / 16'd127;
                rem = num - quo*16'd127;
                m_calc = quo[7:0] + ((rem != 0) ? 8'd1 : 8'd0);
                m_eff  = m_calc;
                if ((m_calc == 8'd0) || (m_calc == 8'd255)) clip_any = 1'b1;
                // [V-PERTOK-M] the image holds ONLY the power of two (m = 1)
                scale_exp[hd] = to_bf16(8'd1, (eb_eff > 8'd7) ? (eb_eff - 8'd7) : 8'd0);
            end else begin
                // later token: re-derive ITS OWN mantissa against the lock
                eb_eff = lock_eb[hd];
                shift = $signed({1'b0,maxmag[14:7]}) - $signed({1'b0,eb_eff});
                if      (shift >  15) lock_num = 23'h7F_FFFF;
                else if (shift < -15) lock_num = 23'd0;
                else if (shift < 0)   lock_num = {15'd0,1'b1,maxmag[6:0]} >> (-shift);
                else                  lock_num = {15'd0,1'b1,maxmag[6:0]} << shift;
                lquo = lock_num / 23'd127;
                lrem = lock_num - lquo*23'd127;
                if ((lquo > 23'd255) || ((lquo == 23'd255) && (lrem != 23'd0))) m9 = 9'd255;
                else m9 = {1'b0,lquo[7:0]} + ((lrem != 0) ? 9'd1 : 9'd0);
                m_eff = m9[7:0];
                // [V-PERTOK-M] the real clip point: the quantiser's mantissa field
                // saturated at MTOP (or underflowed to 0)
                if ((m9 == 9'd0) || (m9 == 9'd255)) clip_any = 1'b1;
            end
            mexp[t*2+hd] = m_eff;
            if ({1'b0,m_eff} > mmax_exp) mmax_exp = m_eff;
            // quantise with (this token's eb, this token's m)
            recip = recip_lut(m_eff);
            for (lane=0; lane<64; lane=lane+1) begin
                kk = vref[base+lane];
                mm = {1'b1, kk[6:0]};
                d  = $signed({1'b0,kk[14:7]}) - $signed({1'b0,eb_eff});
                if (kk[14:7] == 8'd0) nn = 24'd0;
                else if (d >= 0)      nn = ({{16{1'b0}},mm} << d) & 24'hFFFFFF;
                else                  nn = ({{16{1'b0}},mm} >> (-d)) & 24'hFFFFFF;
                pr = nn * {24'd0,recip} + 41'd32768;
                qv = pr >> 16;
                if (qv > 25'd127) begin
                    codes_exp = codes_exp + 1;
                    q = 8'd127;
                end else q = qv[7:0];
                qexp[base+lane] = kk[15] ? (~q + 8'd1) : q;
            end
        end
    endtask

    // ---- DDR capture -------------------------------------------------------
    reg [7:0] ddr [longint];
    integer   beats, addrs_bad, data_bad, checks;
    reg [63:0] aexp;
    integer    t_cap, h_cap, dg_cap, tokseen [0:NTOK*2*4-1];
    // [V-PERTOK-M] per-token mantissa append capture
    reg [7:0]  mgot [0:NTOK*2-1];
    integer    mseen [0:NTOK*2-1];
    integer    m_idx;
    initial for (integer i=0;i<NTOK*2*4;i=i+1) tokseen[i] = 0;
    initial for (integer i=0;i<NTOK*2;i=i+1) begin mgot[i] = 8'd0; mseen[i] = 0; end

    always @(posedge clk) begin
        if (rst_n && o_beat_valid && i_beat_ready) begin
            beats = beats + 1;
            for (t_cap=0;t_cap<NTOK;t_cap=t_cap+1)
                for (h_cap=0;h_cap<2;h_cap=h_cap+1)
                    for (dg_cap=0;dg_cap<4;dg_cap=dg_cap+1) begin
                        aexp = i_v_base + t_cap*128 + h_cap*64 + dg_cap*16;
                        if (o_beat_address == aexp) tokseen[(t_cap*2+h_cap)*4+dg_cap] = tokseen[(t_cap*2+h_cap)*4+dg_cap] + 1;
                    end
            for (integer b=0;b<16;b=b+1) ddr[o_beat_address + b] = o_beat_data[b*8 +: 8];
        end
        if (rst_n && o_m_app_valid && i_m_app_ready) begin
            m_idx = o_m_app_token[3:0]*2 + o_m_app_head;
            if (o_m_app_token < NTOK && o_m_app_mant[15:8] == 8'd0) begin
                mgot[m_idx]  = o_m_app_mant[7:0];
                mseen[m_idx] = mseen[m_idx] + 1;
            end
        end
    end

    task automatic stream_token(input integer t);
        integer hd, b, lane;
        reg [127:0] beat;
        begin
            i_token = t[15:0];
            @(posedge clk);
            while (!o_ready) @(posedge clk);
            i_start <= 1'b1;
            @(posedge clk);
            i_start <= 1'b0;
            for (hd=0; hd<2; hd=hd+1) begin
                for (b=0; b<8; b=b+1) begin
                    beat = 128'd0;
                    for (lane=0; lane<8; lane=lane+1)
                        beat[lane*16 +: 16] = vref[(t*2+hd)*64 + b*8 + lane];
                    i_v_data  <= beat;
                    i_v_valid <= 1'b1;
                    @(posedge clk);
                    while (!o_v_ready) @(posedge clk);
                end
            end
            i_v_valid <= 1'b0;
            while (!o_done) @(posedge clk);
            @(posedge clk);
        end
    endtask

    integer errors;
    task automatic chk(input integer cond, input [255:0] msg);
        begin
            checks = checks + 1;
            if (!cond) begin
                errors = errors + 1;
                $display("FAIL %0s", msg);
            end
        end
    endtask

    integer t, hd, lane, base, w;
    reg [7:0] e_tmp;
    reg [6:0] f_tmp;
    reg [15:0] scale_got [0:1];
    initial begin
        errors = 0; checks = 0; beats = 0; addrs_bad = 0; data_bad = 0;
        clip_exp = 0; codes_exp = 0; mmax_exp = 0;
        for (integer i=0;i<2;i=i+1) begin lock_v[i]=0; lock_eb[i]=0; scale_exp[i]=0; end
        // ---- stimulus: token 0 = base magnitudes, token 1 = one dim just over
        //      token 0's octave, token 2 = one dim well below it, token 3 head 1
        //      = 2^16 over the lock, which is the only way to saturate m now
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1)
                for (lane=0;lane<64;lane=lane+1) begin
                    e_tmp = 8'd124 + ((lane+t+hd) % 4);
                    f_tmp = 7'd40 + ((lane*7+hd*3+t) % 80);
                    vref[(t*2+hd)*64+lane] = {1'b0, e_tmp, f_tmp};
                end
        vref[0] = {1'b0, 8'd130, 7'd0};                 // token 0 head 0 peak
        vref[128] = {1'b0, 8'd131, 7'd50};              // token 1 head 0: +1 octave
        vref[(2*2+0)*64 + 5] = {1'b0, 8'd120, 7'd0};    // token 2 head 0: -1 octave
        vref[(3*2+1)*64] = {1'b0, 8'd141, 7'd0};        // token 3 head 1: m saturates

        // ---- reference
        for (t=0;t<NTOK;t=t+1) begin
            clip_any = 1'b0;
            ref_head(t,0);
            ref_head(t,1);
            if (clip_any) clip_exp = clip_exp + 1;
        end

        // ---- reset
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ---- PHASE1/2: stream the tokens
        for (t=0;t<NTOK;t=t+1) stream_token(t);

        // ---- PHASE1 byte/address checks
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1)
                for (lane=0;lane<64;lane=lane+1) begin
                    base = (t*2+hd)*64 + lane;
                    chk(ddr[i_v_base + t*128 + hd*64 + lane] === qexp[base],
                        "V byte mismatch");
                end
        chk(addrs_bad == 0, "beat address outside the token-major V layout");
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1)
                for (w=0;w<4;w=w+1)
                    chk(tokseen[(t*2+hd)*4+w] == 1, "each (token,head,dg) beat written exactly once");

        $display("  lane chk: t0 hd1 lanes 0..3 ddr=%h %h %h %h exp=%h %h %h %h", ddr[i_v_base+64], ddr[i_v_base+65], ddr[i_v_base+66], ddr[i_v_base+67], qexp[64], qexp[65], qexp[66], qexp[67]);
        // ---- PHASE2: the 256 B scale image carries the layer's power of two
        scale_got[0] = o_scale_bf16[15:0];
        scale_got[1] = o_scale_bf16[31:16];
        chk(o_scale_valid === 1'b1, "scale valid after the first token");
        chk(scale_got[0] === scale_exp[0], "V scale head 0 = 2^(eb_lock-134)");
        chk(scale_got[1] === scale_exp[1], "V scale head 1 = 2^(eb_lock-134)");
        for (w=0;w<16;w=w+1)
            chk(((w[3] ? scale_got[1] : scale_got[0])) === (w[3] ? scale_exp[1] : scale_exp[0]),
                "scale image word");

        // ---- PHASE3a: the per-token mantissa appends
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1) begin
                $display("  m t%0d h%0d dut=%0d exp=%0d seen=%0d", t, hd,
                         mgot[t*2+hd], mexp[t*2+hd], mseen[t*2+hd]);
                chk(mseen[t*2+hd] == 1, "exactly one mantissa append per (token,head)");
                chk(mgot[t*2+hd] === mexp[t*2+hd], "per-token mantissa matches the reference");
            end
        chk(o_m_max === mmax_exp[7:0], "o_m_max = largest published mantissa");

        // ---- PHASE3b: clip accounting
        chk(o_clip_count == clip_exp[15:0], "clip-token count matches the reference");
        chk(o_clip_codes == codes_exp[31:0], "saturated-code count matches the reference");
        chk((o_clip_sticky === 1'b1) == (clip_exp != 0), "clip sticky polarity");
        chk(clip_exp > 0, "stimulus really saturates one token's mantissa");
        chk(codes_exp > 0, "stimulus really saturates codes");
        chk(ddr[i_v_base + 3*128 + 1*64 + 0] === 8'h7f ||
            ddr[i_v_base + 3*128 + 1*64 + 0] === 8'h81, "saturated dim clamps at +-127");
        // and the tokens that only GREW must NOT clip any more (the old rule did)
        chk(mexp[1*2+0] != 8'd255 && mexp[1*2+0] > 8'd9,
            "a grown token gets its own larger mantissa instead of clipping");

        $display("V-CHAIN beats=%0d clip_tok=%0d clip_exp=%0d codes=%0d codes_exp=%0d m=%0d/%0d/%0d/%0d mmax=%0d s0=%h s1=%h",
                 beats, o_clip_count, clip_exp, o_clip_codes, codes_exp,
                 mexp[0], mexp[1], mexp[6], mexp[7], o_m_max, scale_got[0], scale_got[1]);
        if (errors == 0) $display("V-CHAIN PASS: checks=%0d", checks);
        else             $display("V-CHAIN FAIL: checks=%0d errors=%0d", checks, errors);
        $finish;
    end

    // global timeout
    initial begin
        #4000000;
        $display("V-CHAIN TIMEOUT");
        $finish;
    end
endmodule
`default_nettype wire
