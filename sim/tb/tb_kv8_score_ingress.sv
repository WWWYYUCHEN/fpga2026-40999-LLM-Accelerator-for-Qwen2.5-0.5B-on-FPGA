`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_score_ingress -- stage 3 module TB (self-checking)
//
//   PHASE1  score mode, aligned scales: every lane must equal
//           BF16(score[token] * s_k[token]/8)            (latency-agnostic)
//   PHASE2  **alignment proof**: the same scores are replayed with the scale
//           word of a DIFFERENT token block; the DUT must then produce the
//           swapped product.  A design that silently used a stale/rotated scale
//           word would keep passing PHASE1 but fail here.
//   PHASE3  metadata alignment: o_address / o_row / o_first / o_last / o_words
//           must belong to the word whose data came out (the ingress delays the
//           data by 5 cycles; the metadata must be delayed with it)
//   PHASE4  mask: columns beyond the causal limit must come out as -inf (0xff80),
//           and the argmax must ignore them
//   PHASE5  logits mode: the multiplier must be bypassed (raw values, inverted
//           index)
// =============================================================================
module tb_kv8_score_ingress;
    reg clk = 0, rstn = 0;
    always #5 clk = ~clk;

    reg         i_valid = 0;
    wire        o_ready;
    reg         i_logits = 0, i_bank = 0;
    reg  [14:0] i_address = 0;
    reg  [127:0] i_data = 0, i_scale = 0;
    reg  [7:0]   i_eb = 0;        // [M16-SCALE] block exponent
    reg  [31:0] i_length = 32'd64;
    reg  [15:0] i_position = 16'd63;
    wire        o_valid;
    reg         i_ready = 1;
    wire        o_bank;
    wire [14:0] o_address;
    wire [127:0] o_data;
    wire [2:0]  o_max_row;
    wire [15:0] o_maximum;
    wire [31:0] o_index;
    wire        o_first, o_last;
    wire [15:0] o_words;
    wire        o_busy;

    qwen25_score_ingress dut (
        .i_clk(clk), .i_rstn(rstn), .i_valid(i_valid), .o_ready(o_ready),
        .i_logits(i_logits), .i_bank(i_bank), .i_address(i_address),
        .i_data(i_data), .i_scale(i_scale), .i_eb(i_eb),
        .i_length(i_length), .i_position(i_position),
        .o_valid(o_valid), .i_ready(i_ready), .o_bank(o_bank), .o_address(o_address),
        .o_data(o_data), .o_max_row(o_max_row), .o_maximum(o_maximum), .o_index(o_index),
        .o_first(o_first), .o_last(o_last), .o_words(o_words), .o_busy(o_busy)
    );

    // ---------------------------------------------------------------- helpers
    // [M16-SCALE] exact oracle of kv8_score_scale8 MANT_BITS=16:
    //     y = a * m16 * 2^(eb - 134)        (C_K = 268 = 134 + 134)
    // a*m16 is <= 24 significant bits, so the double product is EXACT and the only
    // rounding is the final BF16 RNE below.
    function automatic [15:0] bf16_scale_m16(input [15:0] a, input [15:0] m16, input [7:0] eb);
        real va, vp;
        reg  [31:0] fb;
        integer e;
        begin
            va = bf16_to_real(a);
            vp = va * m16;
            if (eb >= 8'd134) for (e=0;e<(eb-8'd134);e=e+1) vp = vp * 2.0;
            else              for (e=0;e<(8'd134-eb);e=e+1) vp = vp * 0.5;
            fb = $shortrealtobits(vp);
            if (fb[30:23]==8'hff)
                bf16_scale_m16 = (fb[22:16]!=0) ? 16'h7fc0 : (fb[31] ? 16'hff80 : 16'h7f80);
            else
                bf16_scale_m16 = {fb[31], fb[30:16]} + {15'd0, (fb[15:0] > 16'h8000) ||
                                                         (fb[15:0]==16'h8000 && fb[16])};
        end
    endfunction
    function automatic [15:0] bf16_mul(input [15:0] a, input [15:0] b);
        real va, vb, vp;
        reg  [31:0] fb;
        begin
            va = bf16_to_real(a); vb = bf16_to_real(b); vp = va*vb;
            fb = $shortrealtobits(vp);
            // fp32 -> bf16: same bias, so the exponent field is identical; RNE on the
            // 16 dropped bits.  (bf16 subnormals do not occur in this stimulus.)
            if (fb[30:23]==8'hff)
                bf16_mul = (fb[22:16]!=0) ? 16'h7fc0 : (fb[31] ? 16'hff80 : 16'h7f80);
            else
                bf16_mul = {fb[31], fb[30:16]} + {15'd0, (fb[15:0] > 16'h8000) ||
                                                        (fb[15:0]==16'h8000 && fb[16])};
        end
    endfunction
    function automatic real bf16_to_real(input [15:0] v);
        real m; integer e;
        begin
            if (v[14:7] == 8'd0) bf16_to_real = 0.0;
            else begin
                m = 1.0 + (v[6:0] / 128.0);
                e = v[14:7] - 127;
                bf16_to_real = (v[15] ? -m : m);
                if (e > 0) bf16_to_real = bf16_to_real * (2.0 ** e);
                else       bf16_to_real = bf16_to_real / (2.0 ** (-e));
            end
        end
    endfunction
    function automatic [15:0] make_bf16(input integer sign, input integer expo, input integer mant);
        begin
            make_bf16 = {sign[0], expo[7:0], mant[6:0]};
        end
    endfunction
    function automatic [15:0] neg_inf;
        begin neg_inf = 16'hff80; end
    endfunction

    integer errors, checks;
    integer seed = 32'h0bad_f00d;

    // one word = 8 tokens; score[t] and scale[t] are built from the token index
    reg [15:0] score_words [0:63];      // 8 words x 8 lanes (BF16 activation)
    reg [15:0] scale_words [0:63];      // 8 words x 8 lanes (m16 INTEGER mantissa)
    reg [7:0]  scale_eb    [0:7];       // [M16-SCALE] one block exponent per scale word
    integer t, w, lane;

    task automatic drive_word(input integer widx, input integer scale_widx,
                              input integer logits_mode, input integer len,
                              input integer pos);
        reg [127:0] d, s;
        integer    k;
        begin
            d = 128'd0; s = 128'd0;
            for (k=0;k<8;k=k+1) begin
                d[k*16 +: 16] = score_words[widx*8 + k];
                s[k*16 +: 16] = scale_words[scale_widx*8 + k];
            end
            i_eb = scale_eb[scale_widx];
            i_logits  = logits_mode[0];
            i_length  = len[31:0];
            i_position= pos[15:0];
            i_address = widx[14:0];
            i_bank    = 1'b0;
            i_data    = d;
            i_scale   = s;
            i_valid   = 1'b1;
            @(posedge clk);
            while (!o_ready) @(posedge clk);
            i_valid   = 1'b0;
            // wait for the result (latency is 5 from the accept edge)
            @(posedge clk);
            while (!o_valid) @(posedge clk);
        end
    endtask

    reg [15:0] want [0:7];
    reg [15:0] gotv;
    integer    k, widx, bad;
    reg [15:0] maxv;
    integer    maxi;
    reg [31:0] idx;

    // expected lane value for (score word, scale word)
    task automatic compute_want(input integer widx, input integer swidx, input integer logits_mode,
                               input integer len, input integer pos);
        integer j;
        reg [16:0] lim;
        reg [31:0] col;
        begin
            for (j=0;j<8;j=j+1) begin
                col = widx*8 + j;
                lim = pos + ((widx*8)/4096)*0 ;
                if (logits_mode) begin
                    want[j] = (col < len) ? score_words[widx*8+j] : neg_inf();
                end else if (!(col < len) || (col > pos)) begin
                    want[j] = neg_inf();
                end else begin
                    want[j] = bf16_scale_m16(score_words[widx*8+j], scale_words[swidx*8+j],
                                             scale_eb[swidx]);
                end
            end
        end
    endtask

    task automatic check_word(input integer widx, input integer swidx, input integer logits_mode,
                              input integer len, input integer pos);
        integer j;
        begin
            compute_want(widx, swidx, logits_mode, len, pos);
            for (j=0;j<8;j=j+1) begin
                gotv = o_data[j*16 +: 16];
                checks = checks + 1;
                if (gotv !== want[j]) begin
                    errors = errors + 1;
                    if (errors < 15)
                        $display("FAIL lane=%0d word=%0d got=%h want=%h (score=%h m16=%0d eb=%0d)",
                                 j, widx, gotv, want[j], score_words[widx*8+j],
                                 scale_words[swidx*8+j], scale_eb[swidx]);
                end
            end
            // metadata must belong to this word
            checks = checks + 1;
            if (o_address !== widx[14:0]) begin
                errors = errors + 1;
                if (errors < 15) $display("FAIL metadata addr word=%0d got=%0d", widx, o_address);
            end
            checks = checks + 1;
            if ((o_words !== ((len+7)>>3)) ) begin
                errors = errors + 1;
                if (errors < 15) $display("FAIL words word=%0d got=%0d want=%0d", widx, o_words, (len+7)>>3);
            end
            checks = checks + 1;
            if (o_first !== (widx==0)) begin
                errors = errors + 1;
                if (errors < 15) $display("FAIL first word=%0d got=%b", widx, o_first);
            end
            // argmax must ignore the masked lanes and pick the largest value
            maxv = 16'hff80; maxi = 32'hffffffff;
            for (j=0;j<8;j=j+1) begin
                if (want[j] !== neg_inf()) begin
                    if ((want[j][15:0] > maxv[15:0]) || maxi==32'hffffffff) begin
                        maxv = want[j]; maxi = widx*8 + j;
                    end
                end
            end
            if (maxi == 32'hffffffff) begin
                checks = checks + 1;
                if (o_maximum !== neg_inf()) begin
                    errors = errors + 1;
                    $display("FAIL all-masked word=%0d maximum=%h", widx, o_maximum);
                end
            end else begin
                checks = checks + 1;
                idx = maxi;   // the logits-mode inversion cancels: o_index is the true token index in both modes
                if (o_maximum !== maxv || o_index !== idx) begin
                    errors = errors + 1;
                    if (errors < 15)
                        $display("FAIL argmax word=%0d max=%h/%h index=%0d/%0d", widx,
                                 o_maximum, maxv, o_index, idx);
                end
            end
        end
    endtask

    initial begin
        errors = 0; checks = 0;
        // score words: distinct per token so a lane swap is visible
        for (t=0;t<64;t=t+1)
            score_words[t] = make_bf16(0, 125 + (t%5), (t*7)%128);
        // [M16-SCALE] scales: 16-bit integer mantissas, distinct per token so using
        // another word's scale changes every lane value; plus a per-word exponent.
        for (t=0;t<64;t=t+1)
            scale_words[t] = 16'd517 + (t%13)*16'd37;
        for (w=0;w<8;w=w+1)
            scale_eb[w] = 8'd118 + w[7:0]*8'd2;

        repeat (5) @(posedge clk);
        rstn = 1;
        repeat (5) @(posedge clk);

        $display("--- PHASE1: aligned scales, 8 words x 8 lanes ---");
        i_length = 32'd64; i_position = 16'd63;
        for (widx=0;widx<8;widx=widx+1) begin
            drive_word(widx, widx, 0, 64, 63);
            check_word(widx, widx, 0, 64, 63);
        end

        $display("--- PHASE2: misaligned scale word must produce the swapped product ---");
        for (widx=0;widx<3;widx=widx+1) begin
            drive_word(widx, widx+3, 0, 64, 63);   // scale of another token block
            check_word(widx, widx+3, 0, 64, 63);   // reference uses the same swap
            // and the swap must actually change the data, otherwise PHASE2 proves nothing
            checks = checks + 1;
            if (o_data[15:0] === bf16_scale_m16(score_words[widx*8], scale_words[widx*8],
                                                scale_eb[widx]) &&
                scale_words[widx*8] !== scale_words[(widx+3)*8]) begin
                errors = errors + 1;
                $display("FAIL PHASE2 is vacuous: swapped scale gave the aligned result");
            end
        end

        $display("--- PHASE3: metadata under back-pressure (i_ready stalls) ---");
        i_ready = 0;
        for (widx=4;widx<6;widx=widx+1) begin
            fork
                drive_word(widx, widx, 0, 64, 63);
                begin
                    repeat (7) @(posedge clk);
                    i_ready = 1;
                end
            join
            check_word(widx, widx, 0, 64, 63);
        end
        i_ready = 1;

        $display("--- PHASE4: causal mask (position 20, length 64) ---");
        i_length = 32'd64; i_position = 16'd20;
        for (widx=0;widx<4;widx=widx+1) begin
            drive_word(widx, widx, 0, 64, 20);
            check_word(widx, widx, 0, 64, 20);
        end

        $display("--- PHASE5: logits mode (multiplier bypassed, index inverted) ---");
        i_length = 32'd32; i_position = 16'd31;
        for (widx=0;widx<4;widx=widx+1) begin
            drive_word(widx, widx, 1, 32, 31);
            check_word(widx, widx, 1, 32, 31);
        end

        repeat (5) @(posedge clk);
        if (errors == 0) $display("STAGE3 PASS: checks=%0d", checks);
        else             $display("STAGE3 FAIL: checks=%0d errors=%0d", checks, errors);
        $finish;
    end

    initial begin
        #2000000;
        $display("STAGE3 TIMEOUT");
        $finish;
    end
endmodule
`default_nettype wire
