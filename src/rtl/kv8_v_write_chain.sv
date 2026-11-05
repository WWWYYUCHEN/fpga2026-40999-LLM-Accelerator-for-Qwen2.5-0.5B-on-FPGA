`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// kv8_v_write_chain -- the INT8 V producer (write side of the KV cache, V half)
//
// Why it is NOT the K chain with another address
//   K is quantised per (head, token): every token may carry its own mantissa, so
//   the score path only has to multiply by s_k/8 of that token.  V cannot work
//   that way: the PV reduction runs over tokens, so a per-token V scale cannot be
//   factored out of the sum.  A per-token V scale therefore has to enter the
//   reduction itself -- it is folded into the probabilities of that token
//   (p'[t] = p[t] * m[t]) and the common power of two 2^(eb_lock-134) stays in
//   kv8_context_scale on the attention write-back.  Everything below is written
//   for that split:
//
//     [V-PERTOK-M] the layer's EXPONENT is locked by the first token, but the
//     MANTISSA is re-derived for every token -- exactly K's rule
//     (kv8_k_write_chain.sv:540-542 is `eff_eb <= eb_lock; eff_m <= q_m;`).
//     The published per-token scale is therefore
//
//         s_v[t] = m[t] * 2^(eb_lock-134),   m[t] = ceil(M8[t] * 2^(d+MARGIN)/127)
//
//     with d = Eb[t]-eb_lock, i.e. s_v[t] tracks that token's own maximum to
//     within the ceil step (<= 1/m), instead of staying at token 0's value.
//     The codes can only clip when m itself saturates at MTOP (255), which needs
//     a token ~2^(d+MARGIN) > 255*127/128, i.e. d >= 6.
//
//   Measured (REPORT_STAGE5_VFIX.md): the exact mantissa costs 64 DSP48E2
//   (`use_dsp`, ~0 LUT) in kv8_v_pscale; the earlier LUT version was 8 320 LUT,
//   and rounding m up to a power of two (a pure exponent add, 1 664 LUT) was
//   rejected because it costs V rel_rms 0.884% -> 2.561% and 43 saturated codes.
//
//   What the OLD rule did (and why it was wrong): it locked (m, eb) together at
//   the layer's first token and re-quantised later tokens against that frozen
//   scale, so any token that grew past 127*s_lock clipped: 17.7% of all V codes
//   on a 30-token prefill, V reconstruction rel_rms 40.98% (REPORT_STAGE5_GAP.md).
//   The "clip" witness fired only when the re-derived m reached 255, i.e. at
//   ~255/m_lock = 42x over-range, so it reported 2 appends while 13 099 codes
//   saturated.  The witness below now counts the real thing.
//
// Layout produced (token-major, the layout qwen25_attention.v already reads):
//     v_base(layer) + token*128 + head*64 + dim_group*16     (8 x 16 B per token)
//   i.e. the same 16 BF16 beats per token the QKV write-back produces
//   (word w = head*8 + dim_group8, 8 BF16 = 8 dims), quantised and packed 2:1 by
//   kv8_beat_pack2 exactly like K.
//
// Scale image published for the layer (256 B, written to DDR by the caller and
// bulk-loaded back into kv8_scale_ram.vmem):
//     scale_base + layer*256 + head*128 + dim*2,  dim = 0..63
//   = 16 x 16 B beats, beat w = {head, word} = 8 copies of s_v[head].
//   [V-PERTOK-M] The image now carries ONLY the layer's power of two
//   (m = 1, i.e. 2^(eb_lock-134)); the per-token mantissa m[t] is published on
//   o_m_app_* into kv8_scale_ram's per-token V mantissa table instead, because a
//   single 256 B image cannot describe 16384 tokens.
// =============================================================================
module kv8_v_write_chain #(
    parameter integer MARGIN = 2,
    // [PER-LAYER-TABLE] layer slices in the scale RAM; the V lock (the layer's
    // power of two) is kept per layer so a decode append reuses it.
    parameter integer LAYERS = 24
) (
    input  wire         clk,
    input  wire         rst_n,
    // ---- per-token job ----------------------------------------------------
    input  wire         i_start,          // pulse: latch the geometry below
    input  wire [4:0]   i_layer,
    input  wire [15:0]  i_token,          // global token index (ascending)
    input  wire [63:0]  i_v_base,         // layer-inclusive V base of this layer
    output wire         o_ready,          // idle
    // ---- BF16 V beat stream from the controller write-back -----------------
    input  wire         i_v_valid,
    output wire         o_v_ready,
    input  wire [127:0] i_v_data,         // 8 x BF16 (word = head*8 + dim_group8)
    output wire         o_busy,
    output wire         o_done,           // pulse: the token's 8 INT8 beats are out
    // ---- INT8 beat stream to DDR (token-major) ----------------------------
    output wire         o_beat_valid,
    input  wire         i_beat_ready,
    output wire [63:0]  o_beat_address,
    output wire [127:0] o_beat_data,
    output wire         o_beat_last,      // last beat of this token (for wr_last)
    // ---- per-layer V scale publish ----------------------------------------
    output wire         o_scale_valid,    // sticky: the layer's s_v is locked
    output wire [31:0]  o_scale_bf16,     // {head1, head0} BF16 2^(eb_lock-134)
    input  wire         i_scale_clear,    // pulse: the caller has taken the image
    // ---- on-chip V table publish (kv8_scale_ram vmem write port) -----------
    // 16 words per layer = 2 heads x 8 dim-words, 8 copies of the layer's
    // 2^(eb_lock-134) each (the per-token mantissa is NOT in this image);
    // written by the producer the cycle the layer's lock is taken, so the bank
    // (LOADING at that time) never has to be reloaded from DDR.
    output wire         o_v_we,
    output wire [3:0]   o_v_word,
    output wire [127:0] o_v_data,
    // ---- per-token V mantissa append [V-PERTOK-M] --------------------------
    // One entry per (token, head) into kv8_scale_ram's V mantissa table; 2 B
    // each, 8 tokens per 128-bit word (the K mantissa geometry).
    output wire         o_m_app_valid,
    input  wire         i_m_app_ready,
    output wire [15:0]  o_m_app_token,
    output wire         o_m_app_head,
    output wire [15:0]  o_m_app_mant,
    // ---- status -----------------------------------------------------------
    output wire         o_clip_sticky,    // sticky: some append saturated m at MTOP
    output wire [15:0]  o_clip_count,     // THIS LAYER's appends with m at MTOP (or 0)
    output wire [31:0]  o_clip_codes,     // THIS LAYER's saturated V codes (|code| > 127)
    output wire [7:0]   o_m_max,          // largest m published this layer (diagnostic)
    output wire [3:0]   o_token_beats,     // 8 beats (128 B) per token, authoritative
    output wire [3:0]   o_dbg_state
);
    localparam integer HEADS         = 2;
    localparam integer BEATS_PER_HEAD= 8;      // 8 x 8 BF16 dims = 64 dims
    localparam integer BEATS_PER_TOKEN = 8;    // 2 heads x 4 INT8 beats

    localparam [3:0] S_IDLE=4'd0, S_P0=4'd1, S_P0_WAIT=4'd2, S_REST=4'd3,
                     S_P1=4'd4, S_APP=4'd5, S_DONE=4'd6;

    reg [3:0]   state;
    reg [4:0]   layer_r;
    reg [15:0]  token_r;
    reg [63:0]  v_base_r;
    reg         head;
    reg [2:0]   beat;
    reg         p1_tog;
    reg [1:0]   p1_dg;
    reg         p1_half;
    reg         p1_head;
    reg         lock_pend;
    // pass-0 result of the head in flight
    reg [7:0]   eb_raw [0:1];
    reg [7:0]   m_int  [0:1];
    // layer lock: the first token's EXPONENT, fixed for the whole layer, kept
    // per (LAYER, head): a decode step must reuse the power of two the layer's
    // first token locked instead of re-deriving one from the appended token.
    reg [7:0]   eb_lock[0:LAYERS-1][0:1];
    reg [7:0]   m_lock [0:LAYERS-1][0:1];  // kept for diagnostics only
    reg         lock_v [0:LAYERS-1][0:1];
    // the pair used by the quantise pass of the head in flight
    reg [7:0]   eff_eb [0:1];
    reg [7:0]   eff_m  [0:1];
    // clip witness of the token in flight, per head
    reg         clip_r [0:1];

    reg [31:0]  clip_count;        // [V-PERTOK-M] per layer now, not cumulative
    reg [31:0]  clip_codes_r;      // [V-PERTOK-M] saturated codes this layer
    reg [7:0]   m_max_r;           // largest m published this layer
    // per-token mantissa append (one per head per token)
    reg         m_app_pend;
    reg [15:0]  m_app_tok;
    reg         m_app_head;
    reg  [7:0]  m_app_mant;
    reg [3:0]   sat_cnt;
    reg         vs_go;
    reg [3:0]   vs_cnt;
    reg         vs_head;
    reg [15:0]  vs_val;
    reg         clip_sticky_r;
    reg         scale_valid_r;
    reg [31:0]  scale_bf16_r;

    // ---------------------------------------------------------------- buffers
    (* ram_style = "distributed" *) reg [127:0] hold [0:HEADS*BEATS_PER_HEAD-1];

    // ---------------------------------------------------------------- quantiser
    reg         q_clear, q_pass, q_valid, q_last;
    reg [127:0] q_data;
    reg         q_lock_valid;
    reg [7:0]   q_lock;
    reg         q_rest_valid;
    reg [7:0]   q_rest_eb;
    reg [7:0]   q_rest_m;
    // [RECIP-SEQ] one-shot for the restore request.  The quantiser's restore path
    // is now multi-cycle (see kv8_v_quant.sv [RECIP-SEQ]), so the old
    // "re-assert while !done" form would fire a second request every other cycle
    // for the whole 17..41-cycle latency.  Issue exactly once, retire on done.
    reg         rest_issued;
    wire        q_done, q_qv, q_lock_done, q_rest_done;
    wire [7:0]  q_eb, q_m;
    wire [63:0] q_q;
    wire [7:0]  q_sat;

    // [M16-SPLIT] V keeps the shipped 8-bit quantiser (see kv8_v_quant.sv).
    kv8_v_quant #(.MARGIN(MARGIN)) v_quant_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_clear(q_clear), .i_pass(q_pass), .i_valid(q_valid), .i_last(q_last),
        .i_k(q_data),
        .i_eb_lock_valid(q_lock_valid), .i_eb_lock(q_lock),
        .o_lock_done(q_lock_done),
        .i_restore_valid(q_rest_valid), .i_restore_eb(q_rest_eb), .i_restore_m(q_rest_m),
        .o_restore_done(q_rest_done),
        .o_done(q_done), .o_eb(q_eb), .o_m(q_m), .o_maxmag_exp(),
        .o_q_valid(q_qv), .o_q(q_q), .o_sat_lane(q_sat)
    );
    // [V-PERTOK-M] popcount of this beat's saturated lanes (4 bits is enough)
    always @(*) begin
        sat_cnt = 4'd0;
        sat_cnt = {3'd0,q_sat[0]} + {3'd0,q_sat[1]} + {3'd0,q_sat[2]} + {3'd0,q_sat[3]}
                + {3'd0,q_sat[4]} + {3'd0,q_sat[5]} + {3'd0,q_sat[6]} + {3'd0,q_sat[7]};
    end

    // ------------------------------------------------------------- beat pack 2:1
    // [STAGE5-FIX4] 1-deep skid between the quantiser and the packer.
    // kv8_beat_pack2 takes a 64-bit half only while it does not already hold an
    // output beat, and kv8_k_quant's o_q_valid lands one cycle after the pass-1
    // push, so "packer ready now" is not "packer ready when the half arrives".
    // Driving i_valid straight from q_qv with o_ready unconnected dropped the low
    // half of a pair under DMA back-pressure; its partner then arrived with
    // half_pend=0 and was dropped too, so a token never delivered its 8 beats and
    // the 128 B write command wedged the engine (exactly the K chain's old bug).
    wire         pk_valid, pk_err;
    wire [16:0]  pk_index;
    wire [127:0] pk_data;
    wire         pk_ready;
    wire         pk_in_ready;   // packer can take another 64-bit half
    reg          sk_valid;      // a quantised half the packer could not take yet
    reg  [63:0]  sk_q;
    reg  [13:0]  sk_token;
    reg          sk_head;
    reg  [1:0]   sk_dg;
    reg          sk_half;
    wire         pk_in_valid = sk_valid | q_qv;
    wire [63:0]  pk_in_q     = sk_valid ? sk_q        : q_q;
    wire [13:0]  pk_in_token = sk_valid ? sk_token    : token_r[13:0];
    wire         pk_in_head  = sk_valid ? sk_head     : p1_head;
    wire [1:0]   pk_in_dg    = sk_valid ? sk_dg       : p1_dg;
    wire         pk_in_half  = sk_valid ? sk_half     : p1_half;
    kv8_beat_pack2_v pack2_Inst (
        .clk(clk), .rst_n(rst_n), .i_flush(i_start && state == S_IDLE),
        .i_valid(pk_in_valid), .o_ready(pk_in_ready),
        .i_index({pk_in_token, pk_in_head, pk_in_dg}),
        .i_half(pk_in_half),
        .i_q(pk_in_q),
        .o_valid(pk_valid), .i_ready(pk_ready),
        .o_index(pk_index), .o_data(pk_data),
        .o_err_pair(pk_err)
    );

    // [STAGE5-FIX4] skid sequencing + a COUNT for the per-head hand-off: a head
    // may not retire while any half it pushed is still in flight, so "4 beats"
    // can only be believed once 4 beats were actually accepted.
    reg [5:0] push_cnt, acc_cnt;
    wire      q_push     = q_valid && q_pass;   // a pass-1 push (pass 0 has q_pass=0)
    wire      half_taken = pk_in_valid && pk_in_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            sk_valid <= 1'b0; sk_q <= 64'd0; sk_token <= 14'd0;
            sk_head <= 1'b0; sk_dg <= 2'd0; sk_half <= 1'b0;
            push_cnt <= 6'd0; acc_cnt <= 6'd0;
        end else if (i_start && (state == S_IDLE)) begin
            sk_valid <= 1'b0;
            push_cnt <= 6'd0; acc_cnt <= 6'd0;
        end else begin
            if (q_push) push_cnt <= push_cnt + 6'd1;
            if (half_taken) acc_cnt <= acc_cnt + 6'd1;
            if (sk_valid) begin
                if (pk_in_ready) sk_valid <= 1'b0;
            end else if (q_qv && !pk_in_ready) begin
                sk_valid <= 1'b1;
                sk_q     <= q_q;
                sk_token <= token_r[13:0];
                sk_head  <= p1_head;
                sk_dg    <= p1_dg;
                sk_half  <= p1_half;
            end
        end
    end

    // ------------------------------------------------------------- beat output
    // The packer holds the beat until the controller accepts it, so the address is
    // a pure function of the beat index it latched.
    wire [2:0] beat_in_token = {pk_index[2], pk_index[1:0]};
    assign pk_ready        = i_beat_ready;
    assign o_beat_valid    = pk_valid;
    assign o_beat_data     = pk_data;
    assign o_beat_address  = v_base_r + {43'd0, token_r[13:0], 7'd0} +
                             {57'd0, beat_in_token, 4'b0000};
    assign o_beat_last     = pk_valid && (beat_in_token == 3'd7);

    reg [2:0] beat_cnt;
    assign o_ready   = (state == S_IDLE);
    assign o_v_ready = (state == S_P0);
    assign o_busy    = (state != S_IDLE);
    assign o_done    = (state == S_DONE);
    assign o_dbg_state = state;
    assign o_scale_valid = scale_valid_r;
    assign o_v_we   = vs_go;
    assign o_v_word = {vs_head, vs_cnt[2:0]};
    assign o_v_data = {8{vs_val}};
    assign o_scale_bf16  = scale_bf16_r;
    assign o_clip_sticky = clip_sticky_r;
    assign o_clip_count  = clip_count[15:0];
    assign o_clip_codes  = clip_codes_r;
    assign o_m_max       = m_max_r;
    assign o_token_beats = 4'd8;
    // [V-PERTOK-M] per-token mantissa append (2 B entry, {8'h00, m})
    assign o_m_app_valid = m_app_pend;
    assign o_m_app_token = m_app_tok;
    assign o_m_app_head  = m_app_head;
    assign o_m_app_mant  = {8'd0, m_app_mant};

    // kv8_k_quant emits the table convention (eb_internal - 10); the quantiser and
    // the published BF16 both need the internal exponent.
    wire [7:0] q_eb_i = (q_eb > 8'd245) ? 8'd255 : (q_eb + 8'd10);
    // [STAGE5-FIX5] published table value = s_true = m*2^(eb_r-134); q_eb_i is
    // eb_r, and 134-127 = 7, so the published exponent is q_eb_i - 7.  q_eb_i
    // itself must stay eb_r: it is also the quantiser restore value.
    wire [7:0] q_eb_pub = (q_eb_i > 8'd7) ? (q_eb_i - 8'd7) : 8'd0;

    // m * 2^(eb-127) -> BF16, identical to kv8_scale_ram's expander (that function
    // is what the read path will invert, so the two must agree bit for bit).
    function [15:0] to_bf16(input [7:0] m, input [7:0] eb);
        reg [2:0] k;
        reg [8:0] sh;
        reg [7:0] fe;
        begin
            if (m == 8'd0) to_bf16 = 16'h0000;
            else begin
                if      (m[7]) k = 3'd7;
                else if (m[6]) k = 3'd6;
                else if (m[5]) k = 3'd5;
                else if (m[4]) k = 3'd4;
                else if (m[3]) k = 3'd3;
                else if (m[2]) k = 3'd2;
                else if (m[1]) k = 3'd1;
                else           k = 3'd0;
                sh = {1'b0,m} << (3'd7-k);
                fe = eb + {5'd0,k};
                if (fe == 8'd0) to_bf16 = 16'h0000;
                else            to_bf16 = {1'b0,fe,sh[6:0]};
            end
        end
    endfunction

    // [V-PERTOK-M] the token's published mantissa / its clip flag, from the
    // quantiser's exact m.  255 means the quantiser's own mantissa field was
    // saturated (or needed more), which is the real clip point; 0 is a dead scale.
    wire [7:0] m_pub    = q_m;
    wire       m_clip   = (q_m == 8'd0) || (q_m == 8'd255);

    integer h;
    integer l;
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; layer_r <= 5'd0; token_r <= 16'd0; v_base_r <= 64'd0;
            head <= 1'b0; beat <= 3'd0;
            q_clear <= 1'b0; q_pass <= 1'b0; q_valid <= 1'b0; q_last <= 1'b0; q_data <= 128'd0;
            q_lock_valid <= 1'b0; q_lock <= 8'd0;
            p1_tog <= 1'b0; p1_dg <= 2'd0; p1_head <= 1'b0; p1_half <= 1'b0;
            lock_pend <= 1'b0; q_rest_valid <= 1'b0; rest_issued <= 1'b0;
            beat_cnt <= 3'd0;
            clip_count <= 32'd0; clip_sticky_r <= 1'b0;
            clip_codes_r <= 32'd0; m_max_r <= 8'd0;
            m_app_pend <= 1'b0; m_app_tok <= 16'd0; m_app_head <= 1'b0; m_app_mant <= 8'd0;
            vs_go <= 1'b0; vs_cnt <= 4'd0; vs_head <= 1'b0; vs_val <= 16'd0;
            scale_valid_r <= 1'b0; scale_bf16_r <= 32'd0;
            for (h=0;h<HEADS;h=h+1) begin
                eb_raw[h] <= 8'd0; m_int[h] <= 8'd0;
                for (l=0;l<LAYERS;l=l+1) begin eb_lock[l][h] <= 8'd0; m_lock[l][h] <= 8'd0; lock_v[l][h] <= 1'b0; end
                eff_eb[h] <= 8'd0; eff_m[h] <= 8'd0; clip_r[h] <= 1'b0;
            end
        end else begin
            q_clear <= 1'b0;
            q_lock_valid <= 1'b0;
            if (i_scale_clear) scale_valid_r <= 1'b0;
            // [V-PERTOK-M] the append retires as soon as the table takes it
            if (m_app_pend && i_m_app_ready) m_app_pend <= 1'b0;
            // [V-PERTOK-M] saturated-code witness: one count per pass-1 beat
            if (q_push && sat_cnt != 4'd0) begin
                clip_codes_r <= clip_codes_r + {28'd0, sat_cnt};
                clip_sticky_r <= 1'b1;
            end
            if (vs_go) begin
                if (vs_cnt[2:0] == 3'd7) vs_go <= 1'b0;
                vs_cnt <= vs_cnt + 4'd1;
            end
            // ---- packer output -> DDR beat stream (count them for o_done) ----
            if (pk_valid && i_beat_ready) beat_cnt <= beat_cnt + 3'd1;

            case (state)
            S_IDLE: begin
                q_valid <= 1'b0; q_pass <= 1'b0; beat_cnt <= 3'd0;
                if (i_start) begin
                    layer_r <= i_layer; token_r <= i_token; v_base_r <= i_v_base;
                    head <= 1'b0; beat <= 3'd0; beat_cnt <= 3'd0;
                    q_clear <= 1'b1;
                    // a request's first token (token 0) restarts the layer's lock;
                    // a later token of the same layer (the decode append) keeps it
                    if (i_token == 16'd0)
                        for (h=0;h<HEADS;h=h+1)
                            lock_v[i_layer][h] <= 1'b0;
                    // [V-PERTOK-M] a new layer starts the per-layer witnesses over,
                    // so KV8_CLIP prints THIS layer's events instead of a running
                    // total that has to be differenced by hand.
                    if (i_layer != layer_r) begin
                        clip_count   <= 32'd0;
                        clip_codes_r <= 32'd0;
                        m_max_r      <= 8'd0;
                    end
                    state <= S_P0;
                end
            end
            // ------------------------------------------- pass 0: magnitude, 8 beats
            S_P0: begin
                q_pass <= 1'b0;
                if (i_v_valid && o_v_ready) begin
                    hold[{head,beat}] <= i_v_data;
                    q_valid <= 1'b1;
                    q_data  <= i_v_data;
                    q_last  <= (beat == BEATS_PER_HEAD-1);
                    if (beat == BEATS_PER_HEAD-1) state <= S_P0_WAIT;
                    else                          beat  <= beat + 3'd1;
                end else begin
                    q_valid <= 1'b0;
                end
            end
            S_P0_WAIT: begin
                q_valid <= 1'b0; q_last <= 1'b0;
                if (q_done) begin
                    eb_raw[head] <= q_eb_i;
                    m_int[head]  <= q_m;
                    beat <= 3'd0;
                    if (!lock_v[layer_r][head]) begin
                        // ---- first token of the layer: its EXPONENT is the lock,
                        //      its mantissa is simply this token's m
                        lock_v[layer_r][head] <= 1'b1;
                        eb_lock[layer_r][head] <= q_eb_i;
                        m_lock[layer_r][head]  <= q_m;         // diagnostic: token 0's m
                        eff_eb[head]  <= q_eb_i;
                        eff_m[head]   <= m_pub;
                        clip_r[head]  <= m_clip;
                        if (head == 1'b0) scale_bf16_r[15:0]  <= to_bf16(8'd1, q_eb_pub);
                        else              scale_bf16_r[31:16] <= to_bf16(8'd1, q_eb_pub);
                        // publish this head's 8 V-table words: the layer's power of
                        // two only (m = 1); the per-token m goes on o_m_app_*
                        vs_go   <= 1'b1;
                        vs_cnt  <= 4'd0;
                        vs_head <= head;
                        vs_val  <= to_bf16(8'd1, q_eb_pub);
                        if (head == 1'b1) scale_valid_r <= 1'b1;
                        // [V-PERTOK-M] publish this (token, head) mantissa
                        m_app_pend <= 1'b1; m_app_tok <= token_r;
                        m_app_head <= head; m_app_mant <= m_pub;
                        if (m_pub > m_max_r) m_max_r <= m_pub;
                        if (head == 1'b0) begin head <= 1'b1; state <= S_P0; q_clear <= 1'b1; end
                        else              begin head <= 1'b0; beat <= 3'd0; state <= S_REST; end
                    end else begin
                        // ---- locked exponent: re-derive THIS token's mantissa
                        //      against eb_lock (K's rule, kv8_k_write_chain:540-542)
                        q_lock       <= eb_lock[layer_r][head];
                        q_lock_valid <= 1'b1;
                        lock_pend    <= 1'b1;
                    end
                end else if (lock_pend && q_lock_done) begin
                    lock_pend    <= 1'b0;
                    eff_eb[head] <= eb_lock[layer_r][head];   // the layer's exponent is fixed
                    eff_m[head]  <= m_pub;              // the mantissa is this token's own
                    // [V-PERTOK-M] the real clip point: the token needs a mantissa the
                    // 8-bit power-of-two field cannot hold (or one that underflowed to
                    // 0).  Before the per-token mantissa this test was ~42x over-range.
                    clip_r[head] <= m_clip;
                    m_app_pend   <= 1'b1; m_app_tok <= token_r;
                    m_app_head   <= head; m_app_mant <= m_pub;
                    if (m_pub > m_max_r) m_max_r <= m_pub;
                    if (head == 1'b0) begin head <= 1'b1; state <= S_P0; q_clear <= 1'b1; end
                    else              begin head <= 1'b0; beat <= 3'd0; state <= S_REST; end
                end
            end
            // restore this head's own (eb, m): the other head's magnitude pass
            // overwrote the quantiser registers in between
            S_REST: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                beat_cnt <= 3'd0;
                if (!rest_issued) begin
                    q_rest_valid <= 1'b1;
                    q_rest_eb    <= eff_eb[head];
                    q_rest_m     <= eff_m[head];
                    rest_issued  <= 1'b1;
                end else begin
                    q_rest_valid <= 1'b0;
                end
                if (q_rest_done) begin
                    rest_issued <= 1'b0;
                    state       <= S_P1;
                end
            end
            // --------------------------------------------- pass 1: replay + emit
            S_P1: begin
                p1_tog <= ~p1_tog;
                q_valid <= 1'b0; q_pass <= 1'b0;
                // [STAGE5-FIX4] one half in flight at a time: do not push again
                // until the half already handed to the quantiser was accepted
                if (push_cnt == acc_cnt) begin
                if (p1_tog) begin
                    q_valid <= 1'b1;
                    q_pass  <= 1'b1;
                    q_data  <= hold[{head,beat}];
                    q_last  <= 1'b0;
                    p1_dg   <= beat[2:1];
                    p1_half <= beat[0];
                    p1_head <= head;
                    if (beat == BEATS_PER_HEAD-1) state <= S_APP;
                    else                          beat  <= beat + 3'd1;
                end
                end
            end
            S_APP: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                // all four INT8 beats of this head have left the packer, and
                // [STAGE5-FIX4] every half pushed for it was really taken
                if (beat_cnt == 3'd4 && !pk_valid && (push_cnt == acc_cnt)) begin
                    beat_cnt <= 3'd0;
                    if (head == 1'b0) begin
                        head  <= 1'b1;
                        beat  <= 3'd0;
                        state <= S_REST;
                    end else begin
                        // one count per token whose V no longer fits the layer scale
                        state <= S_DONE;
                    end
                end
            end
            S_DONE: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                if (clip_r[0] || clip_r[1]) begin
                    clip_count <= clip_count + 32'd1;
                    clip_sticky_r <= 1'b1;
                end
                state <= S_IDLE;            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
