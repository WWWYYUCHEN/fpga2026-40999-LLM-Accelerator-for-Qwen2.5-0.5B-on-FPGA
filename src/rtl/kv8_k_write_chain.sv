`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// kv8_k_write_chain -- stage 1+2: the complete INT8 K producer chain
//
// What it replaces
//   Today the controller writes one token's K as 16 x 16 B BF16 beats
//   (qwen25_qkv_controller.v post-write path: post_operation==0 ->
//    k_base + layer*8MiB + token*256, 16 beats x 16 B).  With the INT8 KV cache
//   the same token is 8 x 16 B INT8 beats and the per-(head,token) scale has to
//   be produced and published at the same time.
//
// Structure (verified building blocks wired together, plus the two documented
// backward-compatible extensions in kv8_k_quant / kv8_scale_ram)
//   BF16 K beats (8 x BF16 = 8 dims, 8 beats per KV head, 2 heads = 16 beats)
//        |
//        +--> 2 KiB hold buffer (16 x 128 bit)   <- second pass source
//        |
//        +--> kv8_k_quant pass 0 (magnitude) -> eb/m per head
//             kv8_k_quant pass 1 (quantise, replayed from the hold buffer)
//                  -> 8 x INT8 per beat
//                  -> kv8_beat_pack2 (2:1) -> 16 x INT8 = one KV8 beat
//                       -> 4 KiB block buffer   (stage 2, option (b))
//                  -> normalised mantissa + biased exponent -> kv8_scale_ram append
//
// STAGE 2 -- the 32-token reorder (option (b), report section 6)
//   The consumer (kv8_fill_packer MODE_KV=0) needs the K beats of one KV head in
//   "dim_group outer, 32 tokens ascending, 32-aligned" order, because one 256-bit
//   operand-bank word is 32 tokens of one dim.  A token-major INT8 layout cannot
//   supply that with a linear read, so the writer stores K block-major:
//
//     K byte address (one 16 B beat, kv8_layout replaced by this file):
//        k_base + layer*2MiB
//               + (token[13:5])*4096         32-token block
//               + dim_group[1:0]*1024        4 dim groups of 16 dims
//               + head*512                   2 KV heads
//               + token[4:0]*16              32 tokens, contiguous
//
//   -> a linear read of one 32-token block is exactly
//        (dg0,h0,t0..31) (dg0,h1,t0..31) (dg1,h0,t0..31) ... (dg3,h1,t0..31)
//      i.e. 16 descriptors of 16 tokens x 16 B = 256 B each, 4 KiB per block.
//   The cost is one buffer: 32 tokens x 8 beats x 16 B = 4 KiB (the stage-1 hold
//   buffer was 2 KiB, it is now the 4 KiB block buffer; the 2 KiB BF16 hold
//   buffer per token is unchanged).
//
//   Emission policy: a block is written when its 32nd token has been quantised,
//   or on i_flush (partial block, e.g. a 30-token prefill).  While a block is
//   being emitted the chain is not ready for a new token (o_ready=0); one block
//   is 256 beats, one token is ~50 cycles, so the writer is still ~8x faster than
//   the K projection that feeds it.
//
// The block-exponent contract (the one thing the two verified blocks did not
// settle between them): kv8_scale_ram stores ONE exponent per (head, 256-token
// block) and its expander returns m8 * 2^(eb-127); kv8_k_quant instead computes a
// per-token exponent.  Inside one block every token must therefore be quantised
// with the block's exponent, so the chain
//   * takes the first token of a block as the lock (and publishes it once, with a
//     byte enable, so the other 15 blocks of the same 128-bit word are untouched),
//   * re-derives m/recip for that lock on every later token of the block
//     (kv8_k_quant i_eb_lock_*), which is exactly the report's "eb is fixed by the
//     first token of the block; a later token that exceeds 255*2^eb sets clip".
//   With the lock ports tied off both blocks behave exactly as verified.
//
// Scale-table encoding (must match kv8_scale_ram's expander)
//       m8 = m,              stored exponent = eb_internal - 3
//   so the value read back is *exactly* s_k/8 = m * 2^(eb_raw-127-3).
//
// Throughput / latency
//   pass 0 : 1 beat/cycle (16 cycles for both heads)
//   pass 1 : 1 beat per 2 cycles (2:1 packer) -> 32 cycles/token
//   total  : ~50 cycles per token, + ~256 cycles per 32-token block emission
// =============================================================================
module kv8_k_write_chain #(
    parameter integer CONTEXT_LOG2 = 14,
    parameter integer LAYERS       = 24,
    parameter integer MARGIN       = 2,
    parameter integer BLOCK_LOG2   = 8,
    // 0 = stage-2 32-token block32 layout (descriptor emission, TB-verified)
    // 1 = token-major layout written one 128 B beat-run per token, i.e. exactly
    //     the command shape the baseline uses for its 256 B/token K write.  The
    //     E2E uses this mode so the KV8 producer cannot disturb the descriptor/
    //     burst machinery of the shared write engine.
    parameter integer TOKEN_MAJOR  = 0
) (
    input  wire         clk,
    input  wire         rst_n,
    // ---- per-token job ----------------------------------------------------
    input  wire         i_start,          // pulse: latch the geometry below
    input  wire [4:0]   i_layer,
    input  wire [15:0]  i_token,          // global token index (ascending)
    input  wire [63:0]  i_k_base,
    input  wire [63:0]  i_k_scale_base,
    input  wire [63:0]  i_v_scale_base,
    output wire         o_ready,          // idle and block buffer free
    // ---- BF16 K beat stream from the controller write-back -----------------
    input  wire         i_k_valid,
    output wire         o_k_ready,
    input  wire [127:0] i_k_data,
    output wire         o_busy,
    output wire         o_done,           // pulse when the token is published
    // ---- partial-block flush (end of prefill / end of layer) ---------------
    input  wire         i_flush,          // pulse or level: write the partial block
    output wire         o_flush_done,     // pulse when that block is on the bus
    // ---- INT8 DDR beat stream (stage-2 block32 layout) --------------------
    output wire         o_beat_valid,
    input  wire         i_beat_ready,
    output wire [63:0]  o_beat_address,
    output wire [127:0] o_beat_data,
    output wire [6:0]   o_beat_tag,       // {head, dim_group}
    output wire         o_beat_desc,      // first beat of a 256 B descriptor
    output wire [4:0]   o_beat_len,       // tokens in that descriptor (16 or fewer)
    output wire [4:0]   o_beat_token,     // token inside the descriptor (0 = first)
    // ---- scale table ownership / bulk load / read port ---------------------
    input  wire         i_bank_wr,
    input  wire         i_bank_rd,
    input  wire         i_wr_acquire,
    input  wire         i_wr_release,
    input  wire         i_rd_release,
    output wire         o_wr_grant,
    output wire         o_rd_grant,
    output wire         o_owner_error,
    input  wire         i_etab_valid,     // external bulk load of the exponent table
    input  wire [2:0]   i_etab_word,
    input  wire [127:0] i_etab_data,
    input  wire [15:0]  i_etab_be,
    input  wire [2:0]   i_etab_rd_word,   // [16K-ONE-LAYER] image-dump read
    output wire [127:0] o_etab_rd_data,
    input  wire         i_load_valid,
    input  wire [11:0]  i_load_word,
    input  wire [127:0] i_load_mant,
    input  wire         i_vload_valid,    // [16K-ONE-LAYER] V m16 bulk load
    input  wire [11:0]  i_vload_word,
    input  wire [127:0] i_vload_data,
    input  wire         i_rd_en,
    input  wire [11:0]  i_rd_word,
    output wire         o_rd_valid,
    output wire [135:0] o_rd_scale,        // {eb_tab[7:0], 8 x m16}  [M16-SCALE]
    input  wire         i_v_we,
    input  wire [3:0]   i_v_word,
    input  wire [127:0] i_v_data,
    input  wire [3:0]   i_v_rd_word,
    output wire [127:0] o_v_rd_data,
    // ---- V per-token mantissa table [V-PERTOK-M] ---------------------------
    // The scale RAM lives here, so the V producer's per-token append and the
    // attention's 8-token read have to be passed through this module.
    input  wire         i_vt_app_valid,
    output wire         o_vt_app_ready,
    input  wire [15:0]  i_vt_app_token,
    input  wire         i_vt_app_head,
    input  wire [15:0]  i_vt_app_mant,
    output wire         o_vt_app_taken,
    input  wire         i_vt_rd_en,
    input  wire [11:0]  i_vt_rd_word,
    output wire [127:0] o_vt_rd_mant,
    output wire         o_vt_rd_valid,
    // ---- diagnostics -------------------------------------------------------
    output wire         o_app_conflict,
    output wire         o_clip_sticky,
    output wire [31:0]  o_stat_beat,
    output wire [31:0]  o_stat_app,
    output wire [3:0]   o_dbg_state,
    output wire [1:0]   o_dbg_lock_v,
    output wire [15:0]  o_clip_count,     // number of appends that saturated the lock
    output wire [4:0]   o_blk_tok,        // tokens buffered in the current 32-token block
    // How many 16 B beats the current token will deliver.  The controller derives the
    // write command's byte count from THIS, so a command length can never float free
    // of what the producer actually delivers.
    output wire [4:0]   o_token_beats,
    output wire         o_err_seq         // sticky: token stream did not ascend in-block
);
    localparam integer HEADS = 2;
    localparam integer BEATS_PER_HEAD = 8;              // 8 x 8 BF16 dims = 64
    localparam integer BLOCK_TOKENS   = 32;             // stage-2 32-token block
    // [K16] a 16-bit code is one 128-bit beat per 8 dims, so a token is
    // 2 heads x 8 dim_groups = 16 beats of 16 B = 256 B (was 8 x 16 B = 128 B).
    localparam integer BEATS_PER_TOKEN = 16;
    localparam integer GROUPS_PER_HEAD = 8;             // dim_group 0..7

    localparam [3:0] S_IDLE = 4'd0, S_P0 = 4'd1, S_P0_WAIT = 4'd2, S_REST = 4'd3,
                     S_P1 = 4'd4, S_APP = 4'd5, S_DONE = 4'd6, S_EMIT = 4'd7;

    reg [3:0]   state;
    reg [4:0]   layer_r;
    reg [15:0]  token_r;
    reg [63:0]  k_base_r, k_scale_base_r, v_scale_base_r;
    reg         head;                    // KV head in flight
    reg [2:0]   beat;                    // BF16 beat inside the head
    reg         p1_tog;                  // 1 beat / 2 cycles in pass 1
    reg [2:0]   p1_dg;                   // dim_group (8 dims per beat) of the beat
    reg         p1_head;
    reg         lock_pend;
    reg         app_done;
    reg         app_issued;
    // pass-0 result of the head in flight
    reg [7:0]   eb_raw   [0:1];
    reg [15:0]  m_int    [0:1];
    reg [7:0]   maxmag_e [0:1];
    // block-exponent lock, one per (LAYER, KV head).  A decode step revisits
    // the layer and MUST reuse the exponent its first token locked: re-deriving
    // one from the token being appended re-publishes the block exponent and
    // silently rescales every token already in the cache.
    reg [7:0]   eb_lock  [0:LAYERS-1][0:1];
    reg         lock_v   [0:LAYERS-1][0:1];
    // published (normalised) scale per head
    reg [15:0]  mant16   [0:1];   // [M16-SCALE] was mant8[7:0]
    reg [7:0]   eb_tbl   [0:1];
    reg         clip_r   [0:1];
    // effective (exponent, mantissa) used by this token's quantise pass, per head
    reg [7:0]   eff_eb   [0:1];
    reg [15:0]  eff_m    [0:1];

    reg [31:0]  stat_beat, stat_app;
    reg [15:0]  clip_count;
    reg         err_seq_r;

    // ---------------------------------------------------------------- buffers
    // Per-token BF16 replay buffer (pass 0 -> pass 1): 16 x 128 bit = 2 KiB.
    (* ram_style = "distributed" *) reg [127:0] hold [0:HEADS*BEATS_PER_HEAD-1];
    // Stage-2 block buffer: 32 tokens x 2 heads x 4 dim groups x 128 bit = 4 KiB.
    // Physical order is the emission order, so the emitter has no address mux:
    //   blk[{dim_group, head, token[4:0]}] -> 8 contiguous groups of 32 words.
    (* ram_style = "distributed" *) reg [127:0] blk [0:BLOCK_TOKENS*BEATS_PER_TOKEN-1];

    // ---------------------------------------------------------------- K quant
    reg         q_clear, q_pass, q_valid, q_last;
    reg [127:0] q_data;
    reg         q_lock_valid;
    reg [7:0]   q_lock;
    reg         q_rest_valid;
    reg [7:0]   q_rest_eb;
    reg [15:0]  q_rest_m;
    // [RECIP-SEQ] one-shot for the restore request.  The quantiser's restore path
    // is now multi-cycle (see kv8_k_quant.sv [RECIP-SEQ]), so the old
    // "re-assert while !done" form would fire a second request every other cycle
    // for the whole 24..40-cycle latency.  Issue exactly once, retire on done.
    reg         rest_issued;
    wire        q_done, q_qv, q_lock_done, q_rest_done;
    wire [7:0]  q_eb;
    wire [15:0] q_m;
    wire [7:0]  q_maxmag;
    wire [127:0] q_q;

    // [M16-CLIP] K opts into the 16-bit mantissa field (65535); the V chain keeps
    // the quantiser's legacy 255 default, so the V format is untouched.
    // [K16-CODE] CODE_EXTRA=8 keeps the same R/m/eb rule but publishes the 8 extra
    // fractional bits as the {k_hi,k_lo} plane pair (128-bit beat, 8 values).
    kv8_k_quant #(.MARGIN(MARGIN), .MTOP(65535), .RSHIFT(24), .CODE_EXTRA(8)) k_quant_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_clear(q_clear), .i_pass(q_pass), .i_valid(q_valid), .i_last(q_last),
        .i_k(q_data),
        .i_eb_lock_valid(q_lock_valid), .i_eb_lock(q_lock),
        .o_lock_done(q_lock_done),
        .i_restore_valid(q_rest_valid), .i_restore_eb(q_rest_eb), .i_restore_m(q_rest_m),
        .o_restore_done(q_rest_done),
        .o_done(q_done), .o_eb(q_eb), .o_m(q_m), .o_maxmag_exp(q_maxmag),
        .o_q_valid(q_qv), .o_q(q_q)
    );

    // ------------------------------------------------------------- beat pack
    // [K16] the quantiser already emits one complete 128-bit beat (8 x 16-bit
    // value), so the packer is 1:1 and only supplies the skid register.
    //
    // Why the skid still exists (measured, not reasoned): the packer takes a beat
    // only while it does not already hold one, and kv8_k_quant's o_q_valid lands
    // ONE CYCLE after the pass-1 push.  "packer ready now" is therefore not the
    // same as "packer ready when the beat arrives": under DMA back-pressure
    // (i_beat_ready low, TOKEN_MAJOR=1) a beat was silently dropped, which in the
    // E2E leaves a write command forever short of its beats (the orphan-descriptor
    // deadlock).
    wire         pk_ready = TOKEN_MAJOR ? i_beat_ready : 1'b1;
    wire         pk_in_ready;   // packer can take another beat
    wire         pk_valid;
    wire [17:0]  pk_index;
    wire [127:0] pk_data;
    reg          sk_valid;      // a quantised beat the packer could not take yet
    reg  [127:0] sk_q;
    reg  [13:0]  sk_token;
    reg          sk_head;
    reg  [2:0]   sk_dg;
    wire         pk_in_valid = sk_valid | q_qv;
    wire [127:0] pk_in_q     = sk_valid ? sk_q        : q_q;
    wire [13:0]  pk_in_token = sk_valid ? sk_token    : token_r[13:0];
    wire         pk_in_head  = sk_valid ? sk_head     : p1_head;
    wire [2:0]   pk_in_dg    = sk_valid ? sk_dg       : p1_dg;
    kv8_beat_pack2 pack2_Inst (
        .clk(clk), .rst_n(rst_n), .i_flush(i_start && state == S_IDLE),
        .i_valid(pk_in_valid), .o_ready(pk_in_ready),
        .i_index({pk_in_token, pk_in_head, pk_in_dg}),
        .i_q(pk_in_q),
        .o_valid(pk_valid), .i_ready(pk_ready),
        .o_index(pk_index), .o_data(pk_data),
        .o_err_pair()
    );

    // skid sequencing: hold one beat that the packer could not take, and drain it
    // the moment the packer is ready.  Cleared on a new token so nothing can
    // straddle two tokens (a token's beats must never mix with the next one).
    //
    // push_cnt/acc_cnt make token completion a COUNT, not a timing guess: a token
    // may not retire while any beat it pushed is still in flight, so "16 beats"
    // can only be believed once 16 beats were actually accepted.
    reg [5:0] push_cnt, acc_cnt;
    wire      q_push = q_valid && q_pass;   // a pass-1 quantise beat (pass 0 has q_pass=0)
    wire      half_taken = pk_in_valid && pk_in_ready;
    always @(posedge clk) begin
        if (!rst_n) begin
            sk_valid <= 1'b0; sk_q <= 128'd0; sk_token <= 14'd0;
            sk_head <= 1'b0; sk_dg <= 3'd0;
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
            end
        end
    end

    // ------------------------------------------------------------- DDR address
    // Stage-2 block32 layout: one block is 4 KiB, emitted dg-major / head / token.
    reg  [63:0] blk_base_r;              // k_base + layer*4MiB + block*8192
    reg  [8:0]  blk_idx_r;               // token[13:5] of the buffered block
    reg  [15:0] tok_exp_r;               // next expected global token index
    reg  [4:0]  blk_tok;                 // tokens buffered in the current block
    reg  [5:0]  blk_tok_lim;             // tokens to emit (32 or the flushed count)
    reg [3:0]  emit_grp;                 // {dim_group[2:0], head} 0..15
    reg  [4:0]  emit_tok;
    reg         flush_req;

    wire [2:0]  emit_dg   = emit_grp[3:1];
    wire        emit_head = emit_grp[0];
    // physical emission order = {dim_group, head, token}: 32 tokens x 16 beats
    wire [8:0]  blk_waddr = {pk_index[2:0], pk_index[3], token_r[4:0]};
    wire [8:0]  blk_raddr = {emit_dg, emit_head, emit_tok};

    // inside one 32-token block: dim_group*2048 + head*1024 + token*16  (13 bits)
    wire [12:0] emit_intra = {emit_dg, emit_head, emit_tok, 4'b0000};

    // token-major mode: the packer's 128-bit beat goes straight out, one 256 B run
    // per token at k_base + token*256 + head*128 + dim_group*16
    wire [3:0]  tm_beat = {pk_index[3], pk_index[2:0]};
    assign o_beat_valid   = TOKEN_MAJOR ? pk_valid : (state == S_EMIT);
    assign o_beat_data    = TOKEN_MAJOR ? pk_data  : blk[blk_raddr];
    assign o_beat_address = TOKEN_MAJOR ? (k_base_r + {42'd0, token_r[13:0], 8'd0} +
                                          {56'd0, tm_beat, 4'b0000})
                                        : (blk_base_r + {{(64-13){1'b0}}, emit_intra});
    assign o_beat_tag     = TOKEN_MAJOR ? {tm_beat[3], tm_beat[2:0]} : {emit_head, emit_dg};
    // A 32-token run of one (dim_group, head) is 1024 B = four 256 B descriptors of
    // 16 tokens each, so a full block is exactly 32 descriptors.
    assign o_beat_desc    = TOKEN_MAJOR ? (tm_beat == 4'd0) : ((emit_tok == 5'd0) || (emit_tok == 5'd16));
    assign o_beat_token   = TOKEN_MAJOR ? {1'b0,tm_beat} : emit_tok;
    // A descriptor carries 16 tokens, except the tail of a partial block.
    assign o_beat_len     = TOKEN_MAJOR ? 5'd16 :
                            ((blk_tok_lim <= 6'd16) ? blk_tok_lim[4:0]
                          : ((emit_tok < 5'd16) ? 5'd16 : (blk_tok_lim[4:0] - 5'd16)));

    // ------------------------------------------------------- scale RAM instance
    reg         app_valid;
    reg [15:0]  app_mant;
    reg [7:0]   app_exp;
    reg         app_clip;
    wire        app_taken;
    // exponent-table write issued by the chain for a block's first token
    reg         etab_wr;
    reg [2:0]   etab_wr_word;
    reg [15:0]  etab_wr_be;
    reg [127:0] etab_wr_data;

    kv8_scale_ram #(.CONTEXT_LOG2(CONTEXT_LOG2), .BANKS(2), .V_DIM(64), .V_HEADS(HEADS),
                    .BLOCK_LOG2(BLOCK_LOG2), .LAYERS(LAYERS), .TBL_TOK_LOG2(10))
    scale_ram_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_bank_wr(i_bank_wr), .i_bank_rd(i_bank_rd),
        .i_layer_wr(layer_r), .i_layer_rd(layer_r),
        .i_wr_acquire(i_wr_acquire), .i_wr_release(i_wr_release), .i_rd_release(i_rd_release),
        .o_wr_grant(o_wr_grant), .o_rd_grant(o_rd_grant), .o_owner_error(o_owner_error),
        .i_etab_valid(i_etab_valid || etab_wr), .o_etab_ready(),
        .i_etab_word(i_etab_valid ? i_etab_word : etab_wr_word),
        .i_etab_data(i_etab_valid ? i_etab_data : etab_wr_data),
        .i_etab_be(i_etab_valid ? i_etab_be : etab_wr_be),
        .i_etab_rd_word(i_etab_rd_word), .o_etab_rd_data(o_etab_rd_data),
        .i_load_valid(i_load_valid), .o_load_ready(), .i_load_word(i_load_word), .i_load_mant(i_load_mant),
        .i_vload_valid(i_vload_valid), .o_vload_ready(), .i_vload_word(i_vload_word), .i_vload_data(i_vload_data),
        .i_app_valid(app_valid), .o_app_ready(),
        .i_app_token(token_r), .i_app_head(head),
        .i_app_mant(app_mant), .i_app_exp(app_exp), .i_app_clip(app_clip),
        .o_app_taken(app_taken), .o_app_conflict(o_app_conflict),
        .i_rd_en(i_rd_en), .i_rd_word(i_rd_word), .o_rd_valid(o_rd_valid),
        .o_rd_m16(o_rd_scale[127:0]), .o_rd_eb(o_rd_scale[135:128]),
        .i_v_we(i_v_we), .i_v_word(i_v_word), .i_v_data(i_v_data),
        .i_v_rd_word(i_v_rd_word), .o_v_rd_data(o_v_rd_data),
        .i_vt_app_valid(i_vt_app_valid), .o_vt_app_ready(o_vt_app_ready),
        .i_vt_app_token(i_vt_app_token), .i_vt_app_head(i_vt_app_head),
        .i_vt_app_mant(i_vt_app_mant), .o_vt_app_taken(o_vt_app_taken),
        .i_vt_rd_en(i_vt_rd_en), .i_vt_rd_word(i_vt_rd_word),
        .o_vt_rd_mant(o_vt_rd_mant), .o_vt_rd_valid(o_vt_rd_valid),
        .o_clip_sticky(o_clip_sticky), .o_stat_load(), .o_stat_app(o_stat_app), .o_stat_rd()
    );

    // ------------------------------------------------------------ normalisation
    // kv8_scale_ram's expander returns exactly  byte * 2^(etab-127)  for ANY byte
    // value (the internal normalise-by-k / exponent-by-k cancel), and kv8_layout /
    // the report want it to represent s_k/8 = m * 2^(eb_internal-130).  Therefore
    //     stored mantissa byte = m                (the quantiser's own integer, 5..255)
    //     stored exponent byte = eb_internal - 3  (per 256-token block, from the lock)
    // Both are exact: no mantissa rounding is added by the table.
    function [7:0] norm_exp(input [7:0] eb_int);
        reg [9:0] t;
        begin
            t = {2'b00,eb_int};
            if (t <= 10'd3)       norm_exp = 8'd1;
            else if (t > 10'd258) norm_exp = 8'd255;
            else                  norm_exp = t[7:0] - 8'd3;
        end
    endfunction

    // [K16-CODE] The frozen K16 format keeps the SAME (m, eb) rule and C_K = 268,
    // but the code carries 8 extra fractional bits (q16 = 256*q8).  The value is
    // k = (s_k/8) * q16/256, so the dequantisation exponent must drop by 8:
    //     eb_table(K16) = norm_exp(eb_internal) - 8
    // which is exactly "s16 = s8/256".  The clip witness still tests the UNSHIFTED
    // norm_exp, i.e. the same range check as before.
    function [7:0] kexp(input [7:0] eb_int);
        reg [8:0] t;
        begin
            t = {1'b0, norm_exp(eb_int)};
            kexp = (t <= 9'd9) ? 8'd1 : (t[7:0] - 8'd8);
        end
    endfunction

    assign o_ready      = (state == S_IDLE) && !flush_req;
    assign o_busy       = (state != S_IDLE);
    assign o_done       = (state == S_DONE);
    assign o_flush_done = (state == S_EMIT) && i_beat_ready &&
                          (emit_grp == 4'd15) && (({1'b0,emit_tok} + 6'd1) == blk_tok_lim);
    assign o_k_ready    = (state == S_P0);
    assign o_stat_beat  = stat_beat;
    assign o_dbg_state  = state;
    assign o_dbg_lock_v = {lock_v[layer_r][1],lock_v[layer_r][0]};
    assign o_clip_count = clip_count;
    assign o_blk_tok    = blk_tok;
    assign o_token_beats= TOKEN_MAJOR ? 5'd16 : blk_tok_lim[4:0];
    assign o_err_seq    = err_seq_r;

    // kv8_k_quant emits the *table* convention (eb_internal - 10); the quantiser and
    // the normaliser both need the internal exponent, so convert once here.
    wire [7:0] q_eb_i = (q_eb > 8'd245) ? 8'd255 : (q_eb + 8'd10);

    integer h;
    integer l;
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= S_IDLE; layer_r <= 5'd0; token_r <= 16'd0;
            k_base_r <= 64'd0; k_scale_base_r <= 64'd0; v_scale_base_r <= 64'd0;
            head <= 1'b0; beat <= 3'd0;
            q_clear <= 1'b0; q_pass <= 1'b0; q_valid <= 1'b0; q_last <= 1'b0; q_data <= 128'd0;
            q_lock_valid <= 1'b0; q_lock <= 8'd0;
            app_valid <= 1'b0; app_mant <= 16'd0; app_exp <= 8'd0; app_clip <= 1'b0;
            stat_beat <= 32'd0; stat_app <= 32'd0; clip_count <= 16'd0;
            p1_tog <= 1'b0; p1_dg <= 2'd0; p1_head <= 1'b0; lock_pend <= 1'b0;
            app_issued <= 1'b0; app_done <= 1'b0; q_rest_valid <= 1'b0; rest_issued <= 1'b0;
            etab_wr <= 1'b0; etab_wr_word <= 3'd0; etab_wr_be <= 16'd0; etab_wr_data <= 128'd0;
            blk_base_r <= 64'd0; blk_idx_r <= 9'd0; tok_exp_r <= 16'd0;
            blk_tok <= 5'd0; blk_tok_lim <= 6'd0;
            emit_grp <= 4'd0; emit_tok <= 5'd0; flush_req <= 1'b0; err_seq_r <= 1'b0;
            for (h=0;h<HEADS;h=h+1) begin
                eb_raw[h] <= 8'd0; m_int[h] <= 16'd0; maxmag_e[h] <= 8'd0;
                mant16[h] <= 16'd0; eb_tbl[h] <= 8'd0; clip_r[h] <= 1'b0;
                for (l=0;l<LAYERS;l=l+1) begin eb_lock[l][h] <= 8'd0; lock_v[l][h] <= 1'b0; end
            end
        end else begin
            q_clear <= 1'b0;
            q_lock_valid <= 1'b0;
            // ---- packer output -> stage-2 block buffer (always accepted) ----
            if (pk_valid && !TOKEN_MAJOR) blk[blk_waddr] <= pk_data;
            // exponent-table write is a one-cycle pulse
            if (etab_wr) etab_wr <= 1'b0;
            // ---- flush request: level or pulse, serviced when the token is done ----
            if (i_flush && !TOKEN_MAJOR) flush_req <= 1'b1;

            case (state)
            // ------------------------------------------------------------- IDLE
            S_IDLE: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                if (flush_req && blk_tok != 5'd0) begin
                    // partial block: emit exactly the buffered tokens
                    flush_req  <= 1'b0;
                    blk_tok_lim<= {1'b0,blk_tok};
                    emit_grp   <= 4'd0;
                    emit_tok   <= 5'd0;
                    state      <= S_EMIT;
                end else if (flush_req) begin
                    flush_req <= 1'b0;           // nothing buffered
                end else if (i_start) begin
                    layer_r <= i_layer; token_r <= i_token;
                    k_base_r <= i_k_base; k_scale_base_r <= i_k_scale_base; v_scale_base_r <= i_v_scale_base;
                    head <= 1'b0; beat <= 3'd0;
                    q_clear <= 1'b1;                 // fresh magnitude accumulator
                    // Only a real 256-token block start clears the lock.  A new
                    // LAYER no longer does: the lock belongs to (layer, block) and
                    // a decode append of a later token must reuse it.
                    if (i_token[BLOCK_LOG2-1:0] == 0)
                        for (h=0;h<HEADS;h=h+1)
                            lock_v[i_layer][h] <= 1'b0;
                    // stage-2 block base: the caller supplies the per-layer K base
                    // (it owns the layer stride, 8 MiB in this tree's DDR map);
                    // inside a layer the 32-token blocks are 4 KiB apart.
                    if (i_token[4:0] == 5'd0) begin
                        blk_idx_r  <= i_token[13:5];
                        blk_base_r <= i_k_base + {{(64-9-13){1'b0}}, i_token[13:5], 13'd0};
                    end
                    // A flush only *copies out* the tokens buffered so far; it does
                    // not end the block (an incremental prefill flushes at every
                    // row-group boundary and keeps filling the same 4 KiB block, so
                    // the emitted token offsets stay absolute).  The buffer counter
                    // is therefore cleared only when a block really starts or ends.
                    if (i_token[4:0] == 5'd0 || i_layer != layer_r) blk_tok <= 5'd0;
                    // the writer's contract: ascending tokens, one block32 at a time.
                    // A new layer (or token 0 of a new block) re-seeds the expectation,
                    // so the per-layer restart of the E2E prefill is not a sequence error.
                    if ((i_layer != layer_r) || (i_token == 16'd0)) begin
                        tok_exp_r <= i_token + 16'd1;
                    end else begin
                        if (i_token != tok_exp_r)  err_seq_r <= 1'b1;
                        if (i_token[13:5] != blk_idx_r && i_token[4:0] != 5'd0) err_seq_r <= 1'b1;
                        tok_exp_r <= i_token + 16'd1;
                    end
                    state <= S_P0;
                end
            end
            // ------------------------------------------- pass 0: magnitude, 8 beats
            S_P0: begin
                q_pass <= 1'b0;
                if (i_k_valid && o_k_ready) begin
                    hold[{head,beat}] <= i_k_data;
                    q_valid <= 1'b1;
                    q_data  <= i_k_data;
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
                    eb_raw[head]   <= q_eb_i;
                    m_int[head]    <= q_m;
                    maxmag_e[head] <= q_maxmag;
                    beat  <= 3'd0;
                    if (!lock_v[layer_r][head]) begin
                        // first token of the block: its exponent becomes the lock
                        lock_v[layer_r][head]  <= 1'b1;
                        eb_lock[layer_r][head] <= q_eb_i;
                        eff_eb[head]  <= q_eb_i;
                        eff_m[head]   <= q_m;
                        mant16[head]  <= q_m;
                        eb_tbl[head]  <= kexp(q_eb_i);
                        // [M16-CLIP] 16-bit compares: a 16-bit q_m against 8'd255
                        // would never fire, so m16 saturation would read as clip 0.
                        clip_r[head]  <= (q_m == 16'd0) || (q_m == 16'd65535) ||
                                         (norm_exp(q_eb_i) == 8'd255);
                        // publish that exponent for this (head, block) only
                        etab_wr      <= 1'b1;
                        // [16K-ONE-LAYER] the on-chip exponent table is one layer
                        // deep again: 64 blocks x 2 heads, 16 blocks per 128-bit
                        // word, word = {head, block[5:4]}, byte = block[3:0].
                        etab_wr_word <= {head, token_r[CONTEXT_LOG2-1:CONTEXT_LOG2-2]};
                        etab_wr_be   <= 16'h0001 << token_r[BLOCK_LOG2+3:BLOCK_LOG2];
                        etab_wr_data <= {16{kexp(q_eb_i)}};
                        // next: the other head's magnitude pass, then the quantise pass
                        if (head == 1'b0) begin head <= 1'b1; state <= S_P0; q_clear <= 1'b1; end
                        else              begin head <= 1'b0; beat <= 3'd0; state <= S_REST; end
                    end else begin
                        // locked: re-derive m/recip with the block exponent
                        q_lock       <= eb_lock[layer_r][head];
                        q_lock_valid <= 1'b1;
                        lock_pend    <= 1'b1;
                    end
                end else if (lock_pend && q_lock_done) begin
                    lock_pend    <= 1'b0;
                    eff_eb[head] <= eb_lock[layer_r][head];
                    eff_m[head]  <= q_m;
                    mant16[head] <= q_m;
                    eb_tbl[head] <= kexp(eb_lock[layer_r][head]);
                    // [FIX-LOCKWIDTH] q_m == 0 is as broken as a saturated one
                    // (zero published scale) and was not flagged on this path.
                    clip_r[head] <= (q_m == 16'd65535) || (q_m == 16'd0) ||
                                    (norm_exp(eb_lock[layer_r][head]) == 8'd255);
                    if (head == 1'b0) begin head <= 1'b1; state <= S_P0; q_clear <= 1'b1; end
                    else              begin head <= 1'b0; beat <= 3'd0; state <= S_REST; end
                end
            end
            // restore this head's own (eb, m) before its quantise pass: the *other*
            // head's magnitude pass overwrote the quantiser registers in between
            S_REST: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
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
            // The 2:1 packer takes a half only while it does not hold an output
            // beat, and the quantised result of this push lands one cycle later,
            // so pass 1 may only push when the skid is empty: the skid absorbs the
            // "arrived while the packer was busy" case instead of losing the beat.
            S_P1: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                // one half in flight at a time: do not push again until the half
                // already handed to the quantiser has been accepted by the packer
                if (push_cnt == acc_cnt) begin
                p1_tog <= ~p1_tog;
                if (p1_tog) begin
                    q_valid <= 1'b1;
                    q_pass  <= 1'b1;
                    q_data  <= hold[{head,beat}];
                    q_last  <= 1'b0;
                    p1_dg   <= beat[2:0];   // one 8-dim beat = one dim_group
                    p1_head <= head;
                    if (beat == BEATS_PER_HEAD-1) state <= S_APP;
                    else                          beat  <= beat + 3'd1;
                end
                end
            end
            S_APP: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                // one append request per head, issued once and retired once
                if (!app_issued) begin
                    app_valid  <= 1'b1;
                    app_issued <= 1'b1;
                    app_mant   <= mant16[head];
                    app_exp    <= eb_tbl[head];
                    app_clip   <= clip_r[head];
                end else begin
                    app_valid  <= 1'b0;
                end
                if (app_taken) begin
                    app_done <= 1'b1;
                    if (app_clip) clip_count <= clip_count + 16'd1;
                end
                if (app_done && !pk_valid && (push_cnt == acc_cnt)) begin
                    app_done   <= 1'b0;
                    app_issued <= 1'b0;
                    stat_app   <= stat_app + 1'b1;
                    if (head != HEADS-1) begin
                        head  <= head + 1'b1;
                        beat  <= 3'd0;
                        state <= S_REST;
                    end else if (TOKEN_MAJOR) begin
                        state <= S_DONE;
                    end else if (blk_tok == BLOCK_TOKENS-1) begin
                        // 32nd token: the block is complete
                        blk_tok_lim <= 6'd32;
                        emit_grp    <= 4'd0;
                        emit_tok    <= 5'd0;
                        state       <= S_EMIT;
                    end else begin
                        blk_tok <= blk_tok + 5'd1;
                        state   <= S_DONE;
                    end
                end
            end
            // ------------------------------------------ stage-2 block emission ----
            S_EMIT: begin
                if (i_beat_ready) begin
                    stat_beat <= stat_beat + 1'b1;
                    if (({1'b0,emit_tok} + 6'd1) == blk_tok_lim) begin
                        emit_tok <= 5'd0;
                        if (emit_grp == 4'd15) begin
                            // A completed 32-token block starts over; a flush copy-out
                            // keeps the buffered tokens for the next flush.
                            if (blk_tok_lim == 6'd32) blk_tok <= 5'd0;
                            blk_tok_lim <= 6'd0;
                            state       <= S_DONE;
                        end else begin
                            emit_grp <= emit_grp + 4'd1;
                        end
                    end else begin
                        emit_tok <= emit_tok + 5'd1;
                    end
                end
            end
            S_DONE: begin
                q_valid <= 1'b0; q_pass <= 1'b0;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase

        end
    end
endmodule
`default_nettype wire
