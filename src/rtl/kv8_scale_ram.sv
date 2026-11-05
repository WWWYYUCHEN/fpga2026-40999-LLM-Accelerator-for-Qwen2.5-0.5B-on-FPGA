`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_scale_ram -- on-chip KV-cache SCALE table for ONE layer of a 16K context.
//
// Why this RAM exists
//   The KV cache is generated at run time, one token per decode step. A token's
//   K scale s_k = max|k|/127 only exists after that token's K projection, so the
//   table can never be preloaded (unlike the weight scale_ram / qkv_bias path).
//   It must be WRITTEN as tokens are produced and READ BACK for every past token
//   on every later decode step, layer after layer. Hence a true SDP RAM:
//       write port : layer bulk load (DDR -> chip) + one-entry append per token
//       read port  : attention, 8 consecutive tokens of one head per cycle
//
// [16K-ONE-LAYER] 2026-11-03 -- back to the full-context geometry.
//   The [PER-LAYER-TABLE] workaround (24 layers x 1024 tokens resident, layer in
//   the address) fixed the decode corruption but capped the context at 1024
//   tokens: 24*2*128 = 6144 words is 3/4 of what ONE 16K layer of two heads
//   needs (2*2048 = 4096 words) and you cannot have both.  This file keeps the
//   full 16K ONE-LAYER table and gets the layer from a DDR scale image that the
//   controller loads at the start of every layer visit (kv8_layout.sv has the
//   image; the load path is PS_KV8_VLOAD in qwen25_qkv_controller.v):
//
//     DDR image per layer (1 MiB stride, see the E2E seg map)
//       +0x00000  K m16  64 KiB  4096 words x 128 bit  {head, token[13:3]}
//       +0x10000  V m16  64 KiB  4096 words x 128 bit  {head, token[13:3]}
//       +0x20000  V 2^  256 B    16 words x 128 bit    {head, dim_word[2:0]}
//       +0x28000  K etab 256 B   16 words x 128 bit    8 x {head, block} per word
//     (the V 2^ slot is reserved 32 KiB by the DDR map; V's exponent is per
//      layer by the frozen format -- token 0 locks it -- so 256 B is all of it)
//
// Storage format  [M16-SCALE]  (unchanged by [16K-ONE-LAYER])
//   K/V mantissa: one 128-bit word = 8 tokens x m16, word = {head, token[13:3]},
//     lane token[2:0].  The read port returns the mantissas RAW together with the
//     K block exponent.  The BF16 expander (kv8_expand) is GONE from the K read
//     path: rounding a 16-bit scale to BF16's 7-bit significand would put a 0.4%
//     floor under it and throw away the whole gain (REPORT_STAGE5_SLICE_C_MM.md
//     6.1).  The score multiplier consumes m16 directly (kv8_score_scale8
//     MANT_BITS=16).
//   K exponent: 1 B per (head, 256-token block) = 64 blocks x 2 heads = 128 B,
//     i.e. 8 words of 16 bytes (16 blocks each).  Word = {head, block[5:4]},
//     byte = block[3:0].
//   V table: 16 words x 128 bit = 2 heads x 8 dim-words, each word 8 copies of
//     the layer's 2^(eb_lock-134) as BF16 (the per-token mantissa is the V m16
//     table below).
//
// Footprint (16K, one layer, measured by OOC synthesis -- see the report)
//   K mantissa : 4096 x 128 bit = 524,288 bit -> 2 URAM
//   V mantissa : 4096 x 128 bit = 524,288 bit -> 2 URAM
//   K exponent : 8 x 128 bit     = 1,024 bit   -> ~64 LUT6 (distributed)
//   V table    : 16 x 128 bit    = 2,048 bit   -> ~128 LUT6 (distributed)
//   (the old [PER-LAYER-TABLE] build was 8192 x 128 bit x 2 = 4 + 4 URAM, so
//    the 16K geometry is URAM-neutral or cheaper; it is 16x deeper per head.)
//
// Ownership rule (the "no stale scale" guarantee)
//   A bank is EMPTY -> LOADING (i_wr_acquire) -> READY (i_wr_release) -> EMPTY
//   (i_rd_release).  Writes are accepted only in LOADING and reads only in
//   READY, so a bulk load can never land in the table attention is reading; a
//   violating access is dropped and o_owner_error sticks.
//   [16K-ONE-LAYER] the image DUMP (table -> DDR, at the next layer's start)
//   happens while the bank is still READY, before the release, so it needs no
//   extra port privilege: it is an ordinary read.
// -----------------------------------------------------------------------------
module kv8_scale_ram #(
    parameter integer CONTEXT_LOG2 = 14,      // 16384 tokens
    parameter integer BANKS = 2,              // ownership slots (only 0 is used)
    parameter integer V_DIM = 64,             // head_dim
    parameter integer V_HEADS = 2,
    parameter integer BLOCK_LOG2 = 8,         // 256-token exponent block
    // Legacy parameters of the [PER-LAYER-TABLE] workaround.  LAYERS is still
    // used by the write chains' exponent locks (they must stay per layer); the
    // table itself is one layer deep again, so TBL_TOK_LOG2 is unused.
    parameter integer LAYERS = 24,
    parameter integer TBL_TOK_LOG2 = 10
) (
    input  wire         clk,
    input  wire         rst_n,
    // ---------------- bank ownership / layer hand-off ----------------
    input  wire         i_bank_wr,               // bank the producer fills (ownership only)
    input  wire         i_bank_rd,               // bank attention reads (ownership only)
    // [16K-ONE-LAYER] the table is one layer deep: the layer number only feeds
    // the ownership bookkeeping now (kept so the controller/write chains do not
    // have to change their hand-off protocol).
    input  wire [4:0]   i_layer_wr,
    input  wire [4:0]   i_layer_rd,
    input  wire         i_wr_acquire,
    input  wire         i_wr_release,
    input  wire         i_rd_release,
    output wire         o_wr_grant,
    output wire         o_rd_grant,
    output wire         o_owner_error,
    // ---------------- K exponent table (128 B per layer, 16 B/beat) ----------
    input  wire         i_etab_valid,
    output wire         o_etab_ready,
    input  wire [2:0]   i_etab_word,             // {head, block[5:4]}
    input  wire [127:0] i_etab_data,
    // Integration extension: per-byte write enable for the 16 block exponents of
    // one 128-bit word.  The K producer publishes ONE exponent (the block's first
    // token), so it must not clobber the other 15 blocks sharing the word.
    input  wire [15:0]  i_etab_be,
    // [16K-ONE-LAYER] image-dump read port: the whole 16-block word, so the
    // controller can mirror the exponent table into DDR with full 16 B beats.
    input  wire [2:0]   i_etab_rd_word,
    output wire [127:0] o_etab_rd_data,
    // ---------------- K mantissa bulk load (DDR -> on chip) ------------------
    // One beat = 8 tokens x m16 = 128 bit; word = {head, token[13:3]}.
    input  wire         i_load_valid,
    output wire         o_load_ready,
    input  wire [11:0]  i_load_word,
    input  wire [127:0] i_load_mant,
    // ---------------- V mantissa bulk load (DDR -> on chip) ------------------
    // Same geometry as the K mantissa (the V producer's per-token table).
    input  wire         i_vload_valid,
    output wire         o_vload_ready,
    input  wire [11:0]  i_vload_word,
    input  wire [127:0] i_vload_data,
    // ---------------- append the newly produced token -----------------------
    input  wire         i_app_valid,
    output wire         o_app_ready,
    input  wire [15:0]  i_app_token,
    input  wire         i_app_head,
    input  wire [15:0]  i_app_mant,       // m16
    input  wire [7:0]   i_app_exp,
    input  wire         i_app_clip,
    output wire         o_app_taken,
    output wire         o_app_conflict,
    // ---------------- attention read port -----------------------------------
    input  wire         i_rd_en,
    input  wire [11:0]  i_rd_word,        // {head, token[13:3]}
    output wire         o_rd_valid,       // 3 cycles after i_rd_en
    output wire [127:0] o_rd_m16,         // 8 x m16, raw (no BF16 expander)
    output wire [7:0]   o_rd_eb,          // block exponent of that word
    // ---------------- V scale table (per-layer 2^k, 16 words) ---------------
    input  wire         i_v_we,
    input  wire [3:0]   i_v_word,         // {head, dim_word[2:0]}
    input  wire [127:0] i_v_data,
    input  wire [3:0]   i_v_rd_word,
    output wire [127:0] o_v_rd_data,
    // ---------------- V per-token mantissa table [V-PERTOK-M] ----------------
    // 2 B per (head,token), exactly the K mantissa geometry (8 tokens per
    // 128-bit word, one 16-bit lane per append).  The V producer appends the
    // token's own mantissa here, and the attention's PV stage reads 8 tokens at
    // a time to scale the probabilities of those tokens (a per-token V scale
    // cannot be factored out of the PV reduction, so it has to enter the
    // activation).
    input  wire         i_vt_app_valid,
    output wire         o_vt_app_ready,
    input  wire [15:0]  i_vt_app_token,
    input  wire         i_vt_app_head,
    input  wire [15:0]  i_vt_app_mant,
    output wire         o_vt_app_taken,
    input  wire         i_vt_rd_en,          // same cycle as the P read it pairs with
    input  wire [11:0]  i_vt_rd_word,        // {head, token[13:3]}
    output wire [127:0] o_vt_rd_mant,        // 8 x 16-bit mantissa, 1 cycle after i_vt_rd_en
    output wire         o_vt_rd_valid,
    // ---------------- diagnostics -------------------------------------------
    output wire         o_clip_sticky,
    output wire [31:0]  o_stat_load,
    output wire [31:0]  o_stat_app,
    output wire [31:0]  o_stat_rd
);
    localparam integer WORDS   = (1<<(CONTEXT_LOG2-3));           // 2048 words/head (8 tokens each)
    localparam integer KWORDS  = V_HEADS*WORDS;                   // 4096 words = 64 KiB
    localparam integer VTOKWORDS = V_HEADS*WORDS;                 // [V-PERTOK-M] 4096 words
    localparam integer BLOCKS  = (1<<(CONTEXT_LOG2-BLOCK_LOG2));  // 64 blocks/head
    localparam integer ETAB_BYTES = V_HEADS*BLOCKS;               // 128 B
    localparam integer ETAB_WORDS = ETAB_BYTES/16;                // 8 words of 16 blocks
    localparam integer VWORDS  = (V_HEADS*V_DIM)/8;               // 16 words per layer

    // ---------------------------------------------------------------- ownership
    reg  [BANKS-1:0] bank_loading, bank_ready;
    reg              owner_error, clip_sticky, app_conflict;
    wire             wr_ok = bank_loading[i_bank_wr];                  // bulk load window
    wire             rd_ok = bank_ready[i_bank_rd] && !bank_loading[i_bank_rd];
    // An append may also land while the bank is READY: it writes exactly one
    // token entry (16-bit lane enabled) for the token being produced in this
    // layer, which is the last index attention reaches, so it cannot disturb a
    // read that is already in flight.  Bulk loads are restricted to LOADING.
    wire             app_ok = bank_loading[i_bank_wr] || bank_ready[i_bank_wr];
    wire        bad_write = (i_load_valid || i_vload_valid || i_etab_valid) && !wr_ok;
    wire             bad_append = i_app_valid && !app_ok;
    wire             bad_read  = i_rd_en && !rd_ok;

    assign o_wr_grant     = wr_ok;
    assign o_rd_grant     = rd_ok;
    assign o_owner_error  = owner_error;
    assign o_etab_ready   = wr_ok;
    assign o_load_ready   = wr_ok;
    assign o_vload_ready  = wr_ok;
    assign o_app_ready    = app_ok && !i_load_valid && !i_vload_valid;
    assign o_clip_sticky  = clip_sticky;
    assign o_app_conflict = app_conflict;

    // ------------------------------------------------------- read address pipe
    // Both tables are addressed by rd_addr_1 and each has two register stages
    // (memory output -> *q0 -> *q1), so mantissa and exponent reach the consumer
    // on the same cycle: address in cycle c, data in cycle c+3.
    reg  [11:0] rd_addr_1;                    // {head, token[13:3]}
    reg  [2:0]  etab_addr_1;                  // {head, block[5:4]}
    reg  [3:0]  etab_byte_1;                  // block[3:0]
    reg         rd_en_1,rd_en_2,rd_en_3;
    always @(posedge clk) begin : rd_pipe
        if (!rst_n) begin
            rd_en_1<=1'b0; rd_en_2<=1'b0; rd_en_3<=1'b0; rd_addr_1<=12'd0;
            etab_addr_1<=3'd0; etab_byte_1<=4'd0;
        end else begin
            rd_en_1   <= i_rd_en && rd_ok;
            rd_addr_1 <= i_rd_word;                       // {head, token[13:3]}
            etab_addr_1 <= {i_rd_word[11],i_rd_word[10:9]}; // {head, block[5:4]}
            etab_byte_1 <= i_rd_word[8:5];                  // block[3:0]
            rd_en_2   <= rd_en_1;
            rd_en_3   <= rd_en_2;
        end
    end

    // ------------------------------------------------------- exponent table
    // 8 words x 16 bytes = 16 blocks per word, {head, block[5:4]}.
    (* ram_style = "distributed" *) reg [127:0] etab [0:ETAB_WORDS-1];
    reg  [7:0]   etab_q0,etab_q1;
    reg  [127:0] etab_rd_q;
    always @(posedge clk) begin : etab_mem
        integer k;
        if (i_etab_valid && wr_ok) begin
            for (k=0;k<16;k=k+1)
                if (i_etab_be[k])
                    etab[i_etab_word][k*8 +: 8] <= i_etab_data[k*8 +: 8];
        end
        etab_q0 <= etab[etab_addr_1][etab_byte_1*8 +: 8];
        etab_q1 <= etab_q0;
        // [16K-ONE-LAYER] image dump: whole 16-block word, one cycle.
        etab_rd_q <= etab[i_etab_rd_word];
    end
    assign o_etab_rd_data = etab_rd_q;

    // ------------------------------------------------------------- K mantissa
    // [M16-SCALE] 8 tokens x 16-bit mantissa per word; 8 x 16-bit lane enables.
    (* ram_style = "ultra" *) reg [127:0] kmem [0:KWORDS-1];  // 2 heads x 2048 words
    reg  [127:0] kq0,kq1;
    reg  [127:0] kwe_data;
    reg  [7:0]   kwe;
    reg  [11:0]  kwe_addr;
    reg          kwe_en;
    always @(posedge clk) begin : k_mem
        integer l;
        if (kwe_en) begin
            for (l=0;l<8;l=l+1) if (kwe[l]) kmem[kwe_addr][l*16 +: 16] <= kwe_data[l*16 +: 16];
        end
        kq0 <= kmem[rd_addr_1];
        kq1 <= kq0;
    end

    assign o_rd_m16 = kq1;
    assign o_rd_eb  = etab_q1;
    assign o_rd_valid = rd_en_3;

    // ------------------------------------------------------------- write path
    reg        load_d,vload_d,app_s1,app_s2;
    reg [11:0] load_word_a,app_word_a;
    reg [127:0] load_mant_a;
    reg [15:0] app_mant_a;
    reg  [7:0]  app_exp_a;
    reg  [2:0]  app_lane_a;
    reg        app_clip_a;
    wire       load_fire  = i_load_valid && wr_ok;
    wire       vload_fire = i_vload_valid && wr_ok;
    wire       app_fire   = i_app_valid && app_ok && !i_load_valid && !i_vload_valid;
    assign     o_app_taken = app_s2;

    always @(posedge clk) begin : wr_seq
        if (!rst_n) begin
            load_d<=1'b0;vload_d<=1'b0;app_s1<=1'b0;app_s2<=1'b0;kwe_en<=1'b0;kwe<=8'd0;
            clip_sticky<=1'b0;app_conflict<=1'b0;
        end else begin
            kwe_en <= 1'b0;
            // ---- bulk load: one 128-bit word (8 tokens x m16) per cycle ----
            load_d <= load_fire;
            if (load_fire) begin
                load_word_a <= i_load_word;
                load_mant_a <= i_load_mant;
            end
            if (load_d) begin
                kwe_en   <= 1'b1;
                kwe_addr <= load_word_a;
                kwe_data <= load_mant_a;
                kwe      <= 8'hff;
            end
            // ---- single-token append: 16-bit lane write, 2 cycles ----
            app_s1 <= app_fire;
            if (app_fire) begin
                app_word_a <= {i_app_head,i_app_token[CONTEXT_LOG2-1:3]};
                app_mant_a <= i_app_mant;
                app_lane_a <= i_app_token[2:0];
                app_exp_a  <= i_app_exp;
                app_clip_a <= i_app_clip;
            end
            app_s2 <= app_s1 && !load_d;
            if (app_s1 && !load_d) begin
                kwe_en   <= 1'b1;
                kwe_addr <= app_word_a;
                kwe_data <= {8{app_mant_a}};
                kwe      <= 8'h01 << app_lane_a;
            end
            if (app_s1 && !load_d) begin
                if (app_clip_a) clip_sticky <= 1'b1;
                if (etab[{app_word_a[11],app_word_a[10:9]}][app_word_a[8:5]*8 +: 8] != app_exp_a)
                    app_conflict <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------- V table
    (* ram_style = "distributed" *) reg [127:0] vmem [0:VWORDS-1];  // 2 heads x 8 dim-words
    reg [127:0] vq;
    always @(posedge clk) begin : v_mem
        if (i_v_we && wr_ok) vmem[i_v_word] <= i_v_data;
        vq <= vmem[i_v_rd_word];
    end
    assign o_v_rd_data = vq;


    // ------------------------------------------------- V per-token mantissa
    // [V-PERTOK-M] One 16-bit lane per (head, token); the append is the same
    // shape as the K mantissa append (2 bytes, 8 tokens per 128-bit word), and
    // the read returns 8 tokens' mantissas one cycle after the address, which is
    // what the PV activation scaling needs (the P buffer read has the same
    // latency).
    (* ram_style = "ultra" *) reg [127:0] vmm [0:VTOKWORDS-1];  // 2 heads x 2048 words
    reg  [127:0] vmq;
    reg          vm_rd_v;
    reg          vmwe_en;
    reg  [11:0]  vmwe_addr;
    reg  [127:0] vmwe_data;
    reg  [7:0]   vmwe;
    reg          vt_s1, vt_s2;
    reg  [11:0]  vt_word_a;
    reg  [15:0]  vt_mant_a;
    reg  [2:0]   vt_lane_a;
    reg          vload_d2;
    reg  [11:0]  vload_word_a;
    reg  [127:0] vload_data_a;
    reg          vmwe_isload;
    wire         vt_fire = i_vt_app_valid && app_ok && !i_vload_valid;
    always @(posedge clk) begin : v_tok_mem
        integer l;
        if (!rst_n) begin
            vmwe_en <= 1'b0; vmwe <= 8'd0; vt_s1 <= 1'b0; vt_s2 <= 1'b0;
            vm_rd_v <= 1'b0; vmq <= 128'd0;
            vload_d2 <= 1'b0; vmwe_isload <= 1'b0;
        end else begin
            vmwe_en <= 1'b0;
            // ---- bulk load (one 128-bit word per cycle, 2-cycle pipe) ------
            vload_d2 <= vload_fire;
            if (vload_fire) begin
                vload_word_a <= i_vload_word;
                vload_data_a <= i_vload_data;
            end
            // ---- single-token append (one 16-bit lane, 2-cycle pipe) -------
            vt_s1 <= vt_fire;
            if (vt_fire) begin
                vt_word_a <= {i_vt_app_head, i_vt_app_token[CONTEXT_LOG2-1:3]};
                vt_mant_a <= i_vt_app_mant;
                vt_lane_a <= i_vt_app_token[2:0];
            end
            vt_s2 <= vt_s1;
            if (vt_s1) begin
                vmwe_en   <= 1'b1;
                vmwe_addr <= vt_word_a;
                vmwe_data <= {8{vt_mant_a}};
                vmwe      <= 8'h01 << vt_lane_a;
            end
            if (vload_d2) begin
                vmwe_en   <= 1'b1;
                vmwe_addr <= vload_word_a;
                vmwe_data <= vload_data_a;
                vmwe      <= 8'hff;
            end
            if (vmwe_en)
                for (l=0;l<8;l=l+1)
                    if (vmwe[l]) vmm[vmwe_addr][l*16 +: 16] <= vmwe_data[l*16 +: 16];
            vmq <= vmm[i_vt_rd_word];
            vm_rd_v <= i_vt_rd_en && rd_ok;
        end
    end
    assign o_vt_rd_mant = vmq;
    assign o_vt_rd_valid = vm_rd_v;
    assign o_vt_app_ready = app_ok && !i_load_valid && !i_vload_valid;
    assign o_vt_app_taken = vt_s2;

    // ------------------------------------------------------------- statistics
    reg [31:0] stat_load,stat_app,stat_rd;
    assign o_stat_load = stat_load;
    assign o_stat_app  = stat_app;
    assign o_stat_rd   = stat_rd;
    always @(posedge clk) begin : stats
        if (!rst_n) begin
            stat_load<=0;stat_app<=0;stat_rd<=0;
            bank_loading<={BANKS{1'b0}};bank_ready<={BANKS{1'b0}};
            owner_error<=1'b0;
        end else begin
            if (load_fire || vload_fire) stat_load <= stat_load + 1'b1;
            if (app_s2)     stat_app  <= stat_app  + 1'b1;
            if (i_rd_en && rd_ok) stat_rd <= stat_rd + 1'b1;
            if (bad_write || bad_read || bad_append) owner_error <= 1'b1;
            if (i_wr_acquire) begin
                if (bank_ready[i_bank_wr] || bank_loading[i_bank_wr]) owner_error <= 1'b1;
                bank_loading[i_bank_wr] <= 1'b1;
                bank_ready[i_bank_wr]   <= 1'b0;
            end
            if (i_wr_release) begin
                if (!bank_loading[i_bank_wr]) owner_error <= 1'b1;
                bank_loading[i_bank_wr] <= 1'b0;
                bank_ready[i_bank_wr]   <= 1'b1;
            end
            if (i_rd_release) begin
                if (!bank_ready[i_bank_rd]) owner_error <= 1'b1;
                bank_ready[i_bank_rd]   <= 1'b0;
            end
        end
    end
endmodule
`default_nettype wire
