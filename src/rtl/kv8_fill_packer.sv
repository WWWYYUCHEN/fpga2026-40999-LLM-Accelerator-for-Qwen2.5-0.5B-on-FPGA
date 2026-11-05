`timescale 1ns/1ps
`default_nettype none
// ============================================================================
// kv8_fill_packer -- INT8 K^T / V fill packer for the shared 64x128 operand bank
//
// Target: rtl/storage/qwen25_fixed_weight.v (2 banks x 64 rows x 256 columns,
// 256-bit load port, 1024-bit read port) reused *without modification*, with the
// 256-bit load slot reinterpreted as 32 INT8 columns instead of 16 BF16 columns.
//
// [K16] Two plane writes, one DDR read
//   A K16 beat is 8 values of 16 bit: byte 2d = k_lo, byte 2d+1 = k_hi.  One
//   collect phase therefore yields TWO 32-token column words per dim -- the hi
//   plane and the lo plane -- and they are written into the two *seg* halves of
//   the same row (seg is load_word[2], i.e. columns 128..255).  The 128-token K
//   tile only ever uses seg=0, so seg=1 is free storage with exactly the right
//   geometry: a 1024-bit read at seg=1 returns the lo plane of the same
//   128 tokens in the same column order.  No bank is consumed, no capacity is
//   added, and the attention selects the plane with read address bit 0.
//
// Buffer physical model used here (identical slicing to the existing module):
//   write : 256 bit at load_word = {1'b0, row[5:0], seg[2], lane[1:0]}
//           -> memory[{bank,row}] of slice (seg[2], lane)
//   read  : 1024 bit at gemm_word = {1'b0, row[5:0], seg[2]}
//           -> 4 slices (lane 0..3) of one seg half = 128 consecutive columns
//   column index inside a row = seg[2]*128 + lane*32 + byte   (0..255)
//   a 128-token tile uses seg[2]=0 (hi plane), the K16 lo plane uses seg[2]=1.
//
// Mapping (the contract the INT8 GEMM array relies on):
//   K^T  : row = dim = {dim_group[2:0], trans_dim[2:0]}   (8 dims per beat),
//          column = token index inside the tile, seg = plane
//   V    : row = token[5:0] (64-token tile), column = {kv_head, dim[5:0]},
//          seg = 0
//
// Beat index: {token[13:0], head, dim_group[2:0]} (K) / {token[6:0], head,
// dim_group[1:0]} (V), zero-padded to the 18-bit port.
//
// Beat order contract (see report section 6):
//   K^T : dim_group outer, token ascending inner, 32 tokens per group,
//         the 32 tokens of a group must be 32-aligned and ascending.
//   V   : token ascending outer, {head, dim_group} inner (pure stream order).
// ============================================================================
module kv8_fill_packer #(
    parameter integer MODE_KV    = 0,   // 0 = K^T (transpose), 1 = V (in place)
    parameter integer FAULT_MODE = 0    // 0 = correct; 1..4 = planted faults (TB only)
) (
    input  wire         i_clk,
    input  wire         i_rstn,
    // ---- INT8 beat stream (KV8 DDR layout) ---------------------------------
    input  wire         i_beat_valid,
    output wire         o_beat_ready,
    input  wire [17:0]  i_beat_index,   // K: {token[13:0],head,dg[2:0]}
    input  wire [127:0] i_beat_data,    // K16: {k_hi,k_lo} per value, 8 values
    input  wire         i_head_sel,     // K^T: KV head held by the bank
    input  wire         i_bank,         // tile ownership (ping/pong)
    // ---- 256-bit word port to qwen25_fixed_weight --------------------------
    output wire         o_wr_valid,
    input  wire         i_wr_ready,
    output wire         o_wr_bank,
    output wire [9:0]   o_wr_word,      // {1'b0, row[5:0], seg[2], lane[1:0]}
    output wire [255:0] o_wr_data,      // byte b = column lane*32 + b
    // ---- status ------------------------------------------------------------
    output wire         o_err_head,     // sticky: beat of a non-selected head dropped
    output wire         o_err_seq,      // sticky: beat order violated the contract
    output wire         o_busy
);
    localparam integer TOKENS_PER_WORD = 32;   // 256 bit / 8 bit
    localparam integer DIMS_PER_BEAT   = 8;    // [K16] 8 values of 16 bit per beat

    // [K16] the two modes pack different fields into the same 18-bit index.
    wire [13:0] beat_token = (MODE_KV == 0) ? i_beat_index[17:4] : {7'd0,i_beat_index[9:3]};
    wire        beat_head  = (MODE_KV == 0) ? i_beat_index[3]   : i_beat_index[2];
    wire [2:0]  beat_dg    = (MODE_KV == 0) ? i_beat_index[2:0] : {1'b0,i_beat_index[1:0]};
    wire [4:0]  beat_tok   = beat_token[4:0];   // byte lane inside a 32-token word
    wire        beat_dg_lo = beat_dg[0];

    // K^T: one KV head per bank (rows are dims, no head field in the address).
    // V  : both KV heads share a row, distinguished by the column field.
    wire beat_head_ok = (MODE_KV == 0) ? (beat_head == i_head_sel) : 1'b1;
    wire beat_bad_order;

    reg err_head_r, err_seq_r;
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            err_head_r <= 1'b0;
            err_seq_r  <= 1'b0;
        end else begin
            if (i_beat_valid && o_beat_ready && !beat_head_ok) err_head_r <= 1'b1;
            if (i_beat_valid && o_beat_ready &&  beat_head_ok && beat_bad_order) err_seq_r <= 1'b1;
        end
    end
    assign o_err_head = err_head_r;
    assign o_err_seq  = err_seq_r;
    assign o_wr_bank  = i_bank;

    generate
    // =======================================================================
    // K^T : 32 x 8 x 2 planes -> 16 dim rows (8 hi + 8 lo) of 32 tokens
    //   stage_q[p][j][b] = plane p, dim (dg*8+j), token (group base + b)
    //   write word e (0..15): plane = e[3], dim j = e[2:0],
    //                         row = {dg, j}, seg = plane
    // =======================================================================
    if (MODE_KV == 0) begin : g_kt
        localparam [1:0] KT_COLLECT = 2'd0, KT_EMIT = 2'd1;

        reg [7:0] stage_hi [0:DIMS_PER_BEAT-1][0:TOKENS_PER_WORD-1];
        reg [7:0] stage_lo [0:DIMS_PER_BEAT-1][0:TOKENS_PER_WORD-1];
        reg [1:0] kt_state;
        reg [4:0] kt_cnt;
        reg [2:0] kt_dg_q, kt_lane_q;
        reg       kt_half_q;
        reg [3:0] kt_emit;

        wire kt_stage_fire = i_beat_valid && o_beat_ready && beat_head_ok;

        // ---- one write port per byte lane (token), shared by all dims ----
        genvar gb;
        for (gb = 0; gb < TOKENS_PER_WORD; gb = gb + 1) begin : g_lane
            integer j;
            always @(posedge i_clk) begin
                if (kt_stage_fire && (beat_tok == gb)) begin
                    for (j = 0; j < DIMS_PER_BEAT; j = j + 1) begin
                        stage_lo[j][gb] <= i_beat_data[j*16 +: 8];
                        stage_hi[j][gb] <= i_beat_data[j*16+8 +: 8];
                    end
                end
            end
        end

        // ---- emit word mux (16:1 over {plane, dim}) ------------------------
        wire        e_plane = kt_emit[3];
        wire [2:0]  e_dim   = kt_emit[2:0];
        reg [255:0] kt_word;
        integer kb;
        always @(*) begin
            for (kb = 0; kb < TOKENS_PER_WORD; kb = kb + 1)
                kt_word[kb*8 +: 8] = e_plane ? stage_hi[e_dim][kb] : stage_lo[e_dim][kb];
        end

        // ---- row / lane, with the planted faults of the TB -----------------
        // FAULT 1 : row = {td, dg} (bijective permutation, "silently permuted")
        // FAULT 2 : row = dg*8 + td (the legacy 8-dim-group formula: collides)
        // FAULT 3 : lane = ~lane (column block permutation, bijective)
        wire [5:0] kt_row =
            (FAULT_MODE == 1) ? {e_dim, kt_dg_q} :
            (FAULT_MODE == 2) ? ({kt_dg_q, 3'b000} + {3'b000, e_dim}) :
                                {kt_dg_q, e_dim};
        wire [2:0] kt_lane = (FAULT_MODE == 3) ? ~kt_lane_q : kt_lane_q;

        assign o_beat_ready   = (kt_state == KT_COLLECT);
        assign o_wr_valid     = (kt_state == KT_EMIT);
        assign o_wr_data      = kt_word;
        assign o_wr_word      = {1'b0, kt_row, e_plane, kt_lane[1:0]};
        assign o_busy         = (kt_state != KT_COLLECT) || (kt_cnt != 5'd0);
        assign beat_bad_order = (beat_tok != kt_cnt);

        always @(posedge i_clk) begin
            if (!i_rstn) begin
                kt_state  <= KT_COLLECT;
                kt_cnt    <= 5'd0;
                kt_emit   <= 4'd0;
                kt_dg_q   <= 3'd0;
                kt_lane_q <= 3'd0;
                kt_half_q <= 1'b0;
            end else begin
                case (kt_state)
                KT_COLLECT: if (kt_stage_fire) begin
                    kt_cnt <= beat_tok + 5'd1;
                    if (beat_tok == 5'd31) begin
                        kt_state  <= KT_EMIT;
                        kt_emit   <= 4'd0;
                        kt_dg_q   <= beat_dg;
                        kt_lane_q <= beat_token[6:5];
                        kt_half_q <= beat_token[7];
                    end
                end
                KT_EMIT: if (i_wr_ready) begin
                    if (kt_emit == 4'd15) begin
                        kt_state <= KT_COLLECT;
                        kt_cnt   <= 5'd0;
                    end else begin
                        kt_emit <= kt_emit + 4'd1;
                    end
                end
                default: kt_state <= KT_COLLECT;
                endcase
            end
        end
    end
    // =======================================================================
    // V : straight copy, 2 beats (dim_group 2g, 2g+1) -> one 32-column word
    //   word = {beat(dg odd), beat(dg even)}, row = token[5:0],
    //   lane = {head, dim_group[1]}
    // =======================================================================
    else begin : g_v
        reg [127:0] v_lo;          // first beat of the pair
        reg         v_half_pend;
        reg [255:0] v_word_q;      // complete word waiting for the write port
        reg         v_word_pend;
        reg  [1:0]  v_lane_q;
        reg  [5:0]  v_row_q;

        wire v_store   = i_beat_valid && o_beat_ready && !beat_dg_lo;
        wire v_pair    = i_beat_valid && o_beat_ready &&  beat_dg_lo;
        wire v_wr_fire = v_word_pend && i_wr_ready;

        // FAULT 4 : halves swapped inside the 32-column word
        wire [255:0] v_word = (FAULT_MODE == 4) ? {v_lo, i_beat_data}
                                               : {i_beat_data, v_lo};

        assign o_beat_ready   = (!v_word_pend || !beat_dg_lo);
        assign o_wr_valid     = v_word_pend;
        assign o_wr_data      = v_word_q;
        assign o_wr_word      = {1'b0, v_row_q, 1'b0, v_lane_q};
        assign o_busy         = v_word_pend || v_half_pend;
        assign beat_bad_order = (v_store && v_half_pend) || (v_pair && !v_half_pend);

        always @(posedge i_clk) begin
            if (!i_rstn) begin
                v_lo        <= 128'd0;
                v_half_pend <= 1'b0;
                v_word_q    <= 256'd0;
                v_word_pend <= 1'b0;
                v_lane_q    <= 2'd0;
                v_row_q     <= 6'd0;
            end else begin
                if (v_store) begin
                    v_lo        <= i_beat_data;
                    v_half_pend <= 1'b1;
                    v_row_q     <= beat_token[5:0];
                    v_lane_q    <= {beat_head, beat_dg[1]};
                end
                if (v_pair) begin
                    v_word_q    <= v_word;
                    v_word_pend <= 1'b1;
                    v_half_pend <= 1'b0;
                end else if (v_wr_fire) begin
                    v_word_pend <= 1'b0;
                end
            end
        end
    end
    endgenerate
endmodule
`default_nettype wire
