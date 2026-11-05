`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_layout -- INT8 KV-cache DDR layout for a 16K-token context.
//
// Compared with the BF16 layout in rtl/attention/qwen25_attention.v:227-229
// (token stride 256 B, head stride 128 B, 8 dims per 16 B beat, 8 MiB per
// layer), every stride here is a power of two, so the whole intra-layer
// address is a field concatenation and needs no adder:
//
// [K16]/[V16] BOTH K and V are 16 bit per value = {hi,lo}, one 16-bit word per
// dim, so the two images have the SAME geometry (the address formula is shared):
//   K byte address = k_base + layer*4MiB + token*256 + head*128 + dim*2
//   V byte address = v_base + layer*4MiB + token*256 + head*128 + dim*2
//   128-bit word (= 16 B beat, 8 values) inside a layer
//                  = {token[13:0], head, dim_group[2:0]}
//                  = token*16 + head*8 + dim_group       (0 .. 262143)
//
// Per-layer sizes at CONTEXT_LOG2=14 (16384 tokens), 8 MiB layer window:
//   K = 16384*2*64*2 B = 4 MiB      V = 4 MiB      K+V = 8 MiB  (exactly full)
//   24 layers: K 96 MiB, V 96 MiB, total 192 MiB.
//   K and V each own a window strided by 8 MiB from their OWN descriptor base
//   (k_base != v_base), so the two 4 MiB images never share an address: the 8 MiB
//   code sum is per layer-pair, not per window.  No stride change is needed.
//   beats per token = 16 K + 16 V (2 heads x 8 groups of 8 values = 256 B each),
//   so a 128-token K tile needs 1024 beats for the single KV head in flight.
//
// Scale region (DDR side; the on-chip table mirrors one layer at a time):
//   K mantissa 1 B per (head,token) = 32768 B per layer
//       byte  = k_scale_base + layer*32768 + head*16384 + token
//       word  = k_scale_base + layer*32768 + head*16384 + token[13:3]*8  (8 tokens)
//   K block exponent 1 B per (head, 256-token block) = 128 B per layer
//       byte  = k_bexp_base + layer*128 + head*64 + token[13:8]
//       k_bexp_base = k_scale_base + 24*32768
//   V scale BF16 per (head,dim) = 256 B per layer
//       byte  = v_scale_base + layer*256 + head*128 + dim*2
//   scale region total = 24*32768 + 24*128 + 24*256 = 786432+3072+6144 B
//                      = 795,648 B (0.76 MiB)
// -----------------------------------------------------------------------------
module kv8_layout #(
    parameter integer CONTEXT_LOG2 = 14,        // 16384 tokens
    parameter integer KV_HEADS = 2,
    parameter integer F_HEAD_DIM = 64,
    parameter integer LAYERS = 24,
    parameter integer SCALE_BLOCK_LOG2 = 8      // 256-token block exponent
) (
    input  wire [4:0]  i_layer,
    input  wire [15:0] i_token,
    input  wire        i_head,
    input  wire [2:0]  i_dim_group,             // K: 8 values (16 b) per 16 B beat
    input  wire [63:0] i_k_base,
    input  wire [63:0] i_v_base,
    output wire [63:0] o_k_address,             // byte address of the 16 B beat
    output wire [63:0] o_v_address,
    output wire [21:0] o_intra_offset,          // byte offset inside the layer (22 b)
    output wire [17:0] o_word_index,            // 128-bit word / beat index
    // exported per-layer geometry (controller + testbench cross-check)
    output wire [23:0] o_layer_bytes,
    output wire [18:0] o_layer_words,
    output wire [4:0]  o_beats_per_token,
    output wire [7:0]  o_tiles_per_layer,       // 16384 / 128
    // K scale table addresses (the on-chip table mirrors this image)
    input  wire [63:0] i_k_scale_base,
    output wire [63:0] o_scale_byte_address,    // mantissa byte of (head,token)
    output wire [63:0] o_scale_word_address,    // 8-token 64-bit read word
    output wire [11:0] o_scale_word_in_layer,
    output wire [63:0] o_bexp_byte_address,
    output wire [6:0]  o_blocks_per_head,       // 64
    // V scale table address (BF16 per (head,dim), 2 B per entry)
    input  wire [63:0] i_v_scale_base,
    output wire [63:0] o_vscale_byte_address    // entry at the 16-dim group base
);
    // [V16] both images are 2 B/value (4 MiB per layer); the layer window is the
    // DDR map's 8 MiB stride, and each image fills half of its own window.
    localparam integer K_TOKEN_BYTES = KV_HEADS*F_HEAD_DIM*2;          // 256
    localparam integer V_TOKEN_BYTES = KV_HEADS*F_HEAD_DIM*2;          // 256
    localparam integer LAYER_LOG2 = CONTEXT_LOG2 + 8;                  // 22 -> 4 MiB
    localparam integer LAYER_BYTES = (1<<LAYER_LOG2);                  // 4 MiB
    localparam integer BEATS_PER_TOKEN = K_TOKEN_BYTES/16;             // 16
    localparam integer BLOCK_TOKENS = (1<<SCALE_BLOCK_LOG2);           // 256
    localparam integer SCALE_BYTES = (1<<CONTEXT_LOG2)*KV_HEADS;       // 32768
    localparam integer SCALE_WORDS = SCALE_BYTES/8;                    // 4096
    localparam integer BEXP_BYTES = KV_HEADS*((1<<CONTEXT_LOG2)/BLOCK_TOKENS); //128
    localparam integer VSCALE_BYTES = KV_HEADS*F_HEAD_DIM*2;           // 256
    localparam integer SCALE_LOG2 = (CONTEXT_LOG2+1);                  // 15 -> 32 KiB
    localparam integer SCALE_HEAD_LOG2 = CONTEXT_LOG2;                 // 14 -> 16 KiB

    // Intra-layer byte offset of a 16 B beat: 14+1+3+4 = 22 bits, no adder.
    assign o_intra_offset = {i_token[CONTEXT_LOG2-1:0], i_head, i_dim_group, 4'b0000};
    // 128-bit word index: the beat index, 14+1+3 = 18 bits.
    assign o_word_index = {i_token[CONTEXT_LOG2-1:0], i_head, i_dim_group};

    wire [63:0] kv_layer_off = {{(64-5){1'b0}}, i_layer} << LAYER_LOG2;
    assign o_k_address = i_k_base + kv_layer_off + {{(64-22){1'b0}}, o_intra_offset};
    assign o_v_address = i_v_base + kv_layer_off + {{(64-22){1'b0}}, o_intra_offset};

    assign o_layer_bytes = LAYER_BYTES[23:0];
    assign o_layer_words = LAYER_BYTES/16;
    assign o_beats_per_token = BEATS_PER_TOKEN;
    assign o_tiles_per_layer = ((1<<CONTEXT_LOG2)/128);

    // ---- K scale mantissa table -------------------------------------------
    wire [63:0] scale_layer_off = {{(64-5){1'b0}}, i_layer} << SCALE_LOG2;
    wire [63:0] scale_head_off  = {{(64-1){1'b0}}, i_head}  << SCALE_HEAD_LOG2;
    assign o_scale_byte_address = i_k_scale_base + scale_layer_off + scale_head_off
                                + {{(64-16){1'b0}}, i_token};
    assign o_scale_word_in_layer = {i_head, i_token[CONTEXT_LOG2-1:3]};
    assign o_scale_word_address = i_k_scale_base + scale_layer_off + scale_head_off
                                + {{(64-16){1'b0}}, i_token[CONTEXT_LOG2-1:3], 3'b000};

    // ---- K block exponent table (after all mantissa tables) ----------------
    assign o_bexp_byte_address = i_k_scale_base + LAYERS*SCALE_BYTES
        + ({{(64-5){1'b0}}, i_layer} << 7)          // 128 B per layer
        + ({{(64-1){1'b0}}, i_head}  << 6)          // 64 B per head
        + {{(64-7){1'b0}}, i_token[CONTEXT_LOG2-1:SCALE_BLOCK_LOG2]};
    assign o_blocks_per_head = ((1<<CONTEXT_LOG2)/BLOCK_TOKENS);

    // ---- V scale table ----------------------------------------------------
    assign o_vscale_byte_address = i_v_scale_base
        + ({{(64-5){1'b0}}, i_layer} << 8)          // 256 B per layer
        + ({{(64-1){1'b0}}, i_head}  << 7)          // 128 B per (layer,head)
        + {{(64-8){1'b0}}, i_dim_group[1:0], 5'd0};  // 32 B per 16-dim group

endmodule
`default_nettype wire
