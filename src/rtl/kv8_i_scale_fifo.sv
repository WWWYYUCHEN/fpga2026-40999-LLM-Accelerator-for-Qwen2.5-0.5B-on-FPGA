`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_i_scale_fifo -- the stage-3 wiring block: 16 x {block exponent, 8 x m16}
// per (KV head, 128-token tile), filled once per tile from kv8_scale_ram and
// read once per score word.
//
// [M16-SCALE] The word is 136 bit, not 128: the K scale is now a 16-bit integer
// mantissa (token-granular) plus the block exponent (256-token granular), and the
// score multiplier consumes both (kv8_score_scale8 MANT_BITS=16).  The BF16
// expander that used to live in kv8_scale_ram's read path is gone: rounding a
// 16-bit m to BF16's 7-bit significand would put a 0.4% floor under the scale.
//
// Why a buffer and not a direct RAM read
//   The score word and the scale word share the index {head, token[13:3]}, but
//   kv8_scale_ram's read port has 3 cycles of latency while the score word is
//   consumed in the cycle the GEMM result arrives (qwen25_attention.v S_G_STORE).
//   Issuing the RAM read 3 cycles early would couple the ingress to the GEMM
//   schedule, so the 16 words of a tile are fetched once, up front, and then
//   available combinationally.
//
// Why indexed, not popped
//   A 128-token tile is 16 score words, but the QK^T GEMM produces 8 query rows
//   per tile and every row walks the same 16 token words: the scale word for
//   (slice, sub) is used by all 8 rows.  A strict one-pop-per-score-word FIFO
//   would therefore need 128 entries (or would underrun on row 1).  The store is
//   read at index {slice[2:0], sub} = the token word inside the tile, which is
//   the same order the score words are produced in, so it is a FIFO in every
//   respect that matters while costing 16 x 136 bit.
//
// Fill protocol
//   i_fill_start (pulse) -> 16 read requests on the kv8_scale_ram read port,
//   addresses i_fill_base + 0..15, issued one per cycle.  the 16 returning words
//   land in mem[0..15] in order.  o_ready goes high when all 16 have arrived; the
//   consumer must not use o_scale before that.
// -----------------------------------------------------------------------------
module kv8_i_scale_fifo #(
    parameter integer WORDS = 16
) (
    input  wire         clk,
    input  wire         rst_n,
    // ---- fill side (kv8_scale_ram read port) ------------------------------
    input  wire         i_fill_start,
    input  wire [11:0]  i_fill_base,     // {head, token[13:3]} of the tile's word 0
    output wire         o_fill_en,       // 1 = drive i_fill_word to the RAM
    output wire [11:0]  o_fill_word,
    input  wire         i_fill_valid,    // RAM read data valid
    input  wire [127:0] i_fill_data,     // 8 x m16
    input  wire [7:0]   i_fill_eb,       // block exponent of that word
    output wire         o_ready,         // all WORDS words resident
    // ---- read side (one word per score word) ------------------------------
    input  wire [3:0]   i_rd_index,      // {slice[2:0], sub}
    output wire [135:0] o_scale          // {block exponent, 8 x m16}
);
    localparam integer AW = 4;           // 16 words

    (* ram_style = "distributed" *) reg [135:0] mem [0:WORDS-1];
    reg [AW:0]  iss;                     // read requests issued (0..WORDS)
    reg [AW:0]  got;                     // words received
    reg         filling;

    assign o_fill_en   = filling && (iss != WORDS);
    assign o_fill_word = i_fill_base + {{(12-AW){1'b0}}, iss[AW-1:0]};
    assign o_ready     = (got == WORDS);
    assign o_scale     = mem[i_rd_index];

    always @(posedge clk) begin
        if (!rst_n) begin
            iss <= 0; got <= 0; filling <= 1'b0;
        end else begin
            if (i_fill_start) begin
                iss     <= 0;
                got     <= 0;
                filling <= 1'b1;
            end else begin
                if (o_fill_en) iss <= iss + 1'b1;
                // [16K-FIX3] only a beat that belongs to THIS fill may be stored:
                // the 16K image sequencer drives the same read port, so its dump
                // reads also produce i_fill_valid pulses.  Ungated, they pushed
                // `got` past WORDS (measured "got=16/17" during the dump), which
                // raised o_ready on a store full of the PREVIOUS layer's scales.
                if (i_fill_valid && filling) begin
                    mem[got[AW-1:0]] <= {i_fill_eb, i_fill_data};
                    got <= got + 1'b1;
                    if (got + 1'b1 == WORDS) filling <= 1'b0;
                end
            end
        end
    end
endmodule
`default_nettype wire
