`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// kv8_context_scale -- the V-scale application point of the INT8 KV cache.
//
// The PV reduction runs over tokens, so a per-token V scale cannot be factored
// out of the sum (KV8_16K_report.md:175-181).  The V table is one BF16 value per
// (head, dim) for the whole layer, and the only place it can be applied is the
// context write-back:
//
//     context[row][head][dim] = ( sum_t p[t] * v_int8[t][head][dim] ) * s_v[head][dim]
//
// The PV GEMM produces the sum in INT8 units; this block multiplies the 8 lanes
// of one context word by the 8 s_v values of that word, exactly the shape of
// kv8_score_scale8 (8 x BF16 multiply, 2 pipeline stages).
//
// Indexing: a context word is {reg_slice[1:0], cnt_store[0]} = dims [word*8 .. +7]
// of one query head, and the KV head is head>=7, so the V table word is
// {kv_head, context_word} -- precisely kv8_scale_ram's 16 x 128-bit V table.  The
// table read is 1 cycle, the multiply 2, so the word cost is 3 cycles; the
// downstream (qwen25_attention_dma) services one context word every ~7 cycles,
// which this structure hides completely.
//
// Handshake: one word in flight, plus one result register that may be popped on
// the same cycle a new word is accepted (so the flow-through rate is 1 per 3
// cycles).  o_ready is the accept signal the store FSM uses.
// =============================================================================
module kv8_context_scale (
    input  wire         clk,
    input  wire         rst_n,
    // ---- input side: the PV store word of qwen25_attention ----------------
    input  wire         i_valid,
    output wire         o_ready,
    input  wire [127:0] i_data,
    input  wire [2:0]   i_row,
    input  wire [3:0]   i_head,          // query head 0..13
    input  wire [2:0]   i_word,          // context word 0..7 (8 dims each)
    // ---- V scale table read port of kv8_scale_ram -------------------------
    output wire [3:0]   o_scale_word,    // {kv_head, word}; the read is free-running
    input  wire [127:0] i_scale_data,    // valid one cycle after o_scale_word
    // ---- output side: to qwen25_attention_dma -----------------------------
    output wire         o_valid,
    input  wire         i_ready,
    output wire [127:0] o_data,
    output wire [2:0]   o_row,
    output wire [3:0]   o_head,
    output wire [2:0]   o_word,
    // ---- diagnostics ------------------------------------------------------
    // [STAGE5-FIX2] one word is still inside the multiply/result pipeline
    output wire         o_pending,
    output wire         o_overflow,      // sticky: the result register was overwritten
    output wire         o_scale_zero     // sticky: a scale word read as 0 (V table
                                         // not loaded -> the context would be 0)
);
    // [STAGE5-FIX6] kv_head must be (i_head>=7), exactly like the K/V address
    // generation in qwen25_attention.v:205 (7 query heads share one KV head).
    // i_head[3] splits at 8 and sent head 7's data through KV head 0's scale.
    assign o_scale_word = {(i_head >= 4'd7), i_word};

    // ---- pipeline registers -------------------------------------------------
    reg         a_valid;                 // word offered to the multiplier
    reg [127:0] a_data;
    reg [2:0]   m_row [0:2];             // metadata delayed to the multiplier output
    reg [3:0]   m_head[0:2];
    reg [2:0]   m_word[0:2];
    reg         busy;                    // one word inside the multiplier
    reg         out_valid;
    reg [127:0] out_data;
    reg [2:0]   out_row;
    reg [3:0]   out_head;
    reg [2:0]   out_word;
    reg         overflow_r, scale_zero_r;

    wire        mult_valid;
    wire [127:0] mult_data;
    // [M16-SCALE] V stays on the legacy 8-bit BF16 path in the tree (V codes are
    // single-plane INT8 and its 0.055 error is already at the BF16 noise floor),
    // so this instance keeps MANT_BITS=8 / i_eb unused.  When V moves it must use
    // CEX = 261, NOT 268: C_K - C_V = 7 is the 1/sqrt(d), which the K table
    // already carries (K's byte is eb_r-3, V's is eb_r-7).  Putting 268 here
    // would make the context 128x too small.
    kv8_score_scale8 #(.MANT_BITS(8), .CEX(261)) mul_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_valid(a_valid), .i_score(a_data), .i_scale(i_scale_data), .i_eb(8'd0),
        .o_valid(mult_valid), .o_score(mult_data)
    );

    wire pop    = out_valid && i_ready;
    wire accept = i_valid && o_ready;
    // accept when the multiplier is empty and the result register is free (or
    // being popped this cycle: the new product is 3 cycles away, so it cannot
    // collide with the word that is leaving now)
    assign o_ready = !busy && (!out_valid || i_ready);
    assign o_valid = out_valid;
    assign o_data  = out_data;
    assign o_row   = out_row;
    assign o_head  = out_head;
    assign o_word  = out_word;
    // [STAGE5-FIX2] drain indicator for the consumer's `done`
    assign o_pending    = busy || out_valid;
    assign o_overflow   = overflow_r;
    assign o_scale_zero = scale_zero_r;

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            a_valid <= 1'b0; a_data <= 128'd0;
            busy <= 1'b0; out_valid <= 1'b0; out_data <= 128'd0;
            out_row <= 3'd0; out_head <= 4'd0; out_word <= 3'd0;
            overflow_r <= 1'b0; scale_zero_r <= 1'b0;
            for (i=0;i<3;i=i+1) begin
                m_row[i] <= 3'd0; m_head[i] <= 4'd0; m_word[i] <= 3'd0;
            end
        end else begin
            a_valid <= 1'b0;
            if (pop) out_valid <= 1'b0;
            // ---- accept ----------------------------------------------------
            if (accept) begin
                a_valid <= 1'b1;
                a_data  <= i_data;
                m_row[0] <= i_row; m_head[0] <= i_head; m_word[0] <= i_word;
                busy    <= 1'b1;
            end
            if (a_valid && i_scale_data == 128'd0) scale_zero_r <= 1'b1;
            // ---- metadata delay to the multiplier output --------------------
            m_row[1] <= m_row[0]; m_head[1] <= m_head[0]; m_word[1] <= m_word[0];
            m_row[2] <= m_row[1]; m_head[2] <= m_head[1]; m_word[2] <= m_word[1];
            // ---- multiplier output -----------------------------------------
            if (mult_valid) begin
                busy      <= 1'b0;
                out_valid <= 1'b1;
                out_data  <= mult_data;
                out_row   <= m_row[2];
                out_head  <= m_head[2];
                out_word  <= m_word[2];
                if (out_valid && !i_ready) overflow_r <= 1'b1;  // must never happen
            end
        end
    end
endmodule
`default_nettype wire
