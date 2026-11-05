`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_beat_pack2_v -- the shipped 8-bit 2:1 KV8 beat packer, kept verbatim for
// the V chain.
//
// The K chain moved to the frozen K16 format, where kv8_k_quant already emits a
// complete 128-bit beat (8 x 16-bit value) and kv8_beat_pack2 is therefore 1:1.
// V is still one INT8 plane and still needs two 64-bit quantiser beats of the
// same (token, head, dim_group) concatenated into one 128-bit / 16-value beat,
// so this module is the old packer under a V-only name.  Nothing about its
// behaviour changed; keeping it separate is what lets the two formats differ
// without a mode flag inside one module (port widths differ).
//
//   kv8_v_quant.o_q : 64 bit = 8 INT8 = one 8-dim half of a KV8 beat
//   KV8 DDR beat    : 128 bit = 16 INT8 = one 16-dim dim_group
//
//   o_data[ 7:  0] = half0 byte 0   (= dim dg*16 + 0)
//   o_data[15:  8] = half0 byte 1
//   ...
//   o_data[71: 64] = half0 byte 7
//   o_data[79: 72] = half1 byte 0   (= dim dg*16 + 8)
//   ...
//   o_data[127:120] = half1 byte 7
//
// i_index is the KV8 word index {token[13:0], head, dim_group[1:0]} of the beat
// being offered; the index of the *low* half is what gets latched, so the output
// index is self-consistent even if the caller's own counters drift.
//
// Handshake: one 128-bit beat out per two 64-bit beats in, no bubble, one
// outstanding beat (skid register).  Out-of-order halves (odd half first) are
// dropped and reported on o_err_pair (sticky) so a producer bug cannot silently
// produce a byte-swapped KV cache.
// -----------------------------------------------------------------------------
module kv8_beat_pack2_v (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         i_flush,        // drop the held low half (new token)
    // ---- 64-bit quantiser beats -------------------------------------------
    input  wire         i_valid,
    output wire         o_ready,
    input  wire [16:0]  i_index,        // {token[13:0], head, dim_group[1:0]}
    input  wire         i_half,         // 0 = low 8 dims of the group, 1 = high 8
    input  wire [63:0]  i_q,            // 8 INT8, dim = dg*16 + half*8 + byte
    // ---- 128-bit KV8 beats ------------------------------------------------
    output wire         o_valid,
    input  wire         i_ready,
    output wire [16:0]  o_index,
    output wire [127:0] o_data,
    // ---- status -----------------------------------------------------------
    output wire         o_err_pair      // sticky: odd half arrived with no low half
);
    reg         half_pend;              // low half of the current beat is held
    reg [63:0]  lo_q;
    reg [13:0]  lo_token;
    reg         lo_head;
    reg [1:0]   lo_dg;
    reg         out_valid;
    reg [127:0] out_data;
    reg [16:0]  out_index;
    reg         err_pair_r;

    wire        in_fire  = i_valid && o_ready;
    wire        low_half = ~i_half;
    // A beat can be emitted only when the output register is free.
    assign      o_ready  = !out_valid || i_ready;
    assign      o_valid  = out_valid;
    assign      o_data   = out_data;
    assign      o_index  = out_index;
    assign      o_err_pair = err_pair_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            half_pend  <= 1'b0;
            lo_q       <= 64'd0;
            lo_token   <= 14'd0;
            lo_head    <= 1'b0;
            lo_dg      <= 2'd0;
            out_valid  <= 1'b0;
            out_data   <= 128'd0;
            out_index  <= 17'd0;
            err_pair_r <= 1'b0;
        end else begin
            if (i_flush) half_pend <= 1'b0;
            if (out_valid && i_ready) out_valid <= 1'b0;
            if (in_fire) begin
                if (low_half) begin
                    lo_q     <= i_q;
                    lo_token <= i_index[16:3];
                    lo_head  <= i_index[2];
                    lo_dg    <= i_index[1:0];
                    half_pend<= 1'b1;
                end else if (half_pend &&
                             lo_token == i_index[16:3] && lo_head == i_index[2] &&
                             lo_dg    == i_index[1:0]) begin
                    // second half of the pair held in the low register
                    half_pend <= 1'b0;
                    out_data  <= {i_q, lo_q};
                    out_index <= {lo_token, lo_head, lo_dg};   // index of the low half
                    out_valid <= 1'b1;
                end else begin
                    // odd half with no matching low half: never write a mixed beat
                    half_pend  <= 1'b0;
                    err_pair_r <= 1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
