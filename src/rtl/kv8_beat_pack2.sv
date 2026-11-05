`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_beat_pack2 -- KV8 beat packer, K16 revision (stage 1 of the INT8 KV cache)
//
// What changed with the frozen K16 format
//   OLD (8-bit codes) : kv8_k_quant.o_q = 64 bit = 8 INT8 = one 8-dim HALF of a
//                       16-dim beat; this module concatenated two halves into one
//                       128-bit / 16-value beat (a 2:1 packer).
//   NEW (16-bit codes): kv8_k_quant.o_q = 128 bit = 8 x {k_hi,k_lo} = one complete
//                       128-bit beat = 8 values, which is exactly one dim_group
//                       (8 dims) of one KV head.  So the packer is 1:1 and its job
//                       is now only the one-outstanding-beat skid register plus the
//                       index ownership rule (the index latched with the beat is
//                       the one the beat is emitted with, so a drifting caller
//                       counter can never re-label data).
//
// Byte order inside the beat (frozen format, address = dim*2):
//   byte 2d   = k_lo of dim d      byte 2d+1 = k_hi of dim d
//   i.e. value d at bits [16d+15:16d] = {k_hi, k_lo}, which is what the DDR
//   little-endian word at (token*256 + head*128 + dim*2) holds.
//
// Handshake: one 128-bit beat out per 128-bit beat in, no bubble, one
// outstanding beat.  o_err_pair is kept for port compatibility and is always 0
// (there are no halves to mis-pair any more).
// -----------------------------------------------------------------------------
module kv8_beat_pack2 (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         i_flush,        // drop the held beat (new token)
    // ---- 128-bit quantiser beats (8 x 16-bit code) -------------------------
    input  wire         i_valid,
    output wire         o_ready,
    input  wire [17:0]  i_index,        // {token[13:0], head, dim_group[2:0]}
    input  wire [127:0] i_q,            // 8 x {k_hi,k_lo}
    // ---- 128-bit KV8 beats ------------------------------------------------
    output wire         o_valid,
    input  wire         i_ready,
    output wire [17:0]  o_index,
    output wire [127:0] o_data,
    // ---- status -----------------------------------------------------------
    output wire         o_err_pair      // always 0 (kept for port compatibility)
);
    reg         out_valid;
    reg [127:0] out_data;
    reg [17:0]  out_index;

    assign      o_ready  = !out_valid || i_ready;
    assign      o_valid  = out_valid;
    assign      o_data   = out_data;
    assign      o_index  = out_index;
    assign      o_err_pair = 1'b0;

    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_data  <= 128'd0;
            out_index <= 18'd0;
        end else begin
            if (i_flush) out_valid <= 1'b0;
            if (out_valid && i_ready) out_valid <= 1'b0;
            if (i_valid && o_ready) begin
                out_data  <= i_q;
                out_index <= i_index;
                out_valid <= 1'b1;
            end
        end
    end
endmodule
`default_nettype wire
