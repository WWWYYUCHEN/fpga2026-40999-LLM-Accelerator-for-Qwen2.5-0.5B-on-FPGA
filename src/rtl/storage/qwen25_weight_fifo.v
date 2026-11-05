`timescale 1ns / 1ps
`default_nettype wire
// ============================================================================
// qwen25_weight_fifo -- elastic INT8 array-load-word queue for the P x V path
//
// Same role, ports and widths as rtl/storage/qwen25_weight_fifo.v, but the
// storage format changed with the INT8 KV cache:
//   BF16 path : 1024-bit input = 64 BF16 columns (half of a 128-column row);
//               two halves were concatenated into one 2048-bit array word.
//   INT8 path : 1024-bit input = 128 INT8 columns = ONE WHOLE array word
//               (the V scale makes a 1024-bit read 128 columns, see
//               qwen25_attention.v weight read port), so no concatenation;
//               each byte is expanded to a 16-bit array slot {8'h00,code}.
// o_free_halves therefore counts *words* now (the caller uses it the same way).
// ============================================================================
module qwen25_weight_fifo (
    input i_clk,
    input i_rstn,
    input i_flush,
    input i_valid,
    output o_ready,
    output [2:0] o_free_halves,
    input [1023:0] i_data,
    input [127:0] i_meta,
    output o_valid,
    input i_ready,
    output [2047:0] o_data,
    output [127:0] o_meta,
    output o_empty,
    output o_error
);
    localparam integer DEPTH = 4;
    (* ram_style = "distributed" *) reg [1023:0] mem [0:DEPTH-1];
    (* ram_style = "distributed" *) reg [127:0]  met [0:DEPTH-1];
    reg [1:0] wr_ptr, rd_ptr;
    reg [2:0] cnt;

    wire push = i_valid && o_ready;
    wire pop  = o_valid && i_ready;

    assign o_ready       = (cnt != DEPTH[2:0]);
    assign o_valid       = (cnt != 3'd0);
    assign o_empty       = (cnt == 3'd0);
    assign o_free_halves = DEPTH[2:0] - cnt;
    assign o_meta        = met[rd_ptr];
    assign o_error       = 1'b0;

    // one array slot per INT8 column
    genvar b;
    generate
        for (b=0;b<128;b=b+1) begin : g_expand
            assign o_data[b*16 +: 16] = {8'h00, mem[rd_ptr][b*8 +: 8]};
        end
    endgenerate

    always @(posedge i_clk) begin
        if (!i_rstn) begin
            wr_ptr <= 2'd0;
            rd_ptr <= 2'd0;
            cnt    <= 3'd0;
        end else if (i_flush) begin
            wr_ptr <= 2'd0;
            rd_ptr <= 2'd0;
            cnt    <= 3'd0;
        end else begin
            if (push) begin
                mem[wr_ptr] <= i_data;
                met[wr_ptr] <= i_meta;
                wr_ptr      <= wr_ptr + 2'd1;
            end
            if (pop) rd_ptr <= rd_ptr + 2'd1;
            case ({push,pop})
                2'b10:   cnt <= cnt + 3'd1;
                2'b01:   cnt <= cnt - 3'd1;
                default: cnt <= cnt;
            endcase
        end
    end
endmodule
`default_nettype wire
