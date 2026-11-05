`timescale 1ns / 1ps
`default_nettype none

// 一个 INCR burst 的长度规划，固定 16-byte beat，绝不越过 4 KiB 边界。
module qwen25_dma_burst (
    input wire [63:0] address,
    input wire [23:0] remaining_bytes,
    input wire [7:0] maximum_len,
    output wire [8:0] burst_beats,
    output wire [12:0] burst_bytes
);
    wire [8:0] required_beats;
    wire [8:0] boundary_beats, limit_beats, bounded_beats;
    wire [12:0] capacity_bytes;
    wire capacity_filled;
    // burst 最多 4096 bytes；高位非零时无需完整 24-bit 比较。
    assign required_beats = remaining_bytes[12:4] + (|remaining_bytes[3:0]);
    assign boundary_beats = 9'd256 - {1'b0, address[11:4]};
    assign limit_beats = {1'b0, maximum_len} + 9'd1;
    assign bounded_beats = (limit_beats < boundary_beats) ? limit_beats : boundary_beats;
    assign capacity_bytes = {bounded_beats, 4'b0};
    assign capacity_filled = (|remaining_bytes[23:13]) || (remaining_bytes[12:0] >= capacity_bytes);
    assign burst_beats = capacity_filled ? bounded_beats : required_beats;
    assign burst_bytes = capacity_filled ? capacity_bytes : remaining_bytes[12:0];
endmodule

`default_nettype wire
