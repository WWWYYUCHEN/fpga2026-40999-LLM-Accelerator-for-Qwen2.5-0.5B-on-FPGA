`timescale 1ns / 1ps
`default_nettype none

// 命令合法性在任何 AXI VALID 呈现之前验证，DDR 地址高位不截断。
module qwen25_dma_cmd_check (
    input wire [127:0] command,
    input wire [127:0] metadata,
    output wire legal
);
    wire [64:0] range_end;
    wire address_ok, identity_ok, shape_ok;
    assign range_end = {1'b0, command[63:0]} + {41'b0, command[87:64]} - 65'd1;
    assign address_ok = (command[3:0] == 0) && (command[87:64] != 0) && !range_end[64];
    assign identity_ok = (command[127:119] == 0) && (metadata[127:119] == 0) &&
                         (command[115:112] < 8) && (command[117:116] == metadata[78:77]);
    assign shape_ok = (metadata[20:16] < 24) && (metadata[56:53] >= 1) &&
                      (metadata[56:53] <= 8) && (metadata[60:57] <= 8) &&
                      (metadata[76:61] <= 32768) && (metadata[87:84] < 14) &&
                      (metadata[108:106] < 7) && (metadata[118:109] < 896);
    assign legal = address_ok && identity_ok && shape_ok;
endmodule

`default_nettype wire
