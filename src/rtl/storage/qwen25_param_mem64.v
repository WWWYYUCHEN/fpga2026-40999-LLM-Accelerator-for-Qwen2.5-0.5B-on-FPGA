`timescale 1ns / 1ps
// One independent 64-bit XPM port; scale/bias/gamma wrappers cascade this unit.
module qwen25_param_mem64 #(
    parameter integer C_DEPTH = 1024,
    parameter integer C_ADDR_W = 10,
    parameter C_STYLE = "ultra",
    parameter integer C_READ_LATENCY = 1
) (
    input wire i_clk,
    input wire i_rstn,
    input wire i_wr_en,
    input wire [C_ADDR_W-1:0] i_wr_addr,
    input wire [63:0] i_wr_data,
    input wire i_rd_en,
    input wire [C_ADDR_W-1:0] i_rd_addr,
    output wire [63:0] o_rd_data
);
    wire [63:0] wire_rd_data;
    xpm_memory_sdpram #(
        .MEMORY_SIZE(C_DEPTH*64),
        .MEMORY_PRIMITIVE(C_STYLE),
        .CLOCKING_MODE("common_clock"),
        .MEMORY_INIT_FILE("none"),
        .USE_MEM_INIT(0),
        .WRITE_DATA_WIDTH_A(64),
        .BYTE_WRITE_WIDTH_A(64),
        .ADDR_WIDTH_A(C_ADDR_W),
        .READ_DATA_WIDTH_B(64),
        .ADDR_WIDTH_B(C_ADDR_W),
        .READ_LATENCY_B(C_READ_LATENCY),
        .WRITE_MODE_B("read_first"),
        .READ_RESET_VALUE_B("0"),
        .ECC_MODE("no_ecc")
    ) memory_Inst (
        .sleep(1'b0),
        .clka(i_clk), .ena(1'b1), .wea(i_wr_en), .addra(i_wr_addr), .dina(i_wr_data),
        .injectsbiterra(1'b0), .injectdbiterra(1'b0),
        .clkb(i_clk), .rstb(1'b0), .enb(1'b1), .regceb(1'b1), .addrb(i_rd_addr),
        .doutb(wire_rd_data), .sbiterrb(), .dbiterrb()
    );
    assign o_rd_data=wire_rd_data;
endmodule
