`timescale 1ns / 1ps
// 固定数据流：一写一读同步RAM，Controller负责合法地址和生命周期
module qwen25_fixed_ram #(
    parameter DW = 128, // 原生数据位宽
    parameter AW = 10, // 地址位宽
    parameter DEPTH = 608, // 有效字数
    parameter STYLE = "block" // 目标RAM
) (
    input wire clk, // 存储时钟
    input wire wr_en, // 当前生产者写使能
    input wire [AW-1:0] wr_addr, // 物理写字地址
    input wire [DW-1:0] wr_data, // 完整写数据字
    input wire rd_en, // 消费方已预留空间的读使能
    input wire [AW-1:0] rd_addr, // 物理读字地址
    output wire rd_valid, // 同步读结果有效
    output wire [DW-1:0] rd_data // 同步读数据
);

    (* ram_style=STYLE *) reg [DW-1:0] memory [0:DEPTH-1]; // 数据RAM不复位
    reg [DW-1:0] read_data; // BRAM或URAM同步读出
    reg read_valid; // 一拍读使能延迟
    assign rd_data=read_data;
    assign rd_valid=read_valid;
    always @(posedge clk) begin
        read_valid<=rd_en;
        if(wr_en) begin memory[wr_addr]<=wr_data; end
        if(rd_en) begin read_data<=memory[rd_addr]; end
    end
endmodule

