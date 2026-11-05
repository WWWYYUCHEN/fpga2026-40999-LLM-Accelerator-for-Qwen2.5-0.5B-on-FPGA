`timescale 1ns / 1ps
// 固定数据流：单份真双口RAM，无bank owner状态机
module qwen25_fixed_tdp #(
    parameter DW = 128, // 两端口各自宽度
    parameter AW = 10, // 地址宽度
    parameter DEPTH = 608 // 单行有效字数
) (
    input wire clk, // 公共时钟
    input wire a_en, // 端口访问使能
    input wire a_wr, // 端口写选择
    input wire [AW-1:0] a_addr, // 端口字地址
    input wire [DW-1:0] a_data, // 端口写数据
    output wire a_valid, // 读数据有效
    output wire [DW-1:0] a_q, // 同步读取结果
    input wire b_en, // 端口访问使能
    input wire b_wr, // 端口写选择
    input wire [AW-1:0] b_addr, // 端口字地址
    input wire [DW-1:0] b_data, // 端口写数据
    output wire b_valid, // 读数据有效
    output wire [DW-1:0] b_q // 同步读取结果
);

    (* ram_style="block" *) reg [DW-1:0] memory [0:DEPTH-1]; // 真双口共享单份RAM
    reg [DW-1:0] qa,qb;
    reg va,vb;
    assign a_q=qa; assign b_q=qb; assign a_valid=va; assign b_valid=vb;
    always @(posedge clk) begin
        va<=a_en && !a_wr;
        if(a_en) begin
            if(a_wr) begin memory[a_addr]<=a_data; end
            qa<=memory[a_addr];
        end
    end
    always @(posedge clk) begin
        vb<=b_en && !b_wr;
        if(b_en) begin
            if(b_wr) begin memory[b_addr]<=b_data; end
            qb<=memory[b_addr];
        end
    end
endmodule

