`timescale 1ns / 1ps
// 固定数据流：独立projection双bank，Controller安排GEMM写与后处理读
module qwen25_fixed_projection (
    input wire clk, // buffer时钟
    input wire rst_n, // 仅复位full状态
    input wire fill_done, // Controller确认全部写入完成
    input wire fill_bank, // 完成填充的bank
    input wire release_bank, // Controller确认消费完成的bank
    input wire release_en, // 消费结束清full
    output wire [1:0] full, // 两bank独立full状态
    input wire wr_en, // GEMM slice适配或bias写回
    input wire wr_bank, // 写目标bank
    input wire [2:0] wr_row, // token行
    input wire [3:0] wr_word, // 128列内八lane字
    input wire [127:0] wr_data, // 八lane BF16
    input wire rd_en, // 后处理或DDR读使能
    input wire rd_bank, // 消费bank
    input wire [2:0] rd_row, // 消费token行
    input wire [3:0] rd_word, // 消费列字
    output wire rd_valid, // 同步读有效
    output wire [127:0] rd_data // 八lane输出
);

    reg [1:0] full_bits;
    assign full=full_bits;
    always @(posedge clk) begin
        if(!rst_n) begin full_bits<=0; end
        else begin
            if(release_en) begin full_bits[release_bank]<=0; end
            if(fill_done) begin full_bits[fill_bank]<=1; end
        end
    end

    wire [255:0] bank_q; wire [1:0] bank_valid; reg selected_bank; genvar b;
    assign rd_data=selected_bank ? bank_q[255:128] : bank_q[127:0]; assign rd_valid=|bank_valid;
    always @(posedge clk) begin if(rd_en) begin selected_bank<=rd_bank; end end
    generate
        for(b=0;b<2;b=b+1) begin:gen_bank
            qwen25_fixed_ram #(.DW(128),.AW(7),.DEPTH(128),.STYLE("block")) bank_Inst(
                .clk(clk),.wr_en(wr_en && wr_bank==b),.wr_addr({wr_row,wr_word}),.wr_data(wr_data),
                .rd_en(rd_en && rd_bank==b),.rd_addr({rd_row,rd_word}),.rd_valid(bank_valid[b]),.rd_data(bank_q[b*128+:128])
            );
        end
    endgenerate
endmodule

