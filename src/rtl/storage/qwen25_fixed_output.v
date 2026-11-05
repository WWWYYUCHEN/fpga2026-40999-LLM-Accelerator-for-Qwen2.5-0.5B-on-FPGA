`timescale 1ns / 1ps
// 固定数据流：O四bank两pair成对AFU读取，生命周期由FFN/残差Controller控制
module qwen25_fixed_output (
    input wire clk, // O buffer时钟
    input wire rst_n, // 仅复位pair full
    input wire pair_done, // Controller确认当前pair全部写入
    input wire fill_pair, // 当前h/z或g/u pair
    input wire release_en, // Controller等待计算与两路B响应后释放
    input wire release_pair, // 被释放pair
    output wire [1:0] pair_full, // 两pair full状态
    input wire wr_en, // GEMM或AFU固定写端口
    input wire [1:0] wr_bank, // 四个O bank
    input wire [2:0] wr_row, // token行
    input wire [6:0] wr_word, // 896列内128bit字
    input wire [127:0] wr_data, // 八lane BF16
    output wire wr_ready, // 不同bank允许同拍写；同bank AFU优先
    input wire afu_wr_en, // 独立AFU写请求
    input wire [1:0] afu_wr_bank, // AFU目标bank
    input wire [2:0] afu_wr_row, // AFU token行
    input wire [6:0] afu_wr_word, // AFU八元素字地址
    input wire [127:0] afu_wr_data, // AFU八lane结果
    input wire rd_en, // AFU成对读取或DDR写回
    input wire rd_pair, // 消费pair
    input wire [2:0] rd_row, // token行
    input wire [6:0] rd_word, // 896列内128bit字
    output wire rd_valid, // 同步读取有效
    output wire [127:0] rd_x, // pair第一个bank
    output wire [127:0] rd_y // pair第二个bank
);

    reg [1:0] full_bits; reg selected_pair; wire [511:0] bank_q; wire [3:0] val;
    // Capture requests before row arithmetic and the physical BRAM port mux.
    // Data is never reset; read responses and ownership events shift by one clock.
    reg gemm_request,afu_request,read_request;
    reg [1:0] gemm_bank,afu_bank;
    reg [2:0] gemm_row,afu_row,read_row;
    reg [6:0] gemm_word,afu_word,read_word;
    reg [127:0] gemm_data,afu_data;
    reg read_pair;
    reg pair_done_pending,fill_pair_pending,release_pending,release_pair_pending;
    wire [9:0] write_address=gemm_row*10'd112+gemm_word;
    wire [9:0] read_address=read_row*10'd112+read_word;
    wire [9:0] afu_write_address=afu_row*10'd112+afu_word;
    genvar b;
    assign pair_full=full_bits; assign rd_valid=|val;
    assign wr_ready=!(afu_wr_en && wr_bank==afu_wr_bank);
    assign rd_x=selected_pair ? bank_q[383:256] : bank_q[127:0];
    assign rd_y=selected_pair ? bank_q[511:384] : bank_q[255:128];
    always @(posedge clk) begin
        gemm_request<=wr_en && wr_ready;
        afu_request<=afu_wr_en;
        read_request<=rd_en;
        if(wr_en && wr_ready)begin
            gemm_bank<=wr_bank;gemm_row<=wr_row;gemm_word<=wr_word;gemm_data<=wr_data;
        end
        if(afu_wr_en)begin
            afu_bank<=afu_wr_bank;afu_row<=afu_wr_row;afu_word<=afu_wr_word;afu_data<=afu_wr_data;
        end
        if(rd_en)begin read_pair<=rd_pair;read_row<=rd_row;read_word<=rd_word;end
        pair_done_pending<=pair_done;
        release_pending<=release_en;
        if(pair_done)fill_pair_pending<=fill_pair;
        if(release_en)release_pair_pending<=release_pair;
        if(!rst_n) begin full_bits<=0; end
        else begin
            if(release_pending) begin full_bits[release_pair_pending]<=0; end
            if(pair_done_pending) begin full_bits[fill_pair_pending]<=1; end
        end
        if(read_request) begin selected_pair<=read_pair; end
    end
    generate
        for(b=0;b<4;b=b+1) begin:gen_bank
            wire select_afu=afu_request && afu_bank==b;
            wire select_gemm=gemm_request && gemm_bank==b;
            qwen25_fixed_ram #(.DW(128),.AW(10),.DEPTH(896),.STYLE("block")) bank_Inst(
                .clk(clk),.wr_en(select_afu || select_gemm),.wr_addr(select_afu ? afu_write_address : write_address),.wr_data(select_afu ? afu_data : gemm_data),
                .rd_en(read_request && read_pair==(b/2)),.rd_addr(read_address),.rd_valid(val[b]),.rd_data(bank_q[b*128+:128])
            );
        end
    endgenerate
endmodule
