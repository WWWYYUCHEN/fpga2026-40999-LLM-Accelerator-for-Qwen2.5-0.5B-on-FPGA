`timescale 1ns / 1ps
// 固定数据流：P一块八乘32768 BF16，Softmax固定写P乘V固定读。
// [P16K] 行深4096字保持不动：URAM288按位宽（72bit/端口）而不是按深度分配，
//        2048×128仍然是2块URAM288，压深度一块也省不下来（实测见报告）。
// [P16K] LM-head logits流改走stat_only：只喂行最大/argmax，绝不写RAM——
//        logits流没有任何读出方，写进去纯属占用。
// [P16K] 原来的两bank乒乓是冗余的（见下方owner_afu处的说明），删掉整个bank，
//        实例数16→8，P的URAM从32块降到16块。
module qwen25_fixed_attention (
    input wire clk, // buffer时钟
    input wire rst_n, // 复位P块归属和行统计有效状态
    input wire fill_done, // Score流水排空后，将P交给AFU
    input wire release_en, // P×V消费完成，将P交还GEMM
    output wire full, // P块归AFU所有（Softmax/P×V阶段）
    input wire wr_en, // GEMM Score写请求
    input wire [14:0] wr_address, // P行字地址：行[2:0]+行内128bit字[11:0]
    input wire [127:0] wr_data, // 八lane BF16
    input wire stat_only, // 1：本流只做统计不落盘（LM-head logits）
    output wire wr_ready, // 写口授予：stat_only恒接受
    input wire afu_wr_en, // Softmax结果独立写请求
    input wire [14:0] afu_wr_address, // Softmax行和八元素字地址
    input wire [127:0] afu_wr_data, // Softmax八lane BF16结果
    input wire rd_en, // P乘V消费
    input wire [11:0] rd_word, // context八元素字
    output wire rd_valid, // 同步P tile读有效
    output wire [1023:0] rd_tile, // 八query行同context列字
    input wire afu_rd_en, // Softmax独立单行读取
    input wire [2:0] afu_rd_row, // Softmax当前query行
    input wire [11:0] afu_rd_word, // 行内连续八个score的字地址
    output wire afu_rd_valid, // 单行请求寄存后同步读取有效
    output wire [127:0] afu_rd_data, // 同一行八列BF16
    input wire max_wr_en,
    input wire [2:0] max_wr_row,
    input wire [15:0] max_wr_value,
    input wire [31:0] max_wr_index,
    input wire max_wr_first,
    input wire max_wr_last,
    input wire [15:0] max_wr_words,
    output wire [15:0] afu_row_maximum,
    output wire [31:0] afu_row_max_index,
    output wire afu_row_complete
);

    // [P16K] 两bank乒乓是冗余的：attention的FSM按head严格串行（score GEMM→
    // Normalize→P×V→下一head），任何时刻只有一个bank被寻址，另一个bank在
    // 整个head周期里既不读也不写。因此这里只保留一块8行P，URAM从32块降到16块。
    // 归属寄存器：Score填充完成交给AFU，P×V消费完交还GEMM（fill_done优先）。
    reg owner_afu;
    assign full=owner_afu;
    always @(posedge clk)begin
        if(!rst_n)owner_afu<=1'b0;
        else if(fill_done)owner_afu<=1'b1;
        else if(release_en)owner_afu<=1'b0;
    end

    // 统计流与写口解耦：stat_only时写入被丢弃，但握手照常完成，
    // 因此行argmax仍然逐字推进（LM-head的greedy比较器就挂在这条通路上）。
    wire wr_accept=stat_only || !owner_afu;
    assign wr_ready=wr_accept;

    // Statistics live beside each physical buffer row, independent of AFU P writes.
    reg [15:0] row_maximum[0:7];
    reg [31:0] row_max_index[0:7];
    reg [15:0] row_words[0:7];
    reg [7:0] row_started,row_complete;
    wire max_fire=max_wr_en && wr_accept;
    wire [15:0] history_value=max_wr_first ? 16'hff80 : row_maximum[max_wr_row];
    // Normal Score rows have at most 2048 words. The fixed 18992-word logits
    // stream reuses this same comparator and winner register with latest ties.
    wire greedy_latest=max_wr_words==16'd18992;
    wire [31:0] history_index=max_wr_first ? 32'hffffffff :
        (greedy_latest ? ~row_max_index[max_wr_row] : row_max_index[max_wr_row]);
    wire [15:0] merged_value;
    wire [31:0] merged_index,compare_index;
    wire [31:0] incoming_index=greedy_latest ? ~max_wr_index : max_wr_index;
    qwen25_bf16_argmax row_compare_Inst (
        .i_a(history_value),.i_a_index(history_index),
        .i_b(max_wr_value),.i_b_index(incoming_index),.o_value(merged_value),.o_index(compare_index)
    );
    assign merged_index=greedy_latest ? ~compare_index : compare_index;
    assign afu_row_maximum=row_maximum[afu_rd_row];
    assign afu_row_max_index=row_max_index[afu_rd_row];
    assign afu_row_complete=row_complete[afu_rd_row];
    genvar stat_row;
    generate
        for(stat_row=0;stat_row<8;stat_row=stat_row+1)begin:gen_statistics
            always @(posedge clk)begin
                if(!rst_n)begin
                    row_started[stat_row]<=0;row_complete[stat_row]<=0;
                end else if(release_en)begin
                    row_started[stat_row]<=0;row_complete[stat_row]<=0;
                end else if(max_fire && max_wr_row==stat_row)begin
                    row_maximum[stat_row]<=merged_value;
                    row_max_index[stat_row]<=merged_index;
                    row_words[stat_row]<=max_wr_first ? 16'd1 : row_words[stat_row]+1'b1;
                    if(max_wr_first)row_started[stat_row]<=1;
                    row_complete[stat_row]<=max_wr_last &&
                        (max_wr_first ? max_wr_words==1 :
                        (row_started[stat_row] && row_words[stat_row]+16'd1==max_wr_words));
                end
            end
        end
    endgenerate

    wire [1023:0] bank_q; wire [7:0] bank_valid;
    // 在单行/整tile地址选择前寄存完整AFU请求；行统计仍使用实时选择。
    reg afu_read_pending;
    reg [2:0] afu_read_row;
    reg [11:0] afu_read_word;
    reg [2:0] selected_afu_row; reg afu_valid; // 与单行RAM输出对齐
    assign afu_rd_valid=afu_valid;
    assign afu_rd_data=bank_q[selected_afu_row*128+:128];
    always @(posedge clk)begin
        if(!rst_n)begin
            afu_read_pending<=1'b0;
            afu_valid<=1'b0;
        end else begin
            afu_read_pending<=afu_rd_en;
            afu_valid<=afu_read_pending;
        end
        // 无效周期允许载入地址，避免在地址寄存器前增加使能MUX。
        afu_read_row<=afu_rd_row;
        afu_read_word<=afu_rd_word;
        if(afu_read_pending)selected_afu_row<=afu_read_row;
    end
    assign rd_tile=bank_q; assign rd_valid=|bank_valid;
    genvar row_sel;
    generate
        for(row_sel=0;row_sel<8;row_sel=row_sel+1)begin:gen_row
            wire select_afu=afu_wr_en && afu_wr_address[14:12]==row_sel;
            // stat_only的流绝不落盘：logits没有任何读出方。
            wire select_gemm=wr_en && !stat_only && !owner_afu && wr_address[14:12]==row_sel;
            qwen25_fixed_ram #(.DW(128),.AW(12),.DEPTH(4096),.STYLE("ultra")) row_Inst(
                .clk(clk),.wr_en(select_afu || select_gemm),
                .wr_addr(owner_afu ? afu_wr_address[11:0] : wr_address[11:0]),
                .wr_data(owner_afu ? afu_wr_data : wr_data),
                .rd_en(rd_en || (afu_read_pending && afu_read_row==row_sel)),
                .rd_addr(afu_read_pending ? afu_read_word : rd_word),
                .rd_valid(bank_valid[row_sel]),.rd_data(bank_q[row_sel*128+:128])
            );
        end
    endgenerate
endmodule
