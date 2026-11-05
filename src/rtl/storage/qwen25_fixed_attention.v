`timescale 1ns / 1ps
// 固定数据流：P两bank八乘32768 BF16，Softmax固定写P乘V固定读，logits复用物理地址
module qwen25_fixed_attention (
    input wire clk, // buffer时钟
    input wire rst_n, // 复位bank写入阶段和行统计有效状态
    input wire fill_done, // Score流水排空后，将目标bank交给AFU
    input wire fill_bank, // 完成填充的bank
    input wire release_bank, // Controller确认消费完成的bank
    input wire release_en, // P消费结束，将目标bank交还GEMM
    output wire [1:0] full, // 两bank独立full状态
    input wire wr_en, // GEMM Score或logits写请求
    input wire wr_bank, // 写目标P bank
    input wire [14:0] wr_address, // P或logits线性128bit地址
    input wire [127:0] wr_data, // 八lane BF16
    output wire wr_ready, // 只由目标bank的GEMM阶段决定
    input wire afu_wr_en, // Softmax结果独立写请求
    input wire afu_wr_bank, // Softmax所属pingpong bank
    input wire [14:0] afu_wr_address, // Softmax行和八元素字地址
    input wire [127:0] afu_wr_data, // Softmax八lane BF16结果
    input wire rd_en, // P乘V或spill消费
    input wire rd_bank, // 消费bank
    input wire [11:0] rd_word, // context八元素字
    output wire rd_valid, // 同步P tile读有效
    output wire [1023:0] rd_tile, // 八query行同context列字
    input wire afu_rd_en, // Softmax独立单行读取
    input wire afu_rd_bank, // 当前score/P bank
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

    // Each ping-pong bank has one registered write owner. Score drain transfers
    // ownership to AFU; PV release returns it to GEMM. The other bank is independent.
    localparam [1:0] S_GEMM=2'd1,S_AFU=2'd2;
    wire [1:0] bank_gemm,bank_afu;
    assign full=bank_afu;
    assign wr_ready=bank_gemm[wr_bank];
    genvar owner_bank;
    generate
        for(owner_bank=0;owner_bank<2;owner_bank=owner_bank+1)begin:gen_write_owner
            reg [1:0] state_current,state_next;
            reg gemm_enabled,afu_enabled;
            always @(posedge clk)begin
                if(!rst_n)state_current<=S_GEMM;
                else state_current<=state_next;
            end
            always @(*)begin
                state_next=state_current;
                case(state_current)
                    S_GEMM:begin
                        if(fill_done && fill_bank==owner_bank)state_next=S_AFU;
                    end
                    S_AFU:begin
                        if(release_en && release_bank==owner_bank)state_next=S_GEMM;
                        if(fill_done && fill_bank==owner_bank)state_next=S_AFU;
                    end
                    default:state_next=S_GEMM;
                endcase
            end
            always @(*)begin
                gemm_enabled=1'b0;afu_enabled=1'b0;
                case(state_current)
                    S_GEMM:gemm_enabled=1'b1;
                    S_AFU:afu_enabled=1'b1;
                    default:begin end
                endcase
            end
            assign bank_gemm[owner_bank]=gemm_enabled;
            assign bank_afu[owner_bank]=afu_enabled;
        end
    endgenerate

    // Statistics live beside each physical buffer row, independent of AFU P writes.
    reg [15:0] row_maximum[0:15];
    reg [31:0] row_max_index[0:15];
    reg [15:0] row_words[0:15];
    reg [15:0] row_started,row_complete;
    wire [3:0] max_write_slot={wr_bank,max_wr_row};
    wire [3:0] max_read_slot={afu_rd_bank,afu_rd_row};
    wire max_fire=wr_en && wr_ready && max_wr_en;
    wire [15:0] history_value=max_wr_first ? 16'hff80 : row_maximum[max_write_slot];
    // Normal Score rows have at most 4096 words. The fixed 18992-word logits
    // stream reuses this same comparator and winner register with latest ties.
    wire greedy_latest=max_wr_words==16'd18992;
    wire [31:0] history_index=max_wr_first ? 32'hffffffff :
        (greedy_latest ? ~row_max_index[max_write_slot] : row_max_index[max_write_slot]);
    wire [15:0] merged_value;
    wire [31:0] merged_index,compare_index;
    wire [31:0] incoming_index=greedy_latest ? ~max_wr_index : max_wr_index;
    qwen25_bf16_argmax row_compare_Inst (
        .i_a(history_value),.i_a_index(history_index),
        .i_b(max_wr_value),.i_b_index(incoming_index),.o_value(merged_value),.o_index(compare_index)
    );
    assign merged_index=greedy_latest ? ~compare_index : compare_index;
    assign afu_row_maximum=row_maximum[max_read_slot];
    assign afu_row_max_index=row_max_index[max_read_slot];
    assign afu_row_complete=row_complete[max_read_slot];
    genvar stat_row;
    generate
        for(stat_row=0;stat_row<16;stat_row=stat_row+1)begin:gen_statistics
            always @(posedge clk)begin
                if(!rst_n)begin
                    row_started[stat_row]<=0;row_complete[stat_row]<=0;
                end else if(release_en && release_bank==stat_row/8)begin
                    row_started[stat_row]<=0;row_complete[stat_row]<=0;
                end else if(max_fire && max_write_slot==stat_row)begin
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

    wire [2047:0] bank_q; wire [1:0] bank_valid; reg selected_bank; genvar b,r;
    // 在单行/整tile地址选择前寄存完整AFU请求；行统计仍使用实时选择。
    reg afu_read_pending;
    reg afu_read_bank;
    reg [2:0] afu_read_row;
    reg [11:0] afu_read_word;
    reg selected_afu_bank; reg [2:0] selected_afu_row; reg afu_valid; // 与单行RAM输出对齐
    assign afu_rd_valid=afu_valid;
    assign afu_rd_data=bank_q[(selected_afu_bank*8+selected_afu_row)*128+:128];
    always @(posedge clk)begin
        if(!rst_n)begin
            afu_read_pending<=1'b0;
            afu_valid<=1'b0;
        end else begin
            afu_read_pending<=afu_rd_en;
            afu_valid<=afu_read_pending;
        end
        // 无效周期允许载入地址，避免在地址寄存器前增加使能MUX。
        afu_read_bank<=afu_rd_bank;
        afu_read_row<=afu_rd_row;
        afu_read_word<=afu_rd_word;
        if(afu_read_pending)begin selected_afu_bank<=afu_read_bank;selected_afu_row<=afu_read_row;end
    end
    assign rd_tile=selected_bank ? bank_q[2047:1024] : bank_q[1023:0]; assign rd_valid=|bank_valid;
    always @(posedge clk) begin if(rd_en) begin selected_bank<=rd_bank; end end
    generate
        for(b=0;b<2;b=b+1) begin:gen_bank
            wire [7:0] val; assign bank_valid[b]=&val;
            for(r=0;r<8;r=r+1) begin:gen_row
                wire select_afu=bank_afu[b] && afu_wr_en && afu_wr_bank==b && afu_wr_address[14:12]==r;
                wire select_gemm=bank_gemm[b] && wr_en && wr_bank==b && wr_address[14:12]==r;
                qwen25_fixed_ram #(.DW(128),.AW(12),.DEPTH(4096),.STYLE("ultra")) row_Inst(
                    .clk(clk),.wr_en(select_afu || select_gemm),
                    .wr_addr(bank_afu[b] ? afu_wr_address[11:0] : wr_address[11:0]),.wr_data(bank_afu[b] ? afu_wr_data : wr_data),
                    .rd_en((rd_en && rd_bank==b)||(afu_read_pending && afu_read_bank==b && afu_read_row==r)),
                    .rd_addr(afu_read_pending && afu_read_bank==b ? afu_read_word : rd_word),.rd_valid(val[r]),.rd_data(bank_q[b*1024+r*128+:128])
                );
            end
        end
    endgenerate
endmodule
