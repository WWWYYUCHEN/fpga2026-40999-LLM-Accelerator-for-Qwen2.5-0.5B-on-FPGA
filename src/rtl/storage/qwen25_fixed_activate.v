`timescale 1ns / 1ps
`default_nettype none

// Activate: one 4864-word RAM, one 128-bit write and one 128-bit read.
// Activate：token向量紧凑存储，按列组依次读取有效token后组成GEMM tile。
module qwen25_fixed_activate (
    input wire clk, // 所有访问使用同一时钟
    input wire rst_n, // 低有效同步复位事务状态，数据RAM不复位
    input wire wide_layout, // 1：M步长608；0：归一化输入步长112及Context暂存区
    input wire vec_en, // 向量请求
    input wire vec_wr, // 向量写使能；向量读保留但未使用
    input wire [2:0] vec_row, // 写入token编号
    input wire [9:0] vec_word, // M为0..607，归一化输入0..111，Context别名112..127
    input wire [127:0] vec_data, // 一个token的连续八个BF16
    output wire vec_valid, // 向量读未启用，固定0
    output wire [127:0] vec_q, // 向量读未启用，固定0
    input wire tile_en, // 读tile或写Context请求；读请求仅在空闲时提出
    input wire tile_wr, // Context写使能
    input wire [2:0] tile_row, // 写token编号，或读取的起始token编号
    input wire [3:0] tile_rows, // 本次读取的有效token数1..8；起始行加行数不超过8
    input wire [9:0] tile_word, // 同一列组的逻辑字地址
    input wire [127:0] tile_data, // Context写入的八个BF16
    output wire tile_valid, // 所有有效token已收集完成，持续一拍
    output wire [1023:0] tile_q // 每个token占128bit，未读取行置零
);

    // 窄布局中，Z占地址0..895，Context占896..1023；M阶段复用整块RAM。
    wire vector_context = !wide_layout && (vec_word[9:4] == 6'd7);
    wire tile_context = !wide_layout && (tile_word[9:4] == 6'd7);
    wire [12:0] tile_base = wide_layout ? tile_row * 13'd608 :
        (tile_context ? 13'd896 + {6'd0,tile_row,4'd0} : tile_row * 13'd112);
    wire [12:0] tile_offset = tile_context ? {9'd0,tile_word[3:0]} : {3'd0,tile_word};
    wire vector_write = vec_en && vec_wr;
    wire context_write = tile_en && tile_wr;

    // Capture only the raw address fields at the producer boundary. The
    // registered fields feed a narrow address decoder before the RAM write
    // edge, keeping producer/control selection out of the address arithmetic.
    reg [2:0] write_row;
    reg [9:0] write_word;
    reg write_wide_layout;
    wire write_context = !write_wide_layout && (write_word[9:4] == 6'd7);
    reg [8:0] selected_write_row_base;
    always @(*) begin
        case ({write_wide_layout, write_context, write_row})
            5'b10000: selected_write_row_base = 9'd0;
            5'b10001: selected_write_row_base = 9'd38;
            5'b10010: selected_write_row_base = 9'd76;
            5'b10011: selected_write_row_base = 9'd114;
            5'b10100: selected_write_row_base = 9'd152;
            5'b10101: selected_write_row_base = 9'd190;
            5'b10110: selected_write_row_base = 9'd228;
            5'b10111: selected_write_row_base = 9'd266;
            5'b01000: selected_write_row_base = 9'd56;
            5'b01001: selected_write_row_base = 9'd57;
            5'b01010: selected_write_row_base = 9'd58;
            5'b01011: selected_write_row_base = 9'd59;
            5'b01100: selected_write_row_base = 9'd60;
            5'b01101: selected_write_row_base = 9'd61;
            5'b01110: selected_write_row_base = 9'd62;
            5'b01111: selected_write_row_base = 9'd63;
            5'b00000: selected_write_row_base = 9'd0;
            5'b00001: selected_write_row_base = 9'd7;
            5'b00010: selected_write_row_base = 9'd14;
            5'b00011: selected_write_row_base = 9'd21;
            5'b00100: selected_write_row_base = 9'd28;
            5'b00101: selected_write_row_base = 9'd35;
            5'b00110: selected_write_row_base = 9'd42;
            5'b00111: selected_write_row_base = 9'd49;
            default: selected_write_row_base = 9'd0;
        endcase
    end
    wire [5:0] selected_write_word_upper = write_context ? 6'd0 : write_word[9:4];
    wire [8:0] selected_write_address_upper = selected_write_row_base +
        {3'd0, selected_write_word_upper};
    wire [12:0] write_address = {selected_write_address_upper, write_word[3:0]};

    // 写请求先寄存，下一上升沿完成128bit完整字写入，保持原写入吞吐。
    reg write_pending;
    reg [127:0] write_data;

    // 读口每拍处理一个token，地址按阶段步长递推。
    // 读请求先在RAM外壳入口寄存一次；这样上游行数/状态解码只驱动
    // 请求寄存器，不再和物理读地址寄存器处在同一条组合路径上。
    reg read_request_pending;
    reg [2:0] read_request_row;
    reg [9:0] read_request_word;
    reg read_request_wide_layout;
    reg read_request_context;
    reg [12:0] read_request_stride;
    reg [3:0] read_request_remaining;
    wire [12:0] read_request_base = read_request_wide_layout ?
        read_request_row * 13'd608 :
        (read_request_context ? 13'd896 + {6'd0,read_request_row,4'd0} :
        read_request_row * 13'd112);
    wire [12:0] read_request_offset = read_request_context ?
        {9'd0,read_request_word[3:0]} : {3'd0,read_request_word};
    reg read_busy;
    reg read_active;
    reg [12:0] read_address;
    reg [12:0] read_stride;
    reg [3:0] read_remaining;
    reg [2:0] issue_row;
    reg [2:0] response_row;
    reg response_last;
    wire read_start = rst_n && tile_en && !tile_wr && !read_busy && !read_request_pending;
    wire read_launch = rst_n && read_request_pending && !read_busy;
    wire memory_read_valid;
    wire [127:0] memory_read_data;
    reg tile_valid_o;

    assign vec_valid = 1'b0;
    assign vec_q = 128'b0;
    assign tile_valid = tile_valid_o;

    always @(posedge clk) begin
        write_row <= context_write ? tile_row : vec_row;
        write_word <= context_write ? tile_word : vec_word;
        write_wide_layout <= wide_layout;
        if (!rst_n) begin
            write_pending <= 1'b0;
        end else begin
            write_pending <= vector_write || context_write;
            if (context_write || vector_write) begin
                write_data <= context_write ? tile_data : vec_data;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            read_request_pending <= 1'b0;
            read_request_row <= 3'd0;
            read_request_word <= 10'd0;
            read_request_wide_layout <= 1'b0;
            read_request_context <= 1'b0;
            read_request_stride <= 13'd0;
            read_request_remaining <= 4'd0;
            read_busy <= 1'b0;
            read_active <= 1'b0;
            read_address <= 13'd0;
            read_stride <= 13'd0;
            read_remaining <= 4'd0;
            issue_row <= 3'd0;
            response_row <= 3'd0;
            response_last <= 1'b0;
            tile_valid_o <= 1'b0;
        end else begin
            tile_valid_o <= 1'b0;
            if (read_start) begin
                read_request_pending <= 1'b1;
                read_request_row <= tile_row;
                read_request_word <= tile_word;
                read_request_wide_layout <= wide_layout;
                read_request_context <= tile_context;
                read_request_stride <= wide_layout ? 13'd608 : (tile_context ? 13'd16 : 13'd112);
                read_request_remaining <= tile_rows;
            end
            if (read_launch) begin
                read_request_pending <= 1'b0;
                read_busy <= 1'b1;
                read_active <= 1'b1;
                read_address <= read_request_base + read_request_offset;
                read_stride <= read_request_stride;
                read_remaining <= read_request_remaining;
                issue_row <= read_request_row;
            end
            if (read_active) begin
                response_row <= issue_row;
                response_last <= (read_remaining == 4'd1);
                if (read_remaining == 4'd1) begin
                    read_active <= 1'b0;
                end else begin
                    read_address <= read_address + read_stride;
                    issue_row <= issue_row + 3'd1;
                    read_remaining <= read_remaining - 4'd1;
                end
            end
            if (memory_read_valid && response_last) begin
                read_busy <= 1'b0;
                tile_valid_o <= 1'b1;
            end
        end
    end

    // 不增加物理读口：1024bit只是在RAM后收集有效token的输出寄存器。
    genvar token_row;
    generate
        for (token_row=0; token_row<8; token_row=token_row+1) begin : gen_tile_row
            reg [127:0] collected_data;
            assign tile_q[token_row*128+:128] = collected_data;
            always @(posedge clk) begin
                if (read_start) begin
                    collected_data <= 128'd0;
                end else if (memory_read_valid && response_row == token_row && rst_n) begin
                    collected_data <= memory_read_data;
                end
            end
        end
    endgenerate

    qwen25_fixed_ram #(.DW(128),.AW(13),.DEPTH(4864),.STYLE("block")) memory_Inst (
        .clk(clk),
        .wr_en(write_pending && rst_n),.wr_addr(write_address),.wr_data(write_data),
        .rd_en(read_active && rst_n),.rd_addr(read_address),
        .rd_valid(memory_read_valid),.rd_data(memory_read_data)
    );
endmodule
`default_nettype wire
