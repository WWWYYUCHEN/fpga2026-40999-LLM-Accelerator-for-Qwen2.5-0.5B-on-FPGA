`timescale 1ns / 1ps
// 固定数据流：独立固定端口张量与参数存储总封装，无共享读写仲裁
module qwen25_tensor_buffers #(parameter DIRECT_WEIGHT_PORTS=0) (
    input wire clk, // 所有独立buffer同钟域
    input wire rst_n, // 只复位full和参数有效位
    input wire act_wide_layout, // Activate地址步长：1为608字，0为112字及Context区
    input wire act_vec_en, // DMA或AFU向量访问
    input wire act_vec_wr, // DMA或AFU写回
    input wire [2:0] act_vec_row, // 单个token行
    input wire [9:0] act_vec_word, // 每行八元素字0到607
    input wire [127:0] act_vec_data, // 八路BF16
    output wire act_vec_valid, // AFU向量读有效
    output wire [127:0] act_vec_q, // AFU八lane读取
    input wire act_tile_en, // GEMM tile读取或Context写
    input wire act_tile_wr, // P乘V写Context区域
    input wire [2:0] act_tile_row, // Context写回行
    input wire [3:0] act_tile_rows, // Activate读取的有效token数
    input wire [9:0] act_tile_word, // GEMM同列字或Context别名112起
    input wire [127:0] act_tile_data, // Context八lane写数据
    output wire act_tile_valid, // GEMM读取tile有效
    output wire [1023:0] act_tile_q, // 八token各八元素BF16
    input wire weight_fill_done, // Controller确认全部写入完成
    input wire weight_fill_bank, // 完成填充的bank
    input wire weight_release_bank, // Controller确认消费完成的bank
    input wire weight_release_en, // 消费结束清full
    output wire [1:0] weight_full, // 两bank独立full状态
    input wire weight_load_valid, // 解码权重或K转置V loader写有效
    output wire weight_load_ready, // 目标bank未full
    input wire weight_load_bank, // 当前填充bank
    input wire [9:0] weight_load_word, // 256bit字地址
    input wire [255:0] weight_load_data, // 十六BF16
    input wire [3:0] weight_direct_valid, output wire [3:0] weight_direct_ready,
    input wire weight_direct_bank, input wire [31:0] weight_direct_word,
    input wire [1023:0] weight_direct_data,
    input wire weight_gemm_rd_en, // GEMM读取1024bit半行
    input wire weight_gemm_bank, // 当前计算bank
    input wire [7:0] weight_gemm_word, // 1024bit字地址
    output wire weight_gemm_valid, // 半行读取有效
    output wire [1023:0] weight_gemm_q, // 连续64列BF16
    input wire weight_gemm_rd_en_pair, // GEMM第二个1024bit读口
    input wire weight_gemm_bank_pair, // 第二个读口bank
    input wire [7:0] weight_gemm_word_pair, // 第二个读口地址
    output wire weight_gemm_valid_pair, // 第二个读口有效
    output wire [1023:0] weight_gemm_q_pair, // 第二个读口数据
    input wire scale_load_start, // independent scale load_start
    input wire scale_load_valid, // independent scale load_valid
    output wire scale_load_ready, // independent scale load_ready
    input wire [127:0] scale_load_data, // independent scale load_data
    input wire scale_load_last, // independent scale load_last
    output wire scale_load_done, // independent scale load_done
    output wire scale_loaded, // independent scale loaded
    input wire scale_read_en, // independent scale read_en
    input wire [14:0] scale_read_addr, // independent scale read_addr
    output wire scale_read_valid, // independent scale read_valid
    output wire [255:0] scale_read_data, // independent scale read_data
    input wire bias_load_start, // independent bias load_start
    input wire bias_load_valid, // independent bias load_valid
    output wire bias_load_ready, // independent bias load_ready
    input wire [127:0] bias_load_data, // independent bias load_data
    input wire bias_load_last, // independent bias load_last
    output wire bias_load_done, // independent bias load_done
    output wire bias_loaded, // independent bias loaded
    input wire bias_read_en, // independent bias read_en
    input wire [11:0] bias_read_addr, // independent bias read_addr
    output wire bias_read_valid, // independent bias read_valid
    output wire [127:0] bias_read_data, // independent bias read_data
    input wire gamma_load_start, // independent gamma load_start
    input wire gamma_load_valid, // independent gamma load_valid
    output wire gamma_load_ready, // independent gamma load_ready
    input wire [127:0] gamma_load_data, // independent gamma load_data
    input wire gamma_load_last, // independent gamma load_last
    output wire gamma_load_done, // independent gamma load_done
    output wire gamma_loaded, // independent gamma loaded
    input wire gamma_read_en, // independent gamma read_en
    input wire [12:0] gamma_read_addr, // independent gamma read_addr
    output wire gamma_read_valid, // independent gamma read_valid
    output wire [127:0] gamma_read_data, // independent gamma read_data
    input wire proj0_fill_done, // Controller确认全部写入完成
    input wire proj0_fill_bank, // 完成填充的bank
    input wire proj0_release_bank, // Controller确认消费完成的bank
    input wire proj0_release_en, // 消费结束清full
    output wire [1:0] proj0_full, // 两bank独立full状态
    input wire proj0_wr_en, // GEMM slice适配或bias写回
    input wire proj0_wr_bank, // 写目标bank
    input wire [2:0] proj0_wr_row, // token行
    input wire [3:0] proj0_wr_word, // 128列内八lane字
    input wire [127:0] proj0_wr_data, // 八lane BF16
    input wire proj0_rd_en, // 后处理或DDR读使能
    input wire proj0_rd_bank, // 消费bank
    input wire [2:0] proj0_rd_row, // 消费token行
    input wire [3:0] proj0_rd_word, // 消费列字
    output wire proj0_rd_valid, // 同步读有效
    output wire [127:0] proj0_rd_data, // 八lane输出
    input wire proj1_fill_done, // Controller确认全部写入完成
    input wire proj1_fill_bank, // 完成填充的bank
    input wire proj1_release_bank, // Controller确认消费完成的bank
    input wire proj1_release_en, // 消费结束清full
    output wire [1:0] proj1_full, // 两bank独立full状态
    input wire proj1_wr_en, // GEMM slice适配或bias写回
    input wire proj1_wr_bank, // 写目标bank
    input wire [2:0] proj1_wr_row, // token行
    input wire [3:0] proj1_wr_word, // 128列内八lane字
    input wire [127:0] proj1_wr_data, // 八lane BF16
    input wire proj1_rd_en, // 后处理或DDR读使能
    input wire proj1_rd_bank, // 消费bank
    input wire [2:0] proj1_rd_row, // 消费token行
    input wire [3:0] proj1_rd_word, // 消费列字
    output wire proj1_rd_valid, // 同步读有效
    output wire [127:0] proj1_rd_data, // 八lane输出
    input wire attention_fill_done, // Controller确认全部写入完成
    input wire attention_fill_bank, // 完成填充的bank
    input wire attention_release_bank, // Controller确认消费完成的bank
    input wire attention_release_en, // 消费结束清full
    output wire [1:0] attention_full, // 两bank独立full状态
    input wire attention_wr_en, // Softmax或spill reload写
    input wire attention_wr_bank, // 写目标P bank
    input wire [14:0] attention_wr_address, // P或logits线性128bit地址
    input wire [127:0] attention_wr_data, // 八lane BF16
    output wire attention_wr_ready, // GEMM score写口是否接受
    input wire attention_afu_wr_en, // AFU Softmax独立写请求
    input wire attention_afu_wr_bank, // Softmax所属bank
    input wire [14:0] attention_afu_wr_address, // Softmax八lane字地址
    input wire [127:0] attention_afu_wr_data, // Softmax P写数据
    input wire attention_rd_en, // P乘V或spill消费
    input wire attention_rd_bank, // 消费bank
    input wire [11:0] attention_rd_word, // context八元素字
    output wire attention_rd_valid, // 同步P tile读有效
    output wire [1023:0] attention_rd_tile, // 八query行同context列字
    input wire attention_afu_rd_en, // Softmax单行读取使能
    input wire attention_afu_rd_bank, // Softmax当前物理bank
    input wire [2:0] attention_afu_rd_row, // Softmax query行选择
    input wire [11:0] attention_afu_rd_word, // 行内八元素字地址
    output wire attention_afu_rd_valid, // 实际同步读取有效
    output wire [127:0] attention_afu_rd_data, // AFU同一行八个score
    input wire attention_max_wr_en,
    input wire [2:0] attention_max_wr_row,
    input wire [15:0] attention_max_wr_value,
    input wire [31:0] attention_max_wr_index,
    input wire attention_max_wr_first,
    input wire attention_max_wr_last,
    input wire [15:0] attention_max_wr_words,
    output wire [15:0] attention_row_maximum,
    output wire [31:0] attention_row_max_index,
    output wire attention_row_complete,
    input wire output_pair_done, // Controller确认当前pair全部写入
    input wire output_fill_pair, // 当前h/z或g/u pair
    input wire output_release_en, // Controller等待计算与两路B响应后释放
    input wire output_release_pair, // 被释放pair
    output wire [1:0] output_pair_full, // 两pair full状态
    input wire output_wr_en, // GEMM或AFU固定写端口
    input wire [1:0] output_wr_bank, // 四个O bank
    input wire [2:0] output_wr_row, // token行
    input wire [6:0] output_wr_word, // 896列内128bit字
    input wire [127:0] output_wr_data, // 八lane BF16
    output wire output_wr_ready, // 主生产者目标bank的写口授予
    input wire output_afu_wr_en, // AFU独立写请求
    input wire [1:0] output_afu_wr_bank, // AFU目标bank
    input wire [2:0] output_afu_wr_row, // AFU token行
    input wire [6:0] output_afu_wr_word, // AFU字地址
    input wire [127:0] output_afu_wr_data, // AFU八lane结果
    input wire output_rd_en, // AFU成对读取或DDR写回
    input wire output_rd_pair, // 消费pair
    input wire [2:0] output_rd_row, // token行
    input wire [6:0] output_rd_word, // 896列内128bit字
    output wire output_rd_valid, // 同步读取有效
    output wire [127:0] output_rd_x, // pair第一个bank
    output wire [127:0] output_rd_y // pair第二个bank
);

    // 一块4864×128bit RAM，单读单写，逐token收集GEMM tile。
    qwen25_fixed_activate activate_Inst (
        .clk(clk), // Activate时钟
        .rst_n(rst_n), // 只复位读写事务状态
        .wide_layout(act_wide_layout), // 896维与4864维地址步长选择
        .vec_en(act_vec_en), // DMA或AFU向量访问
        .vec_wr(act_vec_wr), // DMA或AFU写回
        .vec_row(act_vec_row), // 单个token行
        .vec_word(act_vec_word), // 每行八元素字0到607
        .vec_data(act_vec_data), // 八路BF16
        .vec_valid(act_vec_valid), // AFU向量读有效
        .vec_q(act_vec_q), // AFU八lane读取
        .tile_en(act_tile_en), // GEMM tile读取或Context写
        .tile_wr(act_tile_wr), // P乘V写Context区域
        .tile_row(act_tile_row), // Context写回行
        .tile_rows(act_tile_rows), // 正常GEMM读取有效行，LM只读取选定行
        .tile_word(act_tile_word), // GEMM同列字或Context别名112起
        .tile_data(act_tile_data), // Context八lane写数据
        .tile_valid(act_tile_valid), // GEMM读取tile有效
        .tile_q(act_tile_q) // 八token各八元素BF16
    );

    // Weight/K转置/V两bank，固定256写1024读，只有full造成装载背压
    qwen25_fixed_weight #(.DIRECT_PORTS(DIRECT_WEIGHT_PORTS)) weight_Inst (
        .clk(clk), // buffer时钟
        .rst_n(rst_n), // 仅复位full状态
        .fill_done(weight_fill_done), // Controller确认全部写入完成
        .fill_bank(weight_fill_bank), // 完成填充的bank
        .release_bank(weight_release_bank), // Controller确认消费完成的bank
        .release_en(weight_release_en), // 消费结束清full
        .full(weight_full), // 两bank独立full状态
        .load_valid(weight_load_valid), // 解码权重或K转置V loader写有效
        .load_ready(weight_load_ready), // 目标bank未full
        .load_bank(weight_load_bank), // 当前填充bank
        .load_word(weight_load_word), // 256bit字地址
        .load_data(weight_load_data), // 十六BF16
        .direct_valid(weight_direct_valid),.direct_ready(weight_direct_ready),
        .direct_bank(weight_direct_bank),.direct_word(weight_direct_word),.direct_data(weight_direct_data),
        .gemm_rd_en(weight_gemm_rd_en), // GEMM读取1024bit半行
        .gemm_bank(weight_gemm_bank), // 当前计算bank
        .gemm_word(weight_gemm_word), // 1024bit字地址
        .gemm_valid(weight_gemm_valid), // 半行读取有效
        .gemm_q(weight_gemm_q), // 连续64列BF16
        .gemm_rd_en_pair(weight_gemm_rd_en_pair),.gemm_bank_pair(weight_gemm_bank_pair),
        .gemm_word_pair(weight_gemm_word_pair),.gemm_valid_pair(weight_gemm_valid_pair),
        .gemm_q_pair(weight_gemm_q_pair)
    );

    // Scale, bias and gamma have independent load and read ports.
    qwen25_fixed_scale scale_Inst (
        .clk(clk), .rst_n(rst_n),
        .load_start(scale_load_start), .load_valid(scale_load_valid), .load_ready(scale_load_ready),
        .load_data(scale_load_data), .load_last(scale_load_last),
        .load_done(scale_load_done), .loaded(scale_loaded),
        .read_en(scale_read_en), .read_addr(scale_read_addr),
        .read_valid(scale_read_valid), .read_data(scale_read_data)
    );
    qwen25_fixed_bias bias_Inst (
        .clk(clk), .rst_n(rst_n),
        .load_start(bias_load_start), .load_valid(bias_load_valid), .load_ready(bias_load_ready),
        .load_data(bias_load_data), .load_last(bias_load_last),
        .load_done(bias_load_done), .loaded(bias_loaded),
        .read_en(bias_read_en), .read_addr(bias_read_addr),
        .read_valid(bias_read_valid), .read_data(bias_read_data)
    );
    qwen25_fixed_gamma gamma_Inst (
        .clk(clk), .rst_n(rst_n),
        .load_start(gamma_load_start), .load_valid(gamma_load_valid), .load_ready(gamma_load_ready),
        .load_data(gamma_load_data), .load_last(gamma_load_last),
        .load_done(gamma_load_done), .loaded(gamma_loaded),
        .read_en(gamma_read_en), .read_addr(gamma_read_addr),
        .read_valid(gamma_read_valid), .read_data(gamma_read_data)
    );

    // 独立projection双bank，Controller安排GEMM写与后处理读
    qwen25_fixed_projection proj_buf0_Inst (
        .clk(clk), // buffer时钟
        .rst_n(rst_n), // 仅复位full状态
        .fill_done(proj0_fill_done), // Controller确认全部写入完成
        .fill_bank(proj0_fill_bank), // 完成填充的bank
        .release_bank(proj0_release_bank), // Controller确认消费完成的bank
        .release_en(proj0_release_en), // 消费结束清full
        .full(proj0_full), // 两bank独立full状态
        .wr_en(proj0_wr_en), // GEMM slice适配或bias写回
        .wr_bank(proj0_wr_bank), // 写目标bank
        .wr_row(proj0_wr_row), // token行
        .wr_word(proj0_wr_word), // 128列内八lane字
        .wr_data(proj0_wr_data), // 八lane BF16
        .rd_en(proj0_rd_en), // 后处理或DDR读使能
        .rd_bank(proj0_rd_bank), // 消费bank
        .rd_row(proj0_rd_row), // 消费token行
        .rd_word(proj0_rd_word), // 消费列字
        .rd_valid(proj0_rd_valid), // 同步读有效
        .rd_data(proj0_rd_data) // 八lane输出
    );

    // 独立projection双bank，Controller安排GEMM写与后处理读
    qwen25_fixed_projection proj_buf1_Inst (
        .clk(clk), // buffer时钟
        .rst_n(rst_n), // 仅复位full状态
        .fill_done(proj1_fill_done), // Controller确认全部写入完成
        .fill_bank(proj1_fill_bank), // 完成填充的bank
        .release_bank(proj1_release_bank), // Controller确认消费完成的bank
        .release_en(proj1_release_en), // 消费结束清full
        .full(proj1_full), // 两bank独立full状态
        .wr_en(proj1_wr_en), // GEMM slice适配或bias写回
        .wr_bank(proj1_wr_bank), // 写目标bank
        .wr_row(proj1_wr_row), // token行
        .wr_word(proj1_wr_word), // 128列内八lane字
        .wr_data(proj1_wr_data), // 八lane BF16
        .rd_en(proj1_rd_en), // 后处理或DDR读使能
        .rd_bank(proj1_rd_bank), // 消费bank
        .rd_row(proj1_rd_row), // 消费token行
        .rd_word(proj1_rd_word), // 消费列字
        .rd_valid(proj1_rd_valid), // 同步读有效
        .rd_data(proj1_rd_data) // 八lane输出
    );

    // P两bank八乘32768 BF16，Softmax固定写P乘V固定读，logits复用物理地址
    qwen25_fixed_attention attention_Inst (
        .clk(clk), // buffer时钟
        .rst_n(rst_n), // 仅复位full状态
        .fill_done(attention_fill_done), // Controller确认全部写入完成
        .fill_bank(attention_fill_bank), // 完成填充的bank
        .release_bank(attention_release_bank), // Controller确认消费完成的bank
        .release_en(attention_release_en), // 消费结束清full
        .full(attention_full), // 两bank独立full状态
        .wr_en(attention_wr_en), // Softmax或spill reload写
        .wr_bank(attention_wr_bank), // 写目标P bank
        .wr_address(attention_wr_address), // P或logits线性128bit地址
        .wr_data(attention_wr_data), // 八lane BF16
        .wr_ready(attention_wr_ready),.afu_wr_en(attention_afu_wr_en),.afu_wr_bank(attention_afu_wr_bank),
        .afu_wr_address(attention_afu_wr_address),.afu_wr_data(attention_afu_wr_data),
        .rd_en(attention_rd_en), // P乘V或spill消费
        .rd_bank(attention_rd_bank), // 消费bank
        .rd_word(attention_rd_word), // context八元素字
        .rd_valid(attention_rd_valid), // 同步P tile读有效
        .rd_tile(attention_rd_tile), // 八query行同context列字
        .afu_rd_en(attention_afu_rd_en),.afu_rd_bank(attention_afu_rd_bank),
        .afu_rd_row(attention_afu_rd_row),.afu_rd_word(attention_afu_rd_word),
        .afu_rd_valid(attention_afu_rd_valid),.afu_rd_data(attention_afu_rd_data),
        .max_wr_en(attention_max_wr_en),.max_wr_row(attention_max_wr_row),
        .max_wr_value(attention_max_wr_value),.max_wr_index(attention_max_wr_index),
        .max_wr_first(attention_max_wr_first),.max_wr_last(attention_max_wr_last),.max_wr_words(attention_max_wr_words),
        .afu_row_maximum(attention_row_maximum),.afu_row_max_index(attention_row_max_index),.afu_row_complete(attention_row_complete)
    );

    // O四bank两pair成对AFU读取，生命周期由FFN/残差Controller控制
    qwen25_fixed_output output_Inst (
        .clk(clk), // O buffer时钟
        .rst_n(rst_n), // 仅复位pair full
        .pair_done(output_pair_done), // Controller确认当前pair全部写入
        .fill_pair(output_fill_pair), // 当前h/z或g/u pair
        .release_en(output_release_en), // Controller等待计算与两路B响应后释放
        .release_pair(output_release_pair), // 被释放pair
        .pair_full(output_pair_full), // 两pair full状态
        .wr_en(output_wr_en), // GEMM或AFU固定写端口
        .wr_bank(output_wr_bank), // 四个O bank
        .wr_row(output_wr_row), // token行
        .wr_word(output_wr_word), // 896列内128bit字
        .wr_data(output_wr_data), // 八lane BF16
        .wr_ready(output_wr_ready),
        .afu_wr_en(output_afu_wr_en),.afu_wr_bank(output_afu_wr_bank),
        .afu_wr_row(output_afu_wr_row),.afu_wr_word(output_afu_wr_word),.afu_wr_data(output_afu_wr_data),
        .rd_en(output_rd_en), // AFU成对读取或DDR写回
        .rd_pair(output_rd_pair), // 消费pair
        .rd_row(output_rd_row), // token行
        .rd_word(output_rd_word), // 896列内128bit字
        .rd_valid(output_rd_valid), // 同步读取有效
        .rd_x(output_rd_x), // pair第一个bank
        .rd_y(output_rd_y) // pair第二个bank
    );

endmodule
