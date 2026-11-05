`timescale 1ns / 1ps
`default_nettype wire

// Company: User project
// Engineer: Codex
// Create Date: 2026-10-08
// Module Name: qwen25_weight_fifo
// Description: 两行弹性权重拼接FIFO，1024bit输入并拼接为2048bit冻结GEMM权重行
// 版权归属: 项目用户
// 开发人员: Codex
// 创建日期: 2026-10-08
// 模块名称: qwen25_weight_fifo
// 模块说明: 两行弹性权重拼接FIFO，1024bit输入并拼接为2048bit冻结GEMM权重行

module qwen25_weight_fifo (
    input i_clk, // 拼接FIFO同步时钟
    input i_rstn, // 同步低有效控制复位
    input i_flush, // 排空确认后清除FIFO身份
    input i_valid, // 1024bit半行有效
    output o_ready, // 可以接受下一个半行
    output [2:0] o_free_halves, // 可预留的1024bit半行槽，包括已收齐整行
    input [1023:0] i_data, // 六十四个连续BF16权重
    input [127:0] i_meta, // 半行所属GEMM身份
    output o_valid, // 一整行2048bit已经收齐
    input i_ready, // 冻结GEMM接受完整权重行
    output [2047:0] o_data, // 按PE列和八槽拼接后的完整行
    output [127:0] o_meta, // 输出行与命令一致的身份
    output o_empty, // 不存在半行或完整行
    output o_error // 兼容接口，固定为零
);

    // Each 128-bit slice below owns its two row entries and one write address.
    reg [127:0] reg_first_meta; // 第一行计算身份
    reg [127:0] reg_second_meta; // 第二行计算身份
    reg [1:0] flag_full; // 两行分别已完整拼接
    reg flag_write_bank; // 当前填充的行槽
    reg flag_read_bank; // 当前待输出的行槽
    // Same-edge pointer copies drive independent data slices, with no extra FIFO latency.
    (* DONT_TOUCH = "yes" *) reg [7:0] read_bank_slice_copies;
    (* DONT_TOUCH = "yes" *) reg [7:0] write_bank_slice_copies;
    reg flag_half; // 已经收齐当前行低64列
    reg [2:0] cnt_halves; // 已接受但尚未输出的半行数量
    wire flag_write_fire; // 实际半行接受事件
    wire flag_read_fire; // 实际整行输出事件
    assign o_ready = !flag_full[flag_write_bank]; // 完整行被接受前禁止覆写
    assign o_free_halves = 3'd4-cnt_halves; // 调用者另行扣除同步RAM未返回请求
    assign o_valid = flag_full[flag_read_bank]; // 只有两半均收齐才可输出
    genvar weight_slice;
    generate for(weight_slice=0;weight_slice<8;weight_slice=weight_slice+1)begin:gen_bank_data_slice
        (* ram_style = "distributed" *) reg [127:0] low_slice[0:1],high_slice[0:1];
        assign o_data[weight_slice*128+:128]=low_slice[read_bank_slice_copies[weight_slice]];
        assign o_data[1024+weight_slice*128+:128]=high_slice[read_bank_slice_copies[weight_slice]];
        always @(posedge i_clk)begin
            if(flag_write_fire && !flag_half)
                low_slice[write_bank_slice_copies[weight_slice]]<=i_data[weight_slice*128+:128];
            if(flag_write_fire && flag_half)
                high_slice[write_bank_slice_copies[weight_slice]]<=i_data[weight_slice*128+:128];
        end
    end endgenerate
    assign o_meta = flag_read_bank ? reg_second_meta : reg_first_meta; // 身份随输出行保持
    assign o_empty = !(|flag_full) && !flag_half; // 半行同样属于在途数据
    assign o_error = 1'b0;
    assign flag_write_fire = i_valid && o_ready; // 输入valid和ready共同推进
    assign flag_read_fire = o_valid && i_ready; // 输出背压不改变计数
    always @(posedge i_clk)begin
        if(!i_rstn || i_flush)begin
            read_bank_slice_copies<=8'd0;
            write_bank_slice_copies<=8'd0;
        end else begin
            if(flag_read_fire)read_bank_slice_copies<={8{!flag_read_bank}};
            if(flag_write_fire && flag_half)write_bank_slice_copies<={8{!flag_write_bank}};
        end
    end

    // FIFO只清valid和指针，权重数据本身保持。
    always @(posedge i_clk) begin
        if (!i_rstn || i_flush) begin
            flag_full <= 0; // 清除已收齐行身份
            flag_write_bank <= 0; // 下一帧从第一行槽填充
            flag_read_bank <= 0; // 下一帧从第一行槽输出
            flag_half <= 0; // 新帧第一拍为低半行
            cnt_halves <= 0;
        end else begin
            case ({flag_write_fire,flag_read_fire})
                2'b10: cnt_halves <= cnt_halves+3'd1;
                2'b01: cnt_halves <= cnt_halves-3'd2;
                2'b11: cnt_halves <= cnt_halves-3'd1;
                default: begin end
            endcase
            if (flag_write_fire) begin
                flag_half <= !flag_half; // 每两次接受构成完整行
                if (!flag_half) begin
                    if (flag_write_bank) begin
                        reg_second_meta <= i_meta; // 第一半锁存第二行身份
                    end else begin
                        reg_first_meta <= i_meta; // 第一半锁存第一行身份
                    end
                end else begin
                    flag_full[flag_write_bank] <= 1'b1; // 第二半完成当前行
                    flag_write_bank <= !flag_write_bank; // 转向另一弹性行槽
                end
            end
            if (flag_read_fire) begin
                flag_full[flag_read_bank] <= 1'b0; // 已接受行释放其FIFO槽
                flag_read_bank <= !flag_read_bank; // 按原顺序输出下一行
            end
        end
    end

    // 高低半行各自具有一个写口，行槽作为地址；保持自然列顺序。
    // Each bank-data slice above retains one write port per half-row.
endmodule
`default_nettype wire
