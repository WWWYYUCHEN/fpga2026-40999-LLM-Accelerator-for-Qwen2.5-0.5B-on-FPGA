`timescale 1ns/1ps
`default_nettype none

// 8项FP32固定平衡树，三个7级加法层；每clock一组，不改变归约顺序。
module qwen_fp32_sum8 (
    input wire clk,
    input wire rst_n,
    input wire clear,
    input wire input_valid,
    input wire [255:0] values,
    output wire [31:0] result,
    output wire output_valid
);
    wire [127:0] pairs;
    wire [63:0] quads;
    wire [3:0] pair_valid;
    wire [1:0] quad_valid;
    for(genvar pair=0;pair<4;pair++)begin:gen_pair
        qwen_fp32_add_valid pair_add_Inst(.clk(clk),.rst_n(rst_n),.clear(clear),.input_valid(input_valid),
            .a(values[pair*64+:32]),.b(values[pair*64+32+:32]),
            .result(pairs[pair*32+:32]),.output_valid(pair_valid[pair]));
    end
    for(genvar quad=0;quad<2;quad++)begin:gen_quad
        qwen_fp32_add_valid quad_add_Inst(.clk(clk),.rst_n(rst_n),.clear(clear),.input_valid(pair_valid[quad*2] && pair_valid[quad*2+1]),
            .a(pairs[quad*64+:32]),.b(pairs[quad*64+32+:32]),
            .result(quads[quad*32+:32]),.output_valid(quad_valid[quad]));
    end
    qwen_fp32_add_valid final_add_Inst(.clk(clk),.rst_n(rst_n),.clear(clear),.input_valid(quad_valid[0] && quad_valid[1]),
        .a(quads[31:0]),.b(quads[63:32]),.result(result),.output_valid(output_valid));
endmodule
`default_nettype wire
