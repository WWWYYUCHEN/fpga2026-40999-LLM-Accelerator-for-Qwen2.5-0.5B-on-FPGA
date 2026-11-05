`timescale 1ns / 1ps
// INT8转BF16是精确转换，入口不执行量化scale乘法。
// 一个弹性寄存器同时保持数据及目标字地址，写口背压不会丢失返回拍。
module qwen25_weight_dequant (
    input i_clk, input i_rstn, input i_valid, output o_ready,
    input [127:0] i_data, input [9:0] i_word,
    input [2047:0] i_scales,
    output o_valid, input i_ready, output [255:0] o_data,
    output [15:0] o_word, output o_busy
);
    reg valid_reg;
    reg [255:0] data_reg;
    reg [9:0] word_reg;
    wire [255:0] converted;
    genvar lane;
    assign o_ready = !valid_reg || i_ready;
    assign o_valid = valid_reg;
    assign o_data = data_reg;
    assign o_word = {6'd0,word_reg};
    assign o_busy = valid_reg;
    generate for(lane=0;lane<16;lane=lane+1) begin: gen_convert
        qwen_int8_to_bf16 convert_Inst(.value(i_data[lane*8+:8]),.result(converted[lane*16+:16]));
    end endgenerate
    always @(posedge i_clk) begin
        if(!i_rstn) valid_reg <= 0;
        else if(o_ready) begin
            valid_reg <= i_valid;
            if(i_valid) begin data_reg <= converted; word_reg <= i_word; end
        end
    end
endmodule
