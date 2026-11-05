`timescale 1ns/1ps
// 十六个BF16权重使用UltraScale专用MUXF7/MUXF8选通，无额外周期。
module qwen_pe_weight_select (
    input wire [255:0] i_weights,
    input wire i_bank,
    input wire [2:0] i_slot,
    output wire [15:0] o_weight
);
    genvar bit_index,group_index;
    generate
        for(bit_index=0;bit_index<16;bit_index=bit_index+1)begin:gen_bit
            wire [3:0] group_data;
            wire bank_zero,bank_one;
            for(group_index=0;group_index<4;group_index=group_index+1)begin:gen_group
                LUT6 #(.INIT(64'hff00f0f0ccccaaaa)) group_Inst(
                    .I0(i_weights[(group_index*4)*16+bit_index]),
                    .I1(i_weights[(group_index*4+1)*16+bit_index]),
                    .I2(i_weights[(group_index*4+2)*16+bit_index]),
                    .I3(i_weights[(group_index*4+3)*16+bit_index]),
                    .I4(i_slot[0]),.I5(i_slot[1]),.O(group_data[group_index]));
            end
            MUXF7 bank_zero_Inst(.I0(group_data[0]),.I1(group_data[1]),.S(i_slot[2]),.O(bank_zero));
            MUXF7 bank_one_Inst(.I0(group_data[2]),.I1(group_data[3]),.S(i_slot[2]),.O(bank_one));
            MUXF8 bank_Inst(.I0(bank_zero),.I1(bank_one),.S(i_bank),.O(o_weight[bit_index]));
        end
    endgenerate
endmodule
