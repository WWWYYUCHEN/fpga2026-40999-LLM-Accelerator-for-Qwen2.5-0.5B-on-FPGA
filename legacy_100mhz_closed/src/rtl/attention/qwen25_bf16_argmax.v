`timescale 1ns / 1ps
`default_nettype none
// Equal values choose the lowest index; a NaN remains visible to downstream math.
module qwen25_bf16_argmax (
    input wire [15:0] i_a,
    input wire [31:0] i_a_index,
    input wire [15:0] i_b,
    input wire [31:0] i_b_index,
    output wire [15:0] o_value,
    output wire [31:0] o_index
);
    wire a_nan=i_a[14:7]==8'hff && i_a[6:0]!=0;
    wire b_nan=i_b[14:7]==8'hff && i_b[6:0]!=0;
    wire both_zero=i_a[14:0]==0 && i_b[14:0]==0;
    wire equal_value=i_a==i_b || both_zero || (a_nan && b_nan);
    wire a_greater=i_a[15]!=i_b[15] ? !i_a[15] :
        (i_a[15] ? i_a[14:0]<i_b[14:0] : i_a[14:0]>i_b[14:0]);
    wire choose_a=equal_value ? i_a_index<=i_b_index :
        (a_nan || (!b_nan && a_greater));
    assign o_value=a_nan || b_nan ? 16'h7fc0 :
        (both_zero ? 16'd0 : (choose_a ? i_a : i_b));
    assign o_index=choose_a ? i_a_index : i_b_index;
endmodule
`default_nettype wire
