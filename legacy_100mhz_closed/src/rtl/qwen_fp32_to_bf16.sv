`timescale 1ns/1ps
`default_nettype none

// 最终buffer边界RND8，普通数只用第八位round bit；特殊类和DAZ显式处理。
module qwen_fp32_to_bf16 (
    input wire [31:0] value,
    output wire [15:0] result
);
    assign result=(value[30:23]==0)?{value[31],15'b0}:
                  ((value[30:23]==8'hff)&&(value[22:0]!=0))?16'h7fc0:
                  value[31:16]+{15'b0,value[15]};
endmodule
`default_nettype wire
