`timescale 1ns/1ps
`default_nettype none

// signed INT8精确浮点转换；不包含descriptor scale，-128也不溢出幅值通路。
module qwen_int8_to_bf16 (
    input wire [7:0] value,
    output wire [15:0] result
);
    wire [7:0] magnitude=value[7]?(~value+8'd1):value;
    logic [2:0] leading;
    wire [7:0] significand=magnitude<<(3'd7-leading);
    wire [7:0] exponent=8'd127+{5'b0,leading};
    always_comb begin
        casez(magnitude)
            8'b1???????:leading=3'd7;
            8'b01??????:leading=3'd6;
            8'b001?????:leading=3'd5;
            8'b0001????:leading=3'd4;
            8'b00001???:leading=3'd3;
            8'b000001??:leading=3'd2;
            8'b0000001?:leading=3'd1;
            default:leading=3'd0;
        endcase
    end
    assign result=(magnitude==0)?16'b0:{value[7],exponent,significand[6:0]};
endmodule
`default_nettype wire
