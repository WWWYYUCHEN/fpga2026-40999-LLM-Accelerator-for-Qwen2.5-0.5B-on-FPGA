// Two-state simulation models for the three UNISIM primitives used by
// qwen_pe_weight_select.  The production RTL still uses the AMD primitives;
// this file only removes event-heavy vendor models from Verilator regression.
module LUT6 #(parameter [63:0] INIT = 64'h0) (
    output wire O,
    input wire I0, input wire I1, input wire I2,
    input wire I3, input wire I4, input wire I5
);
    assign O = INIT[{I5,I4,I3,I2,I1,I0}];
endmodule

module MUXF7 (
    output wire O,
    input wire I0, input wire I1, input wire S
);
    assign O = S ? I1 : I0;
endmodule

module MUXF8 (
    output wire O,
    input wire I0, input wire I1, input wire S
);
    assign O = S ? I1 : I0;
endmodule
