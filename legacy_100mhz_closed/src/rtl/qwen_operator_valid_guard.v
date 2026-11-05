`timescale 1ns / 1ps
`default_nettype none

// Reset only transaction ownership; the arithmetic IP keeps running freely.
module qwen_operator_valid_guard #(
    parameter integer C_LATENCY = 7
) (
    input wire i_clk,
    input wire i_rstn,
    input wire i_issue,
    input wire i_operator_valid,
    output wire o_valid
);
    reg [C_LATENCY-1:0] reg_token;
    assign o_valid = i_rstn && i_operator_valid && reg_token[C_LATENCY-1];
    always @(posedge i_clk) begin
        if (!i_rstn) reg_token <= {C_LATENCY{1'b0}};
        else reg_token <= (reg_token << 1) | i_issue;
    end
endmodule

`default_nettype wire
