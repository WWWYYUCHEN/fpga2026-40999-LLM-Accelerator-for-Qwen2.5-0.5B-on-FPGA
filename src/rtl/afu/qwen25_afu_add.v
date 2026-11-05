`timescale 1ns / 1ps
// AFU-only eighth stage; the frozen GEMM seven-stage operator is unchanged.
module qwen25_afu_add (
    input wire i_clk,
    input wire i_rstn,
    input wire i_clear,
    input wire i_valid,
    input wire [31:0] i_a,
    input wire [31:0] i_b,
    output wire o_valid,
    output wire [31:0] o_result
);
    wire wire_valid;
    wire [31:0] wire_result;
    reg flag_valid_o;
    reg [31:0] reg_result_o;
    assign o_valid = flag_valid_o;
    assign o_result = reg_result_o;
    qwen_fp32_add_valid arithmetic_Inst (
        .clk(i_clk), .rst_n(i_rstn), .clear(i_clear),
        .input_valid(i_valid), .a(i_a), .b(i_b),
        .result(wire_result), .output_valid(wire_valid)
    );
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            flag_valid_o <= 1'b0;
        end else if (i_clear) begin
            flag_valid_o <= 1'b0;
        end else begin
            flag_valid_o <= wire_valid;
            if (wire_valid) begin
                reg_result_o <= wire_result;
            end
        end
    end
endmodule
