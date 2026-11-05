`timescale 1ns / 1ps
// One FP32 MUL/EXP/RECIP lane; the MUL user bit identifies the SwiGLU final stage.
module qwen25_afu_lane (
    input wire i_clk,
    input wire i_rstn,
    input wire i_mul_valid,
    input wire [31:0] i_mul_a,
    input wire [31:0] i_mul_b,
    input wire i_mul_finalize,
    input wire i_mul_tag_enable,
    input wire [15:0] i_gamma,
    input wire i_gamma_enable,
    output wire o_mul_valid,
    output wire [31:0] o_mul_result,
    output wire o_mul_finalize,
    output wire o_gamma_valid,
    output wire [31:0] o_gamma_result,
    input wire i_exp_valid,
    input wire [31:0] i_exp_a,
    output wire o_exp_valid,
    output wire [31:0] o_exp_result,
    input wire i_recip_valid,
    input wire [31:0] i_recip_a,
    output wire o_recip_valid,
    output wire [31:0] o_recip_result
);
    // Stage the complete multiply request before the DSP.  The operation/tag
    // decode is shared by all lanes and otherwise reaches the DSP INMODE pins
    // through a long controller-to-AFU combinational cone.
    reg mul_valid_reg;
    reg [31:0] mul_a_reg,mul_b_reg;
    reg [15:0] mul_user_reg;
    always @(posedge i_clk) begin
        if(!i_rstn) begin
            mul_valid_reg<=1'b0;
            mul_a_reg<=32'd0;
            mul_b_reg<=32'd0;
            mul_user_reg<=16'd0;
        end else begin
            mul_valid_reg<=i_mul_valid;
            mul_a_reg<=i_mul_a;
            mul_b_reg<=i_mul_b;
            mul_user_reg<=i_mul_tag_enable ? {15'd0,i_mul_finalize} : i_gamma;
        end
    end

    wire [15:0] wire_gamma;
    wire [15:0] wire_normalized;
    wire wire_mul_valid_raw,wire_exp_valid_raw,wire_recip_valid_raw;
    qwen_operator_valid_guard #(.C_LATENCY(3)) multiply_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(mul_valid_reg),
        .i_operator_valid(wire_mul_valid_raw),.o_valid(o_mul_valid)
    );
    qwen_operator_valid_guard #(.C_LATENCY(16)) exponential_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(i_exp_valid),
        .i_operator_valid(wire_exp_valid_raw),.o_valid(o_exp_valid)
    );
    qwen_operator_valid_guard #(.C_LATENCY(16)) reciprocal_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(i_recip_valid),
        .i_operator_valid(wire_recip_valid_raw),.o_valid(o_recip_valid)
    );
    assign o_mul_finalize = wire_gamma[0];
    qwen25_afu_fp_mul multiply_Inst (
        .aclk(i_clk),
        .s_axis_a_tvalid(mul_valid_reg), .s_axis_a_tdata(mul_a_reg), .s_axis_a_tuser(mul_user_reg),
        .s_axis_b_tvalid(mul_valid_reg), .s_axis_b_tdata(mul_b_reg),
        .m_axis_result_tvalid(wire_mul_valid_raw), .m_axis_result_tdata(o_mul_result),
        .m_axis_result_tuser(wire_gamma)
    );
    qwen_fp32_to_bf16 normalize_round_Inst (
        .value(o_mul_result), .result(wire_normalized)
    );
    qwen_bf16_mul_valid gamma_Inst (
        .clk(i_clk), .rst_n(i_rstn), .clear(1'b0),
        .input_valid(o_mul_valid && i_gamma_enable),
        .a(wire_normalized), .b(wire_gamma),
        .result(o_gamma_result), .output_valid(o_gamma_valid)
    );
    qwen25_afu_fp_exp exponential_Inst (
        .aclk(i_clk),
        .s_axis_a_tvalid(i_exp_valid), .s_axis_a_tdata(i_exp_a),
        .m_axis_result_tvalid(wire_exp_valid_raw), .m_axis_result_tdata(o_exp_result)
    );
    qwen25_afu_fp_recip reciprocal_Inst (
        .aclk(i_clk),
        .s_axis_a_tvalid(i_recip_valid), .s_axis_a_tdata(i_recip_a),
        .m_axis_result_tvalid(wire_recip_valid_raw), .m_axis_result_tdata(o_recip_result)
    );
endmodule
