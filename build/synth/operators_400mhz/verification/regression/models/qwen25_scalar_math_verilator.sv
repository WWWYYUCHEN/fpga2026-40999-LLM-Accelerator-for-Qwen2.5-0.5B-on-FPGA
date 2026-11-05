`timescale 1ns / 1ps
// Simulation-only IP models. Compile instead of the VHDL scalar IP definitions.
// Payloads cross DPI as binary32 bits; SystemVerilog shortreal is not used.
module qwen25_scalar_math_model #(
    parameter integer LATENCY = 7,
    parameter integer OPERATION = 0
) (
    input wire clk,
    input wire reset_n,
    input wire enable,
    input wire valid,
    input wire [31:0] a,
    input wire [31:0] b,
    input wire [15:0] user,
    output wire result_valid,
    output wire [31:0] result,
    output wire [15:0] result_user
);
    import "DPI-C" pure function int unsigned qwen25_fp32_dpi(
        input int operation, input int unsigned first, input int unsigned second
    );
    reg [LATENCY-1:0] valid_pipe = 0;
    reg [31:0] data_pipe [0:LATENCY-1];
    reg [15:0] user_pipe [0:LATENCY-1];
    integer stage;

    assign result_valid = valid_pipe[LATENCY-1];
    assign result = data_pipe[LATENCY-1];
    assign result_user = user_pipe[LATENCY-1];

    initial begin
        if (LATENCY < 1) $fatal(1, "scalar IP latency must be positive");
    end

    // Reset discards in-flight valid bits. aclken freezes payload, user and valid.
    // Payload registers retain their old value on reset, as in the existing model.
    always @(posedge clk) begin
        if (!reset_n) begin
            valid_pipe <= 0;
        end else if (enable) begin
            for (stage = 1; stage < LATENCY; stage = stage + 1) begin
                valid_pipe[stage] <= valid_pipe[stage-1];
                data_pipe[stage] <= data_pipe[stage-1];
                user_pipe[stage] <= user_pipe[stage-1];
            end
            valid_pipe[0] <= valid;
            if (valid) begin
                data_pipe[0] <= qwen25_fp32_dpi(OPERATION, a, b);
                user_pipe[0] <= user;
            end
        end
    end
endmodule

module floating_point_0 (
    input wire aclk,
    input wire s_axis_a_tvalid,
    input wire [31:0] s_axis_a_tdata,
    input wire s_axis_b_tvalid,
    input wire [31:0] s_axis_b_tdata,
    output wire m_axis_result_tvalid,
    output wire [31:0] m_axis_result_tdata
);
    wire [15:0] unused_user;
    qwen25_scalar_math_model #(.LATENCY(7), .OPERATION(0)) math (
        .clk(aclk), .reset_n(1'b1), .enable(1'b1),
        .valid(s_axis_a_tvalid && s_axis_b_tvalid),
        .a(s_axis_a_tdata), .b(s_axis_b_tdata), .user(16'd0),
        .result_valid(m_axis_result_tvalid), .result(m_axis_result_tdata),
        .result_user(unused_user)
    );
endmodule

module qwen25_afu_fp_mul (
    input wire aclk,
    input wire s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    input wire [15:0] s_axis_a_tuser,
    input wire s_axis_b_tvalid, input wire [31:0] s_axis_b_tdata,
    output wire m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata,
    output wire [15:0] m_axis_result_tuser
);
    qwen25_scalar_math_model #(.LATENCY(3), .OPERATION(1)) math (
        .clk(aclk), .reset_n(1'b1), .enable(1'b1),
        .valid(s_axis_a_tvalid && s_axis_b_tvalid),
        .a(s_axis_a_tdata), .b(s_axis_b_tdata), .user(s_axis_a_tuser),
        .result_valid(m_axis_result_tvalid), .result(m_axis_result_tdata),
        .result_user(m_axis_result_tuser)
    );
endmodule

module qwen25_afu_fp_exp (
    input wire aclk,
    input wire s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    output wire m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    wire [15:0] unused_user;
    qwen25_scalar_math_model #(.LATENCY(16), .OPERATION(2)) math (
        .clk(aclk), .reset_n(1'b1), .enable(1'b1),
        .valid(s_axis_a_tvalid), .a(s_axis_a_tdata), .b(32'd0), .user(16'd0),
        .result_valid(m_axis_result_tvalid), .result(m_axis_result_tdata),
        .result_user(unused_user)
    );
endmodule

module qwen25_afu_fp_div (
    input wire aclk, input wire aresetn, input wire aclken,
    input wire s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    input wire s_axis_b_tvalid, input wire [31:0] s_axis_b_tdata,
    output wire m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    wire [15:0] unused_user;
    qwen25_scalar_math_model #(.LATENCY(28), .OPERATION(3)) math (
        .clk(aclk), .reset_n(aresetn), .enable(aclken),
        .valid(s_axis_a_tvalid && s_axis_b_tvalid),
        .a(s_axis_a_tdata), .b(s_axis_b_tdata), .user(16'd0),
        .result_valid(m_axis_result_tvalid), .result(m_axis_result_tdata),
        .result_user(unused_user)
    );
endmodule

module qwen25_afu_fp_rsqrt (
    input wire aclk,
    input wire s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    output wire m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    wire [15:0] unused_user;
    qwen25_scalar_math_model #(.LATENCY(19), .OPERATION(4)) math (
        .clk(aclk), .reset_n(1'b1), .enable(1'b1),
        .valid(s_axis_a_tvalid), .a(s_axis_a_tdata), .b(32'd0), .user(16'd0),
        .result_valid(m_axis_result_tvalid), .result(m_axis_result_tdata),
        .result_user(unused_user)
    );
endmodule

module qwen25_afu_fp_recip (
    input wire aclk,
    input wire s_axis_a_tvalid, input wire [31:0] s_axis_a_tdata,
    output wire m_axis_result_tvalid, output wire [31:0] m_axis_result_tdata
);
    wire [15:0] unused_user;
    qwen25_scalar_math_model #(.LATENCY(16), .OPERATION(5)) math (
        .clk(aclk), .reset_n(1'b1), .enable(1'b1),
        .valid(s_axis_a_tvalid), .a(s_axis_a_tdata), .b(32'd0), .user(16'd0),
        .result_valid(m_axis_result_tvalid), .result(m_axis_result_tdata),
        .result_user(unused_user)
    );
endmodule
