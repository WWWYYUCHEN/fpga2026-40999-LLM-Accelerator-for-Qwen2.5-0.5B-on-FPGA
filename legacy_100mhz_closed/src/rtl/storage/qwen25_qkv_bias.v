`timescale 1ns / 1ps
`default_nettype wire

// GEMM后处理：每个输出列只做一次BF16 scale乘法，然后和BF16 bias相加。
// Weight buffer因此只保存INT8转BF16的原始权重，不再复制scale乘法器。
module qwen25_qkv_bias (
    input i_clk, input i_rstn, input i_valid, output o_ready,
    input [127:0] i_data, input [127:0] i_scale, input [127:0] i_bias,
    input [127:0] i_meta, output o_valid, input i_ready,
    output [127:0] o_data, output [127:0] o_meta, output o_busy
);
    reg [4:0] cnt_credit, cnt_output;
    reg [3:0] ptr_accept, ptr_produce, ptr_consume;
    reg [127:0] reg_data [0:15], reg_meta [0:15];
    reg [15:0] bias_pipe [0:7][0:2];
    wire [255:0] wire_scaled, wire_added;
    wire [7:0] wire_scale_valid, wire_add_valid;
    wire flag_accept = i_valid && o_ready;
    wire flag_produce = &wire_add_valid;
    wire flag_consume = o_valid && i_ready;
    wire [127:0] rounded_data;
    integer lane, stage;
    assign o_ready = cnt_credit < 16;
    assign o_valid = cnt_output != 0;
    assign o_data = reg_data[ptr_consume];
    assign o_meta = reg_meta[ptr_consume];
    assign o_busy = cnt_credit != 0;
    genvar g;
    generate for(g=0; g<8; g=g+1) begin: gen_lane
        qwen_bf16_mul_valid scale_Inst(
            .clk(i_clk), .rst_n(i_rstn), .clear(1'b0), .input_valid(flag_accept),
            .a(i_data[g*16+:16]), .b(i_scale[g*16+:16]),
            .result(wire_scaled[g*32+:32]), .output_valid(wire_scale_valid[g])
        );
        qwen_fp32_add_valid add_Inst(
            .clk(i_clk), .rst_n(i_rstn), .clear(1'b0), .input_valid(wire_scale_valid[g]),
            .a(wire_scaled[g*32+:32]), .b({bias_pipe[g][2],16'd0}),
            .result(wire_added[g*32+:32]), .output_valid(wire_add_valid[g])
        );
        qwen_fp32_to_bf16 round_Inst(.value(wire_added[g*32+:32]), .result(rounded_data[g*16+:16]));
    end endgenerate
    always @(posedge i_clk) begin
        if(!i_rstn) begin
            cnt_credit<=0; cnt_output<=0; ptr_accept<=0; ptr_produce<=0; ptr_consume<=0;
            for(lane=0; lane<8; lane=lane+1)
                for(stage=0; stage<3; stage=stage+1) bias_pipe[lane][stage]<=0;
        end else begin
            for(lane=0; lane<8; lane=lane+1) begin
                bias_pipe[lane][0] <= i_bias[lane*16+:16];
                bias_pipe[lane][1] <= bias_pipe[lane][0];
                bias_pipe[lane][2] <= bias_pipe[lane][1];
            end
            case({flag_accept,flag_consume})
                2'b10: cnt_credit<=cnt_credit+1'b1;
                2'b01: cnt_credit<=cnt_credit-1'b1;
                default: ;
            endcase
            case({flag_produce,flag_consume})
                2'b10: cnt_output<=cnt_output+1'b1;
                2'b01: cnt_output<=cnt_output-1'b1;
                default: ;
            endcase
            if(flag_accept) begin reg_meta[ptr_accept]<=i_meta; ptr_accept<=ptr_accept+1'b1; end
            if(flag_produce) begin reg_data[ptr_produce]<=rounded_data; ptr_produce<=ptr_produce+1'b1; end
            if(flag_consume) ptr_consume<=ptr_consume+1'b1;
        end
    end
endmodule
`default_nettype wire
