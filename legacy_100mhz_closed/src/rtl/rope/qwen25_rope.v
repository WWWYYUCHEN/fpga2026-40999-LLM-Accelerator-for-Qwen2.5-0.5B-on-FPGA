`timescale 1ns / 1ps
`default_nettype none

// Company: User project
// Engineer: Codex
// Create Date: 2026/10/08
// Design Name: Qwen pair RoPE
// Module Name: qwen25_rope
// Description: description/qwen25_rope_Design.pdf
// Simulations: testbench/vivado/qwen25_rope
// Referrences: Qwen split-half rotary embedding
// Dependencies: FP32 MUL/ADD and ordered sincos operator
// Version: V1.0
// Revision Date: 2026/10/08
// History: 2026/10/08 V1.0 Create paired streaming datapath.
// 版权归属: 项目用户
// 开发人员: Codex
// 创建日期: 2026年10月08日
// 设计名称: Qwen单对RoPE
// 模块名称: qwen25_rope
// 模块说明: Description/qwen25_rope_Design.pdf
// 仿真工程: TestBench/Vivado/qwen25_rope
// 参考资料: Qwen split-half旋转公式
// 依赖文件: FP32乘法加法与保序sin/cos算子
// 当前版本: V1.0
// 修订日期: 2026年10月08日
// 修订历史: 2026年10月08日 V1.0 创建双BF16旋转流水

// 两个BF16组成原子事务，角度同拍提交；结果空间在接受前预留。
module qwen25_rope (
    input wire i_clk, // 统一上升沿时钟
    input wire i_rstn, // 同步低有效复位，保持至少两拍取消旧事务
    input wire i_valid, // 数据对和角度同时有效
    output wire o_ready, // 角度算子可接受且最终输出已预留空间
    input wire [31:0] i_data, // 低16位x0，高16位x1
    input wire [31:0] i_angle, // 当前数据对的FP32弧度
    output wire o_valid, // 当前完整结果对可消费
    input wire i_ready, // 下游接受结果对
    output wire [31:0] o_data, // 低16位y0，高16位y1
    output wire o_busy // 存在角度、算术或输出队列事务
);
    localparam C_DEPTH = 64; // 所有在途数据共用的结果信用总数
    reg [31:0] reg_pairs [0:C_DEPTH-1]; // 等待sin/cos返回的原始BF16对
    reg [31:0] reg_results [0:C_DEPTH-1]; // 已完成的输出对，数据不复位
    reg [5:0] ptr_pair_write; // 接受事务保存位置
    reg [5:0] ptr_pair_read; // 与保序sin/cos结果配对位置
    reg [5:0] ptr_result_write; // 真实算术提交位置
    reg [5:0] ptr_result_read; // 输出消费位置
    reg [6:0] cnt_credit; // 接受至最终消费之间的所有事务数
    reg [6:0] cnt_pairs; // 等待sin/cos结果的事务数
    reg [6:0] cnt_results; // 可供输出消费的结果数
    wire flag_capacity; // 还有一个未占用的最终输出槽
    wire flag_accept; // 数据和角度共同接受
    wire flag_trig_ready; // sin/cos算子输入允许接受
    wire flag_trig_valid; // sin/cos算子完整响应有效
    wire flag_trig_accept; // 配对取出与sin/cos同时消费
    wire flag_trig_result_ready; // 已有数据对等待该保序响应
    wire [31:0] wire_sin; // 实际角度对应的FP32正弦
    wire [31:0] wire_cos; // 实际角度对应的FP32余弦
    wire [31:0] wire_pair; // 队首待旋转数据对
    wire [31:0] wire_first; // x0扩展FP32，非规格数按带符号零
    wire [31:0] wire_second; // x1扩展FP32，非规格数按带符号零
    localparam [2:0] S_IDLE=0,S_MUL_SEND=1,S_MUL_WAIT=2,S_ADD_SEND=3,S_ADD_WAIT=4;
    reg [2:0] state_current,state_next;
    reg [1:0] cnt_mul_issue,cnt_mul_return;
    reg flag_add_issue,flag_add_return;
    reg [31:0] reg_first,reg_second,reg_sine,reg_cosine;
    reg [31:0] reg_first_cos,reg_second_sin,reg_second_cos,reg_first_sin;
    reg [15:0] reg_first_rounded;
    reg [31:0] wire_multiply_a,wire_multiply_b;
    wire [31:0] wire_product;
    wire flag_product_valid;
    wire flag_multiply_issue=i_rstn && state_current==S_MUL_SEND;
    wire flag_sum_issue=i_rstn && state_current==S_ADD_SEND;
    wire [31:0] wire_sum_a=flag_add_issue ? reg_second_cos : reg_first_cos;
    wire [31:0] wire_sum_b=flag_add_issue ? reg_first_sin : {~reg_second_sin[31],reg_second_sin[30:0]};
    wire [31:0] wire_sum;
    wire flag_sum_valid;
    wire [15:0] wire_rounded;
    wire flag_produce; // 第二个和返回后提交完整结果对
    wire flag_pop; // 下游实际消费完整结果对

    // 信用约束覆盖角度等待、算术流水与最终输出，输入背压不能产生新事务。
    assign flag_capacity = cnt_credit < C_DEPTH;
    assign o_ready = i_rstn && flag_capacity && flag_trig_ready;
    assign flag_accept = i_valid && o_ready;
    assign flag_trig_result_ready = i_rstn && cnt_pairs != 0 && state_current==S_IDLE;
    assign flag_trig_accept = flag_trig_valid && flag_trig_result_ready;
    assign o_valid = i_rstn && cnt_results != 0;
    assign o_data = reg_results[ptr_result_read];
    assign o_busy = cnt_credit != 0;
    assign flag_produce = i_rstn && state_current==S_ADD_WAIT && flag_sum_valid && flag_add_return;
    assign flag_pop = o_valid && i_ready;

    // Preserve four separately rounded FP32 products before the two sums.
    assign wire_pair = reg_pairs[ptr_pair_read];
    assign wire_first = wire_pair[14:7] == 0 ? {wire_pair[15],31'd0} : {wire_pair[15:0],16'd0};
    assign wire_second = wire_pair[30:23] == 0 ? {wire_pair[31],31'd0} : {wire_pair[31:16],16'd0};
    always @(*) begin
        wire_multiply_a=reg_first;
        wire_multiply_b=reg_cosine;
        case(cnt_mul_issue)
            2'd0:begin wire_multiply_a=reg_first;wire_multiply_b=reg_cosine;end
            2'd1:begin wire_multiply_a=reg_second;wire_multiply_b=reg_sine;end
            2'd2:begin wire_multiply_a=reg_second;wire_multiply_b=reg_cosine;end
            2'd3:begin wire_multiply_a=reg_first;wire_multiply_b=reg_sine;end
            default:begin end
        endcase
    end

    always @(posedge i_clk) begin
        if(!i_rstn)state_current<=S_IDLE;
        else state_current<=state_next;
    end
    always @(*) begin
        state_next=state_current;
        case(state_current)
            S_IDLE:if(flag_trig_accept)state_next=S_MUL_SEND;
            S_MUL_SEND:if(cnt_mul_issue==3)state_next=S_MUL_WAIT;
            S_MUL_WAIT:if(flag_product_valid && cnt_mul_return==3)state_next=S_ADD_SEND;
            S_ADD_SEND:if(flag_add_issue)state_next=S_ADD_WAIT;
            S_ADD_WAIT:if(flag_produce)state_next=S_IDLE;
            default:state_next=S_IDLE;
        endcase
    end
    // Only one pair occupies the shared arithmetic operators. The angle/output
    // FIFOs retain independent credits, so output stalls cannot overwrite work.
    always @(posedge i_clk) begin
        if(!i_rstn)begin
            cnt_mul_issue<=0;cnt_mul_return<=0;
            flag_add_issue<=0;flag_add_return<=0;
        end else begin
            if(flag_trig_accept)begin
                reg_first<=wire_first;reg_second<=wire_second;
                reg_sine<=wire_sin;reg_cosine<=wire_cos;
                cnt_mul_issue<=0;cnt_mul_return<=0;
                flag_add_issue<=0;flag_add_return<=0;
            end
            if(flag_multiply_issue)cnt_mul_issue<=cnt_mul_issue+1'b1;
            if(flag_product_valid && state_current!=S_IDLE)begin
                cnt_mul_return<=cnt_mul_return+1'b1;
                case(cnt_mul_return)
                    2'd0:reg_first_cos<=wire_product;
                    2'd1:reg_second_sin<=wire_product;
                    2'd2:reg_second_cos<=wire_product;
                    2'd3:reg_first_sin<=wire_product;
                    default:begin end
                endcase
            end
            if(flag_sum_issue)flag_add_issue<=!flag_add_issue;
            if(flag_sum_valid && state_current==S_ADD_WAIT)begin
                flag_add_return<=!flag_add_return;
                if(!flag_add_return)reg_first_rounded<=wire_rounded;
            end
        end
    end

    // 保序角度算子可独立替换为硬件；核心不使用固定等待作为完成依据。
    qwen25_rope_sincos sincos_Inst (
        .i_clk(i_clk), .i_rstn(i_rstn),
        .i_valid(i_valid && flag_capacity && i_rstn), .o_ready(flag_trig_ready),
        .i_angle(i_angle), .o_valid(flag_trig_valid), .i_ready(flag_trig_result_ready),
        .o_sin(wire_sin), .o_cos(wire_cos)
    );
    wire flag_product_valid_raw;
    qwen_operator_valid_guard #(.C_LATENCY(3)) product_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(flag_multiply_issue),
        .i_operator_valid(flag_product_valid_raw),.o_valid(flag_product_valid)
    );
    qwen25_afu_fp_mul multiply_Inst (
        .aclk(i_clk),
        .s_axis_a_tvalid(flag_multiply_issue),.s_axis_a_tdata(wire_multiply_a),
        .s_axis_a_tuser(16'd0),.s_axis_b_tvalid(flag_multiply_issue),.s_axis_b_tdata(wire_multiply_b),
        .m_axis_result_tvalid(flag_product_valid_raw),.m_axis_result_tdata(wire_product),.m_axis_result_tuser()
    );
    qwen_fp32_add_valid add_Inst (
        .clk(i_clk), .rst_n(i_rstn), .clear(1'b0),
        .input_valid(flag_sum_issue),.a(wire_sum_a),.b(wire_sum_b),
        .result(wire_sum), .output_valid(flag_sum_valid)
    );
    qwen_fp32_to_bf16 round_Inst (
        .value(wire_sum), .result(wire_rounded)
    );

    // 同拍入队/退役分别计数；复位清身份，数据RAM保持。
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            ptr_pair_write <= 0;
            ptr_pair_read <= 0;
            ptr_result_write <= 0;
            ptr_result_read <= 0;
            cnt_credit <= 0;
            cnt_pairs <= 0;
            cnt_results <= 0;
        end else begin
            case ({flag_accept,flag_pop})
                2'b10: begin cnt_credit <= cnt_credit+1'b1; end
                2'b01: begin cnt_credit <= cnt_credit-1'b1; end
                default: begin cnt_credit <= cnt_credit; end
            endcase
            case ({flag_accept,flag_trig_accept})
                2'b10: begin cnt_pairs <= cnt_pairs+1'b1; end
                2'b01: begin cnt_pairs <= cnt_pairs-1'b1; end
                default: begin cnt_pairs <= cnt_pairs; end
            endcase
            case ({flag_produce,flag_pop})
                2'b10: begin cnt_results <= cnt_results+1'b1; end
                2'b01: begin cnt_results <= cnt_results-1'b1; end
                default: begin cnt_results <= cnt_results; end
            endcase
            if (flag_accept) begin
                reg_pairs[ptr_pair_write] <= i_data;
                ptr_pair_write <= ptr_pair_write+1'b1;
            end
            if (flag_trig_accept) begin
                ptr_pair_read <= ptr_pair_read+1'b1;
            end
            if (flag_produce) begin
                reg_results[ptr_result_write] <= {wire_rounded,reg_first_rounded};
                ptr_result_write <= ptr_result_write+1'b1;
            end
            if (flag_pop) begin
                ptr_result_read <= ptr_result_read+1'b1;
            end
        end
    end
endmodule
`default_nettype wire
