`timescale 1ns / 1ps
// Shared eight-lane arithmetic. All output space is reserved before input acceptance.
module qwen25_afu (
    input wire i_clk,
    input wire i_rstn,
    input wire i_valid,
    output wire o_ready,
    input wire [3:0] i_command,
    input wire [1:0] i_phase,
    input wire [31:0] i_eps,
    input wire [63:0] i_length,
    input wire [127:0] i_x,
    input wire [127:0] i_y,
    input wire [127:0] i_meta,
    input wire i_last,
    input wire [7:0] i_mask,
    output wire o_valid,
    input wire i_ready,
    output wire [127:0] o_data,
    output wire [127:0] o_meta,
    output wire o_last,
    output wire [7:0] o_mask,
    output wire o_busy,
    output wire o_error
);
    localparam [4:0] ST_IDLE = 0, ST_FIRST = 1, ST_STREAM = 2, ST_DRAIN = 3,
        ST_MEAN_SEND = 4, ST_MEAN_WAIT = 5, ST_EPS_SEND = 6, ST_EPS_WAIT = 7,
        ST_ROOT_SEND = 8, ST_ROOT_WAIT = 9, ST_RECIP_SEND = 10, ST_RECIP_WAIT = 11,
        ST_RESPONSE = 12, ST_EXP_DRAIN = 13, ST_REDUCE = 14, ST_ACC_DRAIN = 15,
        ST_MEAN_MUL_SEND = 16, ST_MEAN_MUL_WAIT = 17;
    reg [4:0] state_current, state_next;
    reg [3:0] reg_operation;
    reg [1:0] reg_phase;
    reg [15:0] reg_length, reg_coefficient_length, reg_maximum_length;
    reg [31:0] reg_epsilon, reg_length_float, reg_sum, reg_mean, reg_root_input;
    reg [31:0] reg_coefficient;
    reg [31:0] reg_length_inverse;
    reg [3:0] reg_coefficient_operation;
    reg [15:0] reg_maximum;
    reg flag_coefficient_valid, flag_maximum_valid, flag_last_seen;
    reg [127:0] reg_first_x, reg_first_y;
    reg [7:0] reg_first_mask;
    (* ram_style = "distributed" *) reg [127:0] reg_meta [0:63];
    reg [7:0] reg_mask [0:63];
    reg reg_last [0:63];
    (* ram_style = "distributed" *) reg [127:0] reg_input_x [0:63];
    (* ram_style = "distributed" *) reg [127:0] reg_input_y [0:63];
    reg [5:0] cnt_silu_multiply;
    reg [5:0] cnt_recip_push, cnt_recip_pop, cnt_silu_push;
    reg [6:0] cnt_recip_pending, cnt_silu_pending;
    (* ram_style = "distributed" *) reg [255:0] reg_recip_queue [0:63];
    (* ram_style = "distributed" *) reg [127:0] reg_silu_queue [0:63];
    reg [127:0] reg_results [0:63];
    reg [5:0] cnt_accept, cnt_produce, cnt_pop, cnt_exp_issue;
    reg [6:0] cnt_credit, cnt_output;
    reg [12:0] cnt_words;
    reg [127:0] reg_terminal_meta;
    reg [7:0] reg_terminal_mask;
    reg [127:0] reg_statistics_data;
    // FP32 exponentials cannot share a BF16 tensor buffer without changing precision.
    (* ram_style = "distributed" *) reg [255:0] reg_exp_block [0:31];
    reg [5:0] cnt_block_accept, cnt_block_produce, cnt_block_reduce;
    reg [31:0] dec_length_float;
    reg [4:0] dec_length_shift;
    reg [23:0] dec_length_mantissa;
    reg [7:0] dec_length_exponent;
    reg [255:0] dec_mul_a, dec_mul_b, dec_exp_a, dec_recip_a;
    reg [255:0] dec_add_a, dec_add_b;
    reg [7:0] dec_mul_valid, dec_exp_valid, dec_recip_valid;
    reg dec_add_valid, dec_add_reduce, dec_add_finish;
    reg [255:0] dec_result;
    reg flag_result;
    wire flag_rms_statistics = reg_operation == 11 && reg_phase == 0;
    wire flag_rms_apply = reg_operation == 11 && reg_phase == 1;
    wire flag_exp_statistics = reg_operation == 1 && reg_phase == 1;
    wire flag_probability = reg_operation == 1 && reg_phase == 2;
    wire flag_swiglu = reg_operation == 13 && reg_phase == 1;
    wire [127:0] wire_silu_bf16;
    wire [7:0] wire_mul_finalize;
    wire flag_silu_finalize = flag_swiglu && cnt_silu_pending != 0;
    wire flag_silu_begin = reg_operation == 13 && cnt_recip_pending != 0 && !flag_silu_finalize;
    wire flag_statistics = flag_rms_statistics || flag_exp_statistics;
    wire flag_accept = i_valid && o_ready;
    wire [5:0] input_write_address = state_current == ST_IDLE ? 6'd0 : cnt_accept;
    wire flag_launch = state_current == ST_FIRST || (state_current == ST_STREAM && flag_accept);
    wire [127:0] wire_x = state_current == ST_FIRST ? reg_first_x : i_x;
    wire [127:0] wire_y = state_current == ST_FIRST ? reg_first_y : i_y;
    wire [7:0] wire_mask = state_current == ST_FIRST ? reg_first_mask : i_mask;
    wire [255:0] wire_mul, wire_gamma, wire_exp, wire_recip, wire_add;
    wire [7:0] wire_mul_valid, wire_gamma_valid, wire_exp_valid, wire_recip_valid;
    wire wire_add_valid, wire_sum_valid, wire_add_busy, wire_add_ready;
    wire [31:0] wire_sum;
    wire wire_root_valid;
    wire [31:0] wire_root_result;
    wire [127:0] wire_rounded;
    wire flag_mul_produce = &wire_mul_valid;
    wire flag_silu_first_result = reg_operation == 13 && flag_mul_produce && !wire_mul_finalize[0];
    wire flag_recip_push = reg_operation == 13 && (&wire_recip_valid);
    wire flag_exp_produce = &wire_exp_valid;
    wire flag_pop = o_valid && i_ready && !flag_statistics;
    wire flag_statistics_retire = flag_rms_statistics ? flag_mul_produce : flag_exp_produce;
    wire flag_release = flag_statistics ? flag_statistics_retire : flag_pop;
    integer bit_index, lane_index;
    genvar lane;

    assign o_ready = i_rstn &&
        (state_current == ST_IDLE || (state_current == ST_STREAM && !flag_last_seen &&
        cnt_credit < 64 && (!flag_exp_statistics || cnt_block_accept < 32)));
    assign o_valid = flag_statistics ? state_current == ST_RESPONSE : cnt_output != 0;
    assign o_data = flag_statistics ? reg_statistics_data : reg_results[cnt_pop];
    assign o_meta = flag_statistics ? reg_terminal_meta : reg_meta[cnt_pop];
    assign o_last = flag_statistics ? 1'b1 : reg_last[cnt_pop];
    assign o_mask = flag_statistics ? reg_terminal_mask : reg_mask[cnt_pop];
    assign o_busy = state_current != ST_IDLE;
    assign o_error = 1'b0;

    always @(*) begin
        dec_length_shift = 0;
        for (bit_index = 0; bit_index < 16; bit_index = bit_index + 1) begin
            if (i_length[bit_index]) begin
                dec_length_shift = bit_index;
            end
        end
        dec_length_mantissa = i_length[15:0] << (23-dec_length_shift);
        dec_length_exponent = 8'd127 + dec_length_shift;
        dec_length_float = {1'b0, dec_length_exponent, dec_length_mantissa[22:0]};
    end

    // Route eight independent operand pairs in elementwise mode.
    always @(*) begin
        dec_mul_a = 0;
        dec_mul_b = 0;
        dec_mul_valid = 0;
        dec_exp_a = 0;
        dec_exp_valid = 0;
        dec_recip_a = 0;
        dec_recip_valid = 0;
        dec_add_a = 0;
        dec_add_b = 0;
        dec_add_valid = 0;
        dec_add_reduce = 0;
        dec_add_finish = 0;
        for (lane_index = 0; lane_index < 8; lane_index = lane_index + 1) begin
            if (reg_operation == 11 || reg_operation == 15) begin
                dec_mul_valid[lane_index] = flag_launch;
                dec_mul_a[lane_index*32 +: 32] = wire_mask[lane_index] ? {wire_x[lane_index*16 +: 16],16'd0} : 0;
                if (flag_rms_statistics) begin
                    dec_mul_b[lane_index*32 +: 32] = dec_mul_a[lane_index*32 +: 32];
                end else if (flag_rms_apply) begin
                    dec_mul_b[lane_index*32 +: 32] = reg_coefficient;
                end else begin
                    dec_mul_b[lane_index*32 +: 32] = {wire_y[lane_index*16 +: 16],16'd0};
                end
            end
            if (flag_probability) begin
                dec_mul_valid[lane_index] = wire_exp_valid[lane_index];
                dec_mul_a[lane_index*32 +: 32] = wire_exp[lane_index*32 +: 32];
                dec_mul_b[lane_index*32 +: 32] = reg_coefficient;
            end
            if (reg_operation == 14 || (reg_operation == 1 && reg_phase != 0)) begin
                dec_add_valid = flag_launch;
                dec_add_a[lane_index*32 +: 32] = wire_mask[lane_index] ? {wire_x[lane_index*16 +: 16],16'd0} : 0;
                dec_add_b[lane_index*32 +: 32] = reg_operation == 14 ?
                    {wire_y[lane_index*16 +: 16],16'd0} :
                    (wire_mask[lane_index] ? {~reg_maximum[15],reg_maximum[14:0],16'd0} : 0);
            end
            if (reg_operation == 1 && reg_phase != 0) begin
                dec_exp_valid[lane_index] = wire_add_valid;
                dec_exp_a[lane_index*32 +: 32] = reg_mask[cnt_exp_issue][lane_index] ?
                    wire_add[lane_index*32 +: 32] : 32'hff800000;
            end
            // Final MUL has priority. Credits bound both queues, including all IP in-flight words.
            if (flag_silu_begin) begin
                dec_mul_valid[lane_index] = 1;
                dec_mul_a[lane_index*32 +: 32] = {reg_input_x[cnt_recip_pop][lane_index*16 +: 16],16'd0};
                dec_mul_b[lane_index*32 +: 32] = reg_recip_queue[cnt_recip_pop][lane_index*32 +: 32];
            end
            if (flag_silu_finalize) begin
                dec_mul_valid[lane_index] = 1;
                dec_mul_a[lane_index*32 +: 32] = {reg_silu_queue[cnt_silu_multiply][lane_index*16 +: 16],16'd0};
                dec_mul_b[lane_index*32 +: 32] = {reg_input_y[cnt_silu_multiply][lane_index*16 +: 16],16'd0};
            end
            if (reg_operation == 13) begin
                dec_exp_valid[lane_index] = flag_launch;
                dec_exp_a[lane_index*32 +: 32] = {~wire_x[lane_index*16+15],wire_x[lane_index*16 +: 15],16'd0};
                dec_add_valid = flag_exp_produce;
                dec_add_a[lane_index*32 +: 32] = wire_exp[lane_index*32 +: 32];
                dec_add_b[lane_index*32 +: 32] = 32'h3f800000;
                dec_recip_valid[lane_index] = wire_add_valid;
                dec_recip_a[lane_index*32 +: 32] = wire_add[lane_index*32 +: 32];
            end
        end
        if (flag_rms_statistics && (state_current == ST_FIRST || state_current == ST_STREAM || state_current == ST_DRAIN)) begin
            dec_add_reduce = 1;
            dec_add_valid = flag_mul_produce;
            dec_add_a = wire_mul;
            dec_add_finish = flag_mul_produce && reg_last[cnt_produce];
        end
        if (state_current == ST_REDUCE || state_current == ST_ACC_DRAIN) begin
            dec_add_reduce = 1;
            dec_add_valid = state_current == ST_REDUCE;
            dec_add_a = reg_exp_block[cnt_block_reduce[4:0]];
            dec_add_finish = state_current == ST_REDUCE && flag_last_seen &&
                             cnt_block_reduce == cnt_block_accept-1'b1;
        end
        if (state_current == ST_EPS_SEND || state_current == ST_EPS_WAIT) begin
            dec_add_valid = state_current == ST_EPS_SEND;
            dec_add_a = {224'd0,reg_mean};
            dec_add_b = {224'd0,reg_epsilon};
        end
        if (state_current == ST_MEAN_SEND) begin
            dec_recip_valid[0] = 1;
            dec_recip_a[31:0] = reg_length_float;
        end
        if (state_current == ST_MEAN_MUL_SEND) begin
            dec_mul_valid[0] = 1;
            dec_mul_a[31:0] = reg_sum;
            dec_mul_b[31:0] = reg_length_inverse;
        end
        if (state_current == ST_RECIP_SEND) begin
            dec_recip_valid[0] = 1;
            dec_recip_a[31:0] = reg_sum == 0 ? 32'h3f800000 : reg_sum;
        end
    end

    always @(*) begin
        dec_result = wire_mul;
        flag_result = flag_mul_produce;
        case (reg_operation)
            11: begin dec_result = wire_gamma; flag_result = &wire_gamma_valid; end
            1: begin dec_result = wire_mul; flag_result = flag_probability && flag_mul_produce; end
            13: begin dec_result = wire_mul; flag_result = flag_mul_produce && (!flag_swiglu || wire_mul_finalize[0]); end
            14: begin dec_result = wire_add; flag_result = wire_add_valid; end
            15: begin end
            default: begin flag_result = 0; end
        endcase
        if (flag_statistics) begin
            flag_result = 0;
        end
    end
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : gen_lane
            qwen25_afu_lane arithmetic_Inst (
                .i_clk(i_clk), .i_rstn(i_rstn),
                .i_mul_valid(dec_mul_valid[lane]), .i_mul_a(dec_mul_a[lane*32 +: 32]),
                .i_mul_b(dec_mul_b[lane*32 +: 32]), .i_gamma(wire_y[lane*16 +: 16]),
                .i_mul_finalize(flag_silu_finalize), .i_mul_tag_enable(reg_operation == 13),
                .i_gamma_enable(flag_rms_apply),
                .o_mul_valid(wire_mul_valid[lane]), .o_mul_result(wire_mul[lane*32 +: 32]),
                .o_mul_finalize(wire_mul_finalize[lane]),
                .o_gamma_valid(wire_gamma_valid[lane]), .o_gamma_result(wire_gamma[lane*32 +: 32]),
                .i_exp_valid(dec_exp_valid[lane]), .i_exp_a(dec_exp_a[lane*32 +: 32]),
                .o_exp_valid(wire_exp_valid[lane]), .o_exp_result(wire_exp[lane*32 +: 32]),
                .i_recip_valid(dec_recip_valid[lane]), .i_recip_a(dec_recip_a[lane*32 +: 32]),
                .o_recip_valid(wire_recip_valid[lane]), .o_recip_result(wire_recip[lane*32 +: 32])
            );
            qwen_fp32_to_bf16 silu_round_Inst (
                .value(wire_mul[lane*32 +: 32]), .result(wire_silu_bf16[lane*16 +: 16])
            );
            qwen_fp32_to_bf16 output_round_Inst (
                .value(dec_result[lane*32 +: 32]), .result(wire_rounded[lane*16 +: 16])
            );
        end
    endgenerate
    qwen25_afu_add_bank add_bank_Inst (
        .i_clk(i_clk), .i_rstn(i_rstn), .i_start(state_current == ST_IDLE && flag_accept),
        .i_reduce(dec_add_reduce), .i_valid(dec_add_valid), .o_ready(wire_add_ready),
        .i_a(dec_add_a), .i_b(dec_add_b), .i_finish(dec_add_finish),
        .o_vector_valid(wire_add_valid), .o_vector(wire_add),
        .o_sum_valid(wire_sum_valid), .o_sum(wire_sum), .o_busy(wire_add_busy)
    );
    wire wire_root_valid_raw;
    qwen_operator_valid_guard #(.C_LATENCY(19)) root_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(state_current == ST_ROOT_SEND),
        .i_operator_valid(wire_root_valid_raw),.o_valid(wire_root_valid)
    );
    qwen25_afu_fp_rsqrt root_Inst (
        .aclk(i_clk),
        .s_axis_a_tvalid(i_rstn && state_current == ST_ROOT_SEND), .s_axis_a_tdata(reg_root_input),
        .m_axis_result_tvalid(wire_root_valid_raw), .m_axis_result_tdata(wire_root_result)
    );

    always @(*) begin
        state_next = state_current;
        case (state_current)
            ST_IDLE: begin
                if (flag_accept) begin
                    state_next = ST_FIRST;
                end
            end
            ST_FIRST: begin
                if (flag_exp_statistics && flag_last_seen) begin
                    state_next = ST_EXP_DRAIN;
                end else begin
                    state_next = flag_last_seen ? ST_DRAIN : ST_STREAM;
                end
            end
            ST_STREAM: begin
                if (flag_accept && flag_exp_statistics && (i_last || cnt_block_accept == 31)) begin
                    state_next = ST_EXP_DRAIN;
                end else if (flag_accept && i_last) begin
                    state_next = ST_DRAIN;
                end
            end
            ST_DRAIN: begin
                if (flag_rms_statistics && wire_sum_valid) begin
                    state_next = ST_MEAN_SEND;
                end else if (!flag_statistics && cnt_credit == 0) begin
                    state_next = ST_IDLE;
                end
            end
            ST_EXP_DRAIN: begin
                if (cnt_block_produce == cnt_block_accept && !wire_add_busy) begin
                    state_next = ST_REDUCE;
                end
            end
            ST_REDUCE: begin
                if (cnt_block_reduce == cnt_block_accept-1'b1 && wire_add_ready) begin
                    state_next = ST_ACC_DRAIN;
                end
            end
            ST_ACC_DRAIN: begin
                if (flag_last_seen && wire_sum_valid) begin
                    state_next = ST_RECIP_SEND;
                end else if (!flag_last_seen && !wire_add_busy) begin
                    state_next = ST_STREAM;
                end
            end
            ST_MEAN_SEND: begin
                state_next = ST_MEAN_WAIT;
            end
            ST_MEAN_WAIT: begin
                if (wire_recip_valid[0]) begin
                    state_next = ST_MEAN_MUL_SEND;
                end
            end
            ST_MEAN_MUL_SEND: begin
                state_next = ST_MEAN_MUL_WAIT;
            end
            ST_MEAN_MUL_WAIT: begin
                if (wire_mul_valid[0]) begin
                    state_next = ST_EPS_SEND;
                end
            end
            ST_EPS_SEND: begin
                state_next = ST_EPS_WAIT;
            end
            ST_EPS_WAIT: begin
                if (wire_add_valid) begin
                    state_next = ST_ROOT_SEND;
                end
            end
            ST_ROOT_SEND: begin
                state_next = ST_ROOT_WAIT;
            end
            ST_ROOT_WAIT: begin
                if (wire_root_valid) begin
                    state_next = ST_RESPONSE;
                end
            end
            ST_RECIP_SEND: begin
                state_next = ST_RECIP_WAIT;
            end
            ST_RECIP_WAIT: begin
                if (wire_recip_valid[0]) begin
                    state_next = ST_RESPONSE;
                end
            end
            ST_RESPONSE: begin
                if (i_ready) begin
                    state_next = ST_IDLE;
                end
            end
            default: begin
                state_next = ST_IDLE;
            end
        endcase
    end
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            state_current <= ST_IDLE;
        end else begin
            state_current <= state_next;
        end
    end
    integer result_index;
    // One write port per queue; only transaction counters reset, stored operands do not.
    always @(posedge i_clk) begin
        if (flag_accept) begin
            reg_meta[input_write_address] <= i_meta;
            reg_input_x[input_write_address] <= i_x;
            reg_input_y[input_write_address] <= i_y;
        end
    end
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            reg_operation <= 0;
            reg_phase <= 0;
            reg_length <= 0;
            reg_coefficient_length <= 0;
            reg_maximum_length <= 0;
            reg_coefficient <= 0;
            reg_coefficient_operation <= 0;
            reg_maximum <= 16'hff80;
            flag_coefficient_valid <= 0;
            flag_maximum_valid <= 0;
            flag_last_seen <= 0;
            cnt_accept <= 0; cnt_produce <= 0; cnt_pop <= 0;
            cnt_exp_issue <= 0; cnt_silu_multiply <= 0;
            cnt_recip_push <= 0; cnt_recip_pop <= 0; cnt_silu_push <= 0;
            cnt_recip_pending <= 0; cnt_silu_pending <= 0;
            cnt_credit <= 0; cnt_output <= 0; cnt_words <= 0;
            cnt_block_accept <= 0; cnt_block_produce <= 0; cnt_block_reduce <= 0;
        end else begin
            if (state_current == ST_IDLE && flag_accept) begin
                reg_operation <= i_command;
                reg_phase <= i_phase;
                reg_length <= i_length[15:0];
                reg_epsilon <= i_eps;
                reg_length_float <= dec_length_float;
                reg_first_x <= i_x; reg_first_y <= i_y; reg_first_mask <= i_mask;
                cnt_accept <= 1; cnt_produce <= 0; cnt_pop <= 0;
                cnt_exp_issue <= 0; cnt_silu_multiply <= 0;
                cnt_recip_push <= 0; cnt_recip_pop <= 0; cnt_silu_push <= 0;
                cnt_recip_pending <= 0; cnt_silu_pending <= 0;
                cnt_credit <= 1; cnt_output <= 0; cnt_words <= 1;
                cnt_block_accept <= 1; cnt_block_produce <= 0; cnt_block_reduce <= 0;
                flag_last_seen <= i_last;
                reg_mask[0] <= i_mask; reg_last[0] <= i_last;
                if (i_last) begin reg_terminal_meta <= i_meta; reg_terminal_mask <= i_mask; end
                // The completed buffer row supplies the maximum with the first EXP word.
                if (i_command == 1 && i_phase == 1) begin
                    reg_maximum <= i_y[15:0]; flag_maximum_valid <= 1; reg_maximum_length <= i_length[15:0];
                    flag_coefficient_valid <= 0;
                end
                if (i_command == 11 && i_phase == 0) begin flag_coefficient_valid <= 0; flag_maximum_valid <= 0; end
            end else begin
                case ({flag_accept, flag_release})
                    2'b10: begin
                        cnt_credit <= cnt_credit+1'b1;
                    end
                    2'b01: begin
                        cnt_credit <= cnt_credit-1'b1;
                    end
                    default: begin
                        cnt_credit <= cnt_credit;
                    end
                endcase
                case ({flag_result, flag_pop})
                    2'b10: begin
                        cnt_output <= cnt_output+1'b1;
                    end
                    2'b01: begin
                        cnt_output <= cnt_output-1'b1;
                    end
                    default: begin
                        cnt_output <= cnt_output;
                    end
                endcase
                if (flag_accept) begin
                    reg_mask[cnt_accept] <= i_mask; reg_last[cnt_accept] <= i_last;
                    cnt_accept <= cnt_accept+1'b1; cnt_words <= cnt_words+1'b1;
                    cnt_block_accept <= cnt_block_accept+1'b1;
                    flag_last_seen <= i_last;
                    if (i_last) begin reg_terminal_meta <= i_meta; reg_terminal_mask <= i_mask; end
                end
                if (flag_result) begin
                    for (result_index = 0; result_index < 8; result_index = result_index+1) begin
                        reg_results[cnt_produce][result_index*16 +: 16] <= reg_mask[cnt_produce][result_index] ?
                            wire_rounded[result_index*16 +: 16] : 16'd0;
                    end
                    cnt_produce <= cnt_produce+1'b1;
                end
                if (flag_pop) begin cnt_pop <= cnt_pop+1'b1; end
                if ((flag_rms_statistics && flag_mul_produce) || (flag_exp_statistics && flag_exp_produce)) begin
                    cnt_produce <= cnt_produce+1'b1;
                end
            end
            if (reg_operation == 1 && reg_phase != 0 && wire_add_valid && state_current != ST_REDUCE && state_current != ST_ACC_DRAIN) begin
                cnt_exp_issue <= cnt_exp_issue+1'b1;
            end
            if (!(state_current == ST_IDLE && flag_accept)) begin
                case ({flag_recip_push,flag_silu_begin})
                    2'b10: cnt_recip_pending <= cnt_recip_pending+1'b1;
                    2'b01: cnt_recip_pending <= cnt_recip_pending-1'b1;
                    default: begin end
                endcase
                case ({flag_swiglu && flag_silu_first_result,flag_silu_finalize})
                    2'b10: cnt_silu_pending <= cnt_silu_pending+1'b1;
                    2'b01: cnt_silu_pending <= cnt_silu_pending-1'b1;
                    default: begin end
                endcase
            end
            if (flag_recip_push) begin
                reg_recip_queue[cnt_recip_push] <= wire_recip;
                cnt_recip_push <= cnt_recip_push+1'b1;
            end
            if (flag_silu_begin) begin cnt_recip_pop <= cnt_recip_pop+1'b1; end
            if (flag_swiglu && flag_silu_first_result) begin
                reg_silu_queue[cnt_silu_push] <= wire_silu_bf16;
                cnt_silu_push <= cnt_silu_push+1'b1;
            end
            if (flag_silu_finalize) begin cnt_silu_multiply <= cnt_silu_multiply+1'b1; end
            if (flag_exp_statistics && flag_exp_produce) begin
                reg_exp_block[cnt_block_produce[4:0]] <= wire_exp;
                cnt_block_produce <= cnt_block_produce+1'b1;
            end
            if (state_current == ST_REDUCE && wire_add_ready) begin cnt_block_reduce <= cnt_block_reduce+1'b1; end
            if (state_current == ST_ACC_DRAIN && state_next == ST_STREAM) begin
                cnt_block_accept <= 0; cnt_block_produce <= 0; cnt_block_reduce <= 0;
            end
            if (wire_sum_valid) begin reg_sum <= wire_sum; end
            if (state_current == ST_MEAN_WAIT && wire_recip_valid[0]) begin reg_length_inverse <= wire_recip[31:0]; end
            if (state_current == ST_MEAN_MUL_WAIT && wire_mul_valid[0]) begin reg_mean <= wire_mul[31:0]; end
            if (state_current == ST_EPS_WAIT && wire_add_valid) begin reg_root_input <= wire_add[31:0]; end
            if (state_current == ST_ROOT_WAIT && wire_root_valid) begin
                reg_coefficient <= wire_root_result;
                reg_statistics_data <= {96'd0,wire_root_result};
                flag_coefficient_valid <= 1;
                reg_coefficient_length <= reg_length;
                reg_coefficient_operation <= reg_operation;
            end
            if (state_current == ST_RECIP_WAIT && wire_recip_valid[0]) begin
                reg_coefficient <= flag_exp_statistics && reg_sum == 0 ? 32'd0 : wire_recip[31:0];
                reg_statistics_data <= {96'd0,(flag_exp_statistics && reg_sum == 0 ? 32'd0 : wire_recip[31:0])};
                flag_coefficient_valid <= 1;
                reg_coefficient_length <= reg_length;
                reg_coefficient_operation <= reg_operation;
            end
        end
    end
endmodule
