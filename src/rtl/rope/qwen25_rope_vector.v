`timescale 1ns / 1ps
`default_nettype none

// Eight split-half lanes share one 32-bit RoPE pipeline.
// Accepted frames complete during abort drainage; only global reset cancels them.
module qwen25_rope_vector (
    input wire i_clk,
    input wire i_rstn,
    input wire i_valid,
    output wire o_ready,
    input wire [127:0] i_x,
    input wire [127:0] i_y,
    input wire [127:0] i_meta,
    input wire i_last,
    input wire [7:0] i_mask,
    output wire o_valid,
    input wire i_ready,
    output wire [127:0] o_data,
    output wire [127:0] o_aux,
    output wire [127:0] o_meta,
    output wire o_last,
    output wire [7:0] o_mask,
    output wire o_busy
);
    reg flag_active;
    reg [127:0] reg_first;
    reg [127:0] reg_second;
    reg [127:0] reg_metadata;
    reg flag_last;
    reg [7:0] reg_mask;
    reg [127:0] reg_output_first;
    reg [127:0] reg_output_second;
    reg [3:0] cnt_angle_issue;
    reg [3:0] cnt_angle_return;
    reg [3:0] cnt_pair_send;
    reg [3:0] cnt_pair_return;
    reg [31:0] reg_angles [0:7];
    reg [7:0] reg_position_exponent;
    reg [31:0] reg_inverse_frequency;
    reg [31:0] reg_normalized_position;
    wire [15:0] wire_position = reg_metadata[104:89];
    wire [31:0] wire_position_float = wire_position == 0 ? 32'd0 :
        {1'b0,reg_position_exponent,reg_normalized_position[22:0]};
    wire [4:0] wire_frequency = {reg_metadata[110:109],cnt_angle_issue[2:0]};
    wire flag_angle_issue = flag_active && cnt_angle_issue < 8;
    wire flag_angle_return;
    wire [31:0] wire_angle;
    wire flag_pair_valid = flag_active && cnt_pair_send < cnt_angle_return;
    wire flag_pair_ready;
    wire flag_pair_fire = flag_pair_valid && flag_pair_ready;
    wire [31:0] wire_pair = {reg_second[cnt_pair_send[2:0]*16 +: 16],
                             reg_first[cnt_pair_send[2:0]*16 +: 16]};
    wire flag_result_valid;
    wire flag_result_ready = flag_active && cnt_pair_return < 8;
    wire flag_result_fire = flag_result_valid && flag_result_ready;
    wire [31:0] wire_result;

    assign o_ready = !flag_active;
    assign o_valid = flag_active && cnt_pair_return == 8;
    assign o_data = reg_output_first;
    assign o_aux = reg_output_second;
    assign o_meta = reg_metadata;
    assign o_last = flag_last;
    assign o_mask = reg_mask;
    assign o_busy = flag_active;

    // All 16-bit unsigned positions are exactly representable in binary32.
    always @(*) begin
        reg_position_exponent = 142;
        reg_normalized_position = 0;
        casez (wire_position)
            16'b1???????????????: begin reg_position_exponent=142; reg_normalized_position=wire_position << 8; end
            16'b01??????????????: begin reg_position_exponent=141; reg_normalized_position=wire_position << 9; end
            16'b001?????????????: begin reg_position_exponent=140; reg_normalized_position=wire_position << 10; end
            16'b0001????????????: begin reg_position_exponent=139; reg_normalized_position=wire_position << 11; end
            16'b00001???????????: begin reg_position_exponent=138; reg_normalized_position=wire_position << 12; end
            16'b000001??????????: begin reg_position_exponent=137; reg_normalized_position=wire_position << 13; end
            16'b0000001?????????: begin reg_position_exponent=136; reg_normalized_position=wire_position << 14; end
            16'b00000001????????: begin reg_position_exponent=135; reg_normalized_position=wire_position << 15; end
            16'b000000001???????: begin reg_position_exponent=134; reg_normalized_position=wire_position << 16; end
            16'b0000000001??????: begin reg_position_exponent=133; reg_normalized_position=wire_position << 17; end
            16'b00000000001?????: begin reg_position_exponent=132; reg_normalized_position=wire_position << 18; end
            16'b000000000001????: begin reg_position_exponent=131; reg_normalized_position=wire_position << 19; end
            16'b0000000000001???: begin reg_position_exponent=130; reg_normalized_position=wire_position << 20; end
            16'b00000000000001??: begin reg_position_exponent=129; reg_normalized_position=wire_position << 21; end
            16'b000000000000001?: begin reg_position_exponent=128; reg_normalized_position=wire_position << 22; end
            16'b0000000000000001: begin reg_position_exponent=127; reg_normalized_position=wire_position << 23; end
            default: begin reg_position_exponent=0; end
        endcase
    end

    // FP32 constants: inverse_frequency[j] = 1000000^(-j/32), head_dim=64.
    always @(*) begin
        case (wire_frequency)
            0: reg_inverse_frequency=32'h3f800000;
            1: reg_inverse_frequency=32'h3f263de0;
            2: reg_inverse_frequency=32'h3ed7e89b;
            3: reg_inverse_frequency=32'h3e8c3504;
            4: reg_inverse_frequency=32'h3e361887;
            5: reg_inverse_frequency=32'h3dec7fd5;
            6: reg_inverse_frequency=32'h3d99940e;
            7: reg_inverse_frequency=32'h3d47763f;
            8: reg_inverse_frequency=32'h3d0186e2;
            9: reg_inverse_frequency=32'h3ca8398b;
            10: reg_inverse_frequency=32'h3c5a7bf1;
            11: reg_inverse_frequency=32'h3c0de12d;
            12: reg_inverse_frequency=32'h3bb8449c;
            13: reg_inverse_frequency=32'h3b6f520d;
            14: reg_inverse_frequency=32'h3b1b690d;
            15: reg_inverse_frequency=32'h3ac9d75c;
            16: reg_inverse_frequency=32'h3a83126f;
            17: reg_inverse_frequency=32'h3a2a3b43;
            18: reg_inverse_frequency=32'h39dd1726;
            19: reg_inverse_frequency=32'h398f9272;
            20: reg_inverse_frequency=32'h393a7753;
            21: reg_inverse_frequency=32'h38f22ce3;
            22: reg_inverse_frequency=32'h389d43a4;
            23: reg_inverse_frequency=32'h384c3fbd;
            24: reg_inverse_frequency=32'h3804a2b3;
            25: reg_inverse_frequency=32'h37ac431d;
            26: reg_inverse_frequency=32'h375fba4f;
            27: reg_inverse_frequency=32'h371148e2;
            28: reg_inverse_frequency=32'h36bcb0c1;
            29: reg_inverse_frequency=32'h36751070;
            30: reg_inverse_frequency=32'h361f23e4;
            31: reg_inverse_frequency=32'h35ceaf79;
            default: reg_inverse_frequency=32'h3f800000;
        endcase
    end

    wire flag_angle_return_raw;
    qwen_operator_valid_guard #(.C_LATENCY(3)) angle_valid_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_issue(flag_angle_issue),
        .i_operator_valid(flag_angle_return_raw),.o_valid(flag_angle_return)
    );
    qwen25_afu_fp_mul angle_multiply_Inst (
        .aclk(i_clk),
        .s_axis_a_tvalid(flag_angle_issue),.s_axis_a_tdata(wire_position_float),.s_axis_a_tuser(16'd0),
        .s_axis_b_tvalid(flag_angle_issue),.s_axis_b_tdata(reg_inverse_frequency),
        .m_axis_result_tvalid(flag_angle_return_raw),.m_axis_result_tdata(wire_angle),.m_axis_result_tuser()
    );
    qwen25_rope pair_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_valid(flag_pair_valid),.o_ready(flag_pair_ready),
        .i_data(wire_pair),.i_angle(reg_angles[cnt_pair_send[2:0]]),
        .o_valid(flag_result_valid),.i_ready(flag_result_ready),.o_data(wire_result),.o_busy()
    );

    // One frame reserves all eight angle and output slots before issuing any work.
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            flag_active <= 0;
            cnt_angle_issue <= 0;
            cnt_angle_return <= 0;
            cnt_pair_send <= 0;
            cnt_pair_return <= 0;
        end else begin
            if (i_valid && o_ready) begin
                flag_active <= 1;
                reg_first <= i_x;
                reg_second <= i_y;
                reg_output_first <= i_x;
                reg_output_second <= i_y;
                reg_metadata <= i_meta;
                flag_last <= i_last;
                reg_mask <= i_mask;
                cnt_angle_issue <= 0;
                cnt_angle_return <= 0;
                cnt_pair_send <= 0;
                cnt_pair_return <= 0;
            end
            if (flag_angle_issue) begin cnt_angle_issue <= cnt_angle_issue+1'b1; end
            if (flag_angle_return && flag_active && cnt_angle_return < 8) begin
                reg_angles[cnt_angle_return[2:0]] <= wire_angle;
                cnt_angle_return <= cnt_angle_return+1'b1;
            end
            if (flag_pair_fire) begin cnt_pair_send <= cnt_pair_send+1'b1; end
            if (flag_result_fire) begin
                if (reg_mask[cnt_pair_return[2:0]]) begin
                    reg_output_first[cnt_pair_return[2:0]*16 +: 16] <= wire_result[15:0];
                    reg_output_second[cnt_pair_return[2:0]*16 +: 16] <= wire_result[31:16];
                end
                cnt_pair_return <= cnt_pair_return+1'b1;
            end
            if (o_valid && i_ready) begin flag_active <= 0; end
        end
    end
endmodule
`default_nettype wire
