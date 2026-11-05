`timescale 1ns / 1ps
`default_nettype none
// Score mode scales/masks; logits mode preserves values and reduces into row zero.
module qwen25_score_ingress (
    input wire i_clk,
    input wire i_rstn,
    input wire i_valid,
    output wire o_ready,
    input wire i_logits,
    input wire i_bank,
    input wire [14:0] i_address,
    input wire [127:0] i_data,
    input wire [31:0] i_length,
    input wire [15:0] i_position,
    output wire o_valid,
    input wire i_ready,
    output wire o_bank,
    output wire [14:0] o_address,
    output wire [127:0] o_data,
    output wire [2:0] o_max_row,
    output wire [15:0] o_maximum,
    output wire [31:0] o_index,
    output wire o_first,
    output wire o_last,
    output wire [15:0] o_words,
    output wire o_busy
);
    reg [4:0] stage_valid;
    wire [4:0] stage_ready;
    reg [127:0] stage_data[0:4];
    reg [14:0] stage_address[0:4];
    reg stage_bank[0:4];
    reg stage_logits[0:4];
    reg [2:0] stage_row[0:4];
    reg stage_first[0:4],stage_last[0:4];
    reg [15:0] stage_words[0:4];
    reg [7:0] scaled_mask;
    reg [255:0] masked_indices;
    reg [63:0] pair_values;
    reg [127:0] pair_indices;
    reg [31:0] quad_values;
    reg [63:0] quad_indices;
    reg [15:0] word_value;
    reg [31:0] word_index;
    wire [63:0] pair_result;
    wire [127:0] pair_index;
    wire [31:0] quad_result;
    wire [63:0] quad_index;
    wire [15:0] word_result;
    wire [31:0] word_result_index;
    wire [31:0] input_base=i_logits ? {14'd0,i_address,3'd0} :
        {17'd0,i_address[11:0],3'd0};
    wire [31:0] input_words=(i_length+32'd7)>>3;
    wire [16:0] causal_limit={1'b0,i_position}+{14'd0,i_address[14:12]};
    assign stage_ready[4]=!stage_valid[4] || i_ready;
    assign o_ready=stage_ready[0];
    assign o_valid=stage_valid[4];
    assign o_bank=stage_bank[4];
    assign o_address=stage_address[4];
    assign o_data=stage_data[4];
    assign o_max_row=stage_row[4];
    assign o_first=stage_first[4];
    assign o_last=stage_last[4];
    assign o_words=stage_words[4];
    assign o_maximum=word_value;
    // Invert only the indices in logits mode: the existing lowest-index tree
    // then chooses the most recently arriving token on equal BF16 values.
    assign o_index=stage_logits[4] ? ~word_index : word_index;
    assign o_busy=|stage_valid;
    genvar stage,lane,pair,quad;
    generate
        for(stage=0;stage<4;stage=stage+1)begin:gen_ready
            assign stage_ready[stage]=!stage_valid[stage] || stage_ready[stage+1];
        end
        for(stage=1;stage<5;stage=stage+1)begin:gen_metadata
            always @(posedge i_clk)begin
                if(stage_ready[stage] && stage_valid[stage-1])begin
                    stage_address[stage]<=stage_address[stage-1];
                    stage_bank[stage]<=stage_bank[stage-1];
                    stage_logits[stage]<=stage_logits[stage-1];
                    stage_row[stage]<=stage_row[stage-1];
                    stage_first[stage]<=stage_first[stage-1];
                    stage_last[stage]<=stage_last[stage-1];
                    stage_words[stage]<=stage_words[stage-1];
                end
            end
        end
        for(lane=0;lane<8;lane=lane+1)begin:gen_lane
            wire [15:0] value=i_data[lane*16 +: 16];
            wire [7:0] exponent=value[14:7];
            wire [7:0] scaled_exponent=exponent-8'd3;
            wire [15:0] scaled_value=exponent==8'hff ?
                (value[6:0]!=0 ? 16'h7fc0 : value) :
                (exponent<=3 ? {value[15],15'd0} : {value[15],scaled_exponent,value[6:0]});
            wire [31:0] column=input_base+lane;
            // The index is prepared in stage zero, including the logits physical-row bits.
            reg [31:0] scaled_index;
            always @(posedge i_clk)begin
                if(stage_ready[0] && i_valid)begin
                    stage_data[0][lane*16 +: 16]<=i_logits ? value : scaled_value;
                    scaled_mask[lane]<=column<i_length && (i_logits || column<={15'd0,causal_limit});
                    scaled_index<=i_logits ? ~column : column;
                end
                if(stage_ready[1] && stage_valid[0])begin
                    stage_data[1][lane*16 +: 16]<=scaled_mask[lane] ? stage_data[0][lane*16 +: 16] : 16'hff80;
                    masked_indices[lane*32 +: 32]<=scaled_mask[lane] ? scaled_index : 32'hffffffff;
                end
            end
        end
        for(pair=0;pair<4;pair=pair+1)begin:gen_pair
            qwen25_bf16_argmax compare_Inst (
                .i_a(stage_data[1][pair*32 +: 16]),.i_a_index(masked_indices[pair*64 +: 32]),
                .i_b(stage_data[1][pair*32+16 +: 16]),.i_b_index(masked_indices[pair*64+32 +: 32]),
                .o_value(pair_result[pair*16 +: 16]),.o_index(pair_index[pair*32 +: 32])
            );
        end
        for(quad=0;quad<2;quad=quad+1)begin:gen_quad
            qwen25_bf16_argmax compare_Inst (
                .i_a(pair_values[quad*32 +: 16]),.i_a_index(pair_indices[quad*64 +: 32]),
                .i_b(pair_values[quad*32+16 +: 16]),.i_b_index(pair_indices[quad*64+32 +: 32]),
                .o_value(quad_result[quad*16 +: 16]),.o_index(quad_index[quad*32 +: 32])
            );
        end
    endgenerate
    qwen25_bf16_argmax word_Inst (
        .i_a(quad_values[15:0]),.i_a_index(quad_indices[31:0]),
        .i_b(quad_values[31:16]),.i_b_index(quad_indices[63:32]),
        .o_value(word_result),.o_index(word_result_index)
    );
    integer valid_stage;
    always @(posedge i_clk)begin
        if(!i_rstn)stage_valid<=0;
        else begin
            if(stage_ready[0])stage_valid[0]<=i_valid;
            for(valid_stage=1;valid_stage<5;valid_stage=valid_stage+1)begin
                if(stage_ready[valid_stage])stage_valid[valid_stage]<=stage_valid[valid_stage-1];
            end
        end
        if(stage_ready[0] && i_valid)begin
            stage_address[0]<=i_address;stage_bank[0]<=i_bank;stage_logits[0]<=i_logits;
            stage_row[0]<=i_logits ? 3'd0 : i_address[14:12];
            stage_first[0]<=input_base==0;
            stage_last[0]<=input_base+32'd8>=i_length;
            stage_words[0]<=input_words[15:0];
        end
        if(stage_ready[2] && stage_valid[1])begin
            stage_data[2]<=stage_data[1];pair_values<=pair_result;pair_indices<=pair_index;
        end
        if(stage_ready[3] && stage_valid[2])begin
            stage_data[3]<=stage_data[2];quad_values<=quad_result;quad_indices<=quad_index;
        end
        if(stage_ready[4] && stage_valid[3])begin
            stage_data[4]<=stage_data[3];word_value<=word_result;word_index<=word_result_index;
        end
    end
endmodule
`default_nettype wire
