`timescale 1ns / 1ps
`default_nettype none
// -----------------------------------------------------------------------------
// qwen25_score_ingress -- INT8-KV variant (stage 3 of the INT8 attention landing)
//
// Difference from rtl/attention/qwen25_score_ingress.v
//   1) the fixed 1/8 exponent decrement of the score word (:90-94) is DELETED --
//      the 1/8 now lives inside the KV scale table (kv8_scale_ram stores s_k/8);
//   2) kv8_score_scale8 is inserted in front of stage 0, so each of the 8 lanes
//      becomes  score * s_k/8  (the per-(head,token) INT8 K dequantisation);
//   3) one new input, i_scale[127:0] = 8 x m16 plus i_eb[7:0] (the block exponent),
//      supplied together with i_data for exactly the same word.  In logits mode
//      i_scale/i_eb are ignored and
//      the raw word is passed through (the multiplier is bypassed, exactly like
//      the original).
//
// Alignment contract (this is the thing that silently kills a corr-0.99 run if
// it is wrong): the 8 scales of i_scale must belong to the 8 tokens of i_data,
// i.e. both words are indexed by {head, token[14:3]}.  The testbench drives them
// from the same word index and checks that the value of every lane is
// score[token] * scale[token] (a swapped/misaligned scale changes the lane
// value, so it cannot pass).
//
// Timing cost: the multiplier is 2 cycles deep with no back-pressure inside, so
// the input side accepts a word only when the previous one has left it
// (o_ready = stage_ready[0] && !inflight).  Score words per tile are tens, so a
// 3x slower ingest is ~100 extra cycles against a GEMM that takes thousands.
// -----------------------------------------------------------------------------
module qwen25_score_ingress (
    input wire i_clk,
    input wire i_rstn,
    input wire i_valid,
    output wire o_ready,
    input wire i_logits,
    input wire i_bank,
    input wire [14:0] i_address,
    input wire [127:0] i_data,
    input wire [127:0] i_scale,        // 8 x m16 (MANT_BITS=16); ignored in logits mode
    input wire [7:0]   i_eb,           // [M16-SCALE] block exponent of that scale word
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

    // ------------------------------------------------------------- input pipe
    reg         inflight;
    reg [14:0]  p_address;
    reg         p_bank, p_logits;
    reg [2:0]   p_row;
    reg         p_first, p_last;
    reg [15:0]  p_words;
    reg [127:0] p_raw0, p_raw1;      // raw word, delayed to the scaler output
    reg [14:0]  p_address1;
    reg         p_bank1, p_logits1;
    reg [2:0]   p_row1;
    reg         p_first1, p_last1;
    reg [15:0]  p_words1;

    wire             mul_valid;
    wire [127:0]     mul_data;
    wire             accept = i_valid && o_ready;
    wire [31:0]      in_base = i_logits ? {14'd0,i_address,3'd0} : {17'd0,i_address[11:0],3'd0};
    wire [31:0]      input_words = (i_length+32'd7)>>3;

    // [M16-SCALE] the K scale arrives as 8 x m16 + the block exponent, so the
    // multiply is Sa(8b) x m16(16b) with one rounding, C = 268 (see the module).
    kv8_score_scale8 #(.MANT_BITS(16), .CEX(268)) scale_Inst (
        .clk(i_clk), .rst_n(i_rstn),
        .i_valid(accept), .i_score(i_data), .i_scale(i_scale), .i_eb(i_eb),
        .o_valid(mul_valid), .o_score(mul_data)
    );

    assign stage_ready[4]=!stage_valid[4] || i_ready;
    assign o_ready=stage_ready[0] && !inflight;
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
    assign o_busy=|stage_valid | inflight;

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
            // ---- stage 0: INT8-K scale applied, mask/index prepared ----
            // score mode : y = score * s_k/8   (kv8_score_scale8, 2 cycles)
            // logits mode: y = score           (multiplier bypassed)
            wire [15:0] lane_scaled = p_logits1 ? p_raw1[lane*16 +: 16] : mul_data[lane*16 +: 16];
            wire [31:0] column = (p_logits1 ? {14'd0,p_address1,3'd0} : {17'd0,p_address1[11:0],3'd0}) + lane;
            wire [16:0] causal_limit = {1'b0,i_position} + {14'd0,p_address1[14:12]};
            reg [31:0] scaled_index;
            always @(posedge i_clk)begin
                if(mul_valid)begin
                    stage_data[0][lane*16 +: 16]<=lane_scaled;
                    scaled_mask[lane]<=column<i_length && (p_logits1 || column<={15'd0,causal_limit});
                    scaled_index<=p_logits1 ? ~column : column;
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
        if(!i_rstn)begin
            stage_valid<=0;
            inflight<=0;
        end else begin
            if(stage_ready[0])stage_valid[0]<=mul_valid;
            for(valid_stage=1;valid_stage<5;valid_stage=valid_stage+1)begin
                if(stage_ready[valid_stage])stage_valid[valid_stage]<=stage_valid[valid_stage-1];
            end
            if(accept && !mul_valid)inflight<=1'b1;
            if(mul_valid)inflight<=1'b0;
        end
        // input-side capture (one word in flight, latency matched to the scaler)
        if(accept)begin
            p_address<=i_address;p_bank<=i_bank;p_logits<=i_logits;
            p_row<=i_logits ? 3'd0 : i_address[14:12];
            p_first<=in_base==0;
            p_last<=in_base+32'd8>=i_length;
            p_words<=input_words[15:0];
            p_raw0<=i_data;
            p_address1<=i_address;p_bank1<=i_bank;p_logits1<=i_logits;
            p_row1<=i_logits ? 3'd0 : i_address[14:12];
            p_first1<=in_base==0;
            p_last1<=in_base+32'd8>=i_length;
            p_words1<=input_words[15:0];
            p_raw1<=i_data;
        end
        // stage 0 receives the metadata of the word whose scaled data just arrived
        if(mul_valid)begin
            stage_address[0]<=p_address1;stage_bank[0]<=p_bank1;stage_logits[0]<=p_logits1;
            stage_row[0]<=p_row1;
            stage_first[0]<=p_first1;
            stage_last[0]<=p_last1;
            stage_words[0]<=p_words1;
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
