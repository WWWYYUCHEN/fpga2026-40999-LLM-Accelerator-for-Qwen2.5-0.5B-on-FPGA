`timescale 1ns / 1ps
`default_nettype none
// Independent full-context normalization worker; its P bank stays owned until PV retires.
module qwen25_attention_normalize (
    input wire i_clk,
    input wire i_rstn,
    input wire i_start,
    input wire i_abort,
    input wire [3:0] i_rows,
    input wire [15:0] i_position,
    input wire [15:0] i_context,
    input wire [127:0] i_identity,
    input wire i_bank,
    input wire [15:0] i_row_maximum,
    input wire i_row_complete,
    output wire o_busy,
    output wire o_done,
    output wire o_bank,
    output wire o_read,
    output wire [2:0] o_row,
    output wire [11:0] o_word,
    input wire i_read_valid,
    input wire [127:0] i_read_data,
    output wire o_write,
    output wire [14:0] o_address,
    output wire [127:0] o_data,
    output wire o_valid,
    input wire i_ready,
    output wire [3:0] o_command,
    output wire [1:0] o_phase,
    output wire [63:0] o_length,
    output wire [127:0] o_x,
    output wire [127:0] o_y,
    output wire [127:0] o_meta,
    output wire o_last,
    output wire [7:0] o_mask,
    input wire i_valid,
    output wire o_ready,
    input wire [127:0] i_data,
    input wire [127:0] i_meta,
    input wire i_busy,
    output wire o_error
);
    localparam [2:0] S_IDLE=0,S_STREAM=1,S_RESULTS_WAIT=2,S_NEXT=3,S_DONE=4;
    reg [2:0] state_current,state_next;
    reg [3:0] reg_rows;
    wire [3:0] reg_command=4'd1;
    reg [15:0] reg_position,reg_context;
    reg [127:0] reg_identity;
    // Credits include queued words and RAM responses still in flight.
    reg [127:0] input_words[0:3];
    reg [1:0] input_write_pointer,input_read_pointer;
    reg [2:0] input_count,input_credit;
    reg [12:0] cnt_issue;
    reg [12:0] reg_words,reg_last_word;
    reg flag_last_word;
    reg reg_bank;
    reg [2:0] cnt_row;
    reg [12:0] cnt_word,cnt_retired;
    reg [1:0] reg_phase;
    reg flag_statistics;
    reg row_complete_reg;
    reg [127:0] dec_metadata;
    reg [7:0] dec_mask;
    integer lane;
    wire [12:0] wire_start_words=({1'b0,i_context}+17'd7)>>3;
    wire [12:0] wire_words=reg_words;
    wire [2:0] wire_return_row=i_meta[115:113];
    wire [11:0] wire_return_word=i_meta[48:37];
    wire flag_result=i_valid && o_ready;
    wire flag_vector=reg_phase==2;
    wire input_fire=o_valid && i_ready;
    assign o_busy=state_current!=S_IDLE && state_current!=S_DONE;
    assign o_done=state_current==S_DONE;
    // Select the upcoming bank/first row while idle so status can be sampled
    // before launch; no memory request is issued in this state.
    assign o_bank=state_current==S_IDLE ? i_bank : reg_bank;
    assign o_read=state_current==S_STREAM && row_complete_reg && cnt_issue<wire_words && input_credit<4;
    assign o_row=state_current==S_IDLE ? 3'd0 : cnt_row;
    assign o_word=cnt_issue[11:0];
    assign o_write=flag_result && flag_vector;
    assign o_address={wire_return_row,wire_return_word};
    assign o_data=i_data;
    assign o_valid=state_current==S_STREAM && row_complete_reg && input_count!=0;
    assign o_command=reg_command;
    assign o_phase=reg_phase;
    assign o_length={48'd0,reg_context};
    assign o_x=input_words[input_read_pointer];
    assign o_y={112'd0,i_row_maximum};
    assign o_meta=dec_metadata;
    assign o_last=flag_last_word;
    assign o_mask=dec_mask;
    assign o_ready=o_busy;
    assign o_error=1'b0;
    // Stage the selected P-row completion before the AFU request/ready cone.
    // A phase change keeps the same row; only a row change discards the sample.
    // Its maximum remains stable while the P bank is owned by this worker.
    always @(posedge i_clk)begin
        if(!i_rstn)
            row_complete_reg<=1'b0;
        else if(state_current==S_NEXT && reg_phase==2 && !i_busy)
            row_complete_reg<=1'b0;
        else
            row_complete_reg<=i_row_complete;
    end
    always @(*)begin
        dec_metadata=reg_identity;
        dec_metadata[83:80]=reg_command;
        dec_metadata[52:37]={3'd0,cnt_word};
        dec_metadata[115:113]=cnt_row;
        dec_mask=0;
        for(lane=0;lane<8;lane=lane+1)begin
            dec_mask[lane]=(cnt_word*8+lane<reg_context) && (cnt_word*8+lane<=reg_position+cnt_row);
        end
    end
    always @(*)begin
        state_next=state_current;
        case(state_current)
            S_IDLE:if(i_start)state_next=S_STREAM;
            S_STREAM:if(input_fire && o_last)state_next=S_RESULTS_WAIT;
            S_RESULTS_WAIT:if(flag_vector ? cnt_retired==wire_words : flag_statistics)state_next=S_NEXT;
            S_NEXT:if(!i_busy)state_next=reg_phase==2 && cnt_row==reg_rows-1 ? S_DONE : S_STREAM;
            S_DONE:state_next=S_IDLE;
            default:state_next=S_IDLE;
        endcase
    end
    always @(posedge i_clk)begin
        if(!i_rstn)state_current<=S_IDLE;else state_current<=state_next;
    end
    always @(posedge i_clk)begin
        if(!i_rstn)begin
            reg_rows<=0;reg_position<=0;reg_context<=0;reg_identity<=0;reg_bank<=0;
            reg_words<=0;reg_last_word<=13'h1fff;flag_last_word<=1'b0;
            reg_phase<=0;cnt_row<=0;cnt_word<=0;cnt_retired<=0;flag_statistics<=0;
            cnt_issue<=0;input_count<=0;input_credit<=0;input_write_pointer<=0;input_read_pointer<=0;
        end
        else begin
            if(state_current==S_IDLE && i_start)begin
                reg_rows<=i_rows;reg_position<=i_position;reg_context<=i_context;reg_identity<=i_identity;reg_bank<=i_bank;
                reg_words<=wire_start_words;reg_last_word<=wire_start_words-1'b1;
                flag_last_word<=wire_start_words==13'd1;
                reg_phase<=1;cnt_row<=0;cnt_word<=0;cnt_retired<=0;flag_statistics<=0;
                cnt_issue<=0;input_count<=0;input_credit<=0;input_write_pointer<=0;input_read_pointer<=0;
            end
            if(o_read)cnt_issue<=cnt_issue+1'b1;
            if(i_read_valid)begin
                input_words[input_write_pointer]<=i_read_data;
                input_write_pointer<=input_write_pointer+1'b1;
            end
            if(input_fire)begin
                cnt_word<=cnt_word+1'b1;input_read_pointer<=input_read_pointer+1'b1;
                if(flag_last_word)flag_last_word<=1'b0;
                else if(cnt_word+13'd1==reg_last_word)flag_last_word<=1'b1;
            end
            case({i_read_valid,input_fire})
                2'b10:input_count<=input_count+1'b1;
                2'b01:input_count<=input_count-1'b1;
                default:begin end
            endcase
            case({o_read,input_fire})
                2'b10:input_credit<=input_credit+1'b1;
                2'b01:input_credit<=input_credit-1'b1;
                default:begin end
            endcase
            if(o_write)cnt_retired<=cnt_retired+1'b1;
            if(flag_result && !flag_vector)flag_statistics<=1;
            if(state_current==S_NEXT && !i_busy)begin
                cnt_word<=0;cnt_retired<=0;flag_statistics<=0;
                cnt_issue<=0;input_count<=0;input_credit<=0;input_write_pointer<=0;input_read_pointer<=0;
                flag_last_word<=reg_last_word==0;
                if(reg_phase!=2)reg_phase<=reg_phase+1'b1;
                else begin
                    reg_phase<=1;cnt_row<=cnt_row+1'b1;
                end
            end
        end
    end
endmodule
`default_nettype wire
