`timescale 1ns / 1ps
`default_nettype none
// Ordered single-word DDR service for Q, K, V and Context writes.
// Attention K/V operands are staged only in the shared Weight buffer.
module qwen25_attention_dma (
    input wire i_clk,
    input wire i_rstn,
    input wire i_clear,
    input wire i_abort,
    input wire [127:0] i_identity,
    input wire [7:0] i_burst_limit,
    input wire [63:0] i_context_base,
    input wire i_cache_invalidate,
    input wire [15:0] i_cache_context,
    input wire [63:0] i_cache_k_base,
    input wire [63:0] i_cache_v_base,
    input wire i_read_valid,
    output wire o_read_ready,
    input wire [63:0] i_read_address,
    input wire [63:0] i_read_base,
    input wire [12:0] i_read_bytes,
    output wire o_data_valid,
    input wire i_data_ready,
    output wire [127:0] o_data,
    input wire i_context_valid,
    output wire o_context_ready,
    input wire [127:0] i_context_data,
    input wire [2:0] i_context_row,
    input wire [3:0] i_context_head,
    input wire [2:0] i_context_word,
    output wire o_busy,
    output wire o_error,
    output wire o_cmd_valid,
    input wire i_cmd_ready,
    output wire [127:0] o_cmd,
    output wire [127:0] o_meta,
    input wire i_rd_valid,
    output wire o_rd_ready,
    input wire [127:0] i_rd_data,
    input wire [15:0] i_rd_keep,
    input wire i_rd_last,
    input wire [127:0] i_rd_meta,
    input wire [15:0] i_rd_tag,
    input wire [3:0] i_rd_resource,
    input wire [1:0] i_rd_beat_error,
    input wire i_rd_done,
    input wire i_rd_error,
    input wire i_rd_aborted,
    output wire o_wr_valid,
    input wire i_wr_ready,
    output wire [127:0] o_wr_data,
    output wire [127:0] o_wr_meta,
    output wire o_wr_last,
    output wire [15:0] o_wr_keep,
    input wire i_wr_done,
    input wire i_wr_error,
    input wire i_wr_aborted
);
    localparam [4:0] S_IDLE=0,S_COMMAND=1,S_READ=2,S_READ_END=3,S_RESPONSE=4,S_WRITE=5,S_WRITE_END=6,
        S_CONTEXT_OFFSET=14,S_CONTEXT_LOW=15,S_CONTEXT_HIGH=16;
    reg [4:0] state_current,state_next;
    reg [127:0] reg_command,reg_metadata,reg_data;
    reg [15:0] cnt_tag;
    reg flag_write,flag_read_done;
    reg [63:0] reg_context_base;
    reg [13:0] reg_context_row_offset;
    reg [10:0] reg_context_head_offset;
    reg [14:0] reg_context_offset;
    reg [32:0] reg_context_address_low;
    wire [31:0] context_address_high=reg_context_base[63:32]+{31'd0,reg_context_address_low[32]};

    assign o_read_ready=i_rstn && state_current==S_IDLE && !i_context_valid;
    assign o_context_ready=i_rstn && state_current==S_IDLE;
    assign o_data_valid=state_current==S_RESPONSE;
    assign o_data=reg_data;
    assign o_busy=state_current!=S_IDLE;
    assign o_error=1'b0;
    assign o_cmd_valid=state_current==S_COMMAND;
    assign o_cmd=reg_command;
    assign o_meta=reg_metadata;
    assign o_rd_ready=state_current==S_READ;
    assign o_wr_valid=state_current==S_WRITE;
    assign o_wr_data=reg_data;
    assign o_wr_meta=reg_metadata;
    assign o_wr_last=1'b1;
    assign o_wr_keep=16'hffff;
    always @(*)begin
        state_next=state_current;
        case(state_current)
            S_IDLE:begin
                if(i_context_valid && o_context_ready)state_next=S_CONTEXT_OFFSET;
                else if(i_read_valid && o_read_ready)state_next=S_COMMAND;
            end
            S_CONTEXT_OFFSET:state_next=S_CONTEXT_LOW;
            S_CONTEXT_LOW:state_next=S_CONTEXT_HIGH;
            S_CONTEXT_HIGH:state_next=S_COMMAND;
            S_COMMAND:if(i_cmd_ready)state_next=flag_write ? S_WRITE : S_READ;
            S_READ:if(i_rd_valid && i_rd_last)state_next=S_READ_END;
            S_READ_END:if(flag_read_done || i_rd_done)state_next=S_RESPONSE;
            S_RESPONSE:if(i_data_ready)state_next=S_IDLE;
            S_WRITE:if(i_wr_ready)state_next=S_WRITE_END;
            S_WRITE_END:if(i_wr_done)state_next=S_IDLE;
            default:state_next=S_IDLE;
        endcase
    end
    always @(posedge i_clk)begin
        if(!i_rstn)state_current<=S_IDLE;else state_current<=state_next;
    end
    always @(posedge i_clk)begin
        if(!i_rstn)begin
            cnt_tag<=0;flag_write<=0;flag_read_done<=0;
            reg_command<=0;reg_metadata<=0;
            reg_context_base<=0;reg_context_row_offset<=0;reg_context_head_offset<=0;reg_context_offset<=0;
            reg_context_address_low<=0;
        end else begin
            if(state_current==S_IDLE)begin
                flag_read_done<=0;
                if((i_read_valid && o_read_ready) || (i_context_valid && o_context_ready))begin
                    flag_write<=i_context_valid;
                    reg_context_base<=i_context_base;
                    reg_context_row_offset<=i_context_row*14'd1792;
                    reg_context_head_offset<={i_context_head,7'd0}+{4'd0,i_context_word,4'd0};
                    reg_command<=0;reg_command[63:0]<=i_read_address;
                    reg_command[87:64]<=24'd16;reg_command[95:88]<=i_burst_limit;
                    reg_command[111:96]<=cnt_tag;reg_command[115:112]<=i_context_valid ? 4'd5 : 4'd2;
                    reg_command[118]<=i_context_valid;reg_metadata<=i_identity;
                    cnt_tag<=cnt_tag+1'b1;
                    if(i_context_valid)reg_data<=i_context_data;
                end
            end
            if(state_current==S_CONTEXT_OFFSET)
                reg_context_offset<={1'b0,reg_context_row_offset}+{4'd0,reg_context_head_offset};
            if(state_current==S_CONTEXT_LOW)
                reg_context_address_low<={1'b0,reg_context_base[31:0]}+{18'd0,reg_context_offset};
            if(state_current==S_CONTEXT_HIGH)
                reg_command[63:0]<={context_address_high,reg_context_address_low[31:0]};
            if(i_rd_done)flag_read_done<=1;
            if(state_current==S_READ && i_rd_valid && o_rd_ready)reg_data<=i_rd_data;
        end
    end
endmodule
`default_nettype wire
