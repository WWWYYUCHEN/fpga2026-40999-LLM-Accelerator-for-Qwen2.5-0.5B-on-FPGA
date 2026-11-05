`timescale 1ns / 1ps
`default_nettype none

// Four independent AXI readers prefetch consecutive bursts into banked BRAM.
// The consumer sees the original descriptor order, irrespective of R latency.
module qwen25_mmu_read #(parameter RS_LV = 0) (
    input wire clk, input wire rst_n,
    input wire [127:0] s_cmd, input wire [127:0] s_cmd_meta,
    input wire s_cmd_valid, output wire s_cmd_ready,
    input wire abort_req, input wire soft_reset_req,
    output wire [255:0] m_axi_araddr, output wire [31:0] m_axi_arlen,
    output wire [11:0] m_axi_arsize, output wire [7:0] m_axi_arburst,
    output wire [3:0] m_axi_arvalid, input wire [3:0] m_axi_arready,
    input wire [15:0] m_axi_rid, input wire [511:0] m_axi_rdata,
    input wire [7:0] m_axi_rresp, input wire [3:0] m_axi_rlast,
    input wire [3:0] m_axi_rvalid, output wire [3:0] m_axi_rready,
    output wire [127:0] m_stream_data, output wire [15:0] m_stream_keep,
    output wire m_stream_last, output wire [127:0] m_stream_meta,
    output wire [15:0] m_stream_tag, output wire [3:0] m_stream_resource,
    output wire [1:0] m_stream_error, output wire m_stream_valid,
    input wire m_stream_ready,
    output wire busy, output wire done, output wire error, output wire aborted,
    output wire [7:0] error_code, output wire [15:0] completion_tag,
    output wire [127:0] completion_meta
);
    localparam [1:0] ST_IDLE=0, ST_RUN=1, ST_DRAIN=2;
    reg [1:0] state_current, state_next;
    reg [127:0] reg_command, reg_metadata;
    reg [63:0] reg_address;
    reg [23:0] reg_issue_remaining, reg_output_remaining;
    reg [1:0] reg_issue_port, reg_output_port;
    reg [3:0] flag_pending, flag_terminal, flag_consumed;
    reg [8:0] reg_chunk_beats [0:3];
    reg flag_cancel, flag_recover;
    reg flag_done_o, flag_aborted_o;
    reg [1:0] reg_output_select;
    reg flag_output_valid_o, flag_output_last_o, flag_chunk_last;
    wire [8:0] planned_beats;
    wire [12:0] planned_bytes;
    wire [3:0] lane_ready, lane_busy, lane_done, lane_aborted;
    wire [3:0] lane_data_valid;
    wire [511:0] lane_data;
    wire [63:0] lane_keep;
    wire [3:0] lane_fifo_available;
    wire [575:0] lane_fifo_data;
    wire [3:0] lane_fifo_last;
    wire issue_fire, output_fire, output_slot, cancel_now, lanes_quiet;
    wire [1:0] selected_port;
    wire read_fifo;
    wire [127:0] chunk_command;
    integer port_index;

    qwen25_dma_burst plan_Inst(.address(reg_address),.remaining_bytes(reg_issue_remaining),
        .maximum_len(reg_command[95:88]),.burst_beats(planned_beats),.burst_bytes(planned_bytes));
    assign chunk_command={reg_command[127:88],11'd0,planned_bytes,reg_address};
    assign cancel_now=flag_cancel || abort_req || soft_reset_req;
    assign s_cmd_ready=(state_current==ST_IDLE) && !abort_req && !soft_reset_req;
    assign issue_fire=(state_current==ST_RUN) && !cancel_now && reg_issue_remaining!=0 &&
        !flag_pending[reg_issue_port] && lane_ready[reg_issue_port];
    assign output_fire=flag_output_valid_o && m_stream_ready;
    assign output_slot=!flag_output_valid_o || m_stream_ready;
    assign selected_port=(output_fire && flag_chunk_last) ? reg_output_port+2'd1 : reg_output_port;
    assign read_fifo=(state_current==ST_RUN) && !cancel_now && output_slot && lane_fifo_available[selected_port];
    assign lanes_quiet=(lane_busy==0) && (lane_data_valid==0) && !flag_output_valid_o;
    assign m_stream_data=lane_fifo_data[reg_output_select*144+:128];
    assign m_stream_keep=lane_fifo_data[reg_output_select*144+128+:16];
    assign m_stream_error=2'b00;
    assign m_stream_valid=flag_output_valid_o;
    assign m_stream_last=flag_output_last_o;
    assign m_stream_meta=reg_metadata;
    assign m_stream_tag=reg_command[111:96];
    assign m_stream_resource=reg_command[115:112];
    assign busy=(state_current==ST_RUN) || (state_current==ST_DRAIN);
    assign done=flag_done_o; assign error=1'b0; assign aborted=flag_aborted_o;
    assign error_code=8'd0;
    assign completion_tag=reg_command[111:96]; assign completion_meta=reg_metadata;

    // One full burst fits each bank. Only that bank's engine writes its RAM.
    genvar lane;
    generate for(lane=0;lane<4;lane=lane+1)begin:gen_reader
        (* ram_style="block" *) reg [143:0] return_memory [0:255];
        reg [143:0] reg_memory_output;
        reg [8:0] count_written, count_read;
        wire start_chunk=issue_fire && reg_issue_port==lane;
        wire take_word=read_fifo && selected_port==lane;
        wire [127:0] unused_meta, unused_completion_meta;
        wire [15:0] unused_tag, unused_completion_tag;
        wire [3:0] unused_resource;
        wire unused_last;
        assign lane_fifo_available[lane]=flag_pending[lane] && count_read<count_written;
        assign lane_fifo_data[lane*144+:144]=reg_memory_output;
        assign lane_fifo_last[lane]=count_read==reg_chunk_beats[lane]-1'b1;
        always @(posedge clk)begin
            if(rst_n==RS_LV)begin count_written<=0;count_read<=0;reg_memory_output<=0;end
            else begin
                if(start_chunk)begin count_written<=0;count_read<=0;end
                else begin
                    if(lane_data_valid[lane] && !flag_cancel)begin
                        return_memory[count_written[7:0]]<={lane_keep[lane*16+:16],lane_data[lane*128+:128]};
                        count_written<=count_written+1'b1;
                    end
                    if(take_word)begin
                        reg_memory_output<=return_memory[count_read[7:0]];
                        count_read<=count_read+1'b1;
                    end
                end
            end
        end
        qwen25_dma_rd #(.RS_LV(RS_LV)) reader_Inst(
            .clk(clk),.rst_n(rst_n),.s_cmd(chunk_command),.s_cmd_meta(reg_metadata),
            .s_cmd_valid(start_chunk),.s_cmd_ready(lane_ready[lane]),
            .abort_req(flag_cancel || abort_req),.soft_reset_req(soft_reset_req),
            .m_axi_araddr(m_axi_araddr[lane*64+:64]),.m_axi_arlen(m_axi_arlen[lane*8+:8]),
            .m_axi_arsize(m_axi_arsize[lane*3+:3]),.m_axi_arburst(m_axi_arburst[lane*2+:2]),
            .m_axi_arvalid(m_axi_arvalid[lane]),.m_axi_arready(m_axi_arready[lane]),
            .m_axi_rid(m_axi_rid[lane*4+:4]),.m_axi_rdata(m_axi_rdata[lane*128+:128]),
            .m_axi_rresp(m_axi_rresp[lane*2+:2]),.m_axi_rlast(m_axi_rlast[lane]),
            .m_axi_rvalid(m_axi_rvalid[lane]),.m_axi_rready(m_axi_rready[lane]),
            .m_stream_data(lane_data[lane*128+:128]),.m_stream_keep(lane_keep[lane*16+:16]),
            .m_stream_error(),.m_stream_valid(lane_data_valid[lane]),
            .m_stream_ready(1'b1),.m_stream_last(unused_last),.m_stream_meta(unused_meta),
            .m_stream_tag(unused_tag),.m_stream_resource(unused_resource),
            .busy(lane_busy[lane]),.done(lane_done[lane]),.error(),
            .aborted(lane_aborted[lane]),.error_code(),
            .completion_tag(unused_completion_tag),.completion_meta(unused_completion_meta));
    end endgenerate

    always @(*)begin
        state_next=state_current;
        case(state_current)
            ST_IDLE:if(s_cmd_valid && s_cmd_ready)begin
                state_next=ST_RUN;
            end
            ST_RUN:begin
                if(cancel_now)state_next=ST_DRAIN;
                else if(reg_issue_remaining==0 && reg_output_remaining==0 && flag_pending==0 && lanes_quiet)state_next=ST_IDLE;
            end
            ST_DRAIN:if(lanes_quiet)begin
                state_next=ST_IDLE;
            end
            default:state_next=ST_IDLE;
        endcase
    end
    always @(posedge clk)begin
        if(rst_n==RS_LV)state_current<=ST_IDLE;else state_current<=state_next;
    end
    always @(posedge clk)begin
        if(rst_n==RS_LV)begin
            reg_command<=0;reg_metadata<=0;reg_address<=0;
            reg_issue_remaining<=0;reg_output_remaining<=0;
            reg_issue_port<=0;reg_output_port<=0;
            flag_pending<=0;flag_terminal<=0;flag_consumed<=0;
            flag_cancel<=0;flag_recover<=0;
            flag_done_o<=0;flag_aborted_o<=0;
            reg_output_select<=0;flag_output_valid_o<=0;flag_output_last_o<=0;flag_chunk_last<=0;
            for(port_index=0;port_index<4;port_index=port_index+1)reg_chunk_beats[port_index]<=0;
        end else begin
            flag_done_o<=0;flag_aborted_o<=0;
            if(output_fire)flag_output_valid_o<=0;
            if(busy && cancel_now)flag_cancel<=1;
            if(busy && soft_reset_req)flag_recover<=1;
            if(read_fifo)begin
                reg_output_select<=selected_port;
                flag_output_valid_o<=1;
                flag_chunk_last<=lane_fifo_last[selected_port];
                // The prior output may retire on the same clock as this read.
                flag_output_last_o<=reg_output_remaining<=(output_fire ? 24'd32 : 24'd16);
            end
            if(output_fire)begin
                reg_output_remaining<=reg_output_remaining>16 ? reg_output_remaining-24'd16 : 0;
                if(flag_chunk_last)begin
                    flag_consumed[reg_output_port]<=1;reg_output_port<=reg_output_port+1'b1;
                end
            end
            for(port_index=0;port_index<4;port_index=port_index+1)begin
                if(lane_done[port_index] || lane_aborted[port_index])flag_terminal[port_index]<=1;
                if(flag_terminal[port_index] && flag_consumed[port_index])flag_pending[port_index]<=0;
            end
            case(state_current)
                ST_IDLE:begin
                    flag_cancel<=0;flag_recover<=0;
                    if(s_cmd_valid && s_cmd_ready)begin
                        reg_command<=s_cmd;reg_metadata<=s_cmd_meta;reg_address<=s_cmd[63:0];
                        reg_issue_remaining<=s_cmd[87:64];reg_output_remaining<=s_cmd[87:64];
                        reg_issue_port<=0;reg_output_port<=0;
                        flag_pending<=0;flag_terminal<=0;flag_consumed<=0;
                    end
                end
                ST_RUN:begin
                    if(issue_fire)begin
                        flag_pending[reg_issue_port]<=1;flag_terminal[reg_issue_port]<=0;flag_consumed[reg_issue_port]<=0;
                        reg_chunk_beats[reg_issue_port]<=planned_beats;
                        reg_issue_port<=reg_issue_port+1'b1;
                        reg_address<=reg_address+{51'd0,planned_beats,4'd0};
                        reg_issue_remaining<=reg_issue_remaining-{11'd0,planned_bytes};
                    end
                    if(state_next==ST_IDLE)flag_done_o<=1;
                end
                ST_DRAIN:if(lanes_quiet)begin
                    flag_pending<=0;
                flag_aborted_o<=1;
                end
                default:begin end
            endcase
        end
    end
endmodule
`default_nettype wire
