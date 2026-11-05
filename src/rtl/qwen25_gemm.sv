`timescale 1ns/1ps
`default_nettype none

// 逐行装载权重、激活广播；每列7个树加法器+1个寄存器反馈累加器。
// K-tail operands must be zero-padded upstream; row/column masks mark output lanes only.
module qwen25_gemm (
    input wire clk,input wire rst_n,
    input wire s_cmd_valid,output wire s_cmd_ready,input wire [127:0] s_cmd,input wire [127:0] s_cmd_meta,
    input wire s_act_valid,output wire s_act_ready,input wire [1023:0] s_act_data,input wire [127:0] s_act_meta,
    input wire s_weight_valid,output wire s_weight_ready,input wire [2047:0] s_weight_data,input wire [127:0] s_weight_meta,
    input wire abort_req,input wire soft_reset_req,
    output wire m_valid,input wire m_ready,
    output wire [2047:0] m_bf16,output wire [4095:0] m_fp32,
    output wire [127:0] m_meta,output wire [2:0] m_slice,
    output wire [7:0] m_row_mask,output wire [15:0] m_col_mask,output wire m_last,
    output wire busy,output wire input_done,output wire partial_done,output wire done,output wire error,output wire aborted,
    output wire [7:0] error_code
);
    typedef enum logic [3:0] {G_IDLE,G_DESC,G_MAC_SCAN,G_SUM_REDUCE,G_READ,G_PACK,G_STREAM,G_FINISH} state_t;
    state_t state_current,state_next;
    // Software cancellation ports are reserved; only rst_n resets the engine.
    wire bundle_valid,bundle_ready,release_active,flush_frontend;
    wire [127:0] command,command_meta,activation_meta,weight_meta;
    wire [1023:0] activation;
    wire weight_load_valid, weight_load_ready;
    wire [2:0] weight_load_row;
    wire weight_load_bank,bundle_weight_bank;
    logic compute_bank;
    // Protect only these control copies; each one drives a different PE K row.
    (* DONT_TOUCH = "yes" *) logic [7:0] compute_bank_row_copies;
    always_ff @(posedge clk)begin
        if(!rst_n)compute_bank_row_copies<=8'd0;
        else if(bundle_valid && bundle_ready)compute_bank_row_copies<={8{bundle_weight_bank}};
    end
    wire input_pending;
    logic [127:0] reg_cmd,reg_meta;
    logic [6:0] issued_count;
    logic [5:0] commit_index;
    logic [2:0] issue_row;
    logic [2:0] issue_slot;
    // Minimum eight issues per block; 64 array credits plus partial pipeline fit 16 identities.
    logic [3:0] block_write_pointer,block_read_pointer;
    logic block_last [0:15];
    wire [3:0] valid_rows=reg_cmd[21:18];
    wire [6:0] block_issue_count={valid_rows,3'b000};
    wire last_issue_row={1'b0,issue_row}==valid_rows-1'b1;
    wire [7:0] valid_columns=reg_cmd[29:22];
    wire first_k=reg_cmd[16],last_k=reg_cmd[17];
    wire array_ready,array_valid,array_busy;
    wire [511:0] array_data;
    wire [159:0] array_meta;
    wire issue_valid=(state_current==G_MAC_SCAN)&&(issued_count<block_issue_count);
    wire issue_fire=issue_valid&&array_ready;
    wire consume_array=(state_current==G_IDLE)||(state_current==G_DESC)||
                       (state_current==G_MAC_SCAN)||(state_current==G_SUM_REDUCE);
    wire array_accept=array_valid&&consume_array;
    wire input_complete=issue_fire && issued_count==block_issue_count-1'b1;
    wire [511:0] partial_read_data;
    wire partial_read_valid;
    wire partial_write,partial_busy;
    logic [2:0] output_slot,output_row;
    wire [5:0] read_address=(state_current==G_READ)?{output_slot,output_row}:array_meta[133:128];
    wire results_complete=!array_busy&&!partial_busy;
    logic [2047:0] reg_bf16;
    logic [4095:0] reg_fp32;
    wire [255:0] rounded_row;
    logic flag_done,flag_input_done,flag_partial_done;
    logic [7:0] output_columns;
    assign bundle_ready=(state_current==G_IDLE);
    // PE operands are sampled by the last issue; arithmetic completion is independent.
    assign release_active=input_complete;
    assign flush_frontend=1'b0;
    assign m_valid=(state_current==G_STREAM);
    assign m_bf16=reg_bf16;
    assign m_fp32=reg_fp32;
    assign m_meta=reg_meta;
    assign m_slice=output_slot;
    assign m_last=(output_slot==7);
    assign m_row_mask=(9'd1<<valid_rows)-9'd1;
    assign m_col_mask=(17'd1<<output_columns)-17'd1;
    assign busy=(state_current!=G_IDLE)||array_busy||partial_busy||input_pending;
    assign input_done=flag_input_done;
    assign partial_done=flag_partial_done;
    assign done=flag_done;
    assign error=1'b0;
    assign aborted=1'b0;
    assign error_code=8'd0;

    qwen_gemm_frontend frontend_Inst(.clk(clk),.rst_n(rst_n),.block_input(1'b0),
        .flush(flush_frontend),.s_cmd_valid(s_cmd_valid),.s_cmd_ready(s_cmd_ready),.s_cmd(s_cmd),.s_cmd_meta(s_cmd_meta),
        .s_act_valid(s_act_valid),.s_act_ready(s_act_ready),.s_act_data(s_act_data),.s_act_meta(s_act_meta),
        .s_weight_valid(s_weight_valid),.s_weight_ready(s_weight_ready),.s_weight_meta(s_weight_meta),
        .m_valid(bundle_valid),.m_ready(bundle_ready),.m_cmd(command),.m_cmd_meta(command_meta),
        .m_act_data(activation),.m_act_meta(activation_meta),.m_weight_meta(weight_meta),
        .weight_meta_error(),.input_pending(input_pending),
        .weight_load_valid(weight_load_valid),.weight_load_ready(weight_load_ready),.weight_load_row(weight_load_row),
        .weight_load_bank(weight_load_bank),.m_weight_bank(bundle_weight_bank),.release_active(release_active));
    qwen_gemm_array #(.DISTRIBUTED_COMPUTE_BANK(1)) array_Inst(.clk(clk),.rst_n(rst_n),.clear(1'b0),
        .s_valid(issue_valid),.s_ready(array_ready),.s_activation(activation[issue_row*128+:128]),
        .s_weight_advance(last_issue_row),
        .s_weight_load_valid(weight_load_valid),.s_weight_load_ready(weight_load_ready),
        .s_weight_load_data(s_weight_data),.s_weight_load_row(weight_load_row),
        .s_weight_load_bank(weight_load_bank),.s_compute_bank(compute_bank),
        .s_compute_bank_rows(compute_bank_row_copies),.s_compute_release(input_complete),
        .s_meta({first_k,25'b0,issue_slot,issue_row,reg_meta}),
        .m_valid(array_valid),.m_ready(consume_array),
        .m_data(array_data),.m_meta(array_meta),.busy(array_busy),.protocol_error());
    qwen_gemm_partial partial_Inst(.clk(clk),.rst_n(rst_n),.clear(1'b0),
        .update_valid(array_accept),.first_k(array_meta[159]),.update_slot(array_meta[133:131]),.update_row(array_meta[130:128]),
        .update_data(array_data),.update_done(partial_write),.busy(partial_busy),.protocol_error(),
        .read_enable(state_current==G_READ),.read_address(read_address),
        .read_data(partial_read_data),.read_valid(partial_read_valid));
    for(genvar col=0;col<16;col++)begin:gen_output_conversion
        qwen_fp32_to_bf16 output_round_Inst(.value(partial_read_data[col*32+:32]),.result(rounded_row[col*16+:16]));
    end

    always_comb begin
        if(valid_columns<=output_slot*16)output_columns=0;
        else if(valid_columns-output_slot*16>=16)output_columns=16;
        else output_columns=valid_columns-output_slot*16;
    end

    always_comb begin
        state_next=state_current;
        case(state_current)
            G_IDLE:begin
                if(bundle_valid&&bundle_ready)state_next=G_DESC;
            end
            G_DESC:begin
                state_next=G_MAC_SCAN;
            end
            G_MAC_SCAN:begin
                if(input_complete)state_next=last_k ? G_SUM_REDUCE : G_IDLE;
            end
            G_SUM_REDUCE:begin
                if(results_complete)begin
                    if(!last_k)state_next=G_FINISH;
                    else state_next=G_READ;
                end
            end
            G_READ:begin
                state_next=G_PACK;
            end
            G_PACK:begin
                if(partial_read_valid)begin
                    if({1'b0,output_row}==valid_rows-1'b1)state_next=G_STREAM;
                    else state_next=G_READ;
                end
            end
            G_STREAM:begin
                if(m_ready)begin
                    if(output_slot==7)state_next=G_FINISH;
                    else state_next=G_READ;
                end
            end
            G_FINISH:begin state_next=G_IDLE;end
            default:begin state_next=G_IDLE;end
        endcase
    end
    always_ff @(posedge clk)begin
        if(!rst_n)state_current<=G_IDLE;
        else state_current<=state_next;
    end

    always_ff @(posedge clk)begin
        if(!rst_n)begin
            reg_cmd<=0;reg_meta<=0;
            compute_bank<=0;
            issued_count<=0;issue_row<=0;issue_slot<=0;commit_index<=0;block_write_pointer<=0;block_read_pointer<=0;
            output_slot<=0;output_row<=0;
            flag_done<=0;flag_input_done<=0;flag_partial_done<=0;
        end else begin
            flag_done<=0;flag_input_done<=0;flag_partial_done<=0;
            // Visit only declared rows, then advance to the next 16-column slot.
            if(issue_fire)begin
                issued_count<=issued_count+7'd1;
                if(last_issue_row)begin issue_row<=0;issue_slot<=issue_slot+3'd1;end
                else issue_row<=issue_row+3'd1;
            end
            if(input_complete)flag_input_done<=1;
            if(partial_write)begin
                commit_index<=commit_index+6'd1;
                if({1'b0,commit_index}==block_issue_count-1'b1)begin
                    commit_index<=0;
                    block_read_pointer<=block_read_pointer+4'd1;
                    if(!block_last[block_read_pointer])flag_partial_done<=1;
                end
            end
            if(bundle_valid&&bundle_ready)begin
                reg_cmd<=command;reg_meta<=command_meta;
                compute_bank<=bundle_weight_bank;
                issued_count<=0;issue_row<=0;issue_slot<=0;output_row<=0;output_slot<=0;
                block_last[block_write_pointer]<=command[17];
                block_write_pointer<=block_write_pointer+4'd1;
            end
            if((state_current==G_PACK) && partial_read_valid)begin


                if({1'b0,output_row}!=valid_rows-1'b1)output_row<=output_row+3'd1;
            end
            if((state_current==G_STREAM)&&m_ready&&(output_slot!=7))begin
                output_row<=0;output_slot<=output_slot+3'd1;
                // Keep the fixed output bus shape; undeclared rows are zero.

            end
            if(state_current==G_FINISH)begin
                if(last_k)flag_done<=1;
                commit_index<=0;block_write_pointer<=0;block_read_pointer<=0;
            end
        end
    end
    // 每个输出行独立推断FF写使能，清空时序与原输出打包合同相同。
    for(genvar pack_row=0;pack_row<8;pack_row++)begin:gen_output_pack
        always_ff @(posedge clk)begin
            if(!rst_n || (bundle_valid&&bundle_ready) ||
               ((state_current==G_STREAM)&&m_ready&&(output_slot!=7)))begin
                reg_bf16[pack_row*256+:256]<=0;
                reg_fp32[pack_row*512+:512]<=0;
            end else if((state_current==G_PACK)&&partial_read_valid&&(output_row==pack_row))begin
                reg_bf16[pack_row*256+:256]<=rounded_row;
                reg_fp32[pack_row*512+:512]<=partial_read_data;
            end
        end
    end
endmodule
`default_nettype wire

