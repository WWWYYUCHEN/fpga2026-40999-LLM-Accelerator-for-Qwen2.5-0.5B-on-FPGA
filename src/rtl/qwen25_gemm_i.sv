`timescale 1ns/1ps
`default_nettype none

// ============================================================================
// qwen25_gemm_i —— qwen25_gemm 的集成版（原文件：rtl/qwen25_gemm.sv）
//
// 与 rtl/qwen25_gemm.sv 的差异（逐处标注 [S1]/[S2]）：
//   [S1] 阵列换成 qwen_gemm_array_i：s_meta 仍是 160 bit（未变），
//        m_meta 变 168 bit（阵列把块指数塞进最高 8 bit），m_data 低 384 bit 是 INT24。
//   [S2] 阵列输出后加 qwen_gemm_int24_bridge（16 列 INT24+块指数 -> FP32，2 拍），
//        替换原来的 qwen_fp32_sum8（21 拍）。
//        * qwen_gemm_partial.update_data 取桥接输出
//        * update_slot/update_row/first_k 走同样的 2 拍元数据延迟（否则 partial
//          会用"当前"身份写一个 2 拍前的数据 —— 实测会错行/错槽）
//        * results_complete / busy 追加桥接在途计数（否则阵列 credit 先清零，
//          上层提前一进 G_READ 就读到还没提交的 partial）
//   其余（前端、partial、输出打包、状态机、寄存器）逐字保持原样。
// ============================================================================
module qwen25_gemm_i (
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
    // [ARRAY-256] 256 PE = COL columns x 8 lanes.  The array covers COL output
    // columns per beat, so one bundle issues SLOTS = 128/COL weight slots and
    // valid_rows*SLOTS beats instead of valid_rows*8.  Every column's
    // arithmetic (alignment group, INT24, block exponent, fp32 accumulate
    // order) is the COL=16 one, so the 8x128 fp32 partial tile and the
    // 8-slice output stream are bit-identical.
    localparam integer COL   = 32;
    localparam integer SLOTS = 128 / COL;          // 4
    localparam integer SLOT_B = $clog2(SLOTS);     // 2
    typedef enum logic [3:0] {G_IDLE,G_DESC,G_MAC_SCAN,G_SUM_REDUCE,G_READ,G_PACK,G_STREAM,G_FINISH} state_t;
    state_t state_current,state_next;
    wire bundle_valid,bundle_ready,release_active,flush_frontend;
    wire [127:0] command, command_meta;
    wire [1023:0] activation;
    logic [1023:0] activation_bundle;
    always_ff @(posedge clk) begin
        if (bundle_valid && bundle_ready) activation_bundle <= activation;
    end
    wire weight_load_valid, weight_load_ready;
    wire [2:0] weight_load_row;
    wire weight_load_bank,bundle_weight_bank;
    logic compute_bank;
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
    logic [3:0] block_write_pointer,block_read_pointer;
    logic block_last [0:15];
    wire [3:0] valid_rows=reg_cmd[21:18];
    wire [6:0] block_issue_count=valid_rows*SLOTS;
    wire last_issue_row={1'b0,issue_row}==valid_rows-1'b1;
    wire [7:0] valid_columns=reg_cmd[29:22];
    wire first_k=reg_cmd[16],last_k=reg_cmd[17];
    wire array_ready,array_valid,array_busy;
    wire [COL*32-1:0] array_data;
    wire [167:0] array_meta;                     // [S1] 168 bit
    wire issue_valid=(state_current==G_MAC_SCAN)&&(issued_count<block_issue_count);
    wire issue_fire=issue_valid&&array_ready;
    wire consume_array=(state_current==G_IDLE)||(state_current==G_DESC)||
                       (state_current==G_MAC_SCAN)||(state_current==G_SUM_REDUCE);
    wire array_accept=array_valid&&consume_array;
    wire input_complete=issue_fire && issued_count==block_issue_count-1'b1;
    wire [COL*32-1:0] partial_read_data;
    wire partial_read_valid;
    wire partial_write,partial_busy;
    logic [2:0] output_slot,output_row;
    // one partial read = COL columns; a slice is 16 of them, so the read slot is
    // output_slot/SLOTS_PER_READ and output_slot[0] picks the 16-column half.
    wire [5:0] read_address=(state_current==G_READ)?
        {{(3-SLOT_B){1'b0}},output_slot[2-:SLOT_B],output_row}:array_meta[133:128];
    wire [COL*16-1:0] partial_slice = partial_read_data[output_slot[0]*(COL*16) +: (COL*16)];
    // [S2] 桥接组：INT24 + 块指数 -> FP32，2 拍
    wire [COL*32-1:0] bridge_fp32;
    wire bridge_valid, bridge_inflight;
    wire results_complete=!array_busy&&!partial_busy&&!bridge_inflight;
    logic [6:0] bridge_meta_pipe [0:1];
    logic [2047:0] reg_bf16;
    logic [4095:0] reg_fp32;
    wire [255:0] rounded_row;
    logic flag_done,flag_input_done,flag_partial_done;
    logic [7:0] output_columns;
    assign bundle_ready=(state_current==G_IDLE);
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
    assign busy=(state_current!=G_IDLE)||array_busy||partial_busy||input_pending||bridge_inflight;
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
        .m_act_data(activation),.m_act_meta(),.m_weight_meta(),
        .weight_meta_error(),.input_pending(input_pending),
        .weight_load_valid(weight_load_valid),.weight_load_ready(weight_load_ready),.weight_load_row(weight_load_row),
        .weight_load_bank(weight_load_bank),.m_weight_bank(bundle_weight_bank),.release_active(release_active));
    qwen_gemm_array_i #(.DISTRIBUTED_COMPUTE_BANK(1),.COL(COL)) array_Inst(
        .clk(clk),.rst_n(rst_n),.clear(1'b0),
        .s_valid(issue_valid),.s_ready(array_ready),.s_activation(activation_bundle[issue_row*128+:128]),
        .s_weight_advance(last_issue_row),
        .s_weight_load_valid(weight_load_valid),.s_weight_load_ready(weight_load_ready),
        .s_weight_load_data(s_weight_data),.s_weight_load_row(weight_load_row),
        .s_weight_load_bank(weight_load_bank),.s_compute_bank(compute_bank),
        .s_compute_bank_rows(compute_bank_row_copies),.s_compute_release(input_complete),
        .s_meta({first_k,25'b0,issue_slot,issue_row,reg_meta}),
        .m_valid(array_valid),.m_ready(consume_array),
        .m_data(array_data),.m_meta(array_meta),.busy(array_busy),.protocol_error());
    qwen_gemm_int24_bridge #(.COL(COL)) bridge_Inst(
        .clk(clk),.rst_n(rst_n),.clear(1'b0),
        .in_valid(array_accept),.int24_flat(array_data[COL*24-1:0]),.block_exp(array_meta[167:160]),
        .fp32_flat(bridge_fp32),.out_valid(bridge_valid),.inflight(bridge_inflight));
    // 身份字段与数据走同样 2 拍
    always_ff @(posedge clk)begin
        if(!rst_n)begin bridge_meta_pipe[0]<=7'd0;bridge_meta_pipe[1]<=7'd0;end
        else begin
            bridge_meta_pipe[0]<={array_meta[159],array_meta[133:131],array_meta[130:128]};
            bridge_meta_pipe[1]<=bridge_meta_pipe[0];
        end
    end
`ifdef NOMETA_DELAY
    // 反证：身份不跟数据一起延迟（用"当前"FIFO 出口的身份）——step2 必须失败
    wire [6:0] meta_sel = {array_meta[159],array_meta[133:131],array_meta[130:128]};
`else
    wire [6:0] meta_sel = bridge_meta_pipe[1];
`endif
    qwen_gemm_partial #(.COL(COL)) partial_Inst(.clk(clk),.rst_n(rst_n),.clear(1'b0),
        .update_valid(bridge_valid),.first_k(meta_sel[6]),
        .update_slot(meta_sel[5:3]),.update_row(meta_sel[2:0]),
        .update_data(bridge_fp32),.update_done(partial_write),.busy(partial_busy),.protocol_error(),
        .read_enable(state_current==G_READ),.read_address(read_address),
        .read_data(partial_read_data),.read_valid(partial_read_valid));
    for(genvar col=0;col<16;col++)begin:gen_output_conversion
        qwen_fp32_to_bf16 output_round_Inst(.value(partial_slice[col*32+:32]),.result(rounded_row[col*16+:16]));
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
            end
            if(state_current==G_FINISH)begin
                if(last_k)flag_done<=1;
                commit_index<=0;block_write_pointer<=0;block_read_pointer<=0;
            end
        end
    end
    for(genvar pack_row=0;pack_row<8;pack_row++)begin:gen_output_pack
        always_ff @(posedge clk)begin
            if(!rst_n)begin
                reg_bf16[pack_row*256+:256]<=0;
                reg_fp32[pack_row*512+:512]<=0;
            end else if((bundle_valid&&bundle_ready) ||
               ((state_current==G_STREAM)&&m_ready&&(output_slot!=7)))begin
                reg_bf16[pack_row*256+:256]<=0;
                reg_fp32[pack_row*512+:512]<=0;
            end else if((state_current==G_PACK)&&partial_read_valid&&(output_row==pack_row))begin
                reg_bf16[pack_row*256+:256]<=rounded_row;
                reg_fp32[pack_row*512+:512]<=partial_slice;
            end
        end
    end
endmodule
`default_nettype wire
