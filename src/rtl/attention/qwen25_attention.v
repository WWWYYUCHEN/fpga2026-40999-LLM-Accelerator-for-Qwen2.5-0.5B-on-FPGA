`timescale 1ns / 1ps
`default_nettype none
// Each of the 14 query heads owns an independent Score, P, and 64-wide Context.
// KV0 serves Q0..6 and KV1 serves Q7..13.
module qwen25_attention #(parameter ENABLE_LOGITS=0) (
    input wire i_clk,
    input wire i_rstn,
    input wire i_start,
    input wire i_abort,
    input wire i_logits_mode,
    input wire i_logits_valid,
    output wire o_logits_ready,
    input wire [14:0] i_logits_address,
    input wire [127:0] i_logits_data,
    output wire o_logits_busy,
    input wire [3:0] i_rows,
    input wire [15:0] i_position,
    input wire [15:0] i_context,
    input wire [4:0] i_layer,
    input wire [15:0] i_task,
    input wire [3:0] i_first_head,
    input wire [3:0] i_last_head,
    input wire [63:0] i_q_base,
    input wire [63:0] i_k_base,
    input wire [63:0] i_v_base,
    output wire o_busy,
    output wire o_done,
    output wire o_error,
    output wire [7:0] o_error_code,
    output wire o_service_abort,
    output wire o_read_valid,
    input wire i_read_ready,
    output wire [63:0] o_read_address,
    output wire [63:0] o_read_base,
    output wire [12:0] o_read_bytes,
    input wire i_data_valid,
    output wire o_data_ready,
    input wire [127:0] i_data,
    input wire i_memory_busy,
    input wire i_memory_error,
    output wire o_context_valid,
    input wire i_context_ready,
    output wire [127:0] o_context_data,
    output wire [2:0] o_context_row,
    output wire [3:0] o_context_head,
    output wire [2:0] o_context_word,
    output wire o_weight_valid,
    input wire i_weight_ready,
    output wire [9:0] o_weight_word,
    output wire [255:0] o_weight_data,
    output wire o_weight_commit,
    output wire o_weight_release,
    output wire o_weight_bank,
    input wire i_weight_full,
    output wire o_weight_read,
    output wire [7:0] o_weight_read_word,
    input wire i_weight_read_valid,
    input wire [1023:0] i_weight_read_data,
    output wire o_p_write,
    input wire i_p_write_ready,
    output wire o_p_afu_write,
    output wire o_p_afu_write_bank,
    output wire [14:0] o_p_afu_address,
    output wire [127:0] o_p_afu_data,
    output wire o_p_write_bank,
    output wire o_p_bank,
    output wire o_p_afu_bank,
    output wire o_p_commit,
    output wire o_p_release,
    input wire [1:0] i_p_full,
    output wire [14:0] o_p_address,
    output wire [127:0] o_p_data,
    output wire o_p_read,
    output wire [11:0] o_p_word,
    input wire i_p_valid,
    input wire [1023:0] i_p_data,
    output wire o_p_afu_read,
    output wire [11:0] o_p_afu_word,
    output wire [2:0] o_p_row,
    input wire i_p_afu_valid,
    input wire [127:0] i_p_afu_data,
    output wire [2:0] o_p_max_row,
    output wire [15:0] o_p_maximum,
    output wire [31:0] o_p_max_index,
    output wire o_p_max_first,
    output wire o_p_max_last,
    output wire [15:0] o_p_max_words,
    input wire [15:0] i_p_row_maximum,
    input wire i_p_row_complete,
    output wire o_gemm_cmd_valid,
    input wire i_gemm_cmd_ready,
    output wire [127:0] o_gemm_command,
    output wire [127:0] o_identity,
    output wire o_gemm_act_valid,
    input wire i_gemm_act_ready,
    output wire [1023:0] o_gemm_activation,
    output wire o_gemm_weight_valid,
    input wire i_gemm_weight_ready,
    output wire [2047:0] o_gemm_weight,
    input wire i_gemm_valid,
    output wire o_gemm_ready,
    input wire [2047:0] i_gemm_data,
    input wire [127:0] i_gemm_meta,
    input wire [2:0] i_gemm_slice,
    input wire i_gemm_last,
    input wire i_gemm_done,
    input wire i_gemm_input_done,
    input wire i_gemm_partial,
    input wire i_gemm_busy,
    input wire i_gemm_error,
    output wire o_afu_valid,
    input wire i_afu_ready,
    output wire [3:0] o_afu_command,
    output wire [1:0] o_afu_phase,
    output wire [63:0] o_afu_length,
    output wire [127:0] o_afu_x,
    output wire [127:0] o_afu_y,
    output wire [127:0] o_afu_meta,
    output wire o_afu_last,
    output wire [7:0] o_afu_mask,
    input wire i_afu_valid,
    output wire o_afu_ready,
    input wire [127:0] i_afu_data,
    input wire [127:0] i_afu_meta,
    input wire i_afu_last,
    input wire [7:0] i_afu_mask,
    input wire i_afu_busy,
    input wire i_afu_error
);
    localparam [5:0] S_IDLE=0,S_Q_REQ=1,S_Q_WAIT=2,S_K_BEGIN=3,S_K_REQ=4,S_K_WAIT=5,
        S_K_PUSH=6,S_K_EMIT=7,S_K_FILL=8,S_G_BEGIN=9,S_G_SEND=10,S_W_REQ=11,S_W_WAIT=12,
        S_W_SEND=13,S_G_RUN=14,S_G_STORE=15,S_G_END=16,S_SCORE_COMMIT=17,
        S_V_BEGIN=23,S_V_REQ=24,S_V_WAIT=25,S_V_WRITE=26,
        S_V_FILL=27,S_P_READ=28,S_P_WAIT=29,S_FINISH=30,S_NEXT_HEAD=31,
        S_NEXT_QUERY=34,S_WAIT_NORMALIZED=35,S_BEGIN_VALUE=36,S_NEXT_SCORE_TILE=38;
    reg [5:0] state_current,state_next;
    reg [3:0] reg_rows,reg_head,reg_last_head,reg_first_head;
    reg [11:0] cnt_score_block;
    reg flag_key_loaded,flag_value_loaded;
    reg [1:0] flag_normalized;
    // P belongs to one query head and row group; release after its PV, then rotate banks.
    reg reg_p_bank;
    reg [15:0] reg_position,reg_context,reg_task,reg_token_base;
    reg [63:0] reg_k_layer_base;
    reg [4:0] reg_layer;
    reg [63:0] reg_q_base,reg_k_base,reg_v_base;
    reg [127:0] reg_query [0:63]; // Current Q head: eight rows by eight BF16 words.
    reg [2:0] cnt_query_row,cnt_query_word,cnt_token_segment;
    reg [3:0] cnt_dimension_group;
    reg [3:0] cnt_micro_token,cnt_inner;
    reg [6:0] cnt_v_token;
    reg [2:0] cnt_v_chunk,cnt_weight_row;
    reg flag_v_half,flag_weight_half,flag_pv,flag_command_sent,flag_activation_sent,flag_terminal;
    reg [127:0] reg_input_word,reg_v_low;
    reg [255:0] reg_v_write;
    reg [2047:0] reg_weight,reg_slice_data;
    reg [1023:0] reg_activation;
    reg [2:0] reg_slice;
    reg flag_slice_last;
    reg [4:0] cnt_store;
    reg [3:0] cnt_weight_sent;
    reg [4:0] cnt_pv_weight_reads;
    reg flag_pv_weight_pending;
    reg [127:0] dec_identity,dec_command;
    integer row,lane;
    wire kv_head=reg_head>=7;
    // A 64x128 K bank holds the current KV head only. V uses at most
    // 64 token rows by 128 dimensions in the other shared operand bank.
    wire flag_packed=1'b0;
    wire [15:0] operand_tile_tokens=flag_pv ? 16'd64 : 16'd128;
    wire [15:0] wire_tile_tokens=reg_context-reg_token_base;
    wire [7:0] wire_columns=wire_tile_tokens>=operand_tile_tokens ? operand_tile_tokens[7:0] : wire_tile_tokens[7:0];
    // P x V has one 128-column output tile; only its token reduction is tiled.
    wire [7:0] wire_value_last_row=(wire_columns-8'd1) | 8'd7;
    wire [12:0] wire_words=({1'b0,reg_context}+17'd7)>>3;
    // 在context字数取整/减一和K结束比较之间寄存，切开到FSM的组合控制链。
    // context只在任务启动时更新，首条GEMM命令前已有多个周期供此级稳定。
    reg [12:0] context_last_word;
    always @(posedge i_clk)begin
        context_last_word<=wire_words-13'd1;
    end
    wire [15:0] wire_micro_token=reg_token_base+{9'd0,cnt_token_segment,4'd0}+cnt_micro_token;
    wire [15:0] wire_v_token=reg_token_base+cnt_v_token;
    wire [12:0] wire_global_k={1'b0,reg_token_base[14:3]}+cnt_inner;
    // Fold the small head/dimension offset into token-index units before the
    // byte shift, avoiding several cascaded 64-bit additions on K requests.
    wire [4:0] wire_k_lane_index=flag_packed ? {1'b0,cnt_dimension_group} :
        ({1'b0,kv_head,3'b000}+{1'b0,cnt_dimension_group});
    wire [20:0] wire_k_word_offset={1'b0,wire_micro_token,4'b0000}+{16'd0,wire_k_lane_index};
    wire [24:0] wire_k_offset={wire_k_word_offset,4'b0000};
    wire [3:0] wire_last_inner=flag_pv ? (wire_tile_tokens>=64 ? 4'd7 : ((wire_tile_tokens+7)>>3)-1'b1) : 4'd7;
    wire flag_last_k=flag_pv ? wire_global_k==context_last_word : cnt_inner==7;
    wire flag_feed=state_current>=S_G_SEND && state_current<=S_G_RUN;
    wire flag_bundle_sent=flag_command_sent && flag_activation_sent && cnt_weight_sent==8;
    wire flag_result_fire=i_gemm_valid && o_gemm_ready;
    wire [2:0] wire_store_row=cnt_store[3:1];
    wire [15:0] wire_score_column=reg_token_base+{9'd0,reg_slice,4'd0}+{12'd0,cnt_store[0],3'd0};
    wire [11:0] wire_score_word=wire_score_column[14:3];
    wire flag_store_skip=!flag_pv && wire_score_column>=reg_context;
    wire norm_busy,norm_done,norm_bank,norm_write,norm_error;
    wire score_input_ready,score_busy;
    wire score_input_valid=state_current==S_G_STORE && !flag_pv && !flag_store_skip;
    wire flag_value_skip=flag_pv && reg_slice[2]!=kv_head;
    wire flag_store_fire=state_current==S_G_STORE &&
        (flag_store_skip || flag_value_skip || (flag_pv ? i_context_ready : score_input_ready));
    wire trans_ready,trans_valid,trans_busy;
    wire [255:0] trans_data;
    wire [2:0] trans_dimension;
    wire [14:0] norm_address;
    wire [127:0] norm_data;
    wire norm_start=state_current==S_SCORE_COMMIT && !score_busy;
    wire [127:0] wire_store_data=reg_slice_data[wire_store_row*256+cnt_store[0]*128 +: 128];
    wire pv_fifo_valid,pv_fifo_ready,pv_fifo_error;
    wire [2:0] pv_fifo_free;
    wire [2047:0] pv_fifo_data;
    wire pv_weight_read=flag_pv && state_current==S_G_SEND && i_weight_full &&
        cnt_pv_weight_reads<16 && pv_fifo_free>{2'd0,flag_pv_weight_pending};

    assign o_busy=state_current!=S_IDLE && state_current!=S_FINISH;
    assign o_done=state_current==S_FINISH;
    assign o_error=1'b0;
    assign o_error_code=8'd0;
    assign o_service_abort=1'b0;
    assign o_read_valid=(state_current==S_Q_REQ ||
        (state_current==S_K_REQ && wire_micro_token<reg_context) ||
        (state_current==S_V_REQ && wire_v_token<reg_context));
    assign o_read_address=state_current==S_Q_REQ ? reg_q_base+cnt_query_row*64'd1792+reg_head*64'd128+cnt_query_word*64'd16 :
        (state_current==S_K_REQ ? reg_k_layer_base+wire_k_offset :
        reg_v_base+reg_layer*64'd8388608+wire_v_token*64'd256+cnt_v_chunk*64'd32+flag_v_half*64'd16);
    assign o_data_ready=state_current==S_Q_WAIT || state_current==S_K_WAIT || state_current==S_V_WAIT;
    assign o_read_base=o_read_address;
    assign o_read_bytes=13'd16;
    assign o_context_valid=state_current==S_G_STORE && flag_pv && !flag_value_skip;
    assign o_context_data=wire_store_data;
    assign o_context_row=wire_store_row;
    // Each query head writes its own 64-wide Context slice.
    assign o_context_head=reg_head;
    assign o_context_word={reg_slice[1:0],cnt_store[0]};
    assign o_weight_valid=(state_current==S_K_EMIT && trans_valid) || state_current==S_V_WRITE;
    assign o_weight_word=flag_pv ? {cnt_v_token,cnt_v_chunk} : {cnt_dimension_group,trans_dimension,cnt_token_segment};
    assign o_weight_data=flag_pv ? reg_v_write : trans_data;
    assign o_weight_commit=state_current==S_K_FILL || state_current==S_V_FILL;
    assign o_weight_release=state_current==S_G_END && (flag_terminal || !flag_last_k) && cnt_inner==wire_last_inner;
    assign o_weight_bank=flag_pv;
    assign o_weight_read=pv_weight_read || (state_current==S_W_REQ && i_weight_full);
    assign o_weight_read_word=flag_pv ? {cnt_inner,cnt_pv_weight_reads[3:0]} :
        (flag_packed ? {kv_head,cnt_inner[2:0],cnt_weight_row,flag_weight_half} :
        {cnt_inner,cnt_weight_row,flag_weight_half});
    assign o_p_read=state_current==S_P_READ && flag_normalized[reg_p_bank] && i_p_full[reg_p_bank];
    assign o_p_word=reg_token_base[14:3]+cnt_inner;
    assign o_p_afu_write=norm_write;
    assign o_p_afu_address=norm_address;
    assign o_p_afu_data=norm_data;
    assign o_p_afu_write_bank=norm_bank;
    assign o_p_bank=reg_p_bank;
    assign o_p_afu_bank=norm_bank;
    assign o_p_commit=state_current==S_SCORE_COMMIT && !score_busy;
    assign o_p_release=state_current==S_NEXT_HEAD;
    assign o_identity=dec_identity;
    assign o_gemm_command=dec_command;
    assign o_gemm_cmd_valid=flag_feed && !flag_command_sent;
    assign o_gemm_act_valid=flag_feed && !flag_activation_sent;
    assign o_gemm_activation=reg_activation;
    assign o_gemm_weight_valid=flag_pv ? state_current==S_G_SEND && pv_fifo_valid : state_current==S_W_SEND;
    assign o_gemm_weight=flag_pv ? pv_fifo_data : reg_weight;
    assign o_gemm_ready=state_current==S_G_RUN;

    // Reserve each synchronous response before issuing the next 64-column half.
    qwen25_weight_fifo pv_weight_fifo_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_flush(state_current==S_G_BEGIN),
        .i_valid(flag_pv && i_weight_read_valid),.o_ready(pv_fifo_ready),.o_free_halves(pv_fifo_free),
        .i_data(i_weight_read_data),.i_meta(dec_identity),
        .o_valid(pv_fifo_valid),.i_ready(flag_pv && state_current==S_G_SEND && i_gemm_weight_ready),
        .o_data(pv_fifo_data),.o_meta(),.o_empty(),.o_error(pv_fifo_error)
    );

    // Register the slow mode control before it fans out through score,
    // normalize, and DMA control logic.  The controller asserts this mode
    // before presenting the first logits beat, so the extra register does not
    // add a bubble to the logits stream.
    reg logits_mode_reg;
    always @(posedge i_clk) begin
        if(!i_rstn) logits_mode_reg<=1'b0;
        else logits_mode_reg<=i_logits_mode;
    end
    wire logits_mode=ENABLE_LOGITS!=0 && logits_mode_reg;
    qwen25_score_ingress score_ingress_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_valid(logits_mode ? i_logits_valid : score_input_valid),.o_ready(score_input_ready),
        .i_logits(logits_mode),.i_bank(logits_mode ? 1'b0 : reg_p_bank),
        .i_address(logits_mode ? i_logits_address : {wire_store_row,wire_score_word}),
        .i_data(logits_mode ? i_logits_data : wire_store_data),
        .i_length(logits_mode ? 32'd151936 : {16'd0,reg_context}),.i_position(reg_position),
        .o_valid(o_p_write),.i_ready(i_p_write_ready),.o_bank(o_p_write_bank),.o_address(o_p_address),.o_data(o_p_data),
        .o_max_row(o_p_max_row),.o_maximum(o_p_maximum),.o_index(o_p_max_index),
        .o_first(o_p_max_first),.o_last(o_p_max_last),.o_words(o_p_max_words),.o_busy(score_busy)
    );
    assign o_logits_ready=score_input_ready;
    assign o_logits_busy=score_busy;

    qwen25_attention_normalize normalize_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_start(norm_start),.i_abort(1'b0),
        .i_rows(reg_rows),.i_position(reg_position),.i_context(reg_context),.i_identity(dec_identity),.i_bank(reg_p_bank),
        .i_row_maximum(i_p_row_maximum),.i_row_complete(i_p_row_complete),
        .o_busy(norm_busy),.o_done(norm_done),.o_bank(norm_bank),
        .o_read(o_p_afu_read),.o_row(o_p_row),.o_word(o_p_afu_word),.i_read_valid(i_p_afu_valid),.i_read_data(i_p_afu_data),
        .o_write(norm_write),.o_address(norm_address),.o_data(norm_data),
        .o_valid(o_afu_valid),.i_ready(i_afu_ready),.o_command(o_afu_command),.o_phase(o_afu_phase),.o_length(o_afu_length),
        .o_x(o_afu_x),.o_y(o_afu_y),.o_meta(o_afu_meta),.o_last(o_afu_last),.o_mask(o_afu_mask),
        .i_valid(i_afu_valid),.o_ready(o_afu_ready),.i_data(i_afu_data),.i_meta(i_afu_meta),.i_busy(i_afu_busy),.o_error(norm_error)
    );

    qwen25_key_transpose transpose_Inst (
        .i_clk(i_clk),.i_rstn(i_rstn),.i_flush(1'b0),
        .i_valid(state_current==S_K_PUSH),.o_ready(trans_ready),.i_data(reg_input_word),
        .o_valid(trans_valid),.i_ready(state_current==S_K_EMIT && i_weight_ready),
        .o_data(trans_data),.o_dimension(trans_dimension),.o_busy(trans_busy)
    );

    // Identities stay constant across all partial K blocks of the same output tile.
    always @(*)begin
        dec_identity=0;
        dec_identity[15:0]=reg_task;dec_identity[20:16]=reg_layer;
        dec_identity[52:37]=reg_token_base;dec_identity[56:53]=reg_rows;dec_identity[60:57]=8;
        dec_identity[76:61]=reg_context;dec_identity[83:80]=flag_pv ? 4'd4 : 4'd3;
        // Each query head has its own identity; only its 64-dimensional dot product is reduced.
        dec_identity[87:84]=reg_head;dec_identity[88]=0;dec_identity[104:89]=reg_position;
        if(flag_pv)dec_identity[52:37]=0;
        dec_command=0;dec_command[3:0]=flag_pv ? 4'd4 : 4'd3;
        dec_command[15:4]=flag_pv ? wire_global_k[11:0] : cnt_score_block;
        dec_command[16]=flag_pv ? wire_global_k==0 : cnt_score_block==0;
        dec_command[17]=flag_last_k;dec_command[21:18]=reg_rows;
        dec_command[29:22]=flag_pv ? 8'd128 : wire_columns;
        dec_command[33:30]=flag_pv && flag_last_k && reg_context[2:0]!=0 ? {1'b0,reg_context[2:0]} : 4'd8;
    end

    always @(*)begin
        state_next=state_current;
        case(state_current)
            S_IDLE:if(i_start)state_next=S_Q_REQ;
            S_Q_REQ:if(o_read_valid && i_read_ready)state_next=S_Q_WAIT;
            S_Q_WAIT:if(i_data_valid)state_next=cnt_query_word==7 && cnt_query_row==reg_rows-1 ? (flag_key_loaded ? S_G_BEGIN : S_K_BEGIN) : S_Q_REQ;
            S_K_BEGIN:state_next=flag_packed && flag_key_loaded ? S_G_BEGIN : S_K_REQ;
            S_K_REQ:if(wire_micro_token>=reg_context)state_next=S_K_PUSH;else if(o_read_valid && i_read_ready)state_next=S_K_WAIT;
            S_K_WAIT:if(i_data_valid)state_next=S_K_PUSH;
            S_K_PUSH:if(trans_ready)state_next=cnt_micro_token==15 ? S_K_EMIT : S_K_REQ;
            S_K_EMIT:if(trans_valid && i_weight_ready && trans_dimension==7)
                state_next=(flag_packed ? cnt_dimension_group==15 : cnt_dimension_group==7) && cnt_token_segment==7 ? S_K_FILL : S_K_REQ;
            S_K_FILL:state_next=S_G_BEGIN;
            S_G_BEGIN:state_next=S_G_SEND;
            S_G_SEND:if(cnt_weight_sent<8)begin
                if(!flag_pv)state_next=S_W_REQ;
            end else if(flag_command_sent && flag_activation_sent)state_next=S_G_RUN;
            S_W_REQ:if(i_weight_full)state_next=S_W_WAIT;
            S_W_WAIT:if(i_weight_read_valid)state_next=flag_weight_half ? S_W_SEND : S_W_REQ;
            S_W_SEND:if(i_gemm_weight_ready)state_next=S_G_SEND;
            S_G_RUN:if(i_gemm_valid)state_next=S_G_STORE;
                else if(!flag_last_k && flag_bundle_sent)state_next=S_G_END;
            S_G_STORE:if(flag_store_fire && cnt_store==reg_rows*2-1)state_next=flag_slice_last ? S_G_END : S_G_RUN;
            S_G_END:if(flag_terminal || !flag_last_k)begin
                if(cnt_inner!=wire_last_inner)state_next=flag_pv ? S_P_READ : S_G_BEGIN;
                else if(reg_token_base+operand_tile_tokens<reg_context)state_next=flag_pv ? S_V_BEGIN : S_NEXT_SCORE_TILE;
                else state_next=flag_pv ? S_NEXT_HEAD : S_SCORE_COMMIT;
            end
            S_SCORE_COMMIT:if(!score_busy)state_next=S_WAIT_NORMALIZED;
            S_NEXT_QUERY:state_next=S_Q_REQ;
            S_WAIT_NORMALIZED:if(flag_normalized[reg_p_bank] && !norm_busy)state_next=S_BEGIN_VALUE;
            S_BEGIN_VALUE:state_next=S_V_BEGIN;
            S_NEXT_SCORE_TILE:state_next=S_Q_REQ;
            S_V_BEGIN:state_next=S_V_REQ;
            S_V_REQ:if(wire_v_token>=reg_context)state_next=S_V_WRITE;
                else if(o_read_valid && i_read_ready)state_next=S_V_WAIT;
            S_V_WAIT:if(i_data_valid)state_next=flag_v_half ? S_V_WRITE : S_V_REQ;
            S_V_WRITE:if(i_weight_ready)state_next=cnt_v_token==wire_value_last_row[6:0] && cnt_v_chunk==7 ? S_V_FILL : S_V_REQ;
            S_V_FILL:state_next=S_P_READ;
            S_P_READ:if(o_p_read)state_next=S_P_WAIT;
            S_P_WAIT:if(i_p_valid)state_next=S_G_BEGIN;
            S_NEXT_HEAD:state_next=reg_head<reg_last_head ? S_Q_REQ : S_FINISH;
            S_FINISH:if(i_start)state_next=S_Q_REQ;
            default:state_next=S_IDLE;
        endcase
    end
    always @(posedge i_clk)begin
        if(!i_rstn)state_current<=S_IDLE;else state_current<=state_next;
    end
    // Only validity/counters reset. Tensor contents are overwritten before use.
    always @(posedge i_clk)begin
        if(!i_rstn)begin
            reg_rows<=0;reg_head<=0;reg_last_head<=0;reg_context<=0;reg_position<=0;reg_task<=0;reg_layer<=0;reg_k_layer_base<=0;
            reg_token_base<=0;cnt_query_row<=0;cnt_query_word<=0;cnt_dimension_group<=0;cnt_token_segment<=0;
            cnt_micro_token<=0;cnt_inner<=0;cnt_v_token<=0;cnt_v_chunk<=0;flag_v_half<=0;
            cnt_weight_row<=0;flag_weight_half<=0;cnt_weight_sent<=0;flag_command_sent<=0;flag_activation_sent<=0;
            flag_terminal<=0;flag_pv<=0;cnt_store<=0;
            reg_first_head<=0;cnt_score_block<=0;flag_normalized<=0;reg_p_bank<=0;
            flag_key_loaded<=0;flag_value_loaded<=0;
            cnt_pv_weight_reads<=0;flag_pv_weight_pending<=0;
        end else begin
            flag_pv_weight_pending<=pv_weight_read;
            if(pv_weight_read)cnt_pv_weight_reads<=cnt_pv_weight_reads+1'b1;
            if(i_start && (state_current==S_IDLE || state_current==S_FINISH))begin
                reg_rows<=i_rows;reg_position<=i_position;reg_context<=i_context;reg_layer<=i_layer;reg_task<=i_task;
                reg_head<=i_first_head;reg_last_head<=i_last_head;reg_q_base<=i_q_base;reg_k_base<=i_k_base;reg_v_base<=i_v_base;
                cnt_query_row<=0;cnt_query_word<=0;reg_token_base<=0;flag_pv<=0;
                reg_first_head<=i_first_head;cnt_score_block<=0;flag_normalized<=0;
                flag_key_loaded<=0;flag_value_loaded<=0;
            end
            if(norm_done)flag_normalized[norm_bank]<=1;
            if(o_p_commit)flag_normalized[reg_p_bank]<=0;
            if(o_p_release)begin
                flag_normalized[reg_p_bank]<=0;
                reg_p_bank<=!reg_p_bank;
            end
            if(o_weight_release)begin
                if(!flag_packed || !flag_pv)flag_key_loaded<=0;
                if(!flag_packed || flag_pv)flag_value_loaded<=0;
            end
            if(state_current==S_Q_WAIT && i_data_valid)begin
                reg_query[{cnt_query_row,cnt_query_word}]<=i_data;
                cnt_query_word<=cnt_query_word+1'b1;
                if(cnt_query_word==7)cnt_query_row<=cnt_query_row+1'b1;
            end
            if(state_current==S_K_BEGIN)begin
                flag_pv<=0;cnt_inner<=0;cnt_score_block<=0;cnt_dimension_group<=0;cnt_token_segment<=0;cnt_micro_token<=0;
                reg_k_layer_base<=reg_k_base+{36'd0,reg_layer,23'd0};
            end
            if(state_current==S_K_FILL)flag_key_loaded<=1;
            if(state_current==S_K_REQ && wire_micro_token>=reg_context)reg_input_word<=0;
            if(state_current==S_K_WAIT && i_data_valid)reg_input_word<=i_data;
            if(state_current==S_K_PUSH && trans_ready)cnt_micro_token<=cnt_micro_token+1'b1;
            if(state_current==S_K_EMIT && trans_valid && i_weight_ready && trans_dimension==7)begin
                cnt_micro_token<=0;cnt_token_segment<=cnt_token_segment+1'b1;
                if(cnt_token_segment==7)cnt_dimension_group<=cnt_dimension_group+1'b1;
            end
            if(state_current==S_G_BEGIN)begin
                cnt_pv_weight_reads<=0;flag_pv_weight_pending<=0;
                flag_command_sent<=0;flag_activation_sent<=0;cnt_weight_sent<=0;cnt_weight_row<=0;flag_weight_half<=0;flag_terminal<=0;
                if(!flag_pv)begin
                    for(row=0;row<8;row=row+1)reg_activation[row*128 +: 128]<=row<reg_rows ? reg_query[row*8+cnt_inner[2:0]] : 128'd0;
                end
            end
            if(o_gemm_cmd_valid && i_gemm_cmd_ready)flag_command_sent<=1;
            if(o_gemm_act_valid && i_gemm_act_ready)flag_activation_sent<=1;
            if(i_gemm_done)flag_terminal<=1;
            if(state_current==S_W_WAIT && i_weight_read_valid)begin
                if(flag_weight_half)reg_weight[2047:1024]<=i_weight_read_data;
                else reg_weight[1023:0]<=i_weight_read_data;
                flag_weight_half<=!flag_weight_half;
            end
            if(o_gemm_weight_valid && i_gemm_weight_ready)begin cnt_weight_row<=cnt_weight_row+1'b1;cnt_weight_sent<=cnt_weight_sent+1'b1;end
            if(flag_result_fire && state_current==S_G_RUN)begin
                reg_slice_data<=i_gemm_data;reg_slice<=i_gemm_slice;flag_slice_last<=i_gemm_last;cnt_store<=0;
            end
            if(flag_store_fire)cnt_store<=cnt_store+1'b1;
            if(state_current==S_G_END && (flag_terminal || !flag_last_k))begin
                if(!flag_pv && !flag_last_k)cnt_score_block<=cnt_score_block+1'b1;
                if(cnt_inner!=wire_last_inner)cnt_inner<=cnt_inner+1'b1;
                else if(reg_token_base+operand_tile_tokens<reg_context)reg_token_base<=reg_token_base+operand_tile_tokens;
            end
            if(state_current==S_NEXT_QUERY)begin
                reg_head<=reg_head+1'b1;cnt_inner<=0;cnt_query_row<=0;cnt_query_word<=0;
            end
            if(state_current==S_BEGIN_VALUE)begin flag_pv<=1;reg_token_base<=0;end
            if(state_current==S_NEXT_SCORE_TILE)begin
                cnt_inner<=0;cnt_query_row<=0;cnt_query_word<=0;cnt_score_block<=0;flag_pv<=0;
            end
            if(state_current==S_NEXT_HEAD && reg_head<reg_last_head)begin
                reg_head<=reg_head+1'b1;flag_pv<=0;reg_token_base<=0;
                cnt_inner<=0;cnt_query_row<=0;cnt_query_word<=0;cnt_score_block<=0;
            end
            if(state_current==S_V_BEGIN)begin cnt_v_token<=0;cnt_v_chunk<=0;flag_v_half<=0;cnt_inner<=0;end
            if(state_current==S_V_REQ && wire_v_token>=reg_context)reg_v_write<=0;
            if(state_current==S_V_WAIT && i_data_valid)begin
                if(flag_v_half)reg_v_write<={i_data,reg_v_low};else reg_v_low<=i_data;
                flag_v_half<=!flag_v_half;
            end
            if(state_current==S_V_WRITE && i_weight_ready)begin
                cnt_v_chunk<=cnt_v_chunk+1'b1;
                if(cnt_v_chunk==7)cnt_v_token<=cnt_v_token+1'b1;
            end
            if(state_current==S_P_WAIT && i_p_valid)begin
                for(row=0;row<8;row=row+1)begin
                    for(lane=0;lane<8;lane=lane+1)reg_activation[(row*8+lane)*16 +: 16]<=
                        row<reg_rows && wire_global_k*8+lane<reg_context ? i_p_data[(row*8+lane)*16 +: 16] : 16'd0;
                end
            end
        end
    end
endmodule
`default_nettype wire
