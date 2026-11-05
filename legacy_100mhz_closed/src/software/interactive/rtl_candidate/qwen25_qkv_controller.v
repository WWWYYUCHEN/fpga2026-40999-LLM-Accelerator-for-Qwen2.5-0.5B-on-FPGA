`timescale 1ns / 1ps
// 固定数据流：AXI-Lite快照驱动固化参数、RMSNorm、K到V到Q、bias、RoPE有效边界和真实DDR写回
module qwen25_qkv_controller #(parameter EXTERNAL_GEMM=0,parameter DIRECT_WEIGHT_READ=0) (
    output wire m_gemm_cmd_valid,
    input wire s_gemm_cmd_ready,
    output wire [127:0] m_gemm_cmd,
    output wire [127:0] m_gemm_cmd_meta,
    output wire m_gemm_act_valid,
    input wire s_gemm_act_ready,
    output wire [1023:0] m_gemm_act_data,
    output wire [127:0] m_gemm_act_meta,
    output wire m_gemm_weight_valid,
    input wire s_gemm_weight_ready,
    output wire [2047:0] m_gemm_weight_data,
    output wire [127:0] m_gemm_weight_meta,
    output wire m_gemm_abort,
    output wire m_gemm_soft_reset,
    input wire s_gemm_valid,
    output wire m_gemm_ready,
    input wire [2047:0] s_gemm_bf16,
    input wire [4095:0] s_gemm_fp32,
    input wire [127:0] s_gemm_meta,
    input wire [2:0] s_gemm_slice,
    input wire [7:0] s_gemm_row_mask,
    input wire [15:0] s_gemm_col_mask,
    input wire s_gemm_last,
    input wire s_gemm_busy,
    input wire s_gemm_input_done,
    input wire s_gemm_partial_done,
    input wire s_gemm_done,
    input wire s_gemm_error,
    input wire s_gemm_aborted,
    input wire [7:0] s_gemm_error_code,
    input wire clk, // 系统300MHz同钟域
    input wire rst_n, // 同步低有效复位
    input wire start, // AXI-Lite完整配置快照提交
    input wire [575:0] cfg, // 不可变任务快照
    input wire abort_req, // 停止新阶段并排空
    input wire soft_reset, // 排空后恢复控制
    output wire dma_cmd_valid, // DMA真实命令有效
    input wire dma_cmd_ready, // DMA命令接受
    output wire [127:0] dma_cmd, // 完整64bit地址描述符
    output wire [127:0] dma_meta, // DMA完整metadata
    output wire dma_abort, // 真实DMA排空请求
    input wire rd_busy, // AXI读事务在途
    input wire rd_done, // AXI读与本地流真实成功
    input wire rd_error, // AXI读失败
    input wire rd_aborted, // AXI读取消完成
    input wire [7:0] rd_error_code, // 读错误原因
    input wire [3:0] direct_rd_valid, output wire [3:0] direct_rd_ready,
    input wire [511:0] direct_rd_data, input wire [39:0] direct_rd_word,
    input wire [63:0] direct_rd_keep, input wire [7:0] direct_rd_error,
    input wire [127:0] direct_rd_meta,
    input wire rd_valid, // DMA本地128bit流有效
    output wire rd_ready, // 接受128bit流
    input wire [127:0] rd_data, // DDR读数据
    input wire [15:0] rd_keep, // 有效byte
    input wire rd_last, // descriptor末拍
    input wire [127:0] rd_meta, // 读数据身份
    input wire [15:0] rd_tag, // descriptor关联
    input wire [3:0] rd_resource, // 读取资源编号
    input wire [1:0] rd_beat_error, // AXI本拍错误
    output wire wr_valid, // 送DMA的128bit流
    input wire wr_ready, // DMA接受写流
    output wire [127:0] wr_data, // DDR写数据
    output wire [15:0] wr_keep, // 八BF16完整byte
    output wire wr_last, // 写descriptor末拍
    output wire [127:0] wr_meta, // 写descriptor身份
    input wire wr_busy, // AXI写事务在途
    input wire wr_done, // 真实B响应成功
    input wire wr_error, // 写失败
    input wire wr_aborted, // 写排空取消
    input wire [7:0] wr_error_code, // 写失败原因
    output wire afu_valid, // 有效算术frame输入
    input wire afu_ready, // 外部AFU接受
    output wire [3:0] afu_command, // 11RMSNorm12RoPE边界
    output wire [20:0] afu_command_4, // 阶段控制保留
    output wire [26:0] afu_sum_command_buffer, // 归约阶段控制保留
    output wire [31:0] afu_eps, // FP32 RMS epsilon
    output wire afu_latch, // 归约参数保持请求
    output wire afu_gelu_sign, // QKV不启用GELU
    output wire [127:0] afu_x, // 八lane BF16 X
    output wire [127:0] afu_y, // gamma或RoPE配对第二半
    output wire [127:0] afu_meta, // 有效行字地址位置身份
    output wire afu_last, // 行或RoPE当前向量结束
    output wire [7:0] afu_mask, // 有效八lane
    output wire [63:0] afu_length, // 896维RMS或64维head
    input wire afu_result_valid, // 外部实际计算结果valid
    output wire afu_result_ready, // Controller接受结果
    input wire [127:0] afu_result_data, // 第一组八lane结果
    input wire [127:0] afu_result_aux, // RoPE第二组结果
    input wire [127:0] afu_result_meta, // 结果完整身份
    input wire afu_result_last, // 结果行末
    input wire [7:0] afu_result_mask,
    input wire afu_busy,
    input wire afu_error, // 结果lane有效
    output wire afu_abort, // Attention错误时取消未完成的AFU frame
    output wire busy, // QKV阶段尚未完成
    output wire qkv_done, // K,V,Q全部真实DDR写回完成
    output wire param_ready, // 三表真实完成后只读
    output wire result_commit, // 最终Greedy结果单拍提交
    output wire [17:0] best_token_id, // 提交前已锁存的词表位置
    output wire error, // 控制错误保持
    output wire [31:0] error_code // 兼容CSR接口；不汇总错误码，固定输出零
);

    localparam [6:0] S_IDLE=0,S_DESC=1,S_RD_CMD=2,S_RD_DATA=3,S_RD_FINISH=4,S_DESC_CHECK=5,S_PARAM=6,S_INPUT=7,
        S_NORM_READ=8,S_NORM_WAIT=9,S_NORM_SEND=10,S_NORM_FINISH=11,S_COEFF_READ=12,S_COEFF_WAIT=13,S_WEIGHT=14,
        S_GEMM_BEGIN=15,S_GEMM_FEED=16,S_BIAS_READ=17,S_BIAS_START=18,S_BIAS_WAIT=19,
        S_ROPE_FIRST=20,S_ROPE_FIRST_WAIT=21,S_ROPE_SECOND=22,S_ROPE_SECOND_WAIT=23,S_ROPE_SEND=24,S_ROPE_WAIT=25,S_ROPE_WRITE=26,
        S_WRITE=27,S_WR_CMD=28,S_WR_READ=29,S_WR_WAIT_DATA=30,S_WR_SEND=31,S_WR_FINISH=32,S_NEXT=33,S_DONE=34,
        S_NORM_STATS=37,S_POST_DRAIN=38,S_ATTN_BEGIN=39,S_ATTN_RUN=40,
        S_O_RESTORE=41,S_O_CONTEXT=42,S_O_NEXT=43,
        S_RES_READ=44,S_RES_WAIT=45,S_RES_SEND=46,S_RES_FINISH=47,
        S_H_START=48,S_H_COMMAND=49,S_H_READ=50,S_H_WAIT=51,S_H_SEND=52,S_H_FINISH=53,
        S_Z_READ=54,S_Z_WAIT=55,S_Z_SEND=56,S_Z_FINISH=57,
        S_Z_COPY_READ=58,S_Z_COPY_WAIT=59,S_Z_COPY_WRITE=60,S_O_FINISH=61,S_FFN_BEGIN=62,S_FFN_NEXT=63,S_FFN_WAIT=64,
        S_M_LOAD=65,S_H_LOAD=66,S_DOWN_BEGIN=67,S_DOWN_NEXT=68,S_LAYER_DONE=69,S_BATCH_NEXT=70,S_DESC_EXTRA=71,S_DESC_EXTRA_CHECK=72,
        S_H_OFFSET=73,S_H_ADDRESS=74,S_H_COMMIT=75,
        S_FINAL_BEGIN=76,S_LM_BEGIN=77,S_LM_NEXT=78,S_LM_DRAIN=79,S_LM_COMMIT=80;
    localparam [3:0] READ_DESC=0,READ_SCALE=1,READ_BIAS=2,READ_GAMMA=3,READ_INPUT=4,READ_WEIGHT=5,READ_CONTEXT=6,READ_M=7,READ_H=8,READ_EXTRA=9;
    // Explicit one-hot constants apply only to the main FSM; post FSM stays binary.
    localparam [80:0] HOT_S_IDLE=81'h000000000000000000001;
    localparam [80:0] HOT_S_DESC=81'h000000000000000000002;
    localparam [80:0] HOT_S_RD_CMD=81'h000000000000000000004;
    localparam [80:0] HOT_S_RD_DATA=81'h000000000000000000008;
    localparam [80:0] HOT_S_RD_FINISH=81'h000000000000000000010;
    localparam [80:0] HOT_S_DESC_CHECK=81'h000000000000000000020;
    localparam [80:0] HOT_S_PARAM=81'h000000000000000000040;
    localparam [80:0] HOT_S_INPUT=81'h000000000000000000080;
    localparam [80:0] HOT_S_NORM_READ=81'h000000000000000000100;
    localparam [80:0] HOT_S_NORM_WAIT=81'h000000000000000000200;
    localparam [80:0] HOT_S_NORM_SEND=81'h000000000000000000400;
    localparam [80:0] HOT_S_NORM_FINISH=81'h000000000000000000800;
    localparam [80:0] HOT_S_COEFF_READ=81'h000000000000000001000;
    localparam [80:0] HOT_S_COEFF_WAIT=81'h000000000000000002000;
    localparam [80:0] HOT_S_WEIGHT=81'h000000000000000004000;
    localparam [80:0] HOT_S_GEMM_BEGIN=81'h000000000000000008000;
    localparam [80:0] HOT_S_GEMM_FEED=81'h000000000000000010000;
    localparam [80:0] HOT_S_BIAS_READ=81'h000000000000000020000;
    localparam [80:0] HOT_S_BIAS_START=81'h000000000000000040000;
    localparam [80:0] HOT_S_BIAS_WAIT=81'h000000000000000080000;
    localparam [80:0] HOT_S_ROPE_FIRST=81'h000000000000000100000;
    localparam [80:0] HOT_S_ROPE_FIRST_WAIT=81'h000000000000000200000;
    localparam [80:0] HOT_S_ROPE_SECOND=81'h000000000000000400000;
    localparam [80:0] HOT_S_ROPE_SECOND_WAIT=81'h000000000000000800000;
    localparam [80:0] HOT_S_ROPE_SEND=81'h000000000000001000000;
    localparam [80:0] HOT_S_ROPE_WAIT=81'h000000000000002000000;
    localparam [80:0] HOT_S_ROPE_WRITE=81'h000000000000004000000;
    localparam [80:0] HOT_S_WRITE=81'h000000000000008000000;
    localparam [80:0] HOT_S_WR_CMD=81'h000000000000010000000;
    localparam [80:0] HOT_S_WR_READ=81'h000000000000020000000;
    localparam [80:0] HOT_S_WR_WAIT_DATA=81'h000000000000040000000;
    localparam [80:0] HOT_S_WR_SEND=81'h000000000000080000000;
    localparam [80:0] HOT_S_WR_FINISH=81'h000000000000100000000;
    localparam [80:0] HOT_S_NEXT=81'h000000000000200000000;
    localparam [80:0] HOT_S_DONE=81'h000000000000400000000;
    localparam [80:0] HOT_S_NORM_STATS=81'h000000000002000000000;
    localparam [80:0] HOT_S_POST_DRAIN=81'h000000000004000000000;
    localparam [80:0] HOT_S_ATTN_BEGIN=81'h000000000008000000000;
    localparam [80:0] HOT_S_ATTN_RUN=81'h000000000010000000000;
    localparam [80:0] HOT_S_O_RESTORE=81'h000000000020000000000;
    localparam [80:0] HOT_S_O_CONTEXT=81'h000000000040000000000;
    localparam [80:0] HOT_S_O_NEXT=81'h000000000080000000000;
    localparam [80:0] HOT_S_RES_READ=81'h000000000100000000000;
    localparam [80:0] HOT_S_RES_WAIT=81'h000000000200000000000;
    localparam [80:0] HOT_S_RES_SEND=81'h000000000400000000000;
    localparam [80:0] HOT_S_RES_FINISH=81'h000000000800000000000;
    localparam [80:0] HOT_S_H_START=81'h000000001000000000000;
    localparam [80:0] HOT_S_H_COMMAND=81'h000000002000000000000;
    localparam [80:0] HOT_S_H_READ=81'h000000004000000000000;
    localparam [80:0] HOT_S_H_WAIT=81'h000000008000000000000;
    localparam [80:0] HOT_S_H_SEND=81'h000000010000000000000;
    localparam [80:0] HOT_S_H_FINISH=81'h000000020000000000000;
    localparam [80:0] HOT_S_Z_READ=81'h000000040000000000000;
    localparam [80:0] HOT_S_Z_WAIT=81'h000000080000000000000;
    localparam [80:0] HOT_S_Z_SEND=81'h000000100000000000000;
    localparam [80:0] HOT_S_Z_FINISH=81'h000000200000000000000;
    localparam [80:0] HOT_S_Z_COPY_READ=81'h000000400000000000000;
    localparam [80:0] HOT_S_Z_COPY_WAIT=81'h000000800000000000000;
    localparam [80:0] HOT_S_Z_COPY_WRITE=81'h000001000000000000000;
    localparam [80:0] HOT_S_O_FINISH=81'h000002000000000000000;
    localparam [80:0] HOT_S_FFN_BEGIN=81'h000004000000000000000;
    localparam [80:0] HOT_S_FFN_NEXT=81'h000008000000000000000;
    localparam [80:0] HOT_S_FFN_WAIT=81'h000010000000000000000;
    localparam [80:0] HOT_S_M_LOAD=81'h000020000000000000000;
    localparam [80:0] HOT_S_H_LOAD=81'h000040000000000000000;
    localparam [80:0] HOT_S_DOWN_BEGIN=81'h000080000000000000000;
    localparam [80:0] HOT_S_DOWN_NEXT=81'h000100000000000000000;
    localparam [80:0] HOT_S_LAYER_DONE=81'h000200000000000000000;
    localparam [80:0] HOT_S_BATCH_NEXT=81'h000400000000000000000;
    localparam [80:0] HOT_S_DESC_EXTRA=81'h000800000000000000000;
    localparam [80:0] HOT_S_DESC_EXTRA_CHECK=81'h001000000000000000000;
    localparam [80:0] HOT_S_H_OFFSET=81'h002000000000000000000;
    localparam [80:0] HOT_S_H_ADDRESS=81'h004000000000000000000;
    localparam [80:0] HOT_S_H_COMMIT=81'h008000000000000000000;
    localparam [80:0] HOT_S_FINAL_BEGIN=81'h010000000000000000000;
    localparam [80:0] HOT_S_LM_BEGIN=81'h020000000000000000000;
    localparam [80:0] HOT_S_LM_NEXT=81'h040000000000000000000;
    localparam [80:0] HOT_S_LM_DRAIN=81'h080000000000000000000;
    localparam [80:0] HOT_S_LM_COMMIT=81'h100000000000000000000;
    (* fsm_encoding = "none" *) reg [80:0] state_current;
    reg [80:0] state_next;
    // One-hot state bits directly drive stage ownership and datapath enables.
    reg state_norm_active,state_residual_active,state_normalized_active;
    reg state_hidden_statistics,state_hidden_write,state_residual_to_norm;
    reg state_before_hidden;
    always @(*)begin
        state_norm_active=1'b0;state_residual_active=1'b0;state_normalized_active=1'b0;
        state_hidden_statistics=1'b0;state_hidden_write=1'b0;state_residual_to_norm=1'b0;
        state_before_hidden=1'b0;
        (* parallel_case *) case(1'b1)
            state_current[S_NORM_READ],state_current[S_NORM_WAIT],state_current[S_NORM_SEND],state_current[S_NORM_FINISH]:state_norm_active=1'b1;
            default:begin end
        endcase
        (* parallel_case *) case(1'b1)
            state_current[S_RES_READ],state_current[S_RES_WAIT],state_current[S_RES_SEND],state_current[S_RES_FINISH]:state_residual_active=1'b1;
            default:begin end
        endcase
        (* parallel_case *) case(1'b1)
            state_current[S_Z_READ],state_current[S_Z_WAIT],state_current[S_Z_SEND],state_current[S_Z_FINISH]:state_normalized_active=1'b1;
            default:begin end
        endcase
        (* parallel_case *) case(1'b1)
            state_current[S_H_READ],state_current[S_H_WAIT],state_current[S_H_SEND],state_current[S_H_FINISH]:state_hidden_statistics=1'b1;
            default:begin end
        endcase
        (* parallel_case *) case(1'b1)
            state_current[S_H_COMMAND],state_current[S_H_READ],state_current[S_H_WAIT],state_current[S_H_SEND],state_current[S_H_FINISH]:state_hidden_write=1'b1;
            default:begin end
        endcase
        (* parallel_case *) case(1'b1)
            state_current[S_RES_READ],state_current[S_RES_WAIT],state_current[S_RES_SEND],state_current[S_RES_FINISH],state_current[S_H_START],state_current[S_H_COMMAND],state_current[S_H_READ],state_current[S_H_WAIT],state_current[S_H_SEND],state_current[S_H_FINISH],state_current[S_Z_READ],state_current[S_Z_WAIT],state_current[S_Z_SEND],state_current[S_Z_FINISH]:
                state_residual_to_norm=1'b1;
            default:begin end
        endcase
        (* parallel_case *) case(1'b1)
            state_current[S_IDLE],state_current[S_DESC],state_current[S_RD_CMD],state_current[S_RD_DATA],state_current[S_RD_FINISH],state_current[S_DESC_CHECK],state_current[S_PARAM],state_current[S_INPUT],state_current[S_NORM_READ],state_current[S_NORM_WAIT],state_current[S_NORM_SEND],state_current[S_NORM_FINISH],state_current[S_COEFF_READ],state_current[S_COEFF_WAIT],state_current[S_WEIGHT],state_current[S_GEMM_BEGIN],state_current[S_GEMM_FEED],state_current[S_BIAS_READ],state_current[S_BIAS_START],state_current[S_BIAS_WAIT],state_current[S_ROPE_FIRST],state_current[S_ROPE_FIRST_WAIT],state_current[S_ROPE_SECOND],state_current[S_ROPE_SECOND_WAIT],state_current[S_ROPE_SEND],state_current[S_ROPE_WAIT],state_current[S_ROPE_WRITE],state_current[S_WRITE],state_current[S_WR_CMD],state_current[S_WR_READ],state_current[S_WR_WAIT_DATA],state_current[S_WR_SEND],state_current[S_WR_FINISH],state_current[S_NEXT],state_current[S_DONE],state_current[S_NORM_STATS],state_current[S_POST_DRAIN],state_current[S_ATTN_BEGIN],state_current[S_ATTN_RUN],state_current[S_O_RESTORE],state_current[S_O_CONTEXT],state_current[S_O_NEXT],state_current[S_RES_READ],state_current[S_RES_WAIT],state_current[S_RES_SEND],state_current[S_RES_FINISH]:state_before_hidden=1'b1;
            default:begin end
        endcase
    end
    // Independent weight producer and projection consumer overlap the GEMM engine.
    localparam [2:0] F_IDLE=0,F_COMMAND=1,F_DATA=2,F_FINISH=3,F_COMPLETE=4,F_SCALE_READ=5,F_SCALE_WAIT=6;
    reg [2:0] fetch_state,fetch_next;
    reg [5:0] fetch_tile;
    reg fetch_enabled,fetch_bank,fetch_scale_ready;
    reg fetch_half; // Select either 64-K half of the original DDR tile.
    reg [2:0] fetch_operation;
    reg [10:0] fetch_output_tile;
    reg [2:0] fetch_scale_word;
    // Column base is prepared during the existing eight scale-read transactions.
    // No extra producer state or GEMM issue bubble is required.
    reg [63:0] fetch_column_source,fetch_column_base;
    reg [15:0] fetch_column_index;
    reg [32:0] fetch_column_low;
    reg [6:0] post_state,post_next;
    reg compute_bank,post_bank;
    reg [1:0] post_operation;
    reg [2:0] post_output_tile;
    reg [2047:0] post_biases;
    reg [2047:0] compute_scales, post_scales;
    reg [127:0] post_identity;
    reg [127:0] write_command_reg,write_metadata_reg;
    reg command_hold_valid,command_hold_write;
    reg [127:0] command_hold,metadata_hold;
    reg rope_outstanding;
    // Four reservations cover both synchronous RAM responses and queued input vectors.
    reg [7:0] bias_issued,bias_retired;
    reg [2:0] bias_input_credit,bias_input_count;
    reg [1:0] bias_input_write,bias_input_read;
    reg [6:0] bias_read_address;
    reg [127:0] bias_input_data[0:3],bias_input_coeff[0:3],bias_input_scale[0:3];
    reg [6:0] bias_input_address[0:3];
    reg [127:0] bias_input_metadata;
    reg [575:0] task_cfg;
    reg [127:0] descriptor[0:9]; // format3末字提供E转置权重基址
    reg [63:0] descriptor_base,first_descriptor,first_input,first_context,first_q;
    reg [15:0] remaining_rows,total_rows,group_offset,first_position;
    reg [2:0] ffn_block;
    reg ffn_block_step_pending;
    reg next_layer_input,layer_transition,attention_sweep;
    reg [63:0] scale_base,bias_base,gamma_base,q_base,k_base,v_base,loaded_model_base;
    reg [4:0] layer;
    reg [3:0] rows;
    // GEMM activation validity is captured at block launch, isolating rows fanout from tile data muxes.
    (* DONT_TOUCH = "yes" *) reg [7:0] activation_row_mask;
    // Registered terminal row removes repeated rows-1 arithmetic from the control fanout.
    reg [3:0] rows_last_reg;
    wire [3:0] rows_last = rows_last_reg;
    reg [15:0] task_id;
    reg [2:0] operation; // 0K、1V、2Q、3O、4Gate、5Up
    // Predecode the next operation on the same edge as operation itself.
    // The one-hot register isolates the wide phase fanout from the opcode path.
    reg [7:0] operation_decode_reg;
    reg [2:0] operation_path_next;
    reg [10:0] output_tile;
    reg [5:0] input_tile;
    reg [3:0] inner_k; // 128输入tile内十六个8K块
    reg [2:0] row_index,norm_row;
    reg [1:0] norm_word_cmd_pending; // [1] clear, [0] increment
    reg [6:0] norm_word_storage;
    // Requests use the pending command immediately; storage follows one cycle later.
    // This keeps the word address aligned with the RAM read and AFU metadata.
    wire [6:0] norm_word=norm_word_cmd_pending[1] ? 7'd0 :
        norm_word_cmd_pending[0] ? norm_word_storage+7'd1 : norm_word_storage;
    reg [1:0] norm_row_cmd_pending; // 01 clear, 10 increment/wrap, 11 set last row
    reg [7:0] norm_results_count; // 仅接受结果时更新，不复位计数存储
    reg norm_results_valid; // FFN/归一化阶段边界拉低，下一结果从1开始计数
    wire [7:0] norm_results=norm_results_valid ? norm_results_count : 8'd0;
    reg norm_apply; // 0平方和统计、1重读并归一化输出
    reg [127:0] norm_x,norm_gamma;
    reg [1:0] coeff_kind; reg [3:0] coeff_word;
    reg [2047:0] scales,biases; // 当前128列参数，无跨层复制
    reg [127:0] command_reg,metadata_reg;
    // Keep layer and batch offsets on separate DSPs. BREG captures the row
    // before multiplication; PREG captures the full product before base addition.
    // Row counts settle before the layer/batch decision states. Use this
    // registered comparison at every boundary decision to cut the carry path.
    reg rows_group_final;
    wire layer_address_request=state_current[S_LAYER_DONE] &&
        rows_group_final && descriptor[8][127:64]!=0;
    wire batch_restore_first=!attention_sweep && rows_group_final;
    wire batch_address_request=state_current[S_BATCH_NEXT] && !batch_restore_first;
    reg [1:0] layer_address_valid,batch_address_valid;
    wire layer_reconfigure_fire=layer_address_request && layer_address_valid[1];
    wire batch_reconfigure_fire=state_current[S_BATCH_NEXT] &&
        (batch_restore_first || batch_address_valid[1]);
    wire [17:0] batch_address_row={2'd0,group_offset}+{14'd0,rows};
    wire [47:0] layer_address_product,batch_address_product;
    always @(posedge clk)begin
        if(!rst_n)begin layer_address_valid<=0;batch_address_valid<=0;rows_group_final<=1'b1;end
        else begin
            layer_address_valid<={layer_address_valid[0],layer_address_request};
            batch_address_valid<={batch_address_valid[0],batch_address_request};
            rows_group_final<=remaining_rows<=rows;
        end
    end
    DSP48E2 #(
        .ACASCREG(0),.ADREG(0),.ALUMODEREG(0),.AMULTSEL("A"),.AREG(0),
        .BCASCREG(1),.BMULTSEL("B"),.BREG(1),.CARRYINREG(0),
        .CARRYINSELREG(0),.CREG(0),.DREG(0),.INMODEREG(0),
        .MREG(0),.OPMODEREG(0),.PREADDINSEL("A"),.PREG(1),
        .USE_MULT("MULTIPLY"),.USE_SIMD("ONE48")
    ) layer_address_offset_Inst (
        .CLK(clk),.A(30'd1792),.D(27'd0),.B({2'd0,first_position}),
        .INMODE(5'd0),.OPMODE(9'b000000101),.ALUMODE(4'd0),.CARRYINSEL(3'd0),
        .C(48'd0),.PCIN(48'd0),.ACIN(30'd0),.BCIN(18'd0),
        .CARRYIN(1'b0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
        .CEA1(1'b0),.CEA2(1'b0),.CEAD(1'b0),.CEALUMODE(1'b0),
        .CEB1(1'b0),.CEB2(1'b1),.CEC(1'b0),.CECARRYIN(1'b0),
        .CECTRL(1'b0),.CED(1'b0),.CEINMODE(1'b0),.CEM(1'b0),.CEP(1'b1),
        .RSTA(1'b0),.RSTALLCARRYIN(1'b0),.RSTALUMODE(1'b0),.RSTB(1'b0),
        .RSTC(1'b0),.RSTCTRL(1'b0),.RSTD(1'b0),.RSTINMODE(1'b0),
        .RSTM(1'b0),.RSTP(1'b0),.P(layer_address_product),
        .ACOUT(),.BCOUT(),.CARRYCASCOUT(),.CARRYOUT(),.MULTSIGNOUT(),
        .OVERFLOW(),.PATTERNBDETECT(),.PATTERNDETECT(),.PCOUT(),.UNDERFLOW(),.XOROUT()
    );
    DSP48E2 #(
        .ACASCREG(0),.ADREG(0),.ALUMODEREG(0),.AMULTSEL("A"),.AREG(0),
        .BCASCREG(1),.BMULTSEL("B"),.BREG(1),.CARRYINREG(0),
        .CARRYINSELREG(0),.CREG(0),.DREG(0),.INMODEREG(0),
        .MREG(0),.OPMODEREG(0),.PREADDINSEL("A"),.PREG(1),
        .USE_MULT("MULTIPLY"),.USE_SIMD("ONE48")
    ) batch_address_offset_Inst (
        .CLK(clk),.A(30'd1792),.D(27'd0),.B(batch_address_row),
        .INMODE(5'd0),.OPMODE(9'b000000101),.ALUMODE(4'd0),.CARRYINSEL(3'd0),
        .C(48'd0),.PCIN(48'd0),.ACIN(30'd0),.BCIN(18'd0),
        .CARRYIN(1'b0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
        .CEA1(1'b0),.CEA2(1'b0),.CEAD(1'b0),.CEALUMODE(1'b0),
        .CEB1(1'b0),.CEB2(1'b1),.CEC(1'b0),.CECARRYIN(1'b0),
        .CECTRL(1'b0),.CED(1'b0),.CEINMODE(1'b0),.CEM(1'b0),.CEP(1'b1),
        .RSTA(1'b0),.RSTALLCARRYIN(1'b0),.RSTALUMODE(1'b0),.RSTB(1'b0),
        .RSTC(1'b0),.RSTCTRL(1'b0),.RSTD(1'b0),.RSTINMODE(1'b0),
        .RSTM(1'b0),.RSTP(1'b0),.P(batch_address_product),
        .ACOUT(),.BCOUT(),.CARRYCASCOUT(),.CARRYOUT(),.MULTSIGNOUT(),
        .OVERFLOW(),.PATTERNBDETECT(),.PATTERNDETECT(),.PCOUT(),.UNDERFLOW(),.XOROUT()
    );
    // The address path midpoint lies inside the DSP, after its multiplier.
    // MREG splits preadder/multiply from DSP ALU and the 64-bit base addition.
    // The largest offset is 65535*9728, which fits in 30 unsigned bits.
    wire [47:0] reload_offset_product;
    wire reload_offset_is_m=state_current[S_M_LOAD];
    wire reload_offset_request=state_current[S_M_LOAD] || state_current[S_H_LOAD];
    reg reload_offset_valid;
    always @(posedge clk)begin
        if(!rst_n)reload_offset_valid<=1'b0;
        else reload_offset_valid<=reload_offset_request;
    end
    DSP48E2 #(
        .ACASCREG(0),.ADREG(0),.ALUMODEREG(0),.AMULTSEL("AD"),.AREG(0),
        .BCASCREG(0),.BMULTSEL("B"),.BREG(0),.CARRYINREG(0),
        .CARRYINSELREG(0),.CREG(0),.DREG(0),.INMODEREG(0),
        .MREG(1),.OPMODEREG(0),.PREADDINSEL("A"),.PREG(0),
        .USE_MULT("MULTIPLY"),.USE_SIMD("ONE48")
    ) reload_offset_Inst (
        .CLK(clk),.A(30'd1792),.D(27'd9728),.B({2'b00,task_cfg[52:37]}),
        .INMODE({2'b00,reload_offset_is_m,reload_offset_is_m,1'b0}),
        .OPMODE(9'b000000101),.ALUMODE(4'b0000),.CARRYINSEL(3'b000),
        .C(48'd0),.PCIN(48'd0),.ACIN(30'd0),.BCIN(18'd0),
        .CARRYIN(1'b0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
        .CEA1(1'b0),.CEA2(1'b0),.CEAD(1'b0),.CEALUMODE(1'b0),
        .CEB1(1'b0),.CEB2(1'b0),.CEC(1'b0),.CECARRYIN(1'b0),
        .CECTRL(1'b0),.CED(1'b0),.CEINMODE(1'b0),.CEM(1'b1),.CEP(1'b0),
        .RSTA(1'b0),.RSTALLCARRYIN(1'b0),.RSTALUMODE(1'b0),.RSTB(1'b0),
        .RSTC(1'b0),.RSTCTRL(1'b0),.RSTD(1'b0),.RSTINMODE(1'b0),
        .RSTM(1'b0),.RSTP(1'b0),.P(reload_offset_product),
        .ACOUT(),.BCOUT(),.CARRYCASCOUT(),.CARRYOUT(),.MULTSIGNOUT(),
        .OVERFLOW(),.PATTERNBDETECT(),.PATTERNDETECT(),.PCOUT(),.UNDERFLOW(),.XOROUT()
    );
    reg [3:0] read_kind;
    reg [16:0] read_count,read_last_index;
    reg stream_checked_valid;
    reg [127:0] stream_checked_data;
    reg stream_checked_last;
    reg [16:0] stream_checked_index;
    reg [2:0] stream_checked_row;
    reg [9:0] stream_checked_word;
    reg [3:0] stream_checked_kind;
    reg read_terminal;
    reg [15:0] weight_written;
    // The terminal flag removes the wide weight-count comparison from the
    // fetch/state-control fanout without changing the acceptance cycle.
    reg weight_load_done;
    // A committed 8x128 subblock can be read while later rows keep arriving.
    reg [4:0] weight_chunks_ready [0:1];
    reg [15:0] weight_chunk_mask[0:1][0:3];
    reg [2:0] input_row; reg [9:0] input_word;
    reg cmd_sent,act_sent,act_available;
    reg [1023:0] activation;
    reg [4:0] weight_reads;
    reg weight_pending;
    // Hold the consumed bank identity across the one-cycle release pipeline.
    reg gemm_release_pending,gemm_release_bank;
    reg [3:0] weight_sent;
    reg gemm_terminal;
    reg [4:0] raw_issued;
    reg raw_busy; reg [2047:0] raw_slice; reg [2:0] raw_number; reg [3:0] raw_vector;
    reg [3:0] post_word; reg [2:0] post_row;
    reg [127:0] post_input;
    reg [2:0] rope_pair;
    reg [127:0] rope_x,rope_y,rope_first,rope_second;
    reg rope_write_half;
    reg [4:0] write_word;
    reg [127:0] write_hold;
    reg [127:0] compute_identity,afu_identity;
    reg [63:0] norm_vector_address;
    // 写回地址分成行号、窄偏移、低32位进位、高32位四段。
    reg [16:0] hidden_absolute_row;
    (* use_dsp = "no" *) reg [26:0] hidden_row_offset;
    reg [63:0] hidden_write_base;
    reg [32:0] hidden_address_low;
    // 原始X保存在O bank 1，Wo输出保存在bank 0；残差后bank 1复用为H。
    // Activate仅保存归一化计算输入、Context暂存和FFN中间向量。
    wire output_enabled=descriptor[5][63:0]!=0;
    wire output_phase=operation_decode_reg[3];
    wire ffn_phase=operation_decode_reg[4] || operation_decode_reg[5];
    wire down_phase=operation_decode_reg[6];
    wire lm_phase=operation_decode_reg[7];
    wire linear_phase=output_phase || ffn_phase || down_phase || lm_phase;
    // Decode the type on the same edge that stores descriptor word zero.
    // Later FFN and controller decisions start from the registered flags.
    reg complete_layer_reg,final_enabled_reg;
    wire complete_layer=complete_layer_reg;
    wire final_enabled=final_enabled_reg;
    wire [3:0] compute_rows=lm_phase ? 4'd1 : rows;
    wire [6:0] ffn_words=complete_layer && ffn_block==5 ? 7'd48 : 7'd112;
    wire [2:0] ffn_last_tile=ffn_words==48 ? 3'd2 : 3'd6;
    wire [5:0] ffn_global_tile=ffn_block*6'd7+output_tile;
    wire [11:0] last_k=down_phase ? 12'd607 : 12'd111;
    wire [5:0] fetch_last_tile=fetch_operation==6 ? 6'd37 : 6'd6;
    wire [5:0] fetch_last_output=ffn_phase ? ffn_block*6'd7+ffn_last_tile : 6'd6;
    wire ffn_enabled=descriptor[6]!=0;
    wire ffn_pair=task_id[0]^ffn_block[0];
    wire [2:0] ffn_block_visible=ffn_block_step_pending ? ffn_block+3'd1 : ffn_block;
    reg [3:0] ffn_up_tiles;
    wire ffn_busy,ffn_done,ffn_read,ffn_read_pair;
    reg ffn_post_valid;
    reg [2:0] ffn_post_block;
    wire ffn_post_start=ffn_phase && operation_decode_reg[5] && output_tile==0 && input_tile==0 && inner_k==0 &&
        state_current[S_WEIGHT] && (!ffn_post_valid || ffn_post_block!=ffn_block) && !ffn_busy;
    wire [2:0] ffn_service_block=ffn_post_start ? ffn_block : ffn_post_block;
    wire ffn_service_pair=task_id[0]^ffn_service_block[0];
    wire [6:0] ffn_service_words=complete_layer && ffn_service_block==5 ? 7'd48 : 7'd112;
    wire ffn_consumer_available=!operation_decode_reg[5] || (ffn_post_valid && ffn_post_block==ffn_block) || !ffn_busy;
    wire ffn_active=state_current[S_FFN_BEGIN] || (ffn_phase && busy);
    wire [2:0] ffn_read_row,ffn_store_row;
    wire [6:0] ffn_read_word,ffn_store_word;
    wire ffn_store_valid,ffn_afu_valid,ffn_afu_last,ffn_result_ready;
    wire [1:0] ffn_store_bank;
    wire [127:0] ffn_store_data,ffn_afu_x,ffn_afu_y,ffn_afu_meta;
    wire ffn_dma_valid,ffn_write_valid,ffn_write_last;
    wire [127:0] ffn_dma_command,ffn_dma_metadata,ffn_write_data,ffn_write_meta;
    wire ffn_store_fire=ffn_active && ffn_store_valid;
    wire output_gemm_write_ready;
    wire bias_ready,bias_valid,bias_busy; wire [127:0] bias_data,bias_meta;
    reg [127:0] raw_input_data;
    wire lm_input_ready,lm_write_valid,lm_complete,lm_busy;
    wire [127:0] lm_write_data;
    wire [14:0] lm_write_address;
    wire [31:0] lm_best_token;
    reg lm_finished;
    wire [14:0] lm_input_address={output_tile,4'd0}+{11'd0,raw_number,1'b0}+{14'd0,raw_vector[0]};
    wire raw_write_fire=raw_busy && (lm_phase ? lm_input_ready : output_gemm_write_ready) &&
        (!linear_phase || bias_valid);
    wire [3:0] compute_opcode=lm_phase ? 4'd9 : down_phase ? 4'd8 : ffn_phase ? (operation_decode_reg[4] ? 4'd6 : 4'd7) :
                               (operation_decode_reg[3] ? 4'd5 : (operation_decode_reg[0] ? 4'd1 : (operation_decode_reg[1] ? 4'd2 : 4'd0)));
    reg residual_x_ready,residual_o_ready,gamma_response_ready,norm_input_ready;
    reg hidden_dma_sent,hidden_afu_sent,hidden_b_done,hidden_stats_done;
    wire output_read_valid;
    wire [127:0] output_read_x,output_read_y;
    wire [1:0] output_pair_full;
    wire residual_result=(output_phase || down_phase) && state_residual_active && afu_result_valid && afu_result_ready;
    wire normalized_result=output_phase && state_normalized_active && afu_result_valid && afu_result_ready;
    wire hidden_both_sent=(hidden_dma_sent || (wr_valid && wr_ready)) && (down_phase || hidden_afu_sent || (afu_valid && afu_ready));
    // Register the complete shared handshake predicate at 2.480/5.102 ns.
    // Existing sent flags suppress duplicate transfers and guard entry into H_SEND.
    reg hidden_send_complete_reg;
    always @(posedge clk)begin
        if(!rst_n)hidden_send_complete_reg<=1'b0;
        else hidden_send_complete_reg<=hidden_both_sent;
    end
    wire hidden_send_advance=state_current[S_H_SEND] && hidden_send_complete_reg &&
        hidden_dma_sent && (down_phase || hidden_afu_sent);

    wire hidden_statistics=output_phase && state_hidden_statistics && afu_result_valid && afu_result_ready;
    wire context_scratch_write;
    wire output_ram_read=state_current[S_NORM_READ] || state_current[S_RES_READ] || state_current[S_H_READ] || state_current[S_Z_READ] || state_current[S_Z_COPY_READ];
    wire output_ram_write=linear_phase && !lm_phase && raw_busy && bias_valid;
    wire [6:0] output_write_word=raw_busy ? {output_tile,4'd0}+{raw_number,1'b0}+raw_vector[0] : norm_results[6:0];
    wire [2:0] output_write_row=raw_busy ? raw_vector[3:1] : norm_row_visible;
    wire [127:0] output_write_data;
    wire [11:0] global_k={2'd0,input_tile,inner_k};
    // Each original input tile has two 64-K chunks, so every output tile
    // starts at bank zero; projection ownership remains independent.
    wire weight_bank=inner_k[3];
    wire projection_bank=compute_bank;
    wire [7:0] rows_times_two={3'd0,compute_rows,1'b0};
    wire [3:0] rope_first_word={rope_pair[2],3'b0}+{2'd0,rope_pair[1:0]};
    wire [3:0] rope_second_word=rope_first_word+4'd4;
    wire stream_fire=rd_valid && rd_ready;
    wire stream_ok=1'b1;
    wire stream_checked_fire=stream_checked_valid;
    // Wo reloads each 128-dimensional Context tile before consuming it.
    wire context_reload_write=output_phase &&
        stream_checked_kind==READ_CONTEXT && stream_checked_fire;
    wire hidden_reload_write=stream_checked_kind==READ_H && stream_checked_fire;
    // READ_INPUT装载原始X到O bank 1，包括Wo前恢复X；DMA按O写口ready推进。
    wire input_load_active=state_current[S_RD_DATA] && read_kind==READ_INPUT;
    wire input_reload_write=stream_checked_kind==READ_INPUT && stream_checked_fire;
    wire [15:0] operation_scale_offset=fetch_operation==6 ? 16'd11776 : (fetch_operation==4 ? 16'd2048 : (fetch_operation==5 ? 16'd6912 :
        (fetch_operation==3 ? 16'd1152 : (fetch_operation==0 ? 16'd896 : (fetch_operation==1 ? 16'd1024 : 16'd0)))));
    wire [15:0] operation_bias_offset=operation_decode_reg[0] ? 16'd896 : (operation_decode_reg[1] ? 16'd1024 : 16'd0);
    wire [63:0] operation_base=fetch_operation==7 ? descriptor[9][63:0] : fetch_operation==6 ? descriptor[7][127:64] : (fetch_operation==4 ? descriptor[6][63:0] : (fetch_operation==5 ? descriptor[6][127:64] :
        (fetch_operation==3 ? descriptor[5][63:0] : (fetch_operation==0 ? k_base : (fetch_operation==1 ? v_base : q_base)))));
    wire [31:0] scale_element=fetch_operation==7 ? 32'd304128+fetch_output_tile*32'd128 :
        layer*32'd12672+operation_scale_offset+fetch_output_tile*32'd128;

    // Keep the decode register aligned with every operation update below.
    // This is a phase tag only; the architectural operation register remains
    // the source of all state and counter updates.
    always @(*) begin
        operation_path_next=operation;
        if((state_current[S_IDLE] || state_current[S_DONE]) && start)
            operation_path_next=3'd0;
        if(state_current[S_M_LOAD])
            operation_path_next=3'd6;
        if(state_current[S_DOWN_BEGIN])
            operation_path_next=3'd6;
        if(layer_reconfigure_fire)
            operation_path_next=3'd0;
        if(batch_reconfigure_fire)
            operation_path_next=3'd0;
        if(state_current[S_O_RESTORE])
            operation_path_next=3'd3;
        if(state_current[S_FFN_BEGIN])
            operation_path_next=3'd4;
        if(state_current[S_FFN_NEXT] && output_tile==ffn_last_tile)
            operation_path_next=3'd5;
        if(state_current[S_FINAL_BEGIN])
            operation_path_next=3'd7;
        if(state_current[S_NEXT] && operation!=3'd2)
            operation_path_next=operation+3'd1;
    end
    wire [15:0] gamma_element={layer,11'd0}-{3'd0,layer,8'd0};
    wire [63:0] token_position={48'd0,task_cfg[52:37]}+row_index;
    wire [2:0] params_loaded;
    wire scale_load_ready,bias_load_ready,gamma_load_ready;
    wire scale_load_done,bias_load_done,gamma_load_done;
    wire scale_read_valid,bias_read_valid,gamma_read_valid;
    wire [255:0] scale_read_data;
    wire [127:0] bias_read_data,gamma_read_data;
    wire params_load_start=state_current[S_PARAM];
    wire params_load_valid=state_current[S_RD_DATA] && stream_checked_kind>=READ_SCALE && stream_checked_kind<=READ_GAMMA && stream_checked_fire;
    wire params_load_ready=read_kind==READ_SCALE ? scale_load_ready : (read_kind==READ_BIAS ? bias_load_ready : gamma_load_ready);
    wire gamma_read_en=state_current[S_NORM_READ] || state_current[S_Z_READ];
    wire scale_read_en=fetch_state==F_SCALE_READ;
    wire bias_address_request=state_current[S_COEFF_READ];
    reg bias_read_en;
    wire [15:0] gamma_read_element=(lm_phase ? 16'd43008 : gamma_element)+{6'd0,norm_word,3'd0}+(output_phase ? 16'd896 : 16'd0);
    wire [31:0] scale_read_element=scale_element+fetch_scale_word*16;
    // The measured path midpoint is after the layer product, before the
    // operation/tile/word addition. PREG keeps that cut inside the existing DSP.
    wire [47:0] bias_layer_product;
    reg [14:0] bias_address_tail;
    wire [14:0] bias_address_tail_next=operation_bias_offset[14:0]+
        {output_tile[7:0],7'd0}+{8'd0,coeff_word,3'd0};
    wire [14:0] bias_read_element=bias_layer_product[14:0]+bias_address_tail;
    always @(posedge clk)begin
        bias_address_tail<=bias_address_tail_next;
        if(!rst_n)bias_read_en<=1'b0;
        else bias_read_en<=bias_address_request;
    end
    (* DONT_TOUCH="yes" *) DSP48E2 #(
        .ACASCREG(0),.ADREG(0),.ALUMODEREG(0),.AMULTSEL("A"),.AREG(0),
        .BCASCREG(0),.BMULTSEL("B"),.BREG(0),.CARRYINREG(0),
        .CARRYINSELREG(0),.CREG(0),.DREG(0),.INMODEREG(0),
        .MREG(0),.OPMODEREG(0),.PREADDINSEL("A"),.PREG(1),
        .USE_MULT("MULTIPLY"),.USE_SIMD("ONE48")
    ) bias_layer_offset_Inst (
        .CLK(clk),.A({25'd0,layer}),.D(27'd0),.B(18'd1152),
        .INMODE(5'd0),.OPMODE(9'b000000101),.ALUMODE(4'd0),.CARRYINSEL(3'd0),
        .C(48'd0),.PCIN(48'd0),.ACIN(30'd0),.BCIN(18'd0),
        .CARRYIN(1'b0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
        .CEA1(1'b0),.CEA2(1'b0),.CEAD(1'b0),.CEALUMODE(1'b0),
        .CEB1(1'b0),.CEB2(1'b0),.CEC(1'b0),.CECARRYIN(1'b0),
        .CECTRL(1'b0),.CED(1'b0),.CEINMODE(1'b0),.CEM(1'b0),.CEP(1'b1),
        .RSTA(1'b0),.RSTALLCARRYIN(1'b0),.RSTALUMODE(1'b0),.RSTB(1'b0),
        .RSTC(1'b0),.RSTCTRL(1'b0),.RSTD(1'b0),.RSTINMODE(1'b0),
        .RSTM(1'b0),.RSTP(1'b0),.P(bias_layer_product),
        .ACOUT(),.BCOUT(),.CARRYCASCOUT(),.CARRYOUT(),.MULTSIGNOUT(),
        .OVERFLOW(),.PATTERNBDETECT(),.PATTERNDETECT(),.PCOUT(),.UNDERFLOW(),.XOROUT()
    );
    wire act_tile_valid; wire [1023:0] act_tile_q;
    wire act_norm_result=norm_apply&&state_norm_active&&afu_result_valid&&afu_result_ready;
    wire norm_results_valid_early_clear=((state_current[S_IDLE] || state_current[S_DONE]) && start) ||
        state_current[S_DOWN_BEGIN] || state_current[S_DOWN_NEXT] || layer_reconfigure_fire || batch_reconfigure_fire;
    wire norm_results_valid_middle_clear=(state_current[S_NORM_FINISH] && norm_results==112) ||
        state_current[S_O_RESTORE] || state_current[S_O_NEXT];
    wire norm_results_valid_late_clear=(state_current[S_RES_FINISH] && !state_next[S_RES_FINISH]) ||
        (state_current[S_H_FINISH] && state_next[S_Z_READ]) || state_current[S_FINAL_BEGIN];
    wire norm_results_valid_result_set=residual_result || normalized_result;
    wire norm_results_valid_next= !norm_results_valid_late_clear &&
        (norm_results_valid_result_set || (!norm_results_valid_middle_clear &&
        (act_norm_result || (!norm_results_valid_early_clear && norm_results_valid))));
    always @(posedge clk)begin
        if(!rst_n)norm_results_valid<=1'b0;
        else norm_results_valid<=norm_results_valid_next;
    end
    wire act_vec_write=(stream_checked_kind==READ_M && stream_checked_fire)||act_norm_result||context_reload_write||state_current[S_Z_COPY_WRITE];
    wire [2:0] act_vec_row=context_reload_write ? row_index : (state_current[S_Z_COPY_WRITE] ? norm_row_visible : (act_norm_result ? norm_row_visible : stream_checked_row));
    wire [9:0] act_vec_word=context_reload_write ? 10'd112+stream_checked_word : (state_current[S_Z_COPY_WRITE] ? {3'd0,norm_word} : (act_norm_result ? {2'd0,norm_results} : stream_checked_word));
    wire [127:0] act_vec_data=state_current[S_Z_COPY_WRITE] ? norm_x : (act_norm_result ? afu_result_data : stream_checked_data);
    wire weight_load_ready,weight_valid,weight_valid_pair; wire [1023:0] weight_q,weight_q_pair; wire [1:0] weight_full;
    wire weight_chunk_ready=weight_full[weight_bank];

    wire dequant_ready,dequant_valid,dequant_busy; wire [255:0] dequant_data; wire [15:0] dequant_word;
    wire weight_read_enable;
    // GEMM ready grants a complete eight-row PE bank before RAM reads start.
    // Registered streaming state removes live GEMM ready from the RAM address.
	localparam [1:0] W_WAIT=2'd0,W_STREAM=2'd1,W_DRAIN=2'd2;
	reg [1:0] weight_stream_state,weight_stream_next;
    wire weight_pair_valid=weight_pending;
    wire [7:0] weight_read_word0=attn_active ? attn_weight_read_word : ({1'b0,inner_k[2:0],4'd0}+{weight_reads,1'b0});
    wire [7:0] weight_read_word1=({1'b0,inner_k[2:0],4'd0}+{weight_reads,1'b0})+8'd1;
    // One dual-port request returns the two adjacent 1024-bit halves that
    // used to be assembled by qwen25_weight_fifo.
    assign weight_read_enable=weight_stream_state==W_STREAM;
    wire [2047:0] weight_data={weight_q_pair,weight_q};
    wire [1:0] proj_zero_full,proj_one_full;
    wire p0_valid,p1_valid;wire [127:0] p0_q,p1_q;
    reg [127:0] selected_bias;
    wire [127:0] selected_scale=post_scales[bias_read_address[3:0]*128+:128];
    wire [127:0] selected_norm_meta={9'd0,10'd0,3'd0,1'b0,16'd0,1'b0,4'd0,4'd11,1'b0,2'd0,16'd0,4'd8,rows,16'd0,16'd0,layer,task_id}; // 后续按位覆写word和query_row
    wire gemm_cmd_ready,gemm_act_ready,gemm_weight_ready,gemm_valid,gemm_done,gemm_input_done,gemm_partial,gemm_error,gemm_aborted,gemm_busy;
    wire [7:0] gemm_error_code;wire [2047:0] gemm_bf16;wire [4095:0] gemm_fp32;wire [127:0] gemm_meta;
    wire [2:0] gemm_slice;wire [7:0] gemm_rows;wire [15:0] gemm_cols;wire gemm_last;
    wire gemm_cmd_valid=state_current[S_GEMM_FEED] && !cmd_sent;
    wire gemm_act_valid=state_current[S_GEMM_FEED] && act_available && !act_sent;
    wire gemm_weight_valid=attn_active ? attn_gemm_weight_valid : state_current[S_GEMM_FEED] && weight_pair_valid;
	always @(*)begin
		weight_stream_next=weight_stream_state;
		if(!state_current[S_GEMM_FEED])begin
			weight_stream_next=W_WAIT;
		end else begin
			case(weight_stream_state)
				W_WAIT:if(gemm_weight_ready)begin
					weight_stream_next=W_STREAM;
				end
				W_STREAM:if(weight_reads==7)begin
					weight_stream_next=W_DRAIN;
				end
				W_DRAIN:begin end
				default:begin
					weight_stream_next=W_WAIT;
				end
			endcase
		end
	end
	always @(posedge clk)begin
		if(!rst_n)begin
			weight_stream_state<=W_WAIT;
		end else begin
			weight_stream_state<=weight_stream_next;
		end
	end
    wire gemm_bundle_sent=cmd_sent && act_sent && weight_sent==8;
    wire gemm_block_complete=global_k==last_k ? gemm_terminal : gemm_bundle_sent;
    // Decode GEMM completion locally instead of feeding back the full next-state cone.
    wire gemm_feed_complete=gemm_block_complete && !raw_busy &&
        (global_k!=last_k || linear_phase || post_state==S_IDLE);
    // This complete wait predicate is the measured LUT midpoint: 2.440/4.938 ns.
    // cmd_sent rejects the initial FEED cycle before the predicate has been sampled.
    // Once complete, the block keeps its K index and bank until the delayed advance.
    reg gemm_feed_blocked_reg;
    always @(posedge clk)begin
        if(!rst_n)gemm_feed_blocked_reg<=1'b0;
        else gemm_feed_blocked_reg<=state_current[S_GEMM_FEED] && !gemm_feed_complete;
    end
    wire gemm_feed_advance=state_current[S_GEMM_FEED] && cmd_sent && !gemm_feed_blocked_reg;
    wire gemm_release_request=gemm_feed_advance && inner_k[2:0]==7;
    wire gemm_output_ready=!raw_busy;
    reg [127:0] gemm_command;
    wire [127:0] raw_vector_data=linear_phase ? bias_data : raw_input_data;
    wire [3:0] raw_select=linear_phase ? raw_issued[3:0] : raw_vector;
    wire [127:0] raw_scale=compute_scales[({raw_number,1'b0}+raw_issued[0])*128+:128];
    wire raw_scale_valid=linear_phase && raw_busy && raw_issued<rows_times_two;
    wire raw_scale_fire=raw_scale_valid && bias_ready;
    wire [3:0] direct_dequant_valid,direct_dequant_ready,direct_dequant_busy;
    wire [1023:0] direct_dequant_data;
    wire [63:0] direct_dequant_word;
    wire [31:0] direct_load_word;
    wire [3:0] direct_write_fire=direct_dequant_valid & direct_dequant_ready;
    wire [2:0] direct_write_count={2'd0,direct_write_fire[0]}+{2'd0,direct_write_fire[1]}+
                                {2'd0,direct_write_fire[2]}+{2'd0,direct_write_fire[3]};
    assign output_write_data=raw_vector_data;
    reg [127:0] afu_metadata;
    // Attention owns the existing services only after QKV has drained.
    wire attn_busy,attn_done,attn_error;wire [7:0] attn_error_code;
    wire attn_service_abort,attn_weight_bank,attn_p_write_bank,attn_p_bank,attn_p_afu_bank,attn_p_commit,attn_p_release;
    wire [1:0] attn_p_full;
    wire [11:0] attn_p_afu_word;
    wire attn_enabled=task_cfg[255:192]!=0 && task_cfg[36:21]!=0;
    wire [16:0] attn_cache_end={1'b0,first_position}+total_rows;
    wire [16:0] attn_group_end={1'b0,task_cfg[52:37]}+rows;
    wire [16:0] attn_visible_end=attention_sweep ? attn_cache_end : attn_group_end;
    wire [15:0] attn_context_limit=task_cfg[36:21]>attn_visible_end ? attn_visible_end[15:0] : task_cfg[36:21];
    // Split visible-end addition/clamping from Attention's launch validation.
    // Group configuration settles before S_ATTN_BEGIN; sample continuously so
    // the added stage keeps the launch cycle unchanged. No data reset is needed.
    reg [15:0] attn_context;
    always @(posedge clk) begin
        attn_context<=attn_context_limit;
    end
    wire attn_cache_invalidate=((state_current[S_IDLE] || state_current[S_DONE]) && start) ||
        layer_reconfigure_fire;
    wire attn_active=state_current[S_ATTN_BEGIN] || state_current[S_ATTN_RUN];
    wire attn_start=state_current[S_ATTN_BEGIN];
    // These state-bit copies have the same D inputs and reset as the ownership state.
    // Each protected control register drives one interface or 256-bit payload group.
    (* DONT_TOUCH = "yes" *) reg [1:0] attn_dma_owner_copies,attn_afu_owner_copies;
    (* DONT_TOUCH = "yes" *) reg [1:0] attn_gemm_control_copies,attn_buffer_owner_copies;
    (* DONT_TOUCH = "yes" *) reg [1:0] attn_activation_owner_copies[0:3];
    (* DONT_TOUCH = "yes" *) reg [1:0] attn_weight_owner_copies[0:7];
    wire attn_dma_owner=|attn_dma_owner_copies;
    wire attn_afu_owner=|attn_afu_owner_copies;
    wire attn_gemm_owner=|attn_gemm_control_copies;
    wire attn_buffer_owner=|attn_buffer_owner_copies;
    integer owner_group;
    always @(posedge clk)begin
        if(!rst_n)begin
            attn_dma_owner_copies<=2'b00;attn_afu_owner_copies<=2'b00;
            attn_gemm_control_copies<=2'b00;attn_buffer_owner_copies<=2'b00;
            for(owner_group=0;owner_group<4;owner_group=owner_group+1)attn_activation_owner_copies[owner_group]<=2'b00;
            for(owner_group=0;owner_group<8;owner_group=owner_group+1)attn_weight_owner_copies[owner_group]<=2'b00;
        end else begin
            attn_dma_owner_copies<={state_next[S_ATTN_RUN],state_next[S_ATTN_BEGIN]};
            attn_afu_owner_copies<={state_next[S_ATTN_RUN],state_next[S_ATTN_BEGIN]};
            attn_gemm_control_copies<={state_next[S_ATTN_RUN],state_next[S_ATTN_BEGIN]};
            attn_buffer_owner_copies<={state_next[S_ATTN_RUN],state_next[S_ATTN_BEGIN]};
            for(owner_group=0;owner_group<4;owner_group=owner_group+1)
                attn_activation_owner_copies[owner_group]<={state_next[S_ATTN_RUN],state_next[S_ATTN_BEGIN]};
            for(owner_group=0;owner_group<8;owner_group=owner_group+1)
                attn_weight_owner_copies[owner_group]<={state_next[S_ATTN_RUN],state_next[S_ATTN_BEGIN]};
        end
    end

    wire attn_read_valid,attn_read_ready,attn_data_valid,attn_data_ready,attn_memory_busy,attn_memory_error;
    wire [63:0] attn_read_address,attn_read_base;wire [12:0] attn_read_bytes;wire [127:0] attn_read_data;
    wire attn_context_valid,attn_context_ready;wire [127:0] attn_context_data;
    wire [2:0] attn_context_row,attn_context_word;wire [3:0] attn_context_head;
    wire attn_weight_valid,attn_weight_commit,attn_weight_release,attn_weight_read;
    wire [9:0] attn_weight_word;wire [255:0] attn_weight_data;wire [7:0] attn_weight_read_word;
    wire attn_p_write,attn_p_read,attn_p_valid,attn_p_afu_read,attn_p_afu_valid;
    wire attn_p_write_ready,attn_p_afu_write,attn_p_afu_write_bank;
    wire [14:0] attn_p_afu_address;
    wire [127:0] attn_p_afu_write_data;
    wire [14:0] attn_p_address;wire [11:0] attn_p_word;wire [2:0] attn_p_row;
    wire [127:0] attn_p_write_data,attn_p_afu_data;wire [1023:0] attn_p_data;
    wire [2:0] attn_p_max_row;
    wire [15:0] attn_p_maximum,attn_p_max_words,attn_p_row_maximum;
    wire [31:0] attn_p_max_index;
    wire attn_p_max_first,attn_p_max_last,attn_p_row_complete;
    wire attn_cmd_valid,attn_act_valid,attn_gemm_weight_valid,attn_gemm_ready;
    wire [127:0] attn_command,attn_identity;wire [1023:0] attn_activation;wire [2047:0] attn_gemm_weight;
    wire attn_afu_valid,attn_afu_last,attn_afu_result_ready;
    wire [3:0] attn_afu_command;wire [1:0] attn_afu_phase;wire [63:0] attn_afu_length;
    wire [127:0] attn_afu_x,attn_afu_y,attn_afu_meta;wire [7:0] attn_afu_mask;
    wire attn_dma_valid,attn_dma_rd_ready,attn_dma_wr_valid,attn_dma_wr_last;
    wire [127:0] attn_dma_command,attn_dma_metadata,attn_dma_wr_data,attn_dma_wr_meta;wire [15:0] attn_dma_wr_keep;

    assign context_scratch_write=attn_active && attn_context_valid && attn_context_ready;
    wire [127:0] afu_expected_meta=afu_identity;
    wire afu_fire=afu_valid && afu_ready;
    // Register request acceptance at the measured 2.247 ns path midpoint.
    // Hold the SEND cursor for this acknowledgement and suppress a second issue.
    reg norm_accept_pending;
    always @(posedge clk)begin
        if(!rst_n)norm_accept_pending<=1'b0;
        else norm_accept_pending<=afu_fire &&
            (state_current[S_RES_SEND] || state_current[S_Z_SEND]);
    end
    // Register the measured midpoint (2.558 ns of the 5.162 ns path).
    // The LUT predicate is SEND acceptance OR WAIT with an incomplete RAM response.
    // Keep the word/data owned until the registered acceptance advances the FSM.
    wire norm_buffers_ready=(norm_input_ready || output_read_valid) &&
        (gamma_response_ready || gamma_read_valid);
    reg norm_frontend_midpoint_reg;
    wire norm_frontend_midpoint_now=(state_current[S_NORM_SEND] && afu_fire) ||
        (state_current[S_NORM_WAIT] && !norm_buffers_ready);
    always @(posedge clk)begin
        if(!rst_n)norm_frontend_midpoint_reg<=1'b0;
        else norm_frontend_midpoint_reg<=norm_frontend_midpoint_now;
    end
    // The operation/state decode is registered before it reaches norm_word.
    // A clear has priority over an increment if both are requested.
    wire norm_word_clear_event=(state_current[S_IDLE] || state_current[S_DONE]) && start ||
        state_current[S_DOWN_BEGIN] || state_current[S_DOWN_NEXT] ||
        layer_reconfigure_fire ||
        batch_reconfigure_fire || state_current[S_O_RESTORE] || state_current[S_O_NEXT] ||
        (state_current[S_NORM_STATS] && afu_result_valid && afu_result_ready) ||
        (state_current[S_NORM_FINISH] && norm_results==112) ||
        (state_current[S_RES_FINISH] && !state_next[S_RES_FINISH]) || state_current[S_H_START] ||
        (down_phase && state_current[S_H_FINISH] && state_next[S_H_START]) ||
        (state_current[S_H_FINISH] && state_next[S_Z_READ]) ||
        (state_current[S_Z_FINISH] && state_next[S_Z_COPY_READ]) || state_current[S_FINAL_BEGIN] ||
        (state_current[S_Z_COPY_WRITE] && norm_word==111);
    wire norm_word_increment_event=(state_current[S_NORM_SEND] && norm_frontend_midpoint_reg && norm_word!=111) ||
        (state_current[S_RES_SEND] && norm_accept_pending && norm_word!=111) ||
        (state_current[S_Z_SEND] && norm_accept_pending && norm_word!=111) ||
        (hidden_send_advance && norm_word!=111) ||
        (state_current[S_Z_COPY_WRITE] && norm_word!=111);
    // Register row-boundary commands separately from the row value.  The
    // visible row applies the pending command combinationally for the current
    // state, while the storage register receives it on the following edge.
    wire norm_row_clear_event=((state_current[S_IDLE] || state_current[S_DONE]) && start) ||
        state_current[S_DOWN_BEGIN] || state_current[S_DOWN_NEXT] ||
        layer_reconfigure_fire ||
        batch_reconfigure_fire || state_current[S_O_RESTORE] || state_current[S_O_NEXT];
    wire norm_row_increment_event=(state_current[S_NORM_FINISH] && norm_results==112 && !lm_phase) ||
        (state_current[S_RES_FINISH] && !state_next[S_RES_FINISH]) ||
        (down_phase && state_current[S_H_FINISH] && state_next[S_H_START]) ||
        (state_current[S_Z_COPY_WRITE] && norm_word==111);
    wire norm_row_last_event=state_current[S_FINAL_BEGIN];
    wire ffn_block_clear_event=((state_current[S_IDLE] || state_current[S_DONE]) && start) ||
        state_current[S_INPUT] ||
        layer_reconfigure_fire ||
        batch_reconfigure_fire;
    wire ffn_block_increment_event=state_current[S_FFN_WAIT] && state_next[S_FFN_BEGIN];
    wire [2:0] norm_row_visible=norm_row_cmd_pending==2'b01 ? 3'd0 :
        norm_row_cmd_pending==2'b10 ? (norm_row==rows_last ? 3'd0 : norm_row+3'd1) :
        norm_row_cmd_pending==2'b11 ? rows_last : norm_row;
    wire read_request=state_current[S_RD_CMD] || fetch_state==F_COMMAND;
    wire write_request=post_state==S_WR_CMD || state_current[S_H_COMMAND] || (ffn_active && ffn_dma_valid);
    wire read_command_fire=command_hold_valid && dma_cmd_ready && !command_hold_write;
    wire write_command_fire=command_hold_valid && dma_cmd_ready && command_hold_write;
    wire fetch_begin=fetch_enabled && fetch_scale_ready && fetch_state==F_IDLE && !weight_full[fetch_bank] && (!output_phase || state_current[S_WEIGHT]);
    wire all_dequant_busy=dequant_busy || (|direct_dequant_busy);
    // Cut the read-completion cone before it joins the main FSM hold reduction.
    // Sample on DMA completion as well as FINISH, then consume once in FINISH.
    reg read_finish_pending;
    reg read_payload_complete;
    always @(*)begin
        read_payload_complete=1'b1;
        case(read_kind)
            READ_SCALE:read_payload_complete=params_loaded[0];
            READ_BIAS:read_payload_complete=params_loaded[1];
            READ_GAMMA:read_payload_complete=params_loaded[2];
            READ_WEIGHT:read_payload_complete=weight_load_done && !all_dequant_busy;
            default:begin end
        endcase
    end
    always @(posedge clk)begin
        if(!rst_n)read_finish_pending<=1'b0;
        else read_finish_pending<=read_payload_complete &&
            ((state_current[S_RD_DATA] && rd_done) ||
             (state_current[S_RD_FINISH] && read_terminal));
    end
    wire weight_direct_write=(DIRECT_WEIGHT_READ!=0) && (direct_write_count!=0);
    wire weight_dequant_write=dequant_valid && weight_load_ready;
    wire fetch_commit=fetch_state==F_FINISH && weight_load_done && !all_dequant_busy;
    always @(posedge clk)begin
        if(!rst_n)weight_load_done<=1'b0;
        else if(weight_direct_write)
            weight_load_done<=(weight_written+{12'd0,direct_write_count}==16'd512);
        else if(weight_dequant_write)
            weight_load_done<=(weight_written==16'd511);
        else if(fetch_begin)weight_load_done<=1'b0;
    end
    // Preserve the original bank-update priority while separating sixteen scale-address loads.
    (* DONT_TOUCH = "yes" *) reg [15:0] fetch_bank_scale_copies;
    reg fetch_bank_copy_next;
    always @(*)begin
        fetch_bank_copy_next=fetch_bank;
        if(state_current[S_O_RESTORE] || state_current[S_FFN_BEGIN] || state_current[S_LM_BEGIN] || state_current[S_DOWN_BEGIN])
            fetch_bank_copy_next=1'b0;
        if(state_current[S_WEIGHT] && output_phase && fetch_state==F_IDLE && !weight_full[weight_bank])
            fetch_bank_copy_next=weight_bank;
        if(fetch_commit && !output_phase)fetch_bank_copy_next=!fetch_bank;
    end
    always @(posedge clk)begin
        if(!rst_n || state_current[S_IDLE] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || (state_current[S_DONE] && start))
            fetch_bank_scale_copies<=16'd0;
        else fetch_bank_scale_copies<={16{fetch_bank_copy_next}};
    end

    wire [2047:0] selected_scale_bank;
    genvar scale_bank_group;
    generate for(scale_bank_group=0;scale_bank_group<16;scale_bank_group=scale_bank_group+1)begin:gen_scale_bank_group
        (* ram_style = "distributed" *) reg [127:0] scale_slice[0:1];
        always @(posedge clk)begin
            if(fetch_begin)scale_slice[fetch_bank_scale_copies[scale_bank_group]]<=scales[scale_bank_group*128+:128];
        end
        assign selected_scale_bank[scale_bank_group*128+:128]=scale_slice[weight_bank];
    end endgenerate
    wire post_launch=!linear_phase && gemm_feed_advance && global_k==last_k;
    wire post_bias_complete=post_state==S_BIAS_READ && bias_retired=={rows,4'b0};
    wire post_finished=post_state==S_NEXT;
    // Register the complete drain predicate at the original timing midpoint:
    // GEMM credits -> busy reduction -> this flag -> state decode -> counter clear.
    // The blocked polarity matches the LUT output at 2.445 ns of the 4.988 ns path.
    wire pipeline_quiet_now=(!linear_phase || !afu_busy) && !ffn_busy && !attn_busy && !attn_memory_busy && !rd_busy && !wr_busy && !gemm_busy && !all_dequant_busy && !bias_busy && !rope_outstanding && !raw_busy;
    reg pipeline_blocked_reg;
    always @(posedge clk)begin
        if(!rst_n)pipeline_blocked_reg<=1'b1;
        else pipeline_blocked_reg<=!pipeline_quiet_now;
    end
    wire pipeline_quiet=!pipeline_blocked_reg;
    wire bias_read_request=post_state==S_BIAS_READ && bias_issued<{rows,4'b0} && bias_input_credit<4;
    wire bias_input_valid=bias_input_count!=0 && !linear_phase;
    wire bias_input_fire=bias_input_valid && bias_ready;
    wire bias_output_fire=bias_valid && !linear_phase && post_state==S_BIAS_READ;

    wire p1_rope_read=post_state==S_ROPE_FIRST||post_state==S_ROPE_SECOND;
    wire p1_dma_read=post_state==S_WR_READ;
    wire p1_write=(post_state==S_BIAS_READ && bias_output_fire)||(post_state==S_ROPE_WRITE);
    wire [3:0] p1_write_word=post_state==S_ROPE_WRITE ? (rope_write_half ? rope_second_word : rope_first_word) : bias_meta[112:109];
    wire [127:0] p1_write_data=post_state==S_ROPE_WRITE ? (rope_write_half ? rope_second : rope_first) : bias_data;

    assign dma_cmd_valid=attn_dma_owner ? attn_dma_valid : command_hold_valid;
    assign dma_cmd=attn_dma_owner ? attn_dma_command : command_hold; assign dma_meta=attn_dma_owner ? attn_dma_metadata : metadata_hold;
    assign dma_abort=1'b0;
    assign afu_abort=1'b0;
    assign rd_ready=attn_dma_owner ? attn_dma_rd_ready : ((state_current[S_RD_DATA] || fetch_state==F_DATA) &&
        ((read_kind>=READ_SCALE && read_kind<=READ_GAMMA) ? params_load_ready :
        (read_kind==READ_WEIGHT ? dequant_ready : (read_kind==READ_INPUT ? output_gemm_write_ready : 1'b1))));
    assign wr_valid=attn_dma_owner ? attn_dma_wr_valid : (ffn_active ? ffn_write_valid : (post_state==S_WR_SEND || (state_current[S_H_SEND] && !hidden_dma_sent)));
    assign wr_data=attn_dma_owner ? attn_dma_wr_data : (ffn_active ? ffn_write_data : write_hold);
    assign wr_keep=attn_dma_owner ? attn_dma_wr_keep : 16'hffff;
    assign wr_last=attn_dma_owner ? attn_dma_wr_last : (ffn_active ? ffn_write_last : ((output_phase || down_phase) ? norm_word==111 : write_word==15));
    assign wr_meta=attn_dma_owner ? attn_dma_wr_meta : (ffn_active ? ffn_write_meta : write_metadata_reg);
    assign busy=!state_current[S_IDLE] && !state_current[S_DONE];
    assign qkv_done=state_current[S_DONE];assign param_ready=&params_loaded;
    assign result_commit=state_current[S_LM_COMMIT];
    assign best_token_id=lm_best_token[17:0];
    assign lm_write_valid=lm_phase && attn_p_write;
    assign lm_write_data=attn_p_write_data;
    assign lm_write_address=attn_p_address;
    assign lm_complete=lm_write_valid && attn_p_write_ready && attn_p_max_last;
    assign error=1'b0;
    // 配置和帧身份由软件保证；Controller不保存错误码或产生全局错误反馈。
    assign error_code=32'd0;
    assign afu_valid=attn_afu_owner ? attn_afu_valid : ffn_active ? ffn_afu_valid : (state_current[S_NORM_SEND] && !norm_frontend_midpoint_reg) || post_state==S_ROPE_SEND || ((state_current[S_RES_SEND] || state_current[S_Z_SEND]) && !norm_accept_pending) || (state_current[S_H_SEND] && !hidden_afu_sent && !down_phase);
    assign afu_command=attn_afu_owner ? attn_afu_command : ffn_active ? 4'd13 : (state_current[S_RES_SEND] ? 4'd14 : (post_state==S_ROPE_SEND ? 4'd12 : 4'd11));
    assign afu_command_4=attn_afu_owner ? {19'd0,attn_afu_phase} : ffn_active ? 21'd1 : state_current[S_RES_SEND] ? 21'd0 : {20'd0,norm_apply};assign afu_sum_command_buffer=0;
    assign afu_eps={task_cfg[95:80],16'd0};assign afu_latch=1'b0;assign afu_gelu_sign=1'b0;
    assign afu_x=attn_afu_owner ? attn_afu_x : ffn_active ? ffn_afu_x : (post_state==S_ROPE_SEND ? rope_x : norm_x);
    assign afu_y=attn_afu_owner ? attn_afu_y : ffn_active ? ffn_afu_y : (post_state==S_ROPE_SEND ? rope_y : norm_gamma);
    assign afu_meta=attn_afu_owner ? attn_afu_meta : (ffn_active ? ffn_afu_meta : afu_metadata);assign afu_last=attn_afu_owner ? attn_afu_last : ffn_active ? ffn_afu_last : post_state==S_ROPE_SEND || norm_word==111;
    assign afu_mask=attn_afu_owner ? attn_afu_mask : 8'hff;assign afu_length=attn_afu_owner ? attn_afu_length : ffn_active ? {54'd0,ffn_service_words,3'd0} : (post_state==S_ROPE_SEND ? 64'd64 : 64'd896);
    assign afu_result_ready=attn_afu_owner ? attn_afu_result_ready : ffn_active ? ffn_result_ready : (state_norm_active && !state_current[S_NORM_READ]) ||
        post_state==S_ROPE_WAIT || state_current[S_NORM_STATS] || ((output_phase || down_phase) && state_residual_to_norm);

    // 在AFU结果写回与O-buffer端口仲裁之间切开控制路径，负载随使能延后一拍。
    reg output_afu_write_pending;
    reg [1:0] output_afu_write_bank;
    reg [2:0] output_afu_write_row;
    reg [6:0] output_afu_write_word;
    reg [127:0] output_afu_write_data;
    always @(posedge clk)begin
        if(!rst_n)output_afu_write_pending<=1'b0;
        else output_afu_write_pending<=residual_result || normalized_result || ffn_store_fire;
        // 无效负载不参与写入；数据寄存器无需复位或额外使能MUX。
        output_afu_write_bank<=ffn_store_fire ? ffn_store_bank : (residual_result ? 2'd1 : 2'd0);
        output_afu_write_row<=ffn_store_fire ? ffn_store_row : norm_row_visible;
        output_afu_write_word<=ffn_store_fire ? ffn_store_word : norm_results[6:0];
        output_afu_write_data<=ffn_store_fire ? ffn_store_data : afu_result_data;
    end

    // 每个buffer独立接口，系统阶段Controller直接产生地址和使能。
    qwen25_tensor_buffers #(.DIRECT_WEIGHT_PORTS(DIRECT_WEIGHT_READ)) buffers_Inst (
        .clk(clk),.rst_n(rst_n),
        .act_wide_layout(down_phase),
        .act_vec_en(act_vec_write),
        .act_vec_wr(act_vec_write),
        .act_vec_row(act_vec_row),
        .act_vec_word(act_vec_word),
        .act_vec_data(act_vec_data),
        .act_vec_valid(),
        .act_vec_q(),
        .act_tile_en(state_current[S_GEMM_BEGIN] || context_scratch_write),
        .act_tile_wr(context_scratch_write),
        .act_tile_row(context_scratch_write ? attn_context_row : (lm_phase ? norm_row_visible : 3'd0)),
        .act_tile_rows(compute_rows),
        // Wo reads all sixteen words of the currently reloaded 128-dimensional Context tile.
        .act_tile_word(context_scratch_write ? 10'd112+{attn_context_head[0],attn_context_word} :
            (output_phase ? 10'd112+inner_k : {input_tile,inner_k})),
        .act_tile_data(attn_context_data),
        .act_tile_valid(act_tile_valid),
        .act_tile_q(act_tile_q),
        .weight_fill_done(attn_buffer_owner ? attn_weight_commit : fetch_commit),
        .weight_fill_bank(attn_buffer_owner ? attn_weight_bank : fetch_bank),
        .weight_release_bank(attn_buffer_owner ? attn_weight_bank : gemm_release_bank),
        .weight_release_en(attn_buffer_owner ? attn_weight_release : gemm_release_pending),
        .weight_full(weight_full),
        .weight_load_valid(attn_buffer_owner ? attn_weight_valid : dequant_valid),
        .weight_load_ready(weight_load_ready),
        .weight_load_bank(attn_buffer_owner ? attn_weight_bank : fetch_bank),
        .weight_load_word(attn_buffer_owner ? attn_weight_word : dequant_word[9:0]),
        .weight_load_data(attn_buffer_owner ? attn_weight_data : dequant_data),
        .weight_direct_valid(direct_dequant_valid),.weight_direct_ready(direct_dequant_ready),
        .weight_direct_bank(fetch_bank),.weight_direct_word(direct_load_word),.weight_direct_data(direct_dequant_data),
        .weight_gemm_rd_en(attn_buffer_owner ? attn_weight_read : weight_read_enable),
        .weight_gemm_bank(attn_buffer_owner ? attn_weight_bank : weight_bank),
        .weight_gemm_word(weight_read_word0),
        .weight_gemm_valid(weight_valid),
        .weight_gemm_q(weight_q),
        .weight_gemm_rd_en_pair(!attn_buffer_owner && weight_read_enable),
        .weight_gemm_bank_pair(weight_bank),
        .weight_gemm_word_pair(weight_read_word1),
        .weight_gemm_valid_pair(weight_valid_pair),
        .weight_gemm_q_pair(weight_q_pair),
        .scale_load_start(params_load_start && read_kind==READ_DESC),
        .scale_load_valid(params_load_valid && stream_checked_kind==READ_SCALE),
        .scale_load_ready(scale_load_ready),
        .scale_load_data(stream_checked_data),
        .scale_load_last(stream_checked_last),
        .scale_load_done(scale_load_done),
        .scale_loaded(params_loaded[0]),
        .scale_read_en(scale_read_en),
        .scale_read_addr(scale_read_element[18:4]),
        .scale_read_valid(scale_read_valid),
        .scale_read_data(scale_read_data),
        .bias_load_start(params_load_start && read_kind==READ_SCALE),
        .bias_load_valid(params_load_valid && stream_checked_kind==READ_BIAS),
        .bias_load_ready(bias_load_ready),
        .bias_load_data(stream_checked_data),
        .bias_load_last(stream_checked_last),
        .bias_load_done(bias_load_done),
        .bias_loaded(params_loaded[1]),
        .bias_read_en(bias_read_en),
        .bias_read_addr(bias_read_element[14:3]),
        .bias_read_valid(bias_read_valid),
        .bias_read_data(bias_read_data),
        .gamma_load_start(params_load_start && read_kind==READ_BIAS),
        .gamma_load_valid(params_load_valid && stream_checked_kind==READ_GAMMA),
        .gamma_load_ready(gamma_load_ready),
        .gamma_load_data(stream_checked_data),
        .gamma_load_last(stream_checked_last),
        .gamma_load_done(gamma_load_done),
        .gamma_loaded(params_loaded[2]),
        .gamma_read_en(gamma_read_en),
        .gamma_read_addr(gamma_read_element[15:3]),
        .gamma_read_valid(gamma_read_valid),
        .gamma_read_data(gamma_read_data),
        .proj0_fill_done(post_launch),
        .proj0_fill_bank(projection_bank),
        .proj0_release_bank(post_bank),
        .proj0_release_en(post_bias_complete),
        .proj0_full(proj_zero_full),
        .proj0_wr_en(raw_busy && !linear_phase),
        .proj0_wr_bank(projection_bank),
        .proj0_wr_row(raw_vector[3:1]),
        .proj0_wr_word({raw_number,1'b0}+raw_vector[0]),
        .proj0_wr_data(raw_vector_data),
        .proj0_rd_en(bias_read_request),
        .proj0_rd_bank(post_bank),
        .proj0_rd_row(bias_issued[6:4]),
        .proj0_rd_word(bias_issued[3:0]),
        .proj0_rd_valid(p0_valid),
        .proj0_rd_data(p0_q),
        .proj1_fill_done(post_bias_complete),
        .proj1_fill_bank(post_bank),
        .proj1_release_bank(post_bank),
        .proj1_release_en(post_finished),
        .proj1_full(proj_one_full),
        .proj1_wr_en(p1_write),
        .proj1_wr_bank(post_bank),
        .proj1_wr_row(post_state==S_ROPE_WRITE ? post_row : bias_meta[115:113]),
        .proj1_wr_word(p1_write_word),
        .proj1_wr_data(p1_write_data),
        .proj1_rd_en(p1_rope_read || p1_dma_read),
        .proj1_rd_bank(post_bank),
        .proj1_rd_row(p1_dma_read ? row_index : post_row),
        .proj1_rd_word(p1_dma_read ? write_word[3:0] : (post_state==S_ROPE_SECOND ? rope_second_word : rope_first_word)),
        .proj1_rd_valid(p1_valid),
        .proj1_rd_data(p1_q),
        .attention_fill_done(lm_complete || (attn_buffer_owner && attn_p_commit)),
        .attention_fill_bank(lm_phase ? 1'b0 : attn_p_bank),
        .attention_release_bank(!attn_buffer_owner ? 1'b0 : attn_p_bank),
        .attention_release_en(state_current[S_DESC_EXTRA_CHECK] || state_current[S_FINAL_BEGIN] || (attn_buffer_owner && attn_p_release)),
        .attention_full(attn_p_full),
        .attention_wr_en(lm_write_valid || (attn_buffer_owner && attn_p_write)),
        .attention_wr_bank(attn_p_write_bank),
        .attention_wr_address(lm_phase ? lm_write_address : attn_p_address),
        .attention_wr_data(lm_phase ? lm_write_data : attn_p_write_data),
        .attention_wr_ready(attn_p_write_ready),.attention_afu_wr_en(attn_p_afu_write),
        .attention_afu_wr_bank(attn_p_afu_write_bank),.attention_afu_wr_address(attn_p_afu_address),.attention_afu_wr_data(attn_p_afu_write_data),
        .attention_rd_en(attn_buffer_owner && attn_p_read),
        .attention_rd_bank(attn_p_bank),
        .attention_rd_word(attn_p_word),
        .attention_rd_valid(attn_p_valid),
        .attention_rd_tile(attn_p_data),
        .attention_afu_rd_en(attn_buffer_owner && attn_p_afu_read),.attention_afu_rd_bank(lm_phase ? 1'b0 : attn_p_afu_bank),
        .attention_afu_rd_row(lm_phase ? 3'd0 : attn_p_row),.attention_afu_rd_word(attn_p_afu_word),
        .attention_afu_rd_valid(attn_p_afu_valid),.attention_afu_rd_data(attn_p_afu_data),
        .attention_max_wr_en((attn_buffer_owner || lm_phase) && attn_p_write),.attention_max_wr_row(attn_p_max_row),
        .attention_max_wr_value(attn_p_maximum),.attention_max_wr_index(attn_p_max_index),
        .attention_max_wr_first(attn_p_max_first),.attention_max_wr_last(attn_p_max_last),.attention_max_wr_words(attn_p_max_words),
        .attention_row_maximum(attn_p_row_maximum),.attention_row_max_index(lm_best_token),.attention_row_complete(attn_p_row_complete),
        .output_pair_done(state_current[S_O_FINISH] || (state_current[S_FFN_NEXT] && operation_decode_reg[5] && output_tile==ffn_last_tile) || state_current[S_LAYER_DONE]),
        .output_fill_pair(ffn_phase ? ffn_pair : 1'd0),
        .output_release_en(state_current[S_O_RESTORE] ||
            (state_current[S_FFN_BEGIN] && ffn_block==0) || state_current[S_H_LOAD] || (ffn_phase && ffn_post_valid && ffn_done)),
        .output_release_pair(ffn_phase && ffn_post_valid && ffn_done ? ffn_service_pair :
            ((ffn_phase || state_current[S_FFN_BEGIN]) ? ffn_pair : 1'd0)),
        .output_pair_full(output_pair_full),
        .output_wr_en(output_ram_write || hidden_reload_write || input_reload_write),
        .output_wr_bank((input_reload_write || input_load_active) ? 2'd1 : hidden_reload_write ? 2'd0 : down_phase ? 2'd1 : ffn_phase ? {ffn_pair,operation_decode_reg[5]} : 2'd0),
        .output_wr_row((hidden_reload_write || input_reload_write) ? stream_checked_row : raw_vector[3:1]),
        .output_wr_word((hidden_reload_write || input_reload_write) ? stream_checked_word[6:0] : {output_tile,4'd0}+{raw_number,1'b0}+raw_vector[0]),
        .output_wr_data((hidden_reload_write || input_reload_write) ? stream_checked_data : raw_vector_data),
        .output_wr_ready(output_gemm_write_ready),
        .output_afu_wr_en(output_afu_write_pending),
        .output_afu_wr_bank(output_afu_write_bank),
        .output_afu_wr_row(output_afu_write_row),
        .output_afu_wr_word(output_afu_write_word),
        .output_afu_wr_data(output_afu_write_data),
        .output_rd_en(ffn_active ? ffn_read : output_ram_read),
        .output_rd_pair(ffn_active ? ffn_read_pair : 1'd0),
        .output_rd_row(ffn_active ? ffn_read_row : norm_row_visible),
        .output_rd_word(ffn_active ? ffn_read_word : norm_word),
        .output_rd_valid(output_read_valid),
        .output_rd_x(output_read_x),
        .output_rd_y(output_read_y)
    );



    genvar direct_lane;
    generate for(direct_lane=0;direct_lane<4;direct_lane=direct_lane+1)begin:gen_direct_decode
        wire lane_ok=1'b1;
        wire lane_accept=(DIRECT_WEIGHT_READ!=0) && fetch_state==F_DATA;
        wire decoder_ready;
        assign direct_rd_ready[direct_lane]=fetch_state==F_DATA && decoder_ready;
        assign direct_load_word[direct_lane*8+:8]=direct_dequant_word[direct_lane*16+2+:8];
        qwen25_weight_dequant convert_Inst(
            .i_clk(clk),.i_rstn(rst_n),.i_valid(direct_rd_valid[direct_lane] && lane_accept),.o_ready(decoder_ready),
            .i_data(direct_rd_data[direct_lane*128+:128]),.i_word(direct_rd_word[direct_lane*10+:10]),.i_scales(2048'd0),
            .o_valid(direct_dequant_valid[direct_lane]),.i_ready(direct_dequant_ready[direct_lane]),
            .o_data(direct_dequant_data[direct_lane*256+:256]),.o_word(direct_dequant_word[direct_lane*16+:16]),.o_busy(direct_dequant_busy[direct_lane]));
    end endgenerate
    qwen25_weight_dequant dequant_Inst(
        .i_clk(clk),.i_rstn(rst_n),.i_valid(DIRECT_WEIGHT_READ==0 && fetch_state==F_DATA && rd_valid && stream_ok),
        .o_ready(dequant_ready),.i_data(rd_data),.i_word(read_count[9:0]),.i_scales(scales),
        .o_valid(dequant_valid),.i_ready(weight_load_ready),.o_data(dequant_data),.o_word(dequant_word),.o_busy(dequant_busy)
    );
    qwen25_qkv_bias bias_Inst(
        .i_clk(clk),.i_rstn(rst_n),.i_valid(linear_phase ? raw_scale_valid : bias_input_valid),.o_ready(bias_ready),
        .i_data(linear_phase ? raw_input_data : bias_input_data[bias_input_read]),
        .i_scale(linear_phase ? raw_scale : bias_input_scale[bias_input_read]),
        .i_bias(linear_phase ? 128'd0 : bias_input_coeff[bias_input_read]),.i_meta(bias_input_metadata),
        .o_valid(bias_valid),.i_ready(linear_phase ? (raw_busy && (lm_phase ? lm_input_ready : output_gemm_write_ready)) : post_state==S_BIAS_READ),.o_data(bias_data),.o_meta(bias_meta),.o_busy(bias_busy)
    );
    assign m_gemm_cmd_valid=attn_gemm_owner ? attn_cmd_valid : gemm_cmd_valid;
    assign m_gemm_cmd=attn_gemm_owner ? attn_command : gemm_command;
    assign m_gemm_cmd_meta=attn_gemm_owner ? attn_identity : compute_identity;
    assign m_gemm_act_valid=attn_gemm_owner ? attn_act_valid : gemm_act_valid;
    genvar activation_group;
    generate for(activation_group=0;activation_group<4;activation_group=activation_group+1)begin:gen_activation_owner_group
        assign m_gemm_act_data[activation_group*256+:256]=(|attn_activation_owner_copies[activation_group]) ?
            attn_activation[activation_group*256+:256] : activation[activation_group*256+:256];
    end endgenerate
    assign m_gemm_act_meta=attn_gemm_owner ? attn_identity : compute_identity;
    assign m_gemm_weight_valid=attn_gemm_owner ? attn_gemm_weight_valid : gemm_weight_valid;
    genvar weight_owner_group;
    generate for(weight_owner_group=0;weight_owner_group<8;weight_owner_group=weight_owner_group+1)begin:gen_weight_owner_group
        assign m_gemm_weight_data[weight_owner_group*256+:256]=(|attn_weight_owner_copies[weight_owner_group]) ?
            attn_gemm_weight[weight_owner_group*256+:256] : weight_data[weight_owner_group*256+:256];
    end endgenerate
    assign m_gemm_weight_meta=attn_gemm_owner ? attn_identity : compute_identity;
    assign m_gemm_abort=1'b0;
    assign m_gemm_soft_reset=1'b0;
    assign m_gemm_ready=attn_gemm_owner ? attn_gemm_ready : gemm_output_ready;
    generate if(EXTERNAL_GEMM)begin:gen_external_gemm
        assign gemm_cmd_ready=s_gemm_cmd_ready;
        assign gemm_act_ready=s_gemm_act_ready;
        assign gemm_weight_ready=s_gemm_weight_ready;
        assign gemm_valid=s_gemm_valid;
        assign gemm_bf16=s_gemm_bf16;
        assign gemm_fp32=s_gemm_fp32;
        assign gemm_meta=s_gemm_meta;
        assign gemm_slice=s_gemm_slice;
        assign gemm_rows=s_gemm_row_mask;
        assign gemm_cols=s_gemm_col_mask;
        assign gemm_last=s_gemm_last;
        assign gemm_busy=s_gemm_busy;
        assign gemm_input_done=s_gemm_input_done;
        assign gemm_partial=s_gemm_partial_done;
        assign gemm_done=s_gemm_done;
        assign gemm_error=s_gemm_error;
        assign gemm_aborted=s_gemm_aborted;
        assign gemm_error_code=s_gemm_error_code;
    end else begin:gen_local_gemm
    qwen25_gemm gemm_Inst(
        .clk(clk),.rst_n(rst_n),.s_cmd_valid(m_gemm_cmd_valid),.s_cmd_ready(gemm_cmd_ready),.s_cmd(m_gemm_cmd),.s_cmd_meta(m_gemm_cmd_meta),
        .s_act_valid(m_gemm_act_valid),.s_act_ready(gemm_act_ready),.s_act_data(m_gemm_act_data),.s_act_meta(m_gemm_act_meta),
        .s_weight_valid(m_gemm_weight_valid),.s_weight_ready(gemm_weight_ready),.s_weight_data(m_gemm_weight_data),.s_weight_meta(m_gemm_weight_meta),
        .abort_req(1'b0),.soft_reset_req(1'b0),.m_valid(gemm_valid),.m_ready(m_gemm_ready),.m_bf16(gemm_bf16),.m_fp32(gemm_fp32),
        .m_meta(gemm_meta),.m_slice(gemm_slice),.m_row_mask(gemm_rows),.m_col_mask(gemm_cols),.m_last(gemm_last),
        .busy(gemm_busy),.input_done(gemm_input_done),.partial_done(gemm_partial),.done(gemm_done),.error(gemm_error),.aborted(gemm_aborted),.error_code(gemm_error_code)
    );
    end endgenerate


    qwen25_attention #(.ENABLE_LOGITS(1)) attention_Inst (
        .i_clk(clk),.i_rstn(rst_n),.i_start(attn_start),.i_abort(1'b0),
        .i_logits_mode(lm_phase),.i_logits_valid(lm_phase && raw_busy && bias_valid),
        .o_logits_ready(lm_input_ready),.i_logits_address(lm_input_address),.i_logits_data(bias_data),.o_logits_busy(lm_busy),
        .i_rows(rows),.i_position(task_cfg[52:37]),.i_context(attn_context),.i_layer(layer),.i_task(task_id),
        .i_first_head(4'd0),.i_last_head(4'd13),.i_q_base(task_cfg[511:448]),.i_k_base(task_cfg[383:320]),.i_v_base(task_cfg[447:384]),
        .o_busy(attn_busy),.o_done(attn_done),.o_error(attn_error),.o_error_code(attn_error_code),
        .o_service_abort(attn_service_abort),
        .o_read_valid(attn_read_valid),.i_read_ready(attn_read_ready),.o_read_address(attn_read_address),
        .o_read_base(attn_read_base),.o_read_bytes(attn_read_bytes),
        .i_data_valid(attn_data_valid),.o_data_ready(attn_data_ready),.i_data(attn_read_data),.i_memory_busy(attn_memory_busy),.i_memory_error(attn_memory_error),
        .o_context_valid(attn_context_valid),.i_context_ready(attn_context_ready),.o_context_data(attn_context_data),
        .o_context_row(attn_context_row),.o_context_head(attn_context_head),.o_context_word(attn_context_word),
        .o_weight_valid(attn_weight_valid),.i_weight_ready(weight_load_ready),.o_weight_word(attn_weight_word),.o_weight_data(attn_weight_data),
        .o_weight_commit(attn_weight_commit),.o_weight_release(attn_weight_release),.o_weight_bank(attn_weight_bank),.i_weight_full(weight_full[attn_weight_bank]),
        .o_weight_read(attn_weight_read),.o_weight_read_word(attn_weight_read_word),.i_weight_read_valid(weight_valid),.i_weight_read_data(weight_q),
        .o_p_write(attn_p_write),.o_p_address(attn_p_address),.o_p_data(attn_p_write_data),.o_p_read(attn_p_read),.o_p_word(attn_p_word),.i_p_valid(attn_p_valid),.i_p_data(attn_p_data),
        .i_p_write_ready(attn_p_write_ready),.o_p_afu_write(attn_p_afu_write),.o_p_afu_write_bank(attn_p_afu_write_bank),
        .o_p_afu_address(attn_p_afu_address),.o_p_afu_data(attn_p_afu_write_data),
        .o_p_write_bank(attn_p_write_bank),.o_p_bank(attn_p_bank),.o_p_afu_bank(attn_p_afu_bank),.o_p_commit(attn_p_commit),.o_p_release(attn_p_release),.i_p_full(attn_p_full),.o_p_afu_word(attn_p_afu_word),
        .o_p_afu_read(attn_p_afu_read),.o_p_row(attn_p_row),.i_p_afu_valid(attn_p_afu_valid),.i_p_afu_data(attn_p_afu_data),
        .o_p_max_row(attn_p_max_row),.o_p_maximum(attn_p_maximum),.o_p_max_index(attn_p_max_index),
        .o_p_max_first(attn_p_max_first),.o_p_max_last(attn_p_max_last),.o_p_max_words(attn_p_max_words),
        .i_p_row_maximum(attn_p_row_maximum),.i_p_row_complete(attn_p_row_complete),
        .o_gemm_cmd_valid(attn_cmd_valid),.i_gemm_cmd_ready(gemm_cmd_ready),.o_gemm_command(attn_command),.o_identity(attn_identity),
        .o_gemm_act_valid(attn_act_valid),.i_gemm_act_ready(gemm_act_ready),.o_gemm_activation(attn_activation),
        .o_gemm_weight_valid(attn_gemm_weight_valid),.i_gemm_weight_ready(gemm_weight_ready),.o_gemm_weight(attn_gemm_weight),
        .i_gemm_valid(gemm_valid),.o_gemm_ready(attn_gemm_ready),.i_gemm_data(gemm_bf16),.i_gemm_meta(gemm_meta),.i_gemm_slice(gemm_slice),.i_gemm_last(gemm_last),
        .i_gemm_done(gemm_done),.i_gemm_input_done(gemm_input_done),.i_gemm_partial(gemm_partial),.i_gemm_busy(gemm_busy),.i_gemm_error(gemm_error),
        .o_afu_valid(attn_afu_valid),.i_afu_ready(afu_ready),.o_afu_command(attn_afu_command),.o_afu_phase(attn_afu_phase),.o_afu_length(attn_afu_length),
        .o_afu_x(attn_afu_x),.o_afu_y(attn_afu_y),.o_afu_meta(attn_afu_meta),.o_afu_last(attn_afu_last),.o_afu_mask(attn_afu_mask),
        .i_afu_valid(afu_result_valid),.o_afu_ready(attn_afu_result_ready),.i_afu_data(afu_result_data),.i_afu_meta(afu_result_meta),.i_afu_last(afu_result_last),
        .i_afu_mask(afu_result_mask),.i_afu_busy(afu_busy),.i_afu_error(afu_error)
    );
    qwen25_attention_dma attention_dma_Inst (
        .i_clk(clk),.i_rstn(rst_n),.i_clear(1'b0),.i_abort(1'b0),
        .i_identity(attn_identity),.i_burst_limit(task_cfg[103:96]),.i_context_base(task_cfg[255:192]),
        .i_cache_invalidate(attn_cache_invalidate),.i_cache_context(attn_context),
        .i_cache_k_base(task_cfg[383:320]+layer*64'd8388608),
        .i_cache_v_base(task_cfg[447:384]+layer*64'd8388608),
        .i_read_valid(attn_read_valid),.o_read_ready(attn_read_ready),.i_read_address(attn_read_address),
        .i_read_base(attn_read_base),.i_read_bytes(attn_read_bytes),
        .o_data_valid(attn_data_valid),.i_data_ready(attn_data_ready),.o_data(attn_read_data),
        .i_context_valid(attn_context_valid),.o_context_ready(attn_context_ready),.i_context_data(attn_context_data),
        .i_context_row(attn_context_row),.i_context_head(attn_context_head),.i_context_word(attn_context_word),
        .o_busy(attn_memory_busy),.o_error(attn_memory_error),.o_cmd_valid(attn_dma_valid),.i_cmd_ready(dma_cmd_ready),.o_cmd(attn_dma_command),.o_meta(attn_dma_metadata),
        .i_rd_valid(rd_valid && attn_dma_owner),.o_rd_ready(attn_dma_rd_ready),.i_rd_data(rd_data),.i_rd_keep(rd_keep),.i_rd_last(rd_last),
        .i_rd_meta(rd_meta),.i_rd_tag(rd_tag),.i_rd_resource(rd_resource),.i_rd_beat_error(rd_beat_error),
        .i_rd_done(rd_done && attn_dma_owner),.i_rd_error(rd_error && attn_dma_owner),.i_rd_aborted(rd_aborted && attn_dma_owner),
        .o_wr_valid(attn_dma_wr_valid),.i_wr_ready(wr_ready),.o_wr_data(attn_dma_wr_data),.o_wr_meta(attn_dma_wr_meta),.o_wr_last(attn_dma_wr_last),.o_wr_keep(attn_dma_wr_keep),
        .i_wr_done(wr_done && attn_dma_owner),.i_wr_error(wr_error && attn_dma_owner),.i_wr_aborted(wr_aborted && attn_dma_owner)
    );

    qwen25_ffn_post ffn_post_Inst (
        .clk(clk),.rst_n(rst_n),.start(ffn_post_start),.cancel(1'b0),
        .rows(rows),.task_id(task_id),.layer(layer),.position(task_cfg[52:37]),.burst_limit(task_cfg[103:96]),
        .base(descriptor[7][63:0]+ffn_service_block*64'd1792),.words(ffn_service_words),.row_stride(complete_layer ? 16'd9728 : 16'd1792),
        .pair(ffn_service_pair),.up_tiles(ffn_up_tiles),.busy(ffn_busy),.done(ffn_done),
        .read_en(ffn_read),.read_pair(ffn_read_pair),.read_row(ffn_read_row),.read_word(ffn_read_word),
        .read_valid(output_read_valid),.read_x(output_read_x),.read_y(output_read_y),
        .store_valid(ffn_store_valid),.store_ready(1'b1),.store_bank(ffn_store_bank),.store_row(ffn_store_row),
        .store_word(ffn_store_word),.store_data(ffn_store_data),
        .afu_valid(ffn_afu_valid),.afu_ready(afu_ready),.afu_x(ffn_afu_x),.afu_y(ffn_afu_y),.afu_meta(ffn_afu_meta),.afu_last(ffn_afu_last),
        .result_valid(afu_result_valid && ffn_active),.result_ready(ffn_result_ready),.result_data(afu_result_data),
        .afu_busy(afu_busy),
        .command_valid(ffn_dma_valid),.command_ready(write_command_fire && ffn_active),.command(ffn_dma_command),.metadata(ffn_dma_metadata),
        .write_valid(ffn_write_valid),.write_ready(wr_ready),.write_data(ffn_write_data),.write_meta(ffn_write_meta),.write_last(ffn_write_last),
        .write_done(wr_done && ffn_active),.write_busy(wr_busy)
    );

    // 元数据仅用于DMA/算子身份绑定，RAM无需每拍metadata比较。
    always @(*) begin
        gemm_command=0;gemm_command[3:0]=compute_opcode;
        gemm_command[15:4]=global_k;gemm_command[16]=global_k==0;gemm_command[17]=global_k==last_k;
        gemm_command[21:18]=compute_rows;gemm_command[29:22]=128;gemm_command[33:30]=8;
        afu_metadata=0;afu_metadata[15:0]=task_id;afu_metadata[20:16]=layer;
        afu_metadata[56:53]=rows;afu_metadata[60:57]=8;
        if(post_state==S_ROPE_SEND) begin
            afu_metadata[52:37]={12'd0,rope_first_word};afu_metadata[83:80]=12;
            afu_metadata[104:89]=task_cfg[52:37]+post_row;
            afu_metadata[118:109]={3'd0,post_output_tile,rope_first_word};
        end else begin
            afu_metadata[52:37]={9'd0,norm_word};afu_metadata[83:80]=(output_phase || down_phase) && state_before_hidden ? 14 : 11;
            afu_metadata[104:89]=task_cfg[52:37]+norm_row_visible;
        end
        bias_input_metadata=post_identity;
        bias_input_metadata[115:109]=bias_input_address[bias_input_read];
        selected_bias=0;
        case(bias_read_address[3:0])
            4'd0:begin selected_bias=post_biases[0+:128];end
            4'd1:begin selected_bias=post_biases[128+:128];end
            4'd2:begin selected_bias=post_biases[256+:128];end
            4'd3:begin selected_bias=post_biases[384+:128];end
            4'd4:begin selected_bias=post_biases[512+:128];end
            4'd5:begin selected_bias=post_biases[640+:128];end
            4'd6:begin selected_bias=post_biases[768+:128];end
            4'd7:begin selected_bias=post_biases[896+:128];end
            4'd8:begin selected_bias=post_biases[1024+:128];end
            4'd9:begin selected_bias=post_biases[1152+:128];end
            4'd10:begin selected_bias=post_biases[1280+:128];end
            4'd11:begin selected_bias=post_biases[1408+:128];end
            4'd12:begin selected_bias=post_biases[1536+:128];end
            4'd13:begin selected_bias=post_biases[1664+:128];end
            4'd14:begin selected_bias=post_biases[1792+:128];end
            4'd15:begin selected_bias=post_biases[1920+:128];end
        endcase
        raw_input_data=0;
        case(raw_select)
            4'd0:begin raw_input_data=raw_slice[0+:128];end
            4'd1:begin raw_input_data=raw_slice[128+:128];end
            4'd2:begin raw_input_data=raw_slice[256+:128];end
            4'd3:begin raw_input_data=raw_slice[384+:128];end
            4'd4:begin raw_input_data=raw_slice[512+:128];end
            4'd5:begin raw_input_data=raw_slice[640+:128];end
            4'd6:begin raw_input_data=raw_slice[768+:128];end
            4'd7:begin raw_input_data=raw_slice[896+:128];end
            4'd8:begin raw_input_data=raw_slice[1024+:128];end
            4'd9:begin raw_input_data=raw_slice[1152+:128];end
            4'd10:begin raw_input_data=raw_slice[1280+:128];end
            4'd11:begin raw_input_data=raw_slice[1408+:128];end
            4'd12:begin raw_input_data=raw_slice[1536+:128];end
            4'd13:begin raw_input_data=raw_slice[1664+:128];end
            4'd14:begin raw_input_data=raw_slice[1792+:128];end
            4'd15:begin raw_input_data=raw_slice[1920+:128];end
        endcase
    end


    // 各阶段只等待真实DMA/AFU/GEMM valid、done和B响应。
    always @(*) begin
        state_next=state_current;
        (* parallel_case *) case(1'b1)
            state_current[S_IDLE]:begin if(start)begin state_next=HOT_S_DESC;end end
            state_current[S_DESC]:begin state_next=HOT_S_RD_CMD;end
            state_current[S_RD_CMD]:begin if(read_command_fire)begin state_next=HOT_S_RD_DATA;end end
            state_current[S_RD_DATA]:begin if(rd_done)begin state_next=HOT_S_RD_FINISH;end end
            state_current[S_RD_FINISH]:begin
                if(read_finish_pending)begin
                    case(read_kind)
                        READ_DESC:state_next=HOT_S_DESC_CHECK;
                        READ_SCALE:state_next=HOT_S_PARAM;
                        READ_BIAS:state_next=HOT_S_PARAM;
                        READ_GAMMA:state_next=HOT_S_INPUT;
                        READ_INPUT:state_next=output_phase ? HOT_S_O_CONTEXT : HOT_S_NORM_READ;
                        READ_EXTRA:state_next=HOT_S_DESC_EXTRA_CHECK;
                        READ_M:state_next=HOT_S_H_LOAD;
                        READ_H:state_next=HOT_S_DOWN_BEGIN;
                        READ_CONTEXT:state_next=row_index==rows_last ? HOT_S_WEIGHT : HOT_S_O_CONTEXT;
                        READ_WEIGHT:state_next=HOT_S_GEMM_BEGIN;
                        default:state_next=HOT_S_IDLE;
                    endcase
                end
            end
            state_current[S_DESC_CHECK]:begin
                state_next=complete_layer ? HOT_S_DESC_EXTRA : (param_ready ? HOT_S_INPUT : HOT_S_PARAM);
            end
            state_current[S_DESC_EXTRA]:state_next=HOT_S_RD_CMD;
            state_current[S_DESC_EXTRA_CHECK]:begin
                state_next=attention_sweep ? HOT_S_ATTN_BEGIN : (param_ready ? HOT_S_INPUT : HOT_S_PARAM);
            end
            state_current[S_PARAM]:begin state_next=HOT_S_RD_CMD;end
            state_current[S_INPUT]:begin state_next=next_layer_input ? HOT_S_NORM_READ : HOT_S_RD_CMD;end
            state_current[S_NORM_READ]:begin state_next=HOT_S_NORM_WAIT;end
            state_current[S_NORM_WAIT]:begin
                if(norm_input_ready && gamma_response_ready && !norm_frontend_midpoint_reg)begin state_next=HOT_S_NORM_SEND;end
            end
            state_current[S_NORM_SEND]:begin if(norm_frontend_midpoint_reg)begin state_next=norm_word==111 ? (norm_apply ? HOT_S_NORM_FINISH : HOT_S_NORM_STATS) : HOT_S_NORM_READ;end end
            state_current[S_NORM_STATS]:begin if(afu_result_valid)begin state_next=HOT_S_NORM_READ;end end
             state_current[S_NORM_FINISH]:begin
                 if(norm_results==112)begin state_next=lm_phase ? HOT_S_LM_BEGIN : (norm_row_visible==rows_last ? HOT_S_COEFF_READ : HOT_S_NORM_READ);end
            end
            state_current[S_COEFF_READ]:begin state_next=HOT_S_COEFF_WAIT;end
            state_current[S_COEFF_WAIT]:begin
                if(bias_read_valid)begin state_next=coeff_word==15 ? HOT_S_WEIGHT : HOT_S_COEFF_READ;end
            end
            state_current[S_WEIGHT]:begin if(weight_chunk_ready && ffn_consumer_available && !proj_zero_full[compute_bank] && !proj_one_full[compute_bank])begin state_next=HOT_S_GEMM_BEGIN;end end
            state_current[S_GEMM_BEGIN]:begin state_next=HOT_S_GEMM_FEED;end
            state_current[S_GEMM_FEED]:begin
                if(gemm_feed_advance)begin
                    if(global_k==last_k)begin
                        if(output_phase)state_next=HOT_S_O_NEXT;else if(ffn_phase)state_next=HOT_S_FFN_NEXT;
                        else if(down_phase)state_next=HOT_S_DOWN_NEXT;
                        else if(lm_phase)state_next=HOT_S_LM_NEXT;
                        else state_next=HOT_S_NEXT;
                    end else begin state_next=inner_k==15 && output_phase ? HOT_S_O_CONTEXT : HOT_S_WEIGHT;end
                end
            end
            state_current[S_NEXT]:begin state_next=operation_decode_reg[2] && output_tile==6 ? HOT_S_POST_DRAIN : HOT_S_COEFF_READ;end
            state_current[S_POST_DRAIN]:begin
                if(post_state==S_IDLE && pipeline_quiet)begin
                    state_next=complete_layer && total_rows>8 && !attention_sweep ? HOT_S_BATCH_NEXT :
                        (attn_enabled ? HOT_S_ATTN_BEGIN : HOT_S_DONE);
                end
            end
            state_current[S_ATTN_BEGIN]:begin state_next=HOT_S_ATTN_RUN;end
            state_current[S_ATTN_RUN]:begin
                // The last Context acceptance is earlier than its DDR write response.
                if(attn_done && !attn_memory_busy && !wr_busy)state_next=output_enabled ? HOT_S_O_RESTORE : HOT_S_DONE;
            end
            state_current[S_O_RESTORE],state_current[S_O_CONTEXT]:state_next=HOT_S_RD_CMD;
            state_current[S_O_NEXT]:state_next=output_tile==6 ? HOT_S_RES_READ : HOT_S_O_CONTEXT;
            state_current[S_RES_READ]:state_next=HOT_S_RES_WAIT;
            state_current[S_RES_WAIT]:if(residual_x_ready && residual_o_ready)state_next=HOT_S_RES_SEND;
            state_current[S_RES_SEND]:if(norm_accept_pending)state_next=norm_word==111 ? HOT_S_RES_FINISH : HOT_S_RES_READ;
             state_current[S_RES_FINISH]:if(norm_results==112 && !afu_busy)state_next=norm_row_visible==rows_last ? HOT_S_H_START : HOT_S_RES_READ;
            state_current[S_H_START]:state_next=HOT_S_H_OFFSET;
            state_current[S_H_OFFSET]:state_next=HOT_S_H_ADDRESS;
            state_current[S_H_ADDRESS]:state_next=HOT_S_H_COMMIT;
            state_current[S_H_COMMIT]:state_next=HOT_S_H_COMMAND;
            state_current[S_H_COMMAND]:if(write_command_fire)state_next=HOT_S_H_READ;
            state_current[S_H_READ]:state_next=HOT_S_H_WAIT;
            state_current[S_H_WAIT]:if(output_read_valid)state_next=HOT_S_H_SEND;
            state_current[S_H_SEND]:if(hidden_send_advance)state_next=norm_word==111 ? HOT_S_H_FINISH : HOT_S_H_READ;
             state_current[S_H_FINISH]:if(hidden_b_done && (down_phase || hidden_stats_done) && !wr_busy && !afu_busy)state_next=down_phase ? (norm_row_visible==rows_last ? HOT_S_LAYER_DONE : HOT_S_H_START) : HOT_S_Z_READ;
            state_current[S_Z_READ]:state_next=HOT_S_Z_WAIT;
            state_current[S_Z_WAIT]:if(residual_o_ready && gamma_response_ready)state_next=HOT_S_Z_SEND;
            state_current[S_Z_SEND]:if(norm_accept_pending)state_next=norm_word==111 ? HOT_S_Z_FINISH : HOT_S_Z_READ;
            state_current[S_Z_FINISH]:if(norm_results==112 && !afu_busy)state_next=HOT_S_Z_COPY_READ;
            state_current[S_Z_COPY_READ]:state_next=HOT_S_Z_COPY_WAIT;
            state_current[S_Z_COPY_WAIT]:if(output_read_valid)state_next=HOT_S_Z_COPY_WRITE;
             state_current[S_Z_COPY_WRITE]:if(norm_word==111)state_next=norm_row_visible==rows_last ? HOT_S_O_FINISH : HOT_S_H_START;else state_next=HOT_S_Z_COPY_READ;
            state_current[S_O_FINISH]:if(pipeline_quiet)state_next=ffn_enabled ? HOT_S_FFN_BEGIN : HOT_S_DONE;
            state_current[S_FFN_BEGIN]:state_next=HOT_S_WEIGHT;
            state_current[S_FFN_NEXT]:state_next=operation_decode_reg[5] && output_tile==ffn_last_tile ? HOT_S_FFN_WAIT : HOT_S_WEIGHT;
            state_current[S_FFN_WAIT]:begin
                if(complete_layer && ffn_block!=5)begin
                    if(!gemm_busy && !raw_busy && !rd_busy && !all_dequant_busy)state_next=HOT_S_FFN_BEGIN;
                end else if(ffn_done && pipeline_quiet)state_next=complete_layer ? HOT_S_M_LOAD : HOT_S_DONE;
            end
            state_current[S_M_LOAD],state_current[S_H_LOAD]:if(reload_offset_valid)state_next=HOT_S_RD_CMD;
            state_current[S_DOWN_BEGIN]:state_next=HOT_S_WEIGHT;
            state_current[S_DOWN_NEXT]:state_next=output_tile==6 ? HOT_S_RES_READ : HOT_S_WEIGHT;
            state_current[S_LAYER_DONE]:begin
                if(!rows_group_final)state_next=HOT_S_BATCH_NEXT;
                else if(descriptor[8][127:64]!=0)begin if(layer_address_valid[1])state_next=HOT_S_DESC;end
                else state_next=final_enabled && layer==23 ? HOT_S_FINAL_BEGIN : HOT_S_DONE;
            end
            state_current[S_BATCH_NEXT]:if(batch_restore_first || batch_address_valid[1])state_next=HOT_S_DESC;
            state_current[S_FINAL_BEGIN]:state_next=HOT_S_NORM_READ;
            state_current[S_LM_BEGIN]:state_next=HOT_S_WEIGHT;
            state_current[S_LM_NEXT]:state_next=output_tile==1186 ? HOT_S_LM_DRAIN : HOT_S_WEIGHT;
            state_current[S_LM_DRAIN]:if(lm_finished && !lm_busy && pipeline_quiet && fetch_state==F_COMPLETE)state_next=HOT_S_LM_COMMIT;
            state_current[S_LM_COMMIT]:state_next=HOT_S_DONE;
            state_current[S_DONE]:begin if(start)begin state_next=HOT_S_DESC;end end
            default:begin state_next=HOT_S_IDLE;end
        endcase
    end
    always @(posedge clk)begin
        if(!rst_n)begin state_current<=HOT_S_IDLE;end else begin state_current<=state_next;end
    end
    // Release one edge after consumption; inner_k may already point to the other bank.
    always @(posedge clk)begin
        if(!rst_n)begin
            gemm_release_pending<=1'b0;
            gemm_release_bank<=1'b0;
        end else begin
            gemm_release_pending<=gemm_release_request;
            if(gemm_release_request)gemm_release_bank<=weight_bank;
        end
    end
    integer r,l;
    always @(posedge clk)begin
        if(!rst_n || state_current[S_IDLE])begin
            weight_chunks_ready[0]<=0;weight_chunks_ready[1]<=0;
        end else begin
            if(gemm_release_pending)
                weight_chunks_ready[gemm_release_bank]<=0;
            if(fetch_begin)weight_chunks_ready[fetch_bank]<=0;
            if(dequant_valid && weight_load_ready && dequant_word[5:0]==63)
                weight_chunks_ready[fetch_bank]<={1'b0,dequant_word[9:6]}+5'd1;
        end
    end
    integer chunk_lane;
    always @(posedge clk)begin
        for(chunk_lane=0;chunk_lane<4;chunk_lane=chunk_lane+1)begin
            if(!rst_n || state_current[S_IDLE] || state_current[S_O_RESTORE] || state_current[S_FFN_BEGIN] || state_current[S_DOWN_BEGIN] || state_current[S_LM_BEGIN])begin
                weight_chunk_mask[0][chunk_lane]<=0;weight_chunk_mask[1][chunk_lane]<=0;
            end else begin
                if(attn_active && attn_weight_release)weight_chunk_mask[attn_weight_bank][chunk_lane]<=0;
                if(!attn_active && gemm_release_pending)
                    weight_chunk_mask[gemm_release_bank][chunk_lane]<=0;
                if(fetch_begin)weight_chunk_mask[fetch_bank][chunk_lane]<=0;
                if(direct_write_fire[chunk_lane] && direct_dequant_word[chunk_lane*16+2+:4]==15)
                    weight_chunk_mask[fetch_bank][chunk_lane][direct_dequant_word[chunk_lane*16+6+:4]]<=1;
            end
        end
        // Scale writes belong to their independently addressed slice above.
        if(state_current[S_GEMM_BEGIN] && global_k==0)compute_scales<=selected_scale_bank;
        if(post_launch)post_scales<=compute_scales;
    end
    always @(posedge clk) begin
        if(!rst_n)
            operation_decode_reg<=8'b00000001;
        else
            operation_decode_reg<=8'b00000001 << operation_path_next;
    end

    // Invalid count storage represents zero. The first accepted word uses
    // address zero and writes count one, without a reset on the eight data FFs.
    always @(posedge clk)begin
        if(act_norm_result || residual_result || normalized_result)
            norm_results_count<=norm_results+8'd1;
    end
    // Controller中所有地址计数依据真实传输接受事件推进。
    always @(posedge clk)begin
        if(!rst_n)begin
            task_cfg<=0;layer<=0;rows<=0;rows_last_reg<=4'hf;activation_row_mask<=0;task_id<=0;operation<=0;output_tile<=0;input_tile<=0;inner_k<=0;
            complete_layer_reg<=0;final_enabled_reg<=0;
            row_index<=0;norm_apply<=0;coeff_kind<=0;coeff_word<=0;
            scale_base<=0;bias_base<=0;gamma_base<=0;q_base<=0;k_base<=0;v_base<=0;loaded_model_base<=0;
            command_reg<=0;metadata_reg<=0;write_command_reg<=0;write_metadata_reg<=0;read_kind<=0;read_count<=0;read_last_index<=0;read_terminal<=0;
            stream_checked_valid<=0;stream_checked_data<=0;stream_checked_last<=0;stream_checked_index<=0;stream_checked_kind<=0;
            stream_checked_row<=0;stream_checked_word<=0;
            weight_written<=0;input_row<=0;input_word<=0;cmd_sent<=0;act_sent<=0;act_available<=0;
            weight_reads<=0;weight_pending<=0;weight_sent<=0;gemm_terminal<=0;raw_busy<=0;raw_number<=0;raw_vector<=0;raw_issued<=0;
            post_word<=0;post_row<=0;rope_pair<=0;rope_write_half<=0;write_word<=0;compute_identity<=0;afu_identity<=0;
            residual_x_ready<=0;residual_o_ready<=0;gamma_response_ready<=0;norm_input_ready<=0;
            hidden_dma_sent<=0;hidden_afu_sent<=0;hidden_b_done<=0;hidden_stats_done<=0;ffn_up_tiles<=0;
            ffn_post_valid<=0;ffn_post_block<=0;
            descriptor_base<=0;first_descriptor<=0;first_input<=0;first_context<=0;first_q<=0;
            remaining_rows<=0;total_rows<=0;group_offset<=0;first_position<=0;
            next_layer_input<=0;layer_transition<=0;attention_sweep<=0;
        end else begin
            if(state_current[S_INPUT])begin ffn_post_valid<=0;ffn_up_tiles<=0;end
            if(ffn_post_start)begin ffn_post_valid<=1;ffn_post_block<=ffn_block;ffn_up_tiles<=0;end
            if((state_current[S_IDLE] || state_current[S_DONE]) && start)begin
                task_cfg<=cfg;rows<=cfg[20:5]>8 ? 4'd8 : cfg[8:5];rows_last_reg<=(cfg[20:5]>8 ? 4'd8 : cfg[8:5])-4'd1;task_id<=task_id+16'd1;
                descriptor_base<=cfg[319:256];first_descriptor<=cfg[319:256];first_input<=cfg[191:128];
                first_context<=cfg[255:192];first_q<=cfg[511:448];first_position<=cfg[52:37];
                remaining_rows<=cfg[20:5];total_rows<=cfg[20:5];group_offset<=0;
                next_layer_input<=0;layer_transition<=0;attention_sweep<=0;
                operation<=0;output_tile<=0;input_tile<=0;inner_k<=0;norm_apply<=0;
                coeff_kind<=0;coeff_word<=0;post_row<=0;post_word<=0;row_index<=0;rope_pair<=0;
            end
            if(state_current[S_DESC] || state_current[S_DESC_EXTRA] || (reload_offset_request && reload_offset_valid) || state_current[S_PARAM] || (state_current[S_INPUT] && !next_layer_input) || state_current[S_O_RESTORE] || state_current[S_O_CONTEXT] || fetch_begin)begin
                command_reg<=0;metadata_reg<=0;read_count<=0;read_terminal<=0;
                command_reg[95:88]<=task_cfg[103:96];command_reg[111:96]<=task_id;
                metadata_reg[15:0]<=task_id;metadata_reg[20:16]<=layer;metadata_reg[56:53]<=rows;metadata_reg[60:57]<=8;
                if(state_current[S_DESC])begin
                    command_reg[63:0]<=descriptor_base;command_reg[87:64]<=128;command_reg[115:112]<=7;read_kind<=READ_DESC;read_last_index<=7;
                    metadata_reg[56:53]<=1;
                end else if(state_current[S_DESC_EXTRA])begin
                    command_reg[63:0]<=descriptor_base+64'd128;command_reg[87:64]<=final_enabled ? 32 : 16;command_reg[115:112]<=7;
                    read_kind<=READ_EXTRA;read_last_index<=final_enabled ? 1 : 0;metadata_reg[56:53]<=1;
                end else if(state_current[S_M_LOAD] || state_current[S_H_LOAD])begin
                    command_reg[63:0]<=(reload_offset_is_m ? descriptor[7][63:0] : descriptor[5][127:64])+{34'd0,reload_offset_product[29:0]};
                    command_reg[87:64]<=state_current[S_M_LOAD] ? rows*24'd9728 : rows*24'd1792;
                    command_reg[115:112]<=state_current[S_M_LOAD] ? 0 : 3;
                    read_kind<=state_current[S_M_LOAD] ? READ_M : READ_H;
                    read_last_index<=(state_current[S_M_LOAD] ? rows*17'd608 : rows*17'd112)-17'd1;
                    input_row<=0;input_word<=0;
                end else if(state_current[S_PARAM])begin
                    command_reg[115:112]<=7;metadata_reg[56:53]<=1;
                    if(read_kind==READ_DESC)begin
                        command_reg[63:0]<=scale_base;command_reg[87:64]<=912128;read_kind<=READ_SCALE;read_last_index<=57007;
                    end else if(read_kind==READ_SCALE)begin
                        command_reg[63:0]<=bias_base;command_reg[87:64]<=55296;read_kind<=READ_BIAS;read_last_index<=3455;
                    end else begin
                        command_reg[63:0]<=gamma_base;command_reg[87:64]<=87808;read_kind<=READ_GAMMA;read_last_index<=5487;
                    end
                end else if(state_current[S_INPUT] || state_current[S_O_RESTORE])begin
                    command_reg[63:0]<=task_cfg[191:128];command_reg[87:64]<=rows*24'd1792;command_reg[115:112]<=0;
                    read_kind<=READ_INPUT;read_last_index<=rows*17'd112-17'd1;input_row<=0;input_word<=0;
                end else if(state_current[S_O_CONTEXT])begin
                    command_reg[63:0]<=task_cfg[255:192]+row_index*64'd1792+input_tile*64'd256;
                    command_reg[87:64]<=256;command_reg[115:112]<=5;
                    read_kind<=READ_CONTEXT;read_last_index<=15;input_word<=0;
                end else begin
                    command_reg[63:0]<={fetch_column_base[63:14]+{44'd0,fetch_tile},fetch_column_base[13:0]}+
                        (fetch_half ? 64'd2048 : 64'd0);
                    command_reg[87:64]<=8192;command_reg[115:112]<=1;
                    command_reg[117:116]<={1'b0,fetch_bank};metadata_reg[78:77]<={1'b0,fetch_bank};read_kind<=READ_WEIGHT;read_last_index<=511;weight_written<=0;
                end
            end
            stream_checked_valid<=stream_fire && (state_current[S_RD_DATA] || fetch_state==F_DATA);
            if(stream_fire && (state_current[S_RD_DATA] || fetch_state==F_DATA))begin
                stream_checked_data<=rd_data;
                // The final-beat marker owns the same buffered payload as its data.
                stream_checked_last<=rd_last;
                stream_checked_index<=read_count;
                // The buffered payload and its pre-increment tensor cursor
                // must reach the RAM write port on the same accepted beat.
                stream_checked_row<=input_row;
                stream_checked_word<=input_word;
                stream_checked_kind<=read_kind;
                read_count<=read_count+17'd1;
                if(read_kind==READ_CONTEXT)begin input_word<=input_word+1'b1;end
                if(read_kind==READ_INPUT || read_kind==READ_M || read_kind==READ_H)begin
                    if(input_word==(read_kind==READ_M ? 607 : 111))begin input_word<=0;input_row<=input_row+3'd1;end else begin input_word<=input_word+1'b1;end
                end
            end
            if(stream_checked_fire && state_current[S_RD_DATA])begin
                if(stream_checked_kind==READ_DESC)begin
                    descriptor[stream_checked_index[2:0]]<=stream_checked_data;
                    if(stream_checked_index[2:0]==0)begin
                        complete_layer_reg<=stream_checked_data[63:32]==32'd2 || stream_checked_data[63:32]==32'd3;
                        final_enabled_reg<=stream_checked_data[63:32]==32'd3;
                    end
                end
                if(stream_checked_kind==READ_EXTRA)descriptor[8+stream_checked_index[0]]<=stream_checked_data;
            end
            if(rd_done)read_terminal<=1;
            if(dequant_valid && weight_load_ready)begin weight_written<=weight_written+16'd1;end
            if(DIRECT_WEIGHT_READ!=0)begin
                if(direct_write_count!=0)weight_written<=weight_written+direct_write_count;
            end
            if(state_current[S_DESC_CHECK])begin
                layer<=descriptor[0][100:96];scale_base<=descriptor[1][63:0];bias_base<=descriptor[1][127:64];
                gamma_base<=descriptor[2][63:0];q_base<=descriptor[2][127:64];k_base<=descriptor[3][63:0];v_base<=descriptor[3][127:64];
                if(!param_ready)begin loaded_model_base<=task_cfg[319:256];end
                layer_transition<=0;
            end
            if(state_current[S_DESC_EXTRA_CHECK])begin
                if(!param_ready)read_kind<=READ_DESC;
            end
            if(state_current[S_M_LOAD])operation<=6;
            if(state_current[S_DOWN_BEGIN])begin
                operation<=6;output_tile<=0;input_tile<=0;inner_k<=0;
                norm_apply<=0;
            end
            if(state_current[S_DOWN_NEXT])begin
                output_tile<=output_tile+1'b1;input_tile<=0;inner_k<=0;
                
            end
            if(layer_reconfigure_fire)begin
                descriptor_base<=descriptor[8][127:64];first_descriptor<=descriptor[8][127:64];
                first_input<=descriptor[8][63:0]+{16'd0,layer_address_product};
                task_cfg[191:128]<=descriptor[8][63:0]+{16'd0,layer_address_product};
                task_cfg[255:192]<=first_context;task_cfg[511:448]<=first_q;
                task_cfg[52:37]<=first_position;
                remaining_rows<=total_rows;group_offset<=0;
                rows<=total_rows>8 ? 4'd8 : total_rows[3:0];rows_last_reg<=(total_rows>8 ? 4'd8 : total_rows[3:0])-4'd1;
                next_layer_input<=total_rows<=8;layer_transition<=1;attention_sweep<=0;
                operation<=0;output_tile<=0;input_tile<=0;inner_k<=0;norm_apply<=0;coeff_word<=0;row_index<=0;
            end
            if(batch_reconfigure_fire)begin
                if(batch_restore_first)begin
                    // All Q/K/V groups are durable: restart at group zero for
                    // Attention, then consume each group before changing layer.
                    attention_sweep<=1;
                    remaining_rows<=total_rows;group_offset<=0;
                    rows<=total_rows>8 ? 4'd8 : total_rows[3:0];rows_last_reg<=(total_rows>8 ? 4'd8 : total_rows[3:0])-4'd1;
                    task_cfg[52:37]<=first_position;
                    task_cfg[191:128]<=first_input;
                    task_cfg[255:192]<=first_context;
                    task_cfg[511:448]<=first_q;
                end else begin
                    remaining_rows<=remaining_rows-rows;group_offset<=group_offset+rows;
                    rows<=remaining_rows-rows>8 ? 4'd8 : remaining_rows-rows;rows_last_reg<=(remaining_rows-rows>8 ? 4'd8 : remaining_rows-rows)-4'd1;
                    task_cfg[52:37]<=first_position+group_offset+rows;
                    task_cfg[191:128]<=first_input+{16'd0,batch_address_product};
                    task_cfg[255:192]<=first_context+{16'd0,batch_address_product};
                    task_cfg[511:448]<=first_q+{16'd0,batch_address_product};
                end
                descriptor_base<=first_descriptor;next_layer_input<=0;task_id<=task_id+1'b1;
                operation<=0;output_tile<=0;input_tile<=0;inner_k<=0;norm_apply<=0;coeff_word<=0;row_index<=0;
            end
            if(state_current[S_RD_FINISH] && output_phase && read_finish_pending)begin
                if(read_kind==READ_INPUT)row_index<=0;
                if(read_kind==READ_CONTEXT)row_index<=row_index==rows_last ? 0 : row_index+1'b1;
            end
            if(state_current[S_NORM_READ])begin norm_input_ready<=0;gamma_response_ready<=0;end
            if(state_current[S_NORM_WAIT] && output_read_valid)begin norm_x<=output_read_y;norm_input_ready<=1;end
            if(state_current[S_NORM_WAIT] && gamma_read_valid)begin norm_gamma<=gamma_read_data;gamma_response_ready<=1;end
            if(afu_fire)begin afu_identity<=afu_metadata;end
            if(state_current[S_NORM_STATS] && afu_result_valid && afu_result_ready)begin
                norm_apply<=1;
            end
            if(act_norm_result)begin
                
            end
            if(state_current[S_NORM_FINISH] && norm_results==112)begin
                norm_apply<=0;
            end
            if(state_current[S_COEFF_WAIT] && bias_read_valid)begin
                biases[coeff_word*128+:128]<=bias_read_data;
                if(coeff_word==15)begin coeff_word<=0;end else begin coeff_word<=coeff_word+4'd1;end
            end
            if(state_current[S_GEMM_BEGIN])begin
                case(compute_rows)
                    4'd0:activation_row_mask<=8'b00000000;
                    4'd1:activation_row_mask<=8'b00000001;
                    4'd2:activation_row_mask<=8'b00000011;
                    4'd3:activation_row_mask<=8'b00000111;
                    4'd4:activation_row_mask<=8'b00001111;
                    4'd5:activation_row_mask<=8'b00011111;
                    4'd6:activation_row_mask<=8'b00111111;
                    4'd7:activation_row_mask<=8'b01111111;
                    default:activation_row_mask<=8'b11111111;
                endcase
                cmd_sent<=0;act_sent<=0;act_available<=0;weight_reads<=0;weight_pending<=0;weight_sent<=0;gemm_terminal<=0;
                compute_identity<=0;compute_identity[15:0]<=task_id;compute_identity[20:16]<=layer;compute_identity[52:37]<=output_tile;
                compute_identity[56:53]<=compute_rows;compute_identity[60:57]<=8;compute_identity[83:80]<=compute_opcode;
            end
            if(act_tile_valid)begin
                for(r=0;r<8;r=r+1)begin
                    for(l=0;l<8;l=l+1)begin activation[(r*8+l)*16+:16]<=lm_phase ? (r==0 ? act_tile_q[(norm_row_visible*8+l)*16+:16] : 16'd0) :
                            (activation_row_mask[r] ? act_tile_q[(r*8+l)*16+:16] : 16'd0);end
                end
                act_available<=1;
            end
            if(gemm_cmd_valid && gemm_cmd_ready)begin cmd_sent<=1;end
            if(gemm_act_valid && gemm_act_ready)begin act_sent<=1;end
            weight_pending<=weight_read_enable;
            if(weight_read_enable)begin weight_reads<=weight_reads+5'd1;end
            if(gemm_weight_valid && gemm_weight_ready)begin weight_sent<=weight_sent+4'd1;end
            if(gemm_done)begin gemm_terminal<=1;end
            if(!attn_active && gemm_valid && gemm_output_ready)begin raw_slice<=gemm_bf16;raw_number<=gemm_slice;raw_vector<=0;raw_busy<=1;raw_issued<=0;end
            else if(raw_write_fire)begin
                if(raw_vector==rows_times_two-1)begin raw_busy<=0;end else begin raw_vector<=raw_vector+4'd1;end
            end
            if(raw_scale_fire)raw_issued<=raw_issued+1'b1;
            if(gemm_feed_advance && global_k!=last_k)begin
                if(inner_k==15)begin inner_k<=0;input_tile<=input_tile+1'b1;end else begin inner_k<=inner_k+4'd1;end
            end
            if(post_launch)begin post_word<=0;post_row<=0;row_index<=0;rope_pair<=0;rope_write_half<=0;end
            if(post_state==S_ROPE_FIRST_WAIT && p1_valid)begin rope_x<=p1_q;end
            if(post_state==S_ROPE_SECOND_WAIT && p1_valid)begin rope_y<=p1_q;end
            if(post_state==S_ROPE_WAIT && afu_result_valid)begin
                rope_first<=afu_result_data;rope_second<=afu_result_aux;rope_write_half<=0;
            end
            if(post_state==S_ROPE_WRITE)begin
                rope_write_half<=!rope_write_half;
                if(rope_write_half)begin
                    if(rope_pair==7)begin rope_pair<=0;post_row<=post_row==rows_last ? 0 : post_row+3'd1;end else begin rope_pair<=rope_pair+3'd1;end
                end
            end
            if(post_state==S_WRITE)begin
                write_command_reg<=0;write_metadata_reg<=0;write_command_reg[118]<=1;write_command_reg[95:88]<=task_cfg[103:96];write_command_reg[111:96]<=task_id;write_command_reg[87:64]<=256;
                write_metadata_reg[15:0]<=task_id;write_metadata_reg[20:16]<=layer;write_metadata_reg[56:53]<=1;write_metadata_reg[60:57]<=8;write_metadata_reg[52:37]<=post_output_tile;
                write_command_reg[115:112]<=2;write_word<=0;
                if(post_operation==0)begin write_command_reg[63:0]<=task_cfg[383:320]+layer*64'd8388608+token_position*64'd256;end
                else if(post_operation==1)begin write_command_reg[63:0]<=task_cfg[447:384]+layer*64'd8388608+token_position*64'd256;end
                else begin write_command_reg[63:0]<=task_cfg[511:448]+row_index*64'd1792+post_output_tile*64'd256;end
            end
            if(post_state==S_WR_WAIT_DATA && p1_valid)begin write_hold<=p1_q;end
            if(post_state==S_WR_SEND && wr_ready && write_word!=15)begin write_word<=write_word+5'd1;end
            if(post_state==S_WR_FINISH && wr_done)begin row_index<=row_index==rows_last ? 0 : row_index+3'd1;end
            if(state_current[S_O_RESTORE])begin
                operation<=3;output_tile<=0;input_tile<=0;inner_k<=0;row_index<=0;
                norm_apply<=0;
            end
            if(state_current[S_O_NEXT])begin
                output_tile<=output_tile+1'b1;input_tile<=0;inner_k<=0;row_index<=0;
                norm_apply<=0;
            end
            if(state_current[S_RES_READ])begin residual_x_ready<=0;residual_o_ready<=0;end
            if(state_current[S_RES_WAIT])begin
                if(output_read_valid)begin
                    norm_gamma<=down_phase ? output_read_y : output_read_x;residual_o_ready<=1;
                    norm_x<=down_phase ? output_read_x : output_read_y;residual_x_ready<=1;
                end
            end
            if(residual_result || normalized_result)begin
                
            end
            if(state_current[S_RES_FINISH] && !state_next[S_RES_FINISH])begin
                
            end
            if(state_current[S_H_START])begin
                norm_apply<=0;hidden_b_done<=0;hidden_stats_done<=0;
                write_command_reg<=0;write_metadata_reg<=0;
                hidden_absolute_row<={1'b0,task_cfg[52:37]}+{14'd0,norm_row_visible};
                hidden_write_base<=down_phase ? descriptor[8][63:0] : descriptor[5][127:64];
                write_command_reg[87:64]<=1792;write_command_reg[95:88]<=task_cfg[103:96];
                write_command_reg[111:96]<=task_id;write_command_reg[115:112]<=3;write_command_reg[118]<=1;
                write_metadata_reg[15:0]<=task_id;write_metadata_reg[20:16]<=layer;
                write_metadata_reg[56:53]<=1;write_metadata_reg[60:57]<=8;write_metadata_reg[83:80]<=14;
                write_metadata_reg[104:89]<=task_cfg[52:37]+norm_row_visible;
            end
            // 1792 = 2048 - 256. Keep this constant product in fabric so the
            // dedicated address stage is not folded into a shared DSP input mux.
            if(state_current[S_H_OFFSET])hidden_row_offset<=
                {hidden_absolute_row[15:0],11'd0}-{hidden_absolute_row,8'd0};
            if(state_current[S_H_ADDRESS])hidden_address_low<={1'b0,hidden_write_base[31:0]}+{6'd0,hidden_row_offset};
            if(state_current[S_H_COMMIT])write_command_reg[63:0]<={hidden_write_base[63:32]+{31'd0,hidden_address_low[32]},hidden_address_low[31:0]};
            if(state_current[S_H_READ])begin hidden_dma_sent<=0;hidden_afu_sent<=0;end
            if(state_current[S_H_WAIT] && output_read_valid)begin norm_x<=output_read_y;write_hold<=output_read_y;end
            if(state_current[S_H_SEND])begin
                if(wr_valid && wr_ready)hidden_dma_sent<=1;
                if(afu_fire)hidden_afu_sent<=1;
            end
            if((output_phase || down_phase) && state_hidden_write && wr_done)hidden_b_done<=1;
            if(hidden_statistics)begin
                hidden_stats_done<=1;
            end
            if(state_current[S_H_FINISH] && state_next[S_Z_READ])begin norm_apply<=1;end
            if(state_current[S_Z_READ])begin residual_o_ready<=0;gamma_response_ready<=0;end
            if(state_current[S_Z_WAIT])begin
                if(output_read_valid)begin norm_x<=output_read_y;residual_o_ready<=1;end
                if(gamma_read_valid)begin norm_gamma<=gamma_read_data;gamma_response_ready<=1;end
            end
            if(state_current[S_Z_COPY_WAIT] && output_read_valid)norm_x<=output_read_x;
            if(state_current[S_FFN_BEGIN])begin
                operation<=4;output_tile<=0;input_tile<=0;inner_k<=0;
            end
            if(state_current[S_FFN_NEXT])begin
                input_tile<=0;inner_k<=0;
                if(operation==5)ffn_up_tiles<={1'b0,output_tile}+1'b1;
                if(output_tile==ffn_last_tile)begin operation<=5;output_tile<=0;end
                else output_tile<=output_tile+1'b1;
            end
            if(state_current[S_FINAL_BEGIN])begin
                operation<=7;norm_apply<=0;
            end
            if(state_current[S_LM_BEGIN])begin output_tile<=0;input_tile<=0;inner_k<=0;end
            if(state_current[S_LM_NEXT])begin output_tile<=output_tile+1'b1;input_tile<=0;inner_k<=0;end
            if(state_current[S_NEXT])begin
                input_tile<=0;inner_k<=0;coeff_kind<=0;coeff_word<=0;
                if(operation==2)begin output_tile<=output_tile+3'd1;end else begin operation<=operation+2'd1;output_tile<=0;end
            end
        end
    end

    // Register the FFN block transition before it reaches the block counter.
    // The visible value keeps the first cycle of the next block aligned while
    // the counter update is isolated from the operation/state decode cone.
    always @(posedge clk)begin
        if(!rst_n)begin
            ffn_block<=0;
            ffn_block_step_pending<=0;
        end else begin
            if(ffn_block_clear_event)ffn_block<=0;
            else if(ffn_block_step_pending)ffn_block<=ffn_block+3'd1;
            ffn_block_step_pending<=ffn_block_increment_event;
        end
    end

    // Register row commands so operation/state decode no longer reaches the
    // row counter D input in the same cycle.  norm_row_visible keeps all
    // consumers aligned with the command while the storage register catches up.
    always @(posedge clk)begin
        if(!rst_n)begin
            norm_row<=0;
            norm_row_cmd_pending<=2'b00;
        end else begin
            case(norm_row_cmd_pending)
                2'b01:norm_row<=0;
                2'b10:norm_row<=norm_row==rows_last ? 0 : norm_row+3'd1;
                2'b11:norm_row<=rows_last;
                default:;
            endcase
            if(norm_row_last_event)norm_row_cmd_pending<=2'b11;
            else if(norm_row_clear_event)norm_row_cmd_pending<=2'b01;
            else if(norm_row_increment_event)norm_row_cmd_pending<=2'b10;
            else norm_row_cmd_pending<=2'b00;
        end
    end

    // Register the combined clear/increment command so operation/state decode
    // no longer drives the norm_word reset pin in the same cycle.
    always @(posedge clk)begin
        if(!rst_n)norm_word_cmd_pending<=2'b00;
        else norm_word_cmd_pending<={norm_word_clear_event,norm_word_increment_event};
    end
    always @(posedge clk)begin
        if(!rst_n)norm_word_storage<=0;
        else norm_word_storage<=norm_word;
    end

    always @(posedge clk)begin
        if(!rst_n || state_current[S_FINAL_BEGIN])lm_finished<=0;
        else if(lm_complete)lm_finished<=1;
    end

    // Hold a complete DMA command until acceptance, even if the other producer changes.
    always @(posedge clk) begin
        if(!rst_n)begin
            command_hold_valid<=0;command_hold_write<=0;command_hold<=0;metadata_hold<=0;
        end else begin
            if(command_hold_valid)begin
                if(dma_cmd_ready)begin command_hold_valid<=0;end
            end else if(read_request)begin
                command_hold_valid<=1;command_hold_write<=0;command_hold<=command_reg;metadata_hold<=metadata_reg;
            end else if(write_request)begin
                command_hold_valid<=1;command_hold_write<=1;command_hold<=ffn_active ? ffn_dma_command : write_command_reg;metadata_hold<=ffn_active ? ffn_dma_metadata : write_metadata_reg;
            end
        end
    end

    // Prefetch the next weight tile while GEMM consumes the other physical bank.
    always @(*) begin
        fetch_next=fetch_state;
        case(fetch_state)
            F_IDLE:begin
                if(fetch_enabled && !fetch_scale_ready)begin fetch_next=F_SCALE_READ;end
                else if(fetch_begin)begin fetch_next=F_COMMAND;end
            end
            F_SCALE_READ:begin fetch_next=F_SCALE_WAIT;end
            F_SCALE_WAIT:begin if(scale_read_valid)begin fetch_next=fetch_scale_word==7 ? F_IDLE : F_SCALE_READ;end end
            F_COMMAND:begin if(read_command_fire)begin fetch_next=F_DATA;end end
            F_DATA:begin if(rd_done)begin fetch_next=F_FINISH;end end
            F_FINISH:begin if(fetch_commit)begin fetch_next=(fetch_half && fetch_tile==fetch_last_tile &&
                ((fetch_operation==2 && fetch_output_tile==6) ||
                 (fetch_operation==5 && fetch_output_tile==fetch_last_output) ||
                 (fetch_operation==6 && fetch_output_tile==6) ||
                 (fetch_operation==7 && fetch_output_tile==1186))) ? F_COMPLETE : F_IDLE;end end
            F_COMPLETE:begin end
            default:begin fetch_next=F_IDLE;end
        endcase
        if(state_current[S_IDLE] || state_current[S_O_RESTORE] || state_current[S_FFN_BEGIN] || state_current[S_DOWN_BEGIN] || state_current[S_LM_BEGIN] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || (state_current[S_DONE] && start))begin fetch_next=F_IDLE;end
    end
    always @(posedge clk)begin
        if(!rst_n)begin fetch_state<=F_IDLE;end else begin fetch_state<=fetch_next;end
    end
    always @(posedge clk)begin
        if(fetch_state==F_SCALE_READ && fetch_scale_word==0)begin
            fetch_column_source<=operation_base;
            // Each input tile occupies 16384 bytes; output strides are 7 or 38 tiles.
            fetch_column_index<=fetch_operation==6 ?
                ({fetch_output_tile,5'd0}+{fetch_output_tile,2'd0}+{fetch_output_tile,1'd0}) :
                ({fetch_output_tile,3'd0}-{3'd0,fetch_output_tile});
        end
        if(fetch_state==F_SCALE_WAIT && fetch_scale_word==0)begin
            fetch_column_low<={1'b0,fetch_column_source[31:0]}+{3'd0,fetch_column_index,14'd0};
            fetch_column_base<={fetch_column_source[63:32]+{31'd0,fetch_column_low[32]},fetch_column_low[31:0]};
        end
    end
    always @(posedge clk)begin
        if(!rst_n || state_current[S_IDLE] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || (state_current[S_DONE] && start))begin
            fetch_tile<=0;fetch_enabled<=0;fetch_bank<=0;fetch_half<=0;fetch_operation<=0;fetch_output_tile<=0;fetch_scale_word<=0;fetch_scale_ready<=0;
        end
        else begin
            if(state_current[S_O_RESTORE])begin
                fetch_tile<=0;fetch_enabled<=0;fetch_bank<=0;fetch_half<=0;fetch_operation<=3;fetch_output_tile<=0;fetch_scale_word<=0;fetch_scale_ready<=0;
            end
            if(state_current[S_O_NEXT])begin fetch_scale_ready<=0;fetch_scale_word<=0;end
            if(state_current[S_FFN_BEGIN])begin
                fetch_tile<=0;fetch_enabled<=0;fetch_bank<=0;fetch_half<=0;fetch_operation<=4;fetch_output_tile<=ffn_block_visible*6'd7;fetch_scale_word<=0;fetch_scale_ready<=0;
            end
            if(state_current[S_LM_BEGIN])begin
                fetch_tile<=0;fetch_enabled<=0;fetch_bank<=0;fetch_half<=0;fetch_operation<=7;fetch_output_tile<=0;fetch_scale_word<=0;fetch_scale_ready<=0;
            end
            if(state_current[S_DOWN_BEGIN])begin
                fetch_tile<=0;fetch_enabled<=0;fetch_bank<=0;fetch_half<=0;fetch_operation<=6;fetch_output_tile<=0;fetch_scale_word<=0;fetch_scale_ready<=0;
            end
            if(state_current[S_WEIGHT] && !output_phase && input_tile==0)begin fetch_enabled<=1;end
            if(state_current[S_WEIGHT] && output_phase && fetch_state==F_IDLE && !weight_full[weight_bank])begin
                fetch_enabled<=1;fetch_operation<=3;fetch_tile<=input_tile;fetch_output_tile<=output_tile;fetch_bank<=weight_bank;fetch_half<=inner_k[3];
            end
            if(fetch_state==F_SCALE_WAIT && scale_read_valid)begin
                scales[fetch_scale_word*256+:256]<=scale_read_data;
                if(fetch_scale_word==7)begin fetch_scale_word<=0;fetch_scale_ready<=1;end
                else begin fetch_scale_word<=fetch_scale_word+1'b1;end
            end
            if(fetch_commit && output_phase)begin fetch_enabled<=0;fetch_half<=!fetch_half;end
            if(fetch_commit && !output_phase)begin
                fetch_bank<=!fetch_bank;
                fetch_half<=!fetch_half;
                if(fetch_half && fetch_tile==fetch_last_tile)begin
                    fetch_tile<=0;fetch_scale_ready<=0;
                    if(fetch_operation==4 || fetch_operation==5)begin
                        if(fetch_output_tile==fetch_last_output)begin fetch_operation<=5;fetch_output_tile<=ffn_block*6'd7;end
                        else fetch_output_tile<=fetch_output_tile+1'b1;
                    end else if(fetch_operation==2 || fetch_operation==6 || fetch_operation==7)begin fetch_output_tile<=fetch_output_tile+1'b1;end
                    else begin fetch_operation<=fetch_operation+1'b1;fetch_output_tile<=0;end
                end else if(fetch_half)begin fetch_tile<=fetch_tile+1'b1;end
            end
        end
    end

    // A postprocessor owns its snapshot and bank independently of the next GEMM job.
    always @(*) begin
        post_next=post_state;
        case(post_state)
            S_IDLE:begin if(post_launch)begin post_next=S_BIAS_READ;end end
            S_BIAS_READ:begin if(post_bias_complete)begin post_next=post_operation==1 ? S_WRITE : S_ROPE_FIRST;end end
            S_ROPE_FIRST:begin post_next=S_ROPE_FIRST_WAIT;end
            S_ROPE_FIRST_WAIT:begin if(p1_valid)begin post_next=S_ROPE_SECOND;end end
            S_ROPE_SECOND:begin post_next=S_ROPE_SECOND_WAIT;end
            S_ROPE_SECOND_WAIT:begin if(p1_valid)begin post_next=S_ROPE_SEND;end end
            S_ROPE_SEND:begin if(afu_ready)begin post_next=S_ROPE_WAIT;end end
            S_ROPE_WAIT:begin if(afu_result_valid)begin post_next=S_ROPE_WRITE;end end
            S_ROPE_WRITE:begin
                if(rope_write_half)begin
                    if(rope_pair==7 && post_row==rows_last)begin post_next=S_WRITE;end
                    else begin post_next=S_ROPE_FIRST;end
                end
            end
            S_WRITE:begin post_next=S_WR_CMD;end
            S_WR_CMD:begin if(write_command_fire)begin post_next=S_WR_READ;end end
            S_WR_READ:begin post_next=S_WR_WAIT_DATA;end
            S_WR_WAIT_DATA:begin if(p1_valid)begin post_next=S_WR_SEND;end end
            S_WR_SEND:begin if(wr_ready)begin post_next=write_word==15 ? S_WR_FINISH : S_WR_READ;end end
            S_WR_FINISH:begin if(wr_done)begin post_next=row_index==rows_last ? S_NEXT : S_WRITE;end end
            S_NEXT:begin post_next=S_IDLE;end
            default:begin post_next=S_IDLE;end
        endcase
    end
    always @(posedge clk)begin
        if(!rst_n)begin post_state<=S_IDLE;end else begin post_state<=post_next;end
    end
    always @(posedge clk)begin
        if(!rst_n)begin
            compute_bank<=0;post_bank<=0;post_operation<=0;post_output_tile<=0;post_identity<=0;rope_outstanding<=0;
        end else begin
            if(state_current[S_IDLE] || state_current[S_O_RESTORE] || state_current[S_FFN_BEGIN] || state_current[S_DOWN_BEGIN] || state_current[S_LM_BEGIN] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || ((state_current[S_DONE]) && start))begin compute_bank<=0;end
            if(state_current[S_NEXT] || state_current[S_FFN_NEXT] || state_current[S_LM_NEXT])begin compute_bank<=!compute_bank;end
            if(post_launch)begin post_bank<=compute_bank;post_operation<=operation;post_output_tile<=output_tile;post_identity<=compute_identity;post_biases<=biases;end
            if(afu_fire && post_state==S_ROPE_SEND)begin rope_outstanding<=1;end
            if(afu_result_valid && afu_result_ready && rope_outstanding)begin rope_outstanding<=0;end
        end
    end

    // Reserve response space before issuing a synchronous RAM read; carry the address with data.
    always @(posedge clk)begin
        if(!rst_n || post_launch)begin
            bias_issued<=0;bias_retired<=0;bias_input_credit<=0;bias_input_count<=0;
            bias_input_write<=0;bias_input_read<=0;bias_read_address<=0;
        end else begin
            case({bias_read_request,bias_input_fire})
                2'b10:begin bias_input_credit<=bias_input_credit+1'b1;end
                2'b01:begin bias_input_credit<=bias_input_credit-1'b1;end
                default:begin end
            endcase
            case({p0_valid,bias_input_fire})
                2'b10:begin bias_input_count<=bias_input_count+1'b1;end
                2'b01:begin bias_input_count<=bias_input_count-1'b1;end
                default:begin end
            endcase
            if(bias_read_request)begin bias_read_address<=bias_issued[6:0];bias_issued<=bias_issued+1'b1;end
            if(p0_valid)begin
                bias_input_data[bias_input_write]<=p0_q;bias_input_coeff[bias_input_write]<=selected_bias;bias_input_scale[bias_input_write]<=selected_scale;
                bias_input_address[bias_input_write]<=bias_read_address;bias_input_write<=bias_input_write+1'b1;
            end
            if(bias_input_fire)begin bias_input_read<=bias_input_read+1'b1;end
            if(bias_output_fire)begin bias_retired<=bias_retired+1'b1;end
        end
    end
endmodule
