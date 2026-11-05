`timescale 1ns / 1ps
// 鍥哄畾鏁版嵁娴侊細AXI-Lite蹇収椹卞姩鍥哄寲鍙傛暟銆丷MSNorm銆並鍒癡鍒癚銆乥ias銆丷oPE鏈夋晥杈圭晫鍜岀湡瀹濪DR鍐欏洖
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
    input wire clk, // 绯荤粺300MHz鍚岄挓鍩?
    input wire rst_n, // 鍚屾浣庢湁鏁堝浣?
    input wire start, // AXI-Lite瀹屾暣閰嶇疆蹇収鎻愪氦
    input wire [575:0] cfg, // 涓嶅彲鍙樹换鍔″揩鐓?
    input wire abort_req, // 鍋滄鏂伴樁娈靛苟鎺掔┖
    input wire soft_reset, // 鎺掔┖鍚庢仮澶嶆帶鍒?
    output wire dma_cmd_valid, // DMA鐪熷疄鍛戒护鏈夋晥
    input wire dma_cmd_ready, // DMA鍛戒护鎺ュ彈
    output wire [127:0] dma_cmd, // 瀹屾暣64bit鍦板潃鎻忚堪绗?
    output wire [127:0] dma_meta, // DMA瀹屾暣metadata
    output wire dma_abort, // 鐪熷疄DMA鎺掔┖璇锋眰
    input wire rd_busy, // AXI璇讳簨鍔″湪閫?
    input wire rd_done, // AXI璇讳笌鏈湴娴佺湡瀹炴垚鍔?
    input wire rd_error, // AXI璇诲け璐?
    input wire rd_aborted, // AXI璇诲彇娑堝畬鎴?
    input wire [7:0] rd_error_code, // 璇婚敊璇師鍥?
    input wire [3:0] direct_rd_valid, output wire [3:0] direct_rd_ready,
    input wire [511:0] direct_rd_data, input wire [39:0] direct_rd_word,
    input wire [63:0] direct_rd_keep, input wire [7:0] direct_rd_error,
    input wire [127:0] direct_rd_meta,
    input wire rd_valid, // DMA鏈湴128bit娴佹湁鏁?
    output wire rd_ready, // 鎺ュ彈128bit娴?
    input wire [127:0] rd_data, // DDR璇绘暟鎹?
    input wire [15:0] rd_keep, // 鏈夋晥byte
    input wire rd_last, // descriptor鏈媿
    input wire [127:0] rd_meta, // 璇绘暟鎹韩浠?
    input wire [15:0] rd_tag, // descriptor鍏宠仈
    input wire [3:0] rd_resource, // 璇诲彇璧勬簮缂栧彿
    input wire [1:0] rd_beat_error, // AXI鏈媿閿欒
    output wire wr_valid, // 閫丏MA鐨?28bit娴?
    input wire wr_ready, // DMA鎺ュ彈鍐欐祦
    output wire [127:0] wr_data, // DDR鍐欐暟鎹?
    output wire [15:0] wr_keep, // 鍏獴F16瀹屾暣byte
    output wire wr_last, // 鍐檇escriptor鏈媿
    output wire [127:0] wr_meta, // 鍐檇escriptor韬唤
    input wire wr_busy, // AXI鍐欎簨鍔″湪閫?
    input wire wr_done, // 鐪熷疄B鍝嶅簲鎴愬姛
    input wire wr_error, // 鍐欏け璐?
    input wire wr_aborted, // 鍐欐帓绌哄彇娑?
    input wire [7:0] wr_error_code, // 鍐欏け璐ュ師鍥?
    output wire afu_valid, // 鏈夋晥绠楁湳frame杈撳叆
    input wire afu_ready, // 澶栭儴AFU鎺ュ彈
    output wire [3:0] afu_command, // 11RMSNorm12RoPE杈圭晫
    output wire [20:0] afu_command_4, // 闃舵鎺у埗淇濈暀
    output wire [26:0] afu_sum_command_buffer, // 褰掔害闃舵鎺у埗淇濈暀
    output wire [31:0] afu_eps, // FP32 RMS epsilon
    output wire afu_latch, // 褰掔害鍙傛暟淇濇寔璇锋眰
    output wire afu_gelu_sign, // QKV涓嶅惎鐢℅ELU
    output wire [127:0] afu_x, // 鍏玪ane BF16 X
    output wire [127:0] afu_y, // gamma鎴朢oPE閰嶅绗簩鍗?
    output wire [127:0] afu_meta, // 鏈夋晥琛屽瓧鍦板潃浣嶇疆韬唤
    output wire afu_last, // 琛屾垨RoPE褰撳墠鍚戦噺缁撴潫
    output wire [7:0] afu_mask, // 鏈夋晥鍏玪ane
    output wire [63:0] afu_length, // 896缁碦MS鎴?4缁磆ead
    input wire afu_result_valid, // 澶栭儴瀹為檯璁＄畻缁撴灉valid
    output wire afu_result_ready, // Controller鎺ュ彈缁撴灉
    input wire [127:0] afu_result_data, // 绗竴缁勫叓lane缁撴灉
    input wire [127:0] afu_result_aux, // RoPE绗簩缁勭粨鏋?
    input wire [127:0] afu_result_meta, // 缁撴灉瀹屾暣韬唤
    input wire afu_result_last, // 缁撴灉琛屾湯
    input wire [7:0] afu_result_mask,
    // Dedicated RoPE channel: private request and result ports so the RoPE
    // engine no longer shares afu_valid/afu_result_valid with the scalar AFU.
    output wire rope_req_valid,
    input wire rope_req_ready,
    output wire rope_req_last,
    output wire [127:0] rope_req_x,
    output wire [127:0] rope_req_y,
    output wire [127:0] rope_req_meta,
    output wire [7:0] rope_req_mask,
    input wire rope_res_valid,
    output wire rope_res_ready,
    input wire rope_res_last,
    input wire [127:0] rope_res_data,
    input wire [127:0] rope_res_aux,
    input wire [127:0] rope_res_meta,
    input wire [7:0] rope_res_mask,
    input wire afu_busy,
    input wire afu_error, // 缁撴灉lane鏈夋晥
    output wire afu_abort, // Attention閿欒鏃跺彇娑堟湭瀹屾垚鐨凙FU frame
    output wire busy, // QKV闃舵灏氭湭瀹屾垚
    output wire qkv_done, // K,V,Q鍏ㄩ儴鐪熷疄DDR鍐欏洖瀹屾垚
    output wire param_ready, // 涓夎〃鐪熷疄瀹屾垚鍚庡彧璇?
    output wire result_commit, // 鏈€缁圙reedy缁撴灉鍗曟媿鎻愪氦
    output wire [17:0] best_token_id, // 鎻愪氦鍓嶅凡閿佸瓨鐨勮瘝琛ㄤ綅缃?
    output wire error, // 鎺у埗閿欒淇濇寔
    output wire [31:0] error_code // 鍏煎CSR鎺ュ彛锛涗笉姹囨€婚敊璇爜锛屽浐瀹氳緭鍑洪浂
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
    // [STAGE5] post-FSM states that drain the INT8 KV producers.  These numbers
    // only live in the 7-bit binary post FSM; the main FSM is one-hot (81 bits)
    // and is deliberately left untouched.
    localparam [6:0] PS_KV8_TAIL=7'd40, PS_KV8_EMIT=7'd41, PS_KV8_CMD=7'd42, PS_KV8_SEND=7'd43,
                     PS_KV8_FLUSH=7'd44, PS_KV8_SCALE=7'd45, PS_KV8_SCALE_SEND=7'd46,
                     PS_KV8_VCMD=7'd47, PS_KV8_VSEND=7'd48, PS_KV8_VLOAD=7'd49,
                     PS_KV8_VLOAD_CMD=7'd50, PS_KV8_VRD=7'd51;
    reg compute_bank,post_bank;
    reg [1:0] post_operation;
    reg [2:0] post_output_tile;
    reg [2047:0] post_biases;
    reg [2047:0] compute_scales, post_scales;
    reg [127:0] post_identity;
    reg [127:0] write_command_reg,write_metadata_reg;
    reg command_hold_valid,command_hold_write;
    reg [127:0] command_hold,metadata_hold;
    // =========================================================================
    // [STAGE5] INT8 KV cache: producer chains + the DDR write/scale plumbing
    // -------------------------------------------------------------------------
    // The K and V producers replace the BF16 post-write of the projection
    // results (post_operation 0 = K, 1 = V) and they own the scale RAM:
    //   K : 16 BF16 beats/token -> block32 INT8 in DDR + per-(head,token) scale
    //       append + per-(head,256-token block) exponent publish
    //   V : 16 BF16 beats/token -> token-major INT8 in DDR + the per-(head,dim)
    //       scale of the whole layer (first token of the layer is the lock),
    //       whose 256 B image goes to DDR once per layer and is bulk-loaded back
    //       into kv8_scale_ram.vmem before the layer's first attention.
    // =========================================================================
    // ---- K producer --------------------------------------------------------
    wire        kv8_k_start, kv8_k_flush;
    wire [4:0]  kv8_k_layer;
    wire [15:0] kv8_k_token;
    wire        kv8_k_valid;
    wire [127:0] kv8_k_data;
    wire        kv8_k_ready, kv8_k_vready, kv8_k_busy, kv8_k_flush_done;
    wire        kv8_k_beat_valid, kv8_k_beat_ready;
    wire [63:0] kv8_k_beat_addr;
    wire [127:0] kv8_k_beat_data;
    wire [4:0]  kv8_k_beat_len, kv8_k_blk_tok, kv8_k_token_beats;
    wire        kv8_k_clip_sticky;
    wire [15:0] kv8_k_clip_count;
    // ---- V producer --------------------------------------------------------
    wire        kv8_v_start;
    wire        kv8_v_valid;
    wire [127:0] kv8_v_data;
    wire        kv8_v_ready, kv8_v_vready;
    wire        kv8_v_beat_valid, kv8_v_beat_ready;
    wire [63:0] kv8_v_beat_addr;
    wire [127:0] kv8_v_beat_data;
    wire        kv8_v_scale_valid;
    wire [31:0] kv8_v_scale_bf16;
    reg         kv8_v_scale_clear;
    wire [3:0]  kv8_v_token_beats;
    wire        kv8_v_clip_sticky;
    wire [15:0] kv8_v_clip_count;
    // [V-PERTOK-M] per-token V mantissa append (producer -> scale RAM) and the
    // per-token mantissa read port the PV activation scaling uses.
    wire        kv8_v_m_app_valid, kv8_v_m_app_ready;
    wire [15:0] kv8_v_m_app_token;
    wire        kv8_v_m_app_head;
    wire [15:0] kv8_v_m_app_mant;
    wire [31:0] kv8_v_clip_codes;
    wire [7:0]  kv8_v_m_max;
    wire        kv8_vm_rd_en;
    wire [11:0] kv8_vm_rd_word;
    wire [127:0] kv8_vm_rd_data;
    reg         kv8_v_scale_written, kv8_v_table_loaded;
    reg         kv8_v_scale_we;
    reg  [3:0]  kv8_v_scale_word;
    reg  [127:0] kv8_v_scale_data;
    wire        kv8_v_we_chain;
    wire [3:0]  kv8_v_word_chain;
    wire [127:0] kv8_v_data_chain;
    wire [3:0]  kv8_v_rd_word;
    wire [127:0] kv8_v_rd_data;
    // ---- scale-RAM bank ownership -----------------------------------------
    // 0 = EMPTY (acquire before the layer's first K token), 1 = LOADING,
    // 2 = READY (attention reads; appends still allowed)
    reg  [1:0]  kv8_bank_state;
    reg         kv8_acq_pending, kv8_own_layer_valid;
    reg  [4:0]  kv8_own_layer;
    reg         kv8_wr_acquire, kv8_wr_release, kv8_rd_release;
    wire        kv8_rd_grant;
    // [STAGE5-FIX] The attention may only read the K scale table while the
    // bank is READY: kv8_scale_ram drops every read (rd_ok = 0) while the bank
    // is LOADING, and the read has no error/retry path.  The V chain writes the
    // V-table words into vmem directly at lock time, so the DDR bulk load
    // (PS_KV8_VLOAD) is not needed -- but the LOADING -> READY hand-off is, and
    // it was the only thing missing (attention S_SCALE_FILL waited forever).
    // Wait only when a bank is actually owned: EMPTY (0) means no KV8 traffic.
    wire        kv8_scale_wait=(kv8_bank_state!=2'd0) && !kv8_rd_grant;
    // ---- DMA command / write-stream arbitration ----------------------------
    reg  [63:0] kv8_cmd_addr;
    reg  [23:0] kv8_cmd_bytes;
    reg         kv8_cmd_is_write;
    reg  [4:0]  kv8_desc_left;          // beats left in the current descriptor
    reg  [3:0]  kv8_img_word;           // V scale image beat counter
    reg  [4:0]  kv8_vbyte_cnt;          // token beat counter (K 16, V 8)
    reg  [3:0]  kv8_vrd_count;          // V table bulk-load beat counter
    reg  [4:0]  kv8_beat_lim;           // beats this token's command carries (from the chain)
    reg  [4:0]  kv8_vone_cnt;           // V table 1.0 fill counter
    wire        kv8_push_ready;
    reg         kv8_flush_pending;
    wire        kv8_cmd_valid;
    wire        kv8_cmd_fire;
    wire        kv8_stream_valid;
    wire [127:0] kv8_stream_data;
    wire [127:0] kv8_scale_stream_data;
    // sticky diagnostics
    reg         kv8_dma_conflict;
    // ---- [16K-ONE-LAYER] per-layer scale-image sequencer -------------------
    localparam [3:0] LS_IDLE=4'd0, LS_D_CMD=4'd1, LS_D_WAIT=4'd2, LS_D_BEAT=4'd3,
                     LS_D_WR=4'd4, LS_D_EBC=4'd5, LS_D_EBB=4'd6, LS_D_EBW=4'd7,
                     LS_OWN=4'd8, LS_L_CMD=4'd9, LS_L_RD=4'd10, LS_L_NEXT=4'd11, LS_FIN=4'd12;
    reg  [3:0]  kv8_ls_state;
    reg  [3:0]  kv8_ls_rgn;          // load region 0..3
    reg  [1:0]  kv8_ls_item;         // dump item 0..3 (K0,K1,V0,V1)
    reg  [1:0]  kv8_ls_phase;        // etab beat phase
    reg  [12:0] kv8_ls_cnt;          // beats / words in flight
    reg  [10:0] kv8_ls_w;            // dump word index
    reg  [3:0]  kv8_ls_i;            // etab on-chip word index
    reg  [15:0] kv8_vis_lo;          // produced token range of the previous visit
    reg  [15:0] kv8_vis_hi;
    reg  [4:0]  kv8_vis_layer;       // layer that produced that range
    reg  [15:0] kv8_own_lo;           // [16K-FIX5] pass_lo of the last visit start
    // A "visit" is one ROW GROUP (<=8 rows), not one layer: a rows>8 task
    // is swept as consecutive groups of the same layer, with task_cfg[52:37]
    // advancing by `rows` at every group (batch_reconfigure_fire).
    reg         kv8_ls_eb_dirty;     // this visit produced a 256-token block start
    wire        kv8_ls_own_go;
    // image sequencer command fields + write-beat interface (the shared dma_cmd /
    // wr_data muxes above are defined before the sequencer, so declare here)
    reg  [63:0]  kv8_ls_cmd_addr;
    reg  [23:0]  kv8_ls_cmd_bytes;
    reg          kv8_ls_cmd_wr;
    wire         kv8_ls_cmd_sel;
    wire         kv8_img_wr;
    wire [127:0] kv8_img_wr_data;
    wire         kv8_img_wr_last;
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
    reg [127:0] descriptor[0:9]; // format3鏈瓧鎻愪緵E杞疆鏉冮噸鍩哄潃
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
    reg [2:0] operation; // 0K銆?V銆?Q銆?O銆?Gate銆?Up
    // Predecode the next operation on the same edge as operation itself.
    // The one-hot register isolates the wide phase fanout from the opcode path.
    reg [7:0] operation_decode_reg;
    reg [2:0] operation_path_next;
    reg [10:0] output_tile;
    reg [5:0] input_tile;
    reg [3:0] inner_k; // 128杈撳叆tile鍐呭崄鍏釜8K鍧?
    reg [2:0] row_index,norm_row;
    reg [1:0] norm_word_cmd_pending; // [1] clear, [0] increment
    reg [6:0] norm_word_storage;
    // Requests use the pending command immediately; storage follows one cycle later.
    // This keeps the word address aligned with the RAM read and AFU metadata.
    wire [6:0] norm_word=norm_word_cmd_pending[1] ? 7'd0 :
        norm_word_cmd_pending[0] ? norm_word_storage+7'd1 : norm_word_storage;
    reg [1:0] norm_row_cmd_pending; // 01 clear, 10 increment/wrap, 11 set last row
    reg [7:0] norm_results_count; // 浠呮帴鍙楃粨鏋滄椂鏇存柊锛屼笉澶嶄綅璁℃暟瀛樺偍
    reg norm_results_valid; // FFN/褰掍竴鍖栭樁娈佃竟鐣屾媺浣庯紝涓嬩竴缁撴灉浠?寮€濮嬭鏁?
    wire [7:0] norm_results=norm_results_valid ? norm_results_count : 8'd0;
    reg norm_apply; // 0骞虫柟鍜岀粺璁°€?閲嶈骞跺綊涓€鍖栬緭鍑?
    reg [127:0] norm_x,norm_gamma;
    reg [1:0] coeff_kind; reg [3:0] coeff_word;
    reg [2047:0] scales,biases; // 褰撳墠128鍒楀弬鏁帮紝鏃犺法灞傚鍒?
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
    // ---- [STAGE5] derived addresses and the producer/handshake wires --------
    // K bytes : k_base + layer*8MiB + token[13:5]*4096 + dg*1024 + head*512 + tok*16
    // V bytes : v_base + layer*8MiB + token*128 + head*64 + dg*16
    // V scale : k_layer_base + 0x8000 (inside the layer's own 64 KiB window)
    wire [63:0] kv8_v_layer_base = task_cfg[447:384]+{36'd0,layer,23'd0};
    // ---- [16K-ONE-LAYER] scale image, 0x3000_0000 + layer*1 MiB -------------
    // The on-chip scale tables are one layer deep and 16384 tokens deep, so the
    // per-layer scales live in DDR and are loaded at every layer visit.  The
    // 0x3000_0000 region is the one the owner reserved for exactly this (24 MiB
    // at a 1 MiB stride, ~129 KiB used per layer):
    //   +0x00000 K m16 64 KiB, +0x10000 V m16 64 KiB,
    //   +0x20000 V 2^  256 B (32 KiB reserved), +0x28000 K etab 256 B
    localparam [63:0] KV8_IMG_BASE = 64'h0000_0000_3000_0000;
    localparam [63:0] KV8_IMG_K    = 64'h0000_0000_0000_0000;
    localparam [63:0] KV8_IMG_V    = 64'h0000_0000_0001_0000;
    localparam [63:0] KV8_IMG_V2   = 64'h0000_0000_0002_0000;
    localparam [63:0] KV8_IMG_EB   = 64'h0000_0000_0002_8000;
    wire [63:0] kv8_img_layer   = KV8_IMG_BASE+{39'd0,layer,20'd0};
    wire [63:0] kv8_img_k_base  = kv8_img_layer+KV8_IMG_K;
    wire [63:0] kv8_img_v_base  = kv8_img_layer+KV8_IMG_V;
    wire [63:0] kv8_img_v2_base = kv8_img_layer+KV8_IMG_V2;
    wire [63:0] kv8_img_eb_base = kv8_img_layer+KV8_IMG_EB;
    // the V 2^ image moved here from k_base+0x8000 (same 256 B, new region)
    wire [63:0] kv8_scale_base   = kv8_img_v2_base;
    wire [15:0] kv8_token_now    = token_position[15:0];
    reg  [127:0] kv8_dma_command, kv8_dma_metadata;
    // [STAGE5-FIX3] the V producer is wired exactly like the K one; parking it
    // leaves the attention reading an INT8 V region nobody wrote and a V scale
    // table of zeros (context = 0).  Restored from qwen25_qkv_controller.v.bak.
    assign kv8_v_start      = (post_state==S_WRITE) && (post_operation==1) && kv8_v_ready;
    assign kv8_k_flush      = (post_state==PS_KV8_FLUSH) && (kv8_k_blk_tok!=5'd0);
    assign kv8_push_ready   = (post_operation==0) ? kv8_k_vready : kv8_v_vready;
    assign kv8_k_valid      = (post_state==S_WR_SEND) && (post_operation==0);
    assign kv8_v_valid      = (post_state==S_WR_SEND) && (post_operation==1);
    assign kv8_k_data       = write_hold;
    assign kv8_v_data       = write_hold;
    assign kv8_k_beat_ready = (post_state==PS_KV8_VSEND) && (post_operation==0) && wr_ready;
    assign kv8_v_beat_ready = (post_state==PS_KV8_VSEND) && (post_operation==1) && wr_ready;
    assign kv8_stream_valid = ((post_state==PS_KV8_VSEND) &&
                               ((post_operation==0) ? kv8_k_beat_valid : kv8_v_beat_valid)) ||
                              (post_state==PS_KV8_SCALE_SEND);
    assign kv8_scale_stream_data = {8{(kv8_img_word[3] ? kv8_v_scale_bf16[31:16]
                                                      : kv8_v_scale_bf16[15:0])}};
    assign kv8_stream_data = (post_state==PS_KV8_SCALE_SEND) ? kv8_scale_stream_data :
                             ((post_operation==0) ? kv8_k_beat_data : kv8_v_beat_data);
    assign kv8_cmd_valid = (post_state==PS_KV8_CMD) || (post_state==PS_KV8_SCALE) ||
                           (post_state==PS_KV8_VCMD) || (post_state==PS_KV8_VLOAD_CMD) ||
                           (kv8_ls_state==LS_D_CMD) || (kv8_ls_state==LS_D_EBC) ||
                           (kv8_ls_state==LS_L_CMD);
    assign kv8_cmd_fire  = kv8_cmd_owned && dma_cmd_ready;
    always @(*)begin
        kv8_dma_command=0;
        kv8_dma_command[63:0]   = kv8_ls_cmd_sel ? kv8_ls_cmd_addr  : kv8_cmd_addr;
        kv8_dma_command[87:64]  = kv8_ls_cmd_sel ? kv8_ls_cmd_bytes : kv8_cmd_bytes;
        kv8_dma_command[95:88]  = task_cfg[103:96];
        kv8_dma_command[111:96] = task_id;
        kv8_dma_command[115:112]= 4'd2;
        kv8_dma_command[118]    = kv8_ls_cmd_sel ? kv8_ls_cmd_wr   : kv8_cmd_is_write;
        kv8_dma_metadata=0;
        kv8_dma_metadata[15:0]  = task_id;
        kv8_dma_metadata[20:16] = layer;
        kv8_dma_metadata[56:53] = rows;
        kv8_dma_metadata[60:57] = 8;
    end
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
    reg [2:0] rope_pair;
    reg [127:0] rope_x,rope_y,rope_first,rope_second;
    reg rope_write_half;
    reg [4:0] write_word;
    reg [127:0] write_hold;
    reg [127:0] compute_identity,afu_identity;
    // 鍐欏洖鍦板潃鍒嗘垚琛屽彿銆佺獎鍋忕Щ銆佷綆32浣嶈繘浣嶃€侀珮32浣嶅洓娈点€?
    reg [16:0] hidden_absolute_row;
    (* use_dsp = "no" *) reg [26:0] hidden_row_offset;
    reg [63:0] hidden_write_base;
    reg [32:0] hidden_address_low;
    // 鍘熷X淇濆瓨鍦∣ bank 1锛學o杈撳嚭淇濆瓨鍦╞ank 0锛涙畫宸悗bank 1澶嶇敤涓篐銆?    // Activate浠呬繚瀛樺綊涓€鍖栬绠楄緭鍏ャ€丆ontext鏆傚瓨鍜孎FN涓棿鍚戦噺銆?
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
    // Hoisted declaration (semantically neutral): xvlog rejects the forward
    // reference used by output_write_row below, so the original file cannot be
    // analyzed by xvlog/xsim at all.  This move makes the module simulatable.
    wire [2:0] norm_row_visible=norm_row_cmd_pending==2'b01 ? 3'd0 :
        norm_row_cmd_pending==2'b10 ? (norm_row==rows_last ? 3'd0 : norm_row+3'd1) :
        norm_row_cmd_pending==2'b11 ? rows_last : norm_row;
    // [WIDTH-689] The tensor_buffers output_wr_word port is 7 bits (128 x 128-bit
    // words = 896 columns, i.e. one output row) while the raw-write expression
    // {output_tile,4'd0}+{raw_number,1'b0}+raw_vector[0] is 15 bits wide because
    // output_tile is declared [10:0].  The upper bits are DEAD, not meaningful:
    //  * this port is written from the output/FFN path only, where output_tile is
    //    the 0..6 output tile (S_O_NEXT/S_FFN_NEXT compare it with 6), raw_number
    //    is 3 bits and raw_vector 4 bits, so the largest value is 6*16+7*2+1 = 111
    //    which fits in 7 bits;
    //  * bit-identically for ANY output_tile: {raw_number,1'b0}+raw_vector[0] <= 15
    //    never carries past bit 3, so bits [6:4] of the sum are exactly
    //    output_tile[2:0] and every bit above 6 was already dropped by the 7-bit
    //    port.  Slicing output_tile to 3 bits therefore changes no reachable value
    //    and only removes the [Synth 8-689] width-mismatch warning.
    // (output_write_word above already truncated the same expression silently.)
    wire [6:0] output_raw_word = {output_tile[2:0],4'd0}+{raw_number,1'b0}+raw_vector[0];
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
    // READ_INPUT瑁呰浇鍘熷X鍒癘 bank 1锛屽寘鎷琖o鍓嶆仮澶峏锛汥MA鎸塐鍐欏彛ready鎺ㄨ繘銆?
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
    wire weight_load_ready, weight_valid; wire [1023:0] weight_q,weight_q_pair; wire [1:0] weight_full;
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
    // 鍚庣画鎸変綅瑕嗗啓word鍜宷uery_row
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
    // The RoPE request carries its own identity word; it must not overwrite the
    // scalar AFU word when the postprocessor and the main FSM run together.
    reg [127:0] rope_metadata;
    // Attention owns the existing services only after QKV has drained.
    wire attn_busy, attn_done;
    // [P16K] P单块：不再有bank选择，只剩一块归属标志。
    wire attn_weight_bank;
    wire attn_p_commit, attn_p_release;
    wire attn_p_full;
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
    // [STAGE5-FIX] hold the attention until the scale-RAM bank is READY
    wire attn_start=state_current[S_ATTN_BEGIN] && !kv8_scale_wait;
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
    wire attn_p_write_ready,attn_p_afu_write;
    wire [14:0] attn_p_afu_address;
    wire [127:0] attn_p_afu_write_data;
    // [P16K] attn_p_address同时是P行的{行[2:0],字[11:0]}线性地址，也是LM-head
    // logits流的字序号（0..18991，集测TB逐字比对lm_write_address）。logits走
    // stat_only，不落盘，所以这两重身份不再冲突。
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
    wire afu_fire=afu_valid && afu_ready;
    // The RoPE channel has its own handshakes and its own outstanding flag.
    wire rope_fire=rope_req_valid && rope_req_ready;
    wire rope_return=rope_res_valid && rope_res_ready;
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
    wire read_request=state_current[S_RD_CMD] || fetch_state==F_COMMAND;
    wire write_request=post_state==S_WR_CMD || state_current[S_H_COMMAND] || (ffn_active && ffn_dma_valid);
    // The hold register owns the bus only when neither the attention DMA nor the
    // INT8 KV producer is presenting a command -- otherwise a shared dma_cmd_ready
    // would retire a command that was never sent.  [16K-ONE-LAYER] the scale-image
    // sequencer additionally borrows the bus for its whole window: the image beats
    // must not interleave with a weight-fetch read descriptor.
    wire kv8_hold_block=(kv8_ls_state!=LS_IDLE) && (kv8_ls_state!=LS_OWN);
    wire command_hold_present=command_hold_valid && !kv8_hold_block;
    wire command_hold_taken=dma_cmd_ready && !attn_dma_owner && command_hold_present;
    wire read_command_fire=command_hold_valid && command_hold_taken && !command_hold_write;
    wire write_command_fire=command_hold_valid && command_hold_taken && command_hold_write;
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
            // weight_full[bank] is the LEVEL form of 'this tile is in the bank'; the
            // pulse-derived weight_load_done can already be 0 here because the next
            // tile's fetch_begin clears it.
            READ_WEIGHT:read_payload_complete=(weight_load_done || weight_full[weight_bank]) && !all_dequant_busy;
            default:begin end
        endcase
    end
    // The flag must be LATCHED, not recomputed every cycle: `weight_load_done` is a
    // pulse-updated flag and `fetch_begin` (the next tile's prefetch) clears it, so a
    // one-cycle-wide coincidence between "payload complete" and FINISH could be
    // missed and the FSM would wait in S_RD_FINISH forever.  Set on any qualifying
    // cycle, consumed (and cleared) when FINISH is left.
    always @(posedge clk)begin
        if(!rst_n)read_finish_pending<=1'b0;
        else if((read_payload_complete || payload_ok_r) &&
            ((state_current[S_RD_DATA] && rd_done) ||
             (state_current[S_RD_FINISH] && read_terminal)))read_finish_pending<=1'b1;
        else if(!state_current[S_RD_FINISH])read_finish_pending<=1'b0;
    end
    // Sticky "this tile's payload has been complete at least once".  `weight_load_done`
    // is only updated on dequant/direct write pulses and is cleared by `fetch_begin`,
    // so the live value can be 0 again long before the FSM reaches FINISH.
    reg payload_ok_r;
    always @(posedge clk)begin
        if(!rst_n)payload_ok_r<=1'b0;
        else if(fetch_begin)payload_ok_r<=1'b0;
        else if(read_payload_complete)payload_ok_r<=1'b1;
    end
    wire weight_direct_write=(DIRECT_WEIGHT_READ!=0) && (direct_write_count!=0);
    wire weight_dequant_write=dequant_valid && weight_load_ready;
    wire fetch_commit=fetch_state==F_FINISH && weight_load_done && !all_dequant_busy;
    always @(posedge clk)begin
        if(!rst_n)weight_load_done<=1'b0;
        else if(weight_direct_write)
            weight_load_done<=(weight_written+{12'd0,direct_write_count}>=16'd512);
        else if(weight_dequant_write)
            weight_load_done<=(weight_written>=16'd511);
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
        if(!rst_n)
            fetch_bank_scale_copies<=16'd0;
        else if(state_current[S_IDLE] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || (state_current[S_DONE] && start))
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

    // =========================================================================
    // [16K-ONE-LAYER] layer-start guard: the scale-image sequencer owns the DMA
    // command/read/write stream for its window, so (a) it only starts from a
    // quiet window and (b) the K producer cannot start until it is done.
    // =========================================================================
    // A pass (one layer sweep over `rows` tokens) starts at task_cfg[52:37]; every
    // layer of the pass produces the SAME token range, so the range a visit will
    // produce is [pass_lo, pass_lo+rows) and the range ALREADY in the cache (which
    // this visit must load) is [0, pass_lo).
    wire [15:0] kv8_pass_lo = task_cfg[52:37];
    wire [15:0] kv8_pass_hi = task_cfg[52:37]+{12'd0,rows};
    wire [11:0] kv8_ld_nw  = {1'b0,kv8_pass_lo[15:3]}+(|kv8_pass_lo[2:0] ? 12'd1 : 12'd0);
    wire [15:0] kv8_ld_blk = (kv8_pass_lo+16'd255)>>8;
    wire [3:0]  kv8_ld_nb  = {1'b0,kv8_ld_blk[5:3]}+(kv8_ld_blk[2:0]!=3'd0 ? 4'd1 : 4'd0);
    wire [15:0] kv8_dp_hi  = kv8_vis_hi-16'd1;
    wire [11:0] kv8_dp_wlo = kv8_vis_lo[15:3];
    wire [11:0] kv8_dp_whi = kv8_dp_hi[15:3];
    wire        kv8_dp_eb  = (kv8_vis_lo[7:0]==8'd0)||(kv8_vis_lo[15:8]!=kv8_dp_hi[15:8]);
    wire        kv8_dp_any = kv8_own_layer_valid && (kv8_vis_hi>kv8_vis_lo);
    wire        kv8_ld_any = (kv8_pass_lo!=16'd0);
    // first K token of a layer visit (same predicate the ownership hand-off used)
    // [16K-FIX5] the sequencer must run at the FIRST ROW OF EVERY GROUP, not only
    // at the first group of a layer.  With rows>8 the controller visits layer L as
    // [0,8) [8,16) [16,24) ... back to back, so a layer-only qualifier skipped every
    // group after the first: those rows were never dumped and the T>8 decode load
    // read unwritten image words (measured: image K/V m16 word 1 = 0 at T=10, and
    // chat /max 6 therefore decoded 9707,78,... instead of 9707,0,...).
    // A second group of the same layer is a new visit because its pass_lo moved.
    wire        kv8_ls_start = (post_state==S_WRITE) && (post_operation==0) &&
                               (!kv8_own_layer_valid || kv8_own_layer!=layer ||
                                kv8_own_lo!=kv8_pass_lo);
    // Quiet = no descriptor in flight, no pending command, no client mid-stream.
    // A read descriptor in flight would otherwise feed its beats into the image
    // loader and its own consumer would wait for them forever.
    // The DUMP only drives the write stream and can start at once.  The LOAD reads
    // through the shared read stream, so it must not begin while another read
    // descriptor is in flight (its beats would be swallowed by the image loader and
    // its own consumer would wait forever) nor while a command is pending.  The
    // wait happens in LS_OWN, where the command bus is deliberately NOT blocked, so
    // a pending fetch command can still fire and complete.
    wire        kv8_ld_quiet = !attn_dma_owner && !rd_busy && !command_hold_valid &&
                               !(state_current[S_RD_CMD] || state_current[S_RD_DATA]) &&
                               !(fetch_state==F_COMMAND) && !(fetch_state==F_DATA);
    // armed = a layer start WITH image work has not been taken yet
    wire        kv8_ls_armed = (kv8_ls_state==LS_IDLE) && kv8_ls_start &&
                               (kv8_dp_any||kv8_ld_any);
    wire        kv8_ls_idle  = (kv8_ls_state==LS_IDLE) && !kv8_ls_armed;
    assign      kv8_k_start  = (post_state==S_WRITE) && (post_operation==0) && kv8_k_ready && kv8_ls_idle;

    // Existing clients (main FSM reads, weight prefetch, hidden/context writes) keep
    // priority: the INT8 producer only borrows the command bus when it is free, so a
    // pending read command can never be starved by the producer.
    wire kv8_cmd_owned = kv8_cmd_valid && !command_hold_present && !attn_dma_owner;
    assign dma_cmd_valid=attn_dma_owner ? attn_dma_valid : (command_hold_present ? 1'b1 : kv8_cmd_valid);
    assign dma_cmd=attn_dma_owner ? attn_dma_command : (command_hold_present ? command_hold : kv8_dma_command);
    assign dma_meta=attn_dma_owner ? attn_dma_metadata : (command_hold_present ? metadata_hold : kv8_dma_metadata);
    assign dma_abort=1'b0;
    assign afu_abort=1'b0;
    assign rd_ready=attn_dma_owner ? attn_dma_rd_ready :
        (kv8_ls_state==LS_L_RD ? 1'b1 :
        ((state_current[S_RD_DATA] || fetch_state==F_DATA) &&
        ((read_kind>=READ_SCALE && read_kind<=READ_GAMMA) ? params_load_ready :
        (read_kind==READ_WEIGHT ? dequant_ready : (read_kind==READ_INPUT ? output_gemm_write_ready : 1'b1)))));
    assign wr_valid=attn_dma_owner ? attn_dma_wr_valid : (ffn_active ? ffn_write_valid :
        ((kv8_stream_valid || kv8_img_wr) ? 1'b1 : ((post_state==S_WR_SEND && post_operation>=1) || (state_current[S_H_SEND] && !hidden_dma_sent))));
    assign wr_data=attn_dma_owner ? attn_dma_wr_data : (ffn_active ? ffn_write_data : (kv8_img_wr ? kv8_img_wr_data : (kv8_stream_valid ? kv8_stream_data : write_hold)));
    assign wr_keep=attn_dma_owner ? attn_dma_wr_keep : 16'hffff;
    assign wr_last=attn_dma_owner ? attn_dma_wr_last : (ffn_active ? ffn_write_last :
        (kv8_img_wr ? kv8_img_wr_last :
        (kv8_stream_valid ? (post_state==PS_KV8_SCALE_SEND ? (kv8_img_word==4'd15) :
                             // WLAST must be the SAME expression as the state exit, otherwise
                             // the engine waits for a WLAST that never comes (owed-beat
                             // contract stated twice = the classic orphan-descriptor wedge).
                             (post_state==PS_KV8_VSEND ? ((kv8_vbyte_cnt+5'd1)==kv8_beat_lim) : (kv8_desc_left==5'd1)))
                          : ((output_phase || down_phase) ? norm_word==111 : write_word==15))));
    assign wr_meta=attn_dma_owner ? attn_dma_wr_meta : (ffn_active ? ffn_write_meta : ((kv8_stream_valid || kv8_img_wr) ? kv8_dma_metadata : write_metadata_reg));
    assign busy=!state_current[S_IDLE] && !state_current[S_DONE];
    assign qkv_done=state_current[S_DONE];assign param_ready=&params_loaded;
    assign result_commit=state_current[S_LM_COMMIT];
    assign best_token_id=lm_best_token[17:0];
    assign lm_write_valid=lm_phase && attn_p_write;
    assign lm_write_data=attn_p_write_data;
    assign lm_write_address=attn_p_address;
    assign lm_complete=lm_write_valid && attn_p_write_ready && attn_p_max_last;
    assign error=1'b0;
    // 閰嶇疆鍜屽抚韬唤鐢辫蒋浠朵繚璇侊紱Controller涓嶄繚瀛橀敊璇爜鎴栦骇鐢熷叏灞€閿欒鍙嶉銆?
    assign error_code=32'd0;
    assign afu_valid=attn_afu_owner ? attn_afu_valid : ffn_active ? ffn_afu_valid : (state_current[S_NORM_SEND] && !norm_frontend_midpoint_reg) || ((state_current[S_RES_SEND] || state_current[S_Z_SEND]) && !norm_accept_pending) || (state_current[S_H_SEND] && !hidden_afu_sent && !down_phase);
    // RoPE owns a private request/result pair; the postprocessor is its only client.
    assign rope_req_valid=post_state==S_ROPE_SEND;
    assign afu_command=attn_afu_owner ? attn_afu_command : ffn_active ? 4'd13 : (state_current[S_RES_SEND] ? 4'd14 : 4'd11);
    assign afu_command_4=attn_afu_owner ? {19'd0,attn_afu_phase} : ffn_active ? 21'd1 : state_current[S_RES_SEND] ? 21'd0 : {20'd0,norm_apply};assign afu_sum_command_buffer=0;
    assign afu_eps={task_cfg[95:80],16'd0};assign afu_latch=1'b0;assign afu_gelu_sign=1'b0;
    assign afu_x=attn_afu_owner ? attn_afu_x : ffn_active ? ffn_afu_x : norm_x;
    assign afu_y=attn_afu_owner ? attn_afu_y : ffn_active ? ffn_afu_y : norm_gamma;
    assign rope_req_x=rope_x;assign rope_req_y=rope_y;assign rope_req_meta=rope_metadata;
    assign rope_req_last=1'b1;assign rope_req_mask=8'hff;
    assign afu_meta=attn_afu_owner ? attn_afu_meta : (ffn_active ? ffn_afu_meta : afu_metadata);assign afu_last=attn_afu_owner ? attn_afu_last : ffn_active ? ffn_afu_last : norm_word==111;
    assign afu_mask=attn_afu_owner ? attn_afu_mask : 8'hff;assign afu_length=attn_afu_owner ? attn_afu_length : ffn_active ? {54'd0,ffn_service_words,3'd0} : 64'd896;
    assign afu_result_ready=attn_afu_owner ? attn_afu_result_ready : ffn_active ? ffn_result_ready : (state_norm_active && !state_current[S_NORM_READ]) ||
        state_current[S_NORM_STATS] || ((output_phase || down_phase) && state_residual_to_norm);
    assign rope_res_ready=post_state==S_ROPE_WAIT;

    // 鍦ˋFU缁撴灉鍐欏洖涓嶰-buffer绔彛浠茶涔嬮棿鍒囧紑鎺у埗璺緞锛岃礋杞介殢浣胯兘寤跺悗涓€鎷嶃€?
    reg output_afu_write_pending;
    reg [1:0] output_afu_write_bank;
    reg [2:0] output_afu_write_row;
    reg [6:0] output_afu_write_word;
    reg [127:0] output_afu_write_data;
    always @(posedge clk)begin
        if(!rst_n)output_afu_write_pending<=1'b0;
        else output_afu_write_pending<=residual_result || normalized_result || ffn_store_fire;
        // 鏃犳晥璐熻浇涓嶅弬涓庡啓鍏ワ紱鏁版嵁瀵勫瓨鍣ㄦ棤闇€澶嶄綅鎴栭澶栦娇鑳組UX銆?
        output_afu_write_bank<=ffn_store_fire ? ffn_store_bank : (residual_result ? 2'd1 : 2'd0);
        output_afu_write_row<=ffn_store_fire ? ffn_store_row : norm_row_visible;
        output_afu_write_word<=ffn_store_fire ? ffn_store_word : norm_results[6:0];
        output_afu_write_data<=ffn_store_fire ? ffn_store_data : afu_result_data;
    end

    // 姣忎釜buffer鐙珛鎺ュ彛锛岀郴缁熼樁娈礐ontroller鐩存帴浜х敓鍦板潃鍜屼娇鑳姐€?
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
        .weight_gemm_valid_pair(),
        .weight_gemm_q_pair(weight_q_pair),
        .scale_load_start(params_load_start && read_kind==READ_DESC),
        .scale_load_valid(params_load_valid && stream_checked_kind==READ_SCALE),
        .scale_load_ready(scale_load_ready),
        .scale_load_data(stream_checked_data),
        .scale_load_last(stream_checked_last),
        .scale_load_done(),
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
        .bias_load_done(),
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
        .gamma_load_done(),
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
        .attention_release_en(state_current[S_DESC_EXTRA_CHECK] || state_current[S_FINAL_BEGIN] || (attn_buffer_owner && attn_p_release)),
        .attention_full(attn_p_full),
        // [P16K] logits不再写入P：stat_only时写口只推进行argmax，数据不落盘。
        .attention_wr_en(attn_buffer_owner && attn_p_write),
        .attention_wr_address(attn_p_address),
        .attention_wr_data(attn_p_write_data),
        .attention_stat_only(lm_phase && !attn_buffer_owner),
        .attention_wr_ready(attn_p_write_ready),.attention_afu_wr_en(attn_p_afu_write),
        .attention_afu_wr_address(attn_p_afu_address),.attention_afu_wr_data(attn_p_afu_write_data),
        .attention_rd_en(attn_buffer_owner && attn_p_read),
        .attention_rd_word(attn_p_word),
        .attention_rd_valid(attn_p_valid),
        .attention_rd_tile(attn_p_data),
        .attention_afu_rd_en(attn_buffer_owner && attn_p_afu_read),
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
        .output_pair_full(),
        .output_wr_en(output_ram_write || hidden_reload_write || input_reload_write),
        .output_wr_bank((input_reload_write || input_load_active) ? 2'd1 : hidden_reload_write ? 2'd0 : down_phase ? 2'd1 : ffn_phase ? {ffn_pair,operation_decode_reg[5]} : 2'd0),
        .output_wr_row((hidden_reload_write || input_reload_write) ? stream_checked_row : raw_vector[3:1]),
        .output_wr_word((hidden_reload_write || input_reload_write) ? stream_checked_word[6:0] : output_raw_word),
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
        // [STAGE4] INT8 -> 16-bit slot packing (port-identical replacement for
        // the BF16 dequantizer; qwen_int8_to_bf16 is gone with the BF16 array)
        qwen25_weight_pack8 convert_Inst(
            .i_clk(clk),.i_rstn(rst_n),.i_valid(direct_rd_valid[direct_lane] && lane_accept),.o_ready(decoder_ready),
            .i_data(direct_rd_data[direct_lane*128+:128]),.i_word(direct_rd_word[direct_lane*10+:10]),.i_scales(2048'd0),
            .o_valid(direct_dequant_valid[direct_lane]),.i_ready(direct_dequant_ready[direct_lane]),
            .o_data(direct_dequant_data[direct_lane*256+:256]),.o_word(direct_dequant_word[direct_lane*16+:16]),.o_busy(direct_dequant_busy[direct_lane]));
    end endgenerate
    // [STAGE4] same replacement on the non-direct-read path (inactive here)
    qwen25_weight_pack8 dequant_Inst(
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
    // [STAGE4] single INT8 array for all four GEMMs (opcodes 3/4 = the two
    // attention GEMMs, 5/6/9 = projections + LM head).  Port-identical.
    qwen25_gemm_i gemm_Inst(
        .clk(clk),.rst_n(rst_n),.s_cmd_valid(m_gemm_cmd_valid),.s_cmd_ready(gemm_cmd_ready),.s_cmd(m_gemm_cmd),.s_cmd_meta(m_gemm_cmd_meta),
        .s_act_valid(m_gemm_act_valid),.s_act_ready(gemm_act_ready),.s_act_data(m_gemm_act_data),.s_act_meta(m_gemm_act_meta),
        .s_weight_valid(m_gemm_weight_valid),.s_weight_ready(gemm_weight_ready),.s_weight_data(m_gemm_weight_data),.s_weight_meta(m_gemm_weight_meta),
        .abort_req(1'b0),.soft_reset_req(1'b0),.m_valid(gemm_valid),.m_ready(m_gemm_ready),.m_bf16(gemm_bf16),.m_fp32(gemm_fp32),
        .m_meta(gemm_meta),.m_slice(gemm_slice),.m_row_mask(gemm_rows),.m_col_mask(gemm_cols),.m_last(gemm_last),
        .busy(gemm_busy),.input_done(gemm_input_done),.partial_done(gemm_partial),.done(gemm_done),.error(gemm_error),.aborted(gemm_aborted),.error_code(gemm_error_code)
    );
    end endgenerate


    qwen25_attention #(.ENABLE_LOGITS(1)) attention_Inst (        .i_clk(clk),.i_rstn(rst_n),.i_start(attn_start),.i_abort(1'b0),
        .i_logits_mode(lm_phase),.i_logits_valid(lm_phase && raw_busy && bias_valid),
        .o_logits_ready(lm_input_ready),.i_logits_address(lm_input_address),.i_logits_data(bias_data),.o_logits_busy(lm_busy),
        .i_rows(rows),.i_position(task_cfg[52:37]),.i_context(attn_context),.i_layer(layer),.i_task(task_id),
        .i_first_head(4'd0),.i_last_head(4'd13),.i_q_base(task_cfg[511:448]),.i_k_base(task_cfg[383:320]),.i_v_base(task_cfg[447:384]),
        .o_busy(attn_busy),.o_done(attn_done),.o_error(),.o_error_code(),
        .o_service_abort(),
        .o_read_valid(attn_read_valid),.i_read_ready(attn_read_ready),.o_read_address(attn_read_address),
        .o_read_base(attn_read_base),.o_read_bytes(attn_read_bytes),
        .i_data_valid(attn_data_valid),.o_data_ready(attn_data_ready),.i_data(attn_read_data),.i_memory_busy(attn_memory_busy),.i_memory_error(attn_memory_error),
        .o_context_valid(attn_context_valid),.i_context_ready(attn_context_ready),.o_context_data(attn_context_data),
        .o_context_row(attn_context_row),.o_context_head(attn_context_head),.o_context_word(attn_context_word),
        .o_weight_valid(attn_weight_valid),.i_weight_ready(weight_load_ready),.o_weight_word(attn_weight_word),.o_weight_data(attn_weight_data),
        .o_weight_commit(attn_weight_commit),.o_weight_release(attn_weight_release),.o_weight_bank(attn_weight_bank),.i_weight_full(weight_full[attn_weight_bank]),
        .o_weight_read(attn_weight_read),.o_weight_read_word(attn_weight_read_word),.i_weight_read_valid(weight_valid),.i_weight_read_data(weight_q),
        // [STAGE3] per-token K scale table read port (kv8_k_write_chain's kv8_scale_ram)
        .o_scale_rd_en(attn_scale_rd_en),.o_scale_rd_word(attn_scale_rd_word),
        .i_scale_rd_valid(attn_scale_rd_valid),.i_scale_rd_data(attn_scale_rd_data),
        // [STAGE5] per-(head,dim) V scale table read port
        .o_v_scale_word(kv8_v_rd_word),.i_v_scale_data(kv8_v_rd_data),
        // [V-PERTOK-M] per-token V mantissa read port (8 tokens per cycle)
        .o_v_m_rd_en(kv8_vm_rd_en),.o_v_m_rd_word(kv8_vm_rd_word),
        .i_v_m_rd_data(kv8_vm_rd_data),
        .o_p_write(attn_p_write),.o_p_address(attn_p_address),.o_p_data(attn_p_write_data),.o_p_read(attn_p_read),.o_p_word(attn_p_word),.i_p_valid(attn_p_valid),.i_p_data(attn_p_data),
        .i_p_write_ready(attn_p_write_ready),.o_p_afu_write(attn_p_afu_write),
        .o_p_afu_address(attn_p_afu_address),.o_p_afu_data(attn_p_afu_write_data),
        .o_p_commit(attn_p_commit),.o_p_release(attn_p_release),.i_p_full(attn_p_full),.o_p_afu_word(attn_p_afu_word),
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
    // =========================================================================
    // [STAGE5] INT8 KV producer chains.
    //
    // kv8_k_write_chain owns the scale RAM (its per-(head,token) append, the
    // per-(head,256-token block) exponent publish and the bank ownership state),
    // so the attention's K-scale read port is exposed by the chain and the
    // stage-3 "walk the bank by hand" stub is gone.
    //
    // kv8_v_write_chain turns the same 16 BF16 beats per token into the
    // token-major INT8 V image and locks the layer's per-head V scale; its 256 B
    // image is written to DDR here and bulk-loaded back into the same V table
    // before the layer's first attention (that is what makes the scale visible
    // to kv8_context_scale on the context write-back).
    // =========================================================================
    wire        attn_scale_rd_en;
    wire [11:0] attn_scale_rd_word;
    wire        attn_scale_rd_valid;
    wire [135:0] attn_scale_rd_data;        // [M16-SCALE] {eb_tab, 8 x m16}

    // [M16-SCALE] MARGIN 4 -> 8 and the 16-bit mantissa are ONE change.
    // MARGIN=9 was measured unsafe by the software regression: the d field reaches
    // 15 on a 300-token prompt and M8=255 at d=15 gives ceil(255*2^(5+9)/127) =
    // 65 796 > 65 535.  Zero saturation was seen only because d=15 and M8=255 never
    // coincided -- luck, not margin.  MARGIN=8 gives worst case
    // ceil(255*2^(5+8)/127) = 16 449 = 3.98x headroom, and costs nothing measurable:
    // scale error 0.013% -> ~0.026% while the 12-bit code's own error is 0.070%.
    // =========================================================================
    // [16K-ONE-LAYER] layer-start scale-image sequencer: DUMP then LOAD.
    //
    // The on-chip tables are one layer deep and 16384 tokens deep, so layer L's
    // per-token scales have to be persisted into, and restored from, the DDR
    // image every time L is visited:
    //   DUMP  [vis_lo, T) of the PREVIOUS layer visit  (T = this visit's first
    //         token), i.e. the words that visit produced, plus the block-exponent
    //         table when a 256-token block boundary fell inside that span;
    //   LOAD  [0, T) of THIS layer, plus its V 2^ word and its etab words.
    // Both run before the visit's first append, so a load can never clobber a
    // token the visit has just produced.  With T = 0 (the first prefill pass)
    // nothing is moved at all, which is what keeps the /max 1 prefill path
    // cycle-identical to the pre-16K build.
    //
    // The shared DMA engines are borrowed for the whole window: the main/fetch
    // command hold register is blocked while kv8_ls_state!=LS_IDLE, so image
    // read beats can never be mistaken for a weight fetch's payload.  The MMU
    // transport admits ONE write descriptor at a time (flag_write_active in
    // qwen25_mmu_dma), so every dump descriptor is followed to wr_done before
    // the next command is issued.
    // =========================================================================
    // A pass (one layer sweep over `rows` tokens) starts at task_cfg[52:37]; every
    // layer of the pass produces the SAME token range, so the range a visit will
    // produce is [pass_lo, pass_lo+rows) and the range ALREADY in the cache (which
    // this visit must load) is [0, pass_lo).
    // [16K-ONE-LAYER] the image is HEAD-STRIDED: K/V m16 put head 1 at +0x8000 and
    // the etab puts head 1 at +128, so a load descriptor never spans both heads.
    // 7 regions: K h0/h1, V h0/h1, V 2^, etab h0/h1.
    wire        kv8_rgn_k    = (kv8_ls_rgn<=4'd1);
    wire        kv8_rgn_v    = (kv8_ls_rgn==4'd2)||(kv8_ls_rgn==4'd3);
    wire        kv8_rgn_v2   = (kv8_ls_rgn==4'd4);
    wire        kv8_rgn_eb   = (kv8_ls_rgn>=4'd5);
    wire        kv8_rgn_head = kv8_ls_rgn[0];
    wire [12:0] kv8_rgn_beats= kv8_rgn_v2 ? 13'd16
                             : kv8_rgn_eb ? {9'd0,kv8_ld_nb}
                             :              {1'b0,kv8_ld_nw};
    // the NEXT region's descriptor, set up while the current one streams
    wire [3:0]  kv8_rgn_n    = kv8_ls_rgn+4'd1;
    wire        kv8_n_k      = (kv8_rgn_n<=4'd1);
    wire        kv8_n_v      = (kv8_rgn_n==4'd2)||(kv8_rgn_n==4'd3);
    wire        kv8_n_v2     = (kv8_rgn_n==4'd4);
    wire        kv8_n_eb     = (kv8_rgn_n>=4'd5);
    wire        kv8_n_head   = kv8_rgn_n[0];
    wire [63:0] kv8_n_addr   = kv8_n_k  ? (kv8_img_k_base+(kv8_n_head ? 64'h0000_8000 : 64'd0))
                             : kv8_n_v  ? (kv8_img_v_base+(kv8_n_head ? 64'h0000_8000 : 64'd0))
                             : kv8_n_v2 ? kv8_img_v2_base
                             :            (kv8_img_eb_base+(kv8_n_head ? 64'd128 : 64'd0));
    wire [23:0] kv8_n_bytes  = kv8_n_v2 ? 24'd256
                             : kv8_n_eb ? {15'd0,kv8_ld_nb,4'd0}
                             :            {7'd0,kv8_ld_nw,4'd0};
    // the ownership hand-off pulse: pre-16K timing when there is nothing to move
    assign      kv8_ls_own_go = (kv8_ls_state==LS_OWN) ||
                                ((kv8_ls_state==LS_IDLE)&&kv8_ls_start&&!(kv8_dp_any||kv8_ld_any));
    // ---- cells the sequencer reads back ------------------------------------
    wire [127:0] kv8_etab_rd_data;                 // 16 block exponents of one word
    wire [127:0] kv8_dp_k_word = attn_scale_rd_data[127:0];   // K m16 word
    wire [127:0] kv8_dp_v_word = kv8_vm_rd_data;              // V m16 word
    wire [63:0]  kv8_eb_b0 = kv8_etab_rd_data[63:0];
    wire [63:0]  kv8_eb_b1 = kv8_etab_rd_data[127:64];
    // [16K-FIX4] the image beat is a byte stream: entry n (image offset 2n..2n+1,
    // low byte first) is block n, and kv8_eb_pk below unpacks it in exactly that
    // order.  The concatenation used to run the other way, so block 0 was written
    // at image offset 14 and the load mapped block 7 into on-chip block 0 -- the
    // measured symptom was "decode layer 0 head 0 reads eb=0x00 instead of 0x73".
    wire [127:0] kv8_eb_lo = {{8'h00,kv8_eb_b0[63:56]},{8'h00,kv8_eb_b0[55:48]},{8'h00,kv8_eb_b0[47:40]},{8'h00,kv8_eb_b0[39:32]},
                              {8'h00,kv8_eb_b0[31:24]},{8'h00,kv8_eb_b0[23:16]},{8'h00,kv8_eb_b0[15:8]},{8'h00,kv8_eb_b0[7:0]}};
    wire [127:0] kv8_eb_hi = {{8'h00,kv8_eb_b1[63:56]},{8'h00,kv8_eb_b1[55:48]},{8'h00,kv8_eb_b1[47:40]},{8'h00,kv8_eb_b1[39:32]},
                              {8'h00,kv8_eb_b1[31:24]},{8'h00,kv8_eb_b1[23:16]},{8'h00,kv8_eb_b1[15:8]},{8'h00,kv8_eb_b1[7:0]}};
    // DDR etab image beat = 8 x 16-bit entries {8'h00, eb}: keep the low bytes
    wire [63:0]  kv8_eb_pk = {rd_data[112+:8],rd_data[96+:8],rd_data[80+:8],rd_data[64+:8],
                              rd_data[48+:8],rd_data[32+:8],rd_data[16+:8],rd_data[0+:8]};
    // ---- sequencer command fields (own mux, so kv8_cmd_* stay untouched) ----
    assign       kv8_ls_cmd_sel = (kv8_ls_state==LS_D_CMD)||(kv8_ls_state==LS_D_EBC)||(kv8_ls_state==LS_L_CMD);
    // dump item (K0,K1,V0,V1) address, and the address of the next item.
    // [16K-ONE-LAYER] the DUMP targets the layer that PRODUCED the range
    // (kv8_vis_layer), the LOAD targets the layer being visited -- using one base
    // for both would write layer 23's slice into layer 0's image.
    wire [1:0]   kv8_ls_item_n  = kv8_ls_item+2'd1;
    wire [63:0]  kv8_dp_lbase   = KV8_IMG_BASE+{39'd0,kv8_vis_layer,20'd0};
    wire [63:0]  kv8_dp_k_base  = kv8_dp_lbase+KV8_IMG_K;
    wire [63:0]  kv8_dp_v_base  = kv8_dp_lbase+KV8_IMG_V;
    wire [63:0]  kv8_dp_eb_base = kv8_dp_lbase+KV8_IMG_EB;
    wire [63:0]  kv8_dp_nxt_addr= ((kv8_ls_item_n[1]==1'b0) ? kv8_dp_k_base : kv8_dp_v_base)
                                  + (kv8_ls_item_n[0] ? 64'h0000_8000 : 64'd0)
                                  + {52'd0,kv8_ls_w,4'd0};
    // ---- load-port beat pointers ------------------------------------------
    wire         kv8_img_ld_valid = (kv8_ls_state==LS_L_RD)&&kv8_rgn_k&&rd_valid;
    wire [11:0]  kv8_img_ld_word  = {kv8_rgn_head,kv8_ls_cnt[10:0]};
    wire [127:0] kv8_img_ld_data  = rd_data;
    wire         kv8_img_vld_valid= (kv8_ls_state==LS_L_RD)&&kv8_rgn_v&&rd_valid;
    wire [11:0]  kv8_img_vld_word = {kv8_rgn_head,kv8_ls_cnt[10:0]};
    wire [127:0] kv8_img_vld_data = rd_data;
    wire         kv8_img_v2_we    = (kv8_ls_state==LS_L_RD)&&kv8_rgn_v2&&rd_valid;
    wire [3:0]   kv8_img_v2_word  = kv8_ls_cnt[3:0];
    wire         kv8_img_eb_valid = (kv8_ls_state==LS_L_RD)&&kv8_rgn_eb&&rd_valid;
    // [16K-FIX2] on-chip etab word = {head, block[5:4]}: 4 words per head, so the
    // word index is {head, cnt[2:1]}.  It used to be {head, cnt[3:1]}, a FOUR-bit
    // concatenation truncated into the 3-bit port -- the head bit was dropped, so
    // the head-1 etab image beat was written over head 0's word 0 and head 1's
    // word 4 kept the previous layer's exponent (measured: head 0 read back 0x00,
    // head 1 0x6e = layer 23's value, instead of 0x73/0x72).
    wire [2:0]   kv8_img_eb_word  = {kv8_rgn_head,kv8_ls_cnt[2:1]};
    wire [15:0]  kv8_img_eb_be    = kv8_ls_cnt[0] ? 16'hff00 : 16'h00ff;
    wire [127:0] kv8_img_eb_data  = {2{kv8_eb_pk}};
    wire         kv8_img_rd_en    = (kv8_ls_state==LS_D_CMD)&&(kv8_ls_item[1]==1'b0);
    wire [11:0]  kv8_img_rd_word  = {kv8_ls_item[0],kv8_ls_w};
    wire         kv8_img_vrd_en   = (kv8_ls_state==LS_D_CMD)&&(kv8_ls_item[1]==1'b1);
    wire [11:0]  kv8_img_vrd_word = {kv8_ls_item[0],kv8_ls_w};
    // [16K-FIX] The dump read ADDRESS must stay selected until the write beat has
    // taken its data, not only for the single cycle the read is issued.
    // kv8_scale_ram's read data ports are shift registers that follow the address
    // input EVERY cycle (rd_addr_1 for K m16/etab, vmq for V m16); only the valid
    // is gated by the read enable.  That is correct for the attention because
    // kv8_i_scale_fifo issues its 16 words back-to-back, but it means a one-cycle
    // select hands the beat whatever address the ATTENTION left behind -- for a
    // 1-token prompt that is head 1's word 0, i.e. head 0 was dumped with head 1's
    // value.  Holding the select over CMD+WAIT+BEAT keeps the port pointing at the
    // dump word for the whole 3-cycle read latency.
    wire         kv8_img_win      = (kv8_ls_state==LS_D_CMD)||(kv8_ls_state==LS_D_WAIT)||
                                    (kv8_ls_state==LS_D_BEAT);
    wire         kv8_img_k_sel    = kv8_img_win && (kv8_ls_item[1]==1'b0);
    wire         kv8_img_v_sel    = kv8_img_win && (kv8_ls_item[1]==1'b1);
    wire [2:0]   kv8_img_ebrd_word= kv8_ls_i[2:0];
    // ---- write side: one 16 B beat per dump item, 16 beats for the etab -----
    assign       kv8_img_wr       = (kv8_ls_state==LS_D_BEAT) ||
                                    ((kv8_ls_state==LS_D_EBB)&&((kv8_ls_phase==2'd1)||(kv8_ls_phase==2'd2)));
    assign       kv8_img_wr_data  = (kv8_ls_state==LS_D_EBB)
                                    ? ((kv8_ls_phase==2'd1) ? kv8_eb_lo : kv8_eb_hi)
                                    : (kv8_ls_item[1] ? kv8_dp_v_word : kv8_dp_k_word);
    assign       kv8_img_wr_last  = (kv8_ls_state==LS_D_BEAT) ||
                                    ((kv8_ls_state==LS_D_EBB)&&(kv8_ls_phase==2'd2)&&(kv8_ls_i==4'd7));

    always @(posedge clk)begin
        if(!rst_n)begin
            kv8_ls_state<=LS_IDLE; kv8_ls_rgn<=4'd0; kv8_ls_item<=2'd0; kv8_ls_phase<=2'd0;
            kv8_ls_cnt<=13'd0; kv8_ls_w<=11'd0; kv8_ls_i<=4'd0;
            kv8_ls_eb_dirty<=1'b0;
            kv8_ls_cmd_addr<=64'd0; kv8_ls_cmd_bytes<=24'd0; kv8_ls_cmd_wr<=1'b1;
        end else begin
            case(kv8_ls_state)
                // ------------------------------------------------ start
                LS_IDLE:begin
                    if(kv8_ls_armed)begin
                        kv8_ls_item<=2'd0; kv8_ls_phase<=2'd0; kv8_ls_i<=4'd0;
                        kv8_ls_eb_dirty<=kv8_dp_eb;
                        if(kv8_dp_any)begin
                            kv8_ls_w<=kv8_dp_wlo;
                            kv8_ls_cmd_addr<=kv8_dp_k_base+{52'd0,kv8_dp_wlo,4'd0};
                            kv8_ls_cmd_bytes<=24'd16; kv8_ls_cmd_wr<=1'b1;
                            kv8_ls_state<=LS_D_CMD;
                        end else begin
                            kv8_ls_state<=LS_OWN;
                        end
                    end
                end
                // ---- DUMP: 4 single-beat items: K h0/h1 then V h0/h1 ------
                LS_D_CMD:begin
                    if(kv8_cmd_fire)begin
                        kv8_ls_cnt<=13'd0;
                        kv8_ls_state<=LS_D_WAIT;
                    end
                end
                LS_D_WAIT:begin
                    kv8_ls_cnt<=kv8_ls_cnt+13'd1;
                    if(kv8_ls_cnt==13'd2) kv8_ls_state<=LS_D_BEAT;
                end
                LS_D_BEAT:begin
                    if(wr_ready) kv8_ls_state<=LS_D_WR;   // beat taken; wait for B
                end
                LS_D_WR:begin
                    // one write descriptor at a time: the transport refuses a new
                    // write command while flag_write_active is set.
                    if(wr_done)begin
                        if(kv8_ls_item==2'd3)begin
                            if(kv8_ls_w<kv8_dp_whi)begin
                                kv8_ls_w<=kv8_ls_w+11'd1;
                                kv8_ls_item<=2'd0; kv8_ls_phase<=2'd0;
                                kv8_ls_cmd_addr<=kv8_dp_k_base+{52'd0,kv8_ls_w+11'd1,4'd0};
                                kv8_ls_cmd_bytes<=24'd16; kv8_ls_cmd_wr<=1'b1;
                                kv8_ls_state<=LS_D_CMD;
                            end else begin
                                kv8_ls_phase<=2'd0; kv8_ls_i<=4'd0;
                                if(kv8_ls_eb_dirty)begin
                                    kv8_ls_cmd_addr<=kv8_dp_eb_base;
                                    kv8_ls_cmd_bytes<=24'd256; kv8_ls_cmd_wr<=1'b1;
                                    kv8_ls_state<=LS_D_EBC;
                                end else begin
                                    kv8_ls_state<=LS_OWN;
                                end
                            end
                        end else begin
                            kv8_ls_item<=kv8_ls_item+2'd1;
                            kv8_ls_cmd_addr<=kv8_dp_nxt_addr;
                            kv8_ls_cmd_bytes<=24'd16; kv8_ls_cmd_wr<=1'b1;
                            kv8_ls_state<=LS_D_CMD;
                        end
                    end
                end
                // ---- DUMP the block-exponent table: one 256 B descriptor ---
                LS_D_EBC:begin
                    if(kv8_cmd_fire)begin
                        kv8_ls_phase<=2'd0; kv8_ls_i<=4'd0;
                        kv8_ls_state<=LS_D_EBB;
                    end
                end
                LS_D_EBB:begin
                    // 2 beats per on-chip word; every beat must be ACCEPTED
                    // (wr_ready) or the 256 B descriptor would owe a beat forever
                    if(kv8_ls_phase==2'd0) kv8_ls_phase<=2'd1;
                    else if(kv8_ls_phase==2'd1) begin
                        if(wr_ready) kv8_ls_phase<=2'd2;
                    end else begin
                        if(wr_ready) begin
                            kv8_ls_phase<=2'd0;
                            if(kv8_ls_i==4'd7) kv8_ls_state<=LS_D_EBW;
                            else kv8_ls_i<=kv8_ls_i+4'd1;
                        end
                    end
                end
                LS_D_EBW:begin
                    if(wr_done) kv8_ls_state<=LS_OWN;
                end
                // ---- ownership hand-off (release old bank, acquire new) ---
                LS_OWN:begin
                    if(kv8_ls_phase==2'd0)begin
                        kv8_ls_phase<=2'd1;
                    end else if(!kv8_ld_any)begin
                        kv8_ls_phase<=2'd0;
                        kv8_ls_state<=LS_FIN;
                    end else if(kv8_ld_quiet)begin
                        kv8_ls_phase<=2'd0; kv8_ls_cnt<=13'd0; kv8_ls_rgn<=4'd0;
                        kv8_ls_cmd_addr<=kv8_img_k_base;
                        kv8_ls_cmd_bytes<={7'd0,kv8_ld_nw,4'd0};
                        kv8_ls_cmd_wr<=1'b0;
                        kv8_ls_state<=LS_L_CMD;
                    end
                    // else: stay here until the read stream is free
                end
                // ---- LOAD: 4 read descriptors (K m16, V m16, V 2^, etab) --
                LS_L_CMD:begin
                    if(kv8_cmd_fire)begin
                        kv8_ls_cnt<=13'd0;
                        kv8_ls_state<=LS_L_RD;
                    end
                end
                LS_L_RD:begin
                    if(rd_valid)begin
                        kv8_ls_cnt<=kv8_ls_cnt+13'd1;
                        if(kv8_ls_cnt+13'd1==kv8_rgn_beats) kv8_ls_state<=LS_L_NEXT;
                    end
                end
                LS_L_NEXT:begin
                    kv8_ls_rgn<=kv8_ls_rgn+4'd1;
                    kv8_ls_cmd_addr<=kv8_n_addr;
                    kv8_ls_cmd_bytes<=kv8_n_bytes;
                    kv8_ls_cmd_wr<=1'b0;
                    if(kv8_ls_rgn>=4'd6) kv8_ls_state<=LS_FIN;
                    else kv8_ls_state<=LS_L_CMD;
                end
                LS_FIN:begin
                    kv8_ls_state<=LS_IDLE;
                end
                default:kv8_ls_state<=LS_IDLE;
            endcase
        end
    end
    kv8_k_write_chain #(.CONTEXT_LOG2(14),.LAYERS(24),.MARGIN(8),.BLOCK_LOG2(8),.TOKEN_MAJOR(1))
    kv8_k_chain_Inst (
        .clk(clk),.rst_n(rst_n),
        .i_start(kv8_k_start),.i_layer(kv8_k_layer),.i_token(kv8_k_token),
        .i_k_base(task_cfg[383:320]+{36'd0,layer,23'd0}),
        .i_k_scale_base(kv8_scale_base),.i_v_scale_base(kv8_scale_base),
        .o_ready(kv8_k_ready),
        .i_k_valid(kv8_k_valid),.o_k_ready(kv8_k_vready),.i_k_data(kv8_k_data),
        .o_busy(kv8_k_busy),.o_done(),
        .i_flush(kv8_k_flush),.o_flush_done(kv8_k_flush_done),
        .o_beat_valid(kv8_k_beat_valid),.i_beat_ready(kv8_k_beat_ready),
        .o_beat_address(kv8_k_beat_addr),.o_beat_data(kv8_k_beat_data),
        .o_beat_tag(),.o_beat_desc(),
        .o_beat_len(kv8_k_beat_len),.o_beat_token(),
        .i_bank_wr(1'b0),.i_bank_rd(1'b0),
        .i_wr_acquire(kv8_wr_acquire),.i_wr_release(kv8_wr_release),.i_rd_release(kv8_rd_release),
        .o_wr_grant(),.o_rd_grant(kv8_rd_grant),.o_owner_error(),
        .i_etab_valid(kv8_img_eb_valid),.i_etab_word(kv8_img_eb_word),.i_etab_data(kv8_img_eb_data),.i_etab_be(kv8_img_eb_be),
        .i_etab_rd_word(kv8_img_ebrd_word),.o_etab_rd_data(kv8_etab_rd_data),
        .i_load_valid(kv8_img_ld_valid),.i_load_word(kv8_img_ld_word),.i_load_mant(kv8_img_ld_data),
        .i_vload_valid(kv8_img_vld_valid),.i_vload_word(kv8_img_vld_word),.i_vload_data(kv8_img_vld_data),
        .i_rd_en(attn_scale_rd_en|kv8_img_rd_en),.i_rd_word(kv8_img_k_sel ? kv8_img_rd_word : attn_scale_rd_word),
        .o_rd_valid(attn_scale_rd_valid),.o_rd_scale(attn_scale_rd_data),
        .i_v_we(kv8_img_v2_we | kv8_v_we_chain | kv8_v_scale_we),
        .i_v_word(kv8_img_v2_we ? kv8_img_v2_word : (kv8_v_we_chain ? kv8_v_word_chain : kv8_v_scale_word)),
        .i_v_data(kv8_img_v2_we ? rd_data : (kv8_v_we_chain ? kv8_v_data_chain : kv8_v_scale_data)),
        .i_v_rd_word(kv8_v_rd_word),.o_v_rd_data(kv8_v_rd_data),
        // [V-PERTOK-M] V per-token mantissa table: the V producer appends one
        // entry per (token, head), the PV stage reads 8 tokens at a time.
        .i_vt_app_valid(kv8_v_m_app_valid),.o_vt_app_ready(kv8_v_m_app_ready),
        .i_vt_app_token(kv8_v_m_app_token),.i_vt_app_head(kv8_v_m_app_head),
        .i_vt_app_mant(kv8_v_m_app_mant),.o_vt_app_taken(),
        .i_vt_rd_en(kv8_vm_rd_en|kv8_img_vrd_en),.i_vt_rd_word(kv8_img_v_sel ? kv8_img_vrd_word : kv8_vm_rd_word),
        .o_vt_rd_mant(kv8_vm_rd_data),.o_vt_rd_valid(),
        .o_app_conflict(),.o_clip_sticky(kv8_k_clip_sticky),
        .o_stat_beat(),.o_stat_app(),.o_dbg_state(),.o_dbg_lock_v(),
        .o_clip_count(kv8_k_clip_count),.o_blk_tok(kv8_k_blk_tok),.o_token_beats(kv8_k_token_beats),
        .o_err_seq()
    );

    assign kv8_k_layer = layer;
    assign kv8_k_token = kv8_token_now;

    kv8_v_write_chain #(.MARGIN(2))
    kv8_v_chain_Inst (
        .clk(clk),.rst_n(rst_n),
        .i_start(kv8_v_start),.i_layer(layer),.i_token(kv8_token_now),
        .i_v_base(kv8_v_layer_base),.o_ready(kv8_v_ready),
        .i_v_valid(kv8_v_valid),.o_v_ready(kv8_v_vready),.i_v_data(kv8_v_data),
        .o_busy(),.o_done(),
        .o_beat_valid(kv8_v_beat_valid),.i_beat_ready(kv8_v_beat_ready),
        .o_beat_address(kv8_v_beat_addr),.o_beat_data(kv8_v_beat_data),
        .o_beat_last(),
        .o_scale_valid(kv8_v_scale_valid),.o_scale_bf16(kv8_v_scale_bf16),
        .i_scale_clear(kv8_v_scale_clear),
        .o_v_we(kv8_v_we_chain),.o_v_word(kv8_v_word_chain),.o_v_data(kv8_v_data_chain),
        // [V-PERTOK-M] per-token mantissa append into the scale RAM's V table
        .o_m_app_valid(kv8_v_m_app_valid),.i_m_app_ready(kv8_v_m_app_ready),
        .o_m_app_token(kv8_v_m_app_token),.o_m_app_head(kv8_v_m_app_head),
        .o_m_app_mant(kv8_v_m_app_mant),
        .o_clip_sticky(kv8_v_clip_sticky),.o_clip_count(kv8_v_clip_count),
        .o_clip_codes(kv8_v_clip_codes),.o_m_max(kv8_v_m_max),
        .o_token_beats(kv8_v_token_beats),.o_dbg_state()
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

    // 鍏冩暟鎹粎鐢ㄤ簬DMA/绠楀瓙韬唤缁戝畾锛孯AM鏃犻渶姣忔媿metadata姣旇緝銆?
    always @(*) begin
        gemm_command=0;gemm_command[3:0]=compute_opcode;
        gemm_command[15:4]=global_k;gemm_command[16]=global_k==0;gemm_command[17]=global_k==last_k;
        gemm_command[21:18]=compute_rows;gemm_command[29:22]=128;gemm_command[33:30]=8;
        afu_metadata=0;afu_metadata[15:0]=task_id;afu_metadata[20:16]=layer;
        afu_metadata[56:53]=rows;afu_metadata[60:57]=8;
        afu_metadata[52:37]={9'd0,norm_word};afu_metadata[83:80]=(output_phase || down_phase) && state_before_hidden ? 14 : 11;
        afu_metadata[104:89]=task_cfg[52:37]+norm_row_visible;
        rope_metadata=0;rope_metadata[15:0]=task_id;rope_metadata[20:16]=layer;
        rope_metadata[56:53]=rows;rope_metadata[60:57]=8;
        rope_metadata[52:37]={12'd0,rope_first_word};rope_metadata[83:80]=12;
        rope_metadata[104:89]=task_cfg[52:37]+post_row;
        rope_metadata[118:109]={3'd0,post_output_tile,rope_first_word};
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


    // 鍚勯樁娈靛彧绛夊緟鐪熷疄DMA/AFU/GEMM valid銆乨one鍜孊鍝嶅簲銆?
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
            // [STAGE5-FIX] a handshake, not a timing assumption: S_ATTN_BEGIN is
            // held until kv8_scale_ram reports READY (kv8_rd_grant).
            state_current[S_ATTN_BEGIN]:begin state_next=kv8_scale_wait ? HOT_S_ATTN_BEGIN : HOT_S_ATTN_RUN;end
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
        if(!rst_n)begin
            weight_chunks_ready[0]<=0;weight_chunks_ready[1]<=0;
        end else if(state_current[S_IDLE])begin
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
            if(!rst_n)begin
                weight_chunk_mask[0][chunk_lane]<=0;weight_chunk_mask[1][chunk_lane]<=0;
            end else if(state_current[S_IDLE] || state_current[S_O_RESTORE] || state_current[S_FFN_BEGIN] || state_current[S_DOWN_BEGIN] || state_current[S_LM_BEGIN])begin
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
    // Controller涓墍鏈夊湴鍧€璁℃暟渚濇嵁鐪熷疄浼犺緭鎺ュ彈浜嬩欢鎺ㄨ繘銆?
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
            if(post_state==S_ROPE_WAIT && rope_res_valid)begin
                rope_first<=rope_res_data;rope_second<=rope_res_aux;rope_write_half<=0;
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
            // [STAGE5-FIX3] op 0 and 1 both push into a producer (kv8_push_ready is muxed)
            if(post_state==S_WR_SEND && (post_operation<=1 ? kv8_push_ready : wr_ready) && write_word!=15)begin write_word<=write_word+5'd1;end
            // [STAGE5-FIX3] op 1 has no DDR write response either (its beats go to the
            // V producer), so the row walk must not wait for one.
            if(post_state==S_WR_FINISH && (wr_done || post_operation<=1))begin row_index<=row_index==rows_last ? 0 : row_index+3'd1;end
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
        if(!rst_n)lm_finished<=0;
        else if(state_current[S_FINAL_BEGIN])lm_finished<=0;
        else if(lm_complete)lm_finished<=1;
    end

    // Hold a complete DMA command until acceptance, even if the other producer changes.
    always @(posedge clk) begin
        if(!rst_n)begin
            command_hold_valid<=0;command_hold_write<=0;command_hold<=0;metadata_hold<=0;
        end else begin
            if(command_hold_valid)begin
                if(command_hold_taken)begin command_hold_valid<=0;end
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
        if(!rst_n)begin
            fetch_tile<=0;fetch_enabled<=0;fetch_bank<=0;fetch_half<=0;fetch_operation<=0;fetch_output_tile<=0;fetch_scale_word<=0;fetch_scale_ready<=0;
        end else if(state_current[S_IDLE] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || (state_current[S_DONE] && start))begin
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
            S_ROPE_SEND:begin if(rope_req_ready)begin post_next=S_ROPE_WAIT;end end
            S_ROPE_WAIT:begin if(rope_res_valid)begin post_next=S_ROPE_WRITE;end end
            S_ROPE_WRITE:begin
                if(rope_write_half)begin
                    if(rope_pair==7 && post_row==rows_last)begin post_next=S_WRITE;end
                    else begin post_next=S_ROPE_FIRST;end
                end
            end
            S_WRITE:begin
                // [STAGE5] op 0/1 no longer write DDR from here: the token's 16 BF16
                // beats are pushed into the INT8 producer and the producer's own
                // beats are written later (PS_KV8_*).  Wait for it to be idle first.
                // [STAGE5-FIX3] wait for the producer that will consume this token
                // [16K-ONE-LAYER] at a layer's first K token the scale-image
                // sequencer (dump previous layer + load this layer) runs first and
                // holds the producer off; with nothing to move it is idle on the
                // same cycle, so the /max 1 prefill path keeps its exact cycle count.
                if(post_operation==0)begin post_next=(kv8_k_ready && kv8_ls_idle) ? S_WR_READ : S_WRITE; end
                else if(post_operation==1)begin post_next=kv8_v_ready ? S_WR_READ : S_WRITE; end
                else begin post_next=S_WR_CMD; end
            end
            S_WR_CMD:begin if(write_command_fire)begin post_next=S_WR_READ;end end
            S_WR_READ:begin post_next=S_WR_WAIT_DATA;end
            S_WR_WAIT_DATA:begin if(p1_valid)begin post_next=S_WR_SEND;end end
            S_WR_SEND:begin
                // [STAGE5-FIX3] op 0 and op 1 both feed a producer: 16 beats/token
                if(post_operation<=1)begin
                    if(kv8_push_ready)begin post_next = write_word==15 ? PS_KV8_TAIL : S_WR_READ; end
                end else if(wr_ready)begin post_next=write_word==15 ? S_WR_FINISH : S_WR_READ;end
            end
            S_WR_FINISH:begin
                if(post_operation==0)begin
                    // the token's 128 B run is already on the bus (one command/token)
                    post_next = row_index==rows_last ? S_NEXT : S_WRITE;
                end else if(post_operation==1)begin
                    // [STAGE5-FIX3] same shape as K: the V run is already on the bus
                    post_next = row_index==rows_last ? S_NEXT : S_WRITE;
                end else if(wr_done)begin
                    if(row_index!=rows_last)begin post_next=S_WRITE; end
                    else begin post_next=S_NEXT; end
                end
            end
            // ---------------- [STAGE5] INT8 producer drain ----------------------
            PS_KV8_TAIL:begin
                if(post_operation==0)begin
                    if(kv8_k_beat_valid)begin post_next=PS_KV8_VCMD; end
                end else begin
                    if(kv8_v_scale_valid && !kv8_v_scale_written)begin post_next=PS_KV8_SCALE; end
                    else if(kv8_v_beat_valid)begin post_next=PS_KV8_VCMD; end
                end
            end
            PS_KV8_EMIT:begin
                if(kv8_k_beat_valid)begin post_next=PS_KV8_CMD; end
                // do not leave the producer while the write engine still owns a
                // descriptor: the next client (weight prefetch / tensor write) shares it
                else if(kv8_k_busy || kv8_flush_pending || wr_busy)begin post_next=PS_KV8_EMIT; end
                else begin post_next=S_WR_FINISH; end
            end
            PS_KV8_CMD:begin if(kv8_cmd_fire)begin post_next=PS_KV8_SEND; end end
            PS_KV8_SEND:begin
                if(wr_done || (kv8_stream_valid && wr_ready && kv8_desc_left==5'd1))begin post_next=PS_KV8_EMIT; end
            end
            PS_KV8_FLUSH:begin post_next=PS_KV8_EMIT; end
            PS_KV8_SCALE:begin if(kv8_cmd_fire)begin post_next=PS_KV8_SCALE_SEND; end end
            PS_KV8_SCALE_SEND:begin
                if(wr_ready && kv8_img_word==4'd15)begin post_next=PS_KV8_TAIL; end
            end
            PS_KV8_VCMD:begin if(kv8_cmd_fire)begin post_next=PS_KV8_VSEND; end end
            PS_KV8_VSEND:begin
                // owed-beat exit, plus wr_done: a mismatch must never wedge the whole run
                if(wr_done || (kv8_stream_valid && wr_ready && (kv8_vbyte_cnt+4'd1==kv8_beat_lim)))begin post_next=S_WR_FINISH; end
            end
            PS_KV8_VLOAD:begin post_next=PS_KV8_VLOAD_CMD; end
            PS_KV8_VLOAD_CMD:begin if(kv8_cmd_fire)begin post_next=PS_KV8_VRD; end end
            PS_KV8_VRD:begin if(rd_done)begin post_next=S_WR_FINISH; end end
            S_NEXT:begin post_next=S_IDLE;end
            default:begin post_next=S_IDLE;end
        endcase
    end
    always @(posedge clk)begin
        if(!rst_n)begin post_state<=S_IDLE;end else begin post_state<=post_next;end
    end

    // =========================================================================
    // [STAGE5] INT8 KV producer sequencing: start pulses, descriptor latch,
    // beat counters, per-layer V scale image / bulk load, scale-RAM ownership.
    //
    // The producers keep the DDR layout the attention already reads:
    //   K : k_base + layer*8MiB + token*256 + head*128 + dg*16   (K16, 16 beats/token)
    //   V : v_base + layer*8MiB + token*128 + head*64 + dg*16
    // The V scale image lives inside the layer's own 64 KiB window at +0x8000
    // (the E2E DDR map gives exactly 64 KiB per layer at an 8 MiB stride).
    // =========================================================================
    always @(posedge clk)begin
        if(!rst_n)begin
            kv8_cmd_addr<=64'd0;kv8_cmd_bytes<=24'd0;kv8_cmd_is_write<=1'b1;
            kv8_desc_left<=5'd0;kv8_img_word<=4'd0;kv8_vbyte_cnt<=5'd0;kv8_vrd_count<=4'd0;
            kv8_flush_pending<=1'b0;kv8_bank_state<=2'd0;kv8_acq_pending<=1'b0;
            kv8_wr_acquire<=1'b0;kv8_wr_release<=1'b0;kv8_rd_release<=1'b0;
            kv8_own_layer_valid<=1'b0;kv8_own_layer<=5'd0;
            kv8_vis_lo<=16'd0;kv8_own_lo<=16'd0;
            kv8_v_scale_written<=1'b0;kv8_v_table_loaded<=1'b0;kv8_v_scale_clear<=1'b0;
            kv8_v_scale_we<=1'b0;kv8_v_scale_word<=4'd0;kv8_v_scale_data<=128'd0;
            kv8_dma_conflict<=1'b0;kv8_vone_cnt<=5'd0;
        end else begin
            kv8_wr_acquire<=1'b0;kv8_wr_release<=1'b0;kv8_rd_release<=1'b0;
            kv8_v_scale_we<=1'b0;kv8_v_scale_clear<=1'b0;
            // ---- scale-RAM ownership ------------------------------------
            // A new layer owns the (single) bank from its first K token and the
            // bank is handed to the attention (READY) only after the V table has
            // been populated.  The next layer first drops the old one to EMPTY.
            // [16K-ONE-LAYER] the hand-off is driven by the image sequencer:
            // kv8_ls_own_go is the no-work fast path (identical timing to the
            // pre-16K build) or the sequenced path after the image dump.
            if(kv8_ls_own_go)begin
                kv8_vone_cnt<=5'd16;
                kv8_own_layer<=layer;kv8_own_layer_valid<=1'b1;
                kv8_v_scale_written<=1'b0;kv8_v_table_loaded<=1'b0;
                if(kv8_bank_state==2'd2)begin
                    kv8_rd_release<=1'b1;kv8_bank_state<=2'd0;kv8_acq_pending<=1'b1;
                end else if(kv8_bank_state==2'd0)begin
                    kv8_wr_acquire<=1'b1;kv8_bank_state<=2'd1;
                end
                kv8_vis_lo<=kv8_pass_lo;
                kv8_vis_hi<=kv8_pass_hi;
                kv8_vis_layer<=layer;
                kv8_own_lo<=kv8_pass_lo;
            end
            if(kv8_acq_pending)begin kv8_wr_acquire<=1'b1;kv8_acq_pending<=1'b0;kv8_bank_state<=2'd1; end
            // [STAGE5-FIX] the layer's K/V production is over and the attention is
            // asking for the scale table: hand the bank over (LOADING -> READY).
            // One pulse per layer; the state bit makes it self-terminating.
            if(kv8_bank_state==2'd1 && state_current[S_ATTN_BEGIN])begin
                kv8_wr_release<=1'b1;kv8_bank_state<=2'd2;
            end
            // ---- K descriptor latch: one 256 B (or shorter) descriptor/command
            if(post_state==PS_KV8_EMIT && kv8_k_beat_valid)begin
                kv8_cmd_addr<=kv8_k_beat_addr;
                kv8_cmd_bytes<={kv8_k_beat_len,4'b0000};
                kv8_cmd_is_write<=1'b1;
                kv8_desc_left<=kv8_k_beat_len;
            end
            if(post_state==PS_KV8_SEND && kv8_stream_valid && wr_ready)
                kv8_desc_left<=kv8_desc_left-5'd1;
            // ---- K flush bookkeeping
            if(post_state==PS_KV8_FLUSH && kv8_k_blk_tok!=5'd0) kv8_flush_pending<=1'b1;
            if(kv8_k_flush_done) kv8_flush_pending<=1'b0;
            // ---- V scale image (256 B) write, once per layer
            if(post_state==PS_KV8_TAIL && post_operation==1 && kv8_v_scale_valid && !kv8_v_scale_written)begin
                kv8_cmd_addr<=kv8_scale_base;
                kv8_cmd_bytes<=24'd256;
                kv8_cmd_is_write<=1'b1;
                kv8_img_word<=4'd0;
            end
            if(post_state==PS_KV8_SCALE_SEND && wr_ready)begin
                if(kv8_img_word==4'd15)begin
                    kv8_v_scale_written<=1'b1;
                    kv8_v_scale_clear<=1'b1;
                end else begin
                    kv8_img_word<=kv8_img_word+4'd1;
                end
            end
            // ---- one 128 B command per V token
            // one 128 B command per token: K (token-major) and V use the same shape
            if(post_state==PS_KV8_TAIL &&
               ((post_operation==0 && kv8_k_beat_valid) ||
                (post_operation==1 && !(kv8_v_scale_valid && !kv8_v_scale_written) && kv8_v_beat_valid)))begin
                // the length comes from the producer's own beat count, never from a constant
                kv8_cmd_addr<=(post_operation==0) ? kv8_k_beat_addr : kv8_v_beat_addr;
                kv8_cmd_bytes<=(post_operation==0)
                               ? {15'd0,(kv8_k_token_beats[4:0]),4'b0000}
                               : {16'd0,(kv8_v_token_beats),4'b0000};
                kv8_cmd_is_write<=1'b1;
                kv8_vbyte_cnt<=5'd0;
                kv8_beat_lim<=(post_operation==0) ? kv8_k_token_beats[4:0] : kv8_v_token_beats;
            end
            if(post_state==PS_KV8_VSEND && kv8_stream_valid && wr_ready)
                kv8_vbyte_cnt<=kv8_vbyte_cnt+5'd1;
            // ---- V table bulk load: DDR -> vmem, then the bank becomes READY
            if(post_state==PS_KV8_VLOAD)begin
                kv8_cmd_addr<=kv8_scale_base;
                kv8_cmd_bytes<=24'd256;
                kv8_cmd_is_write<=1'b0;
                kv8_vrd_count<=4'd0;
            end
            if(post_state==PS_KV8_VRD && rd_valid && rd_ready)begin
                kv8_v_scale_we<=1'b1;
                kv8_v_scale_word<=kv8_vrd_count;
                kv8_v_scale_data<=rd_data;
                kv8_vrd_count<=kv8_vrd_count+4'd1;
                if(kv8_vrd_count==4'd15) kv8_v_table_loaded<=1'b1;
            end
            if(post_state==PS_KV8_VRD && rd_done)begin
                kv8_wr_release<=1'b1;
                kv8_bank_state<=2'd2;
            end
            // ---- diagnostics
            if(kv8_cmd_valid && attn_dma_owner) kv8_dma_conflict<=1'b1;
        end
    end


    always @(posedge clk)begin
        if(!rst_n)begin
            compute_bank<=0;post_bank<=0;post_operation<=0;post_output_tile<=0;post_identity<=0;rope_outstanding<=0;
        end else begin
            if(state_current[S_IDLE] || state_current[S_O_RESTORE] || state_current[S_FFN_BEGIN] || state_current[S_DOWN_BEGIN] || state_current[S_LM_BEGIN] || state_current[S_LAYER_DONE] || state_current[S_BATCH_NEXT] || ((state_current[S_DONE]) && start))begin compute_bank<=0;end
            if(state_current[S_NEXT] || state_current[S_FFN_NEXT] || state_current[S_LM_NEXT])begin compute_bank<=!compute_bank;end
            if(post_launch)begin post_bank<=compute_bank;post_operation<=operation;post_output_tile<=output_tile;post_identity<=compute_identity;post_biases<=biases;end
            if(rope_fire)begin rope_outstanding<=1;end
            if(rope_return && rope_outstanding)begin rope_outstanding<=0;end
        end
    end

    // Reserve response space before issuing a synchronous RAM read; carry the address with data.
    always @(posedge clk)begin
        if(!rst_n)begin
            bias_issued<=0;bias_retired<=0;bias_input_credit<=0;bias_input_count<=0;
            bias_input_write<=0;bias_input_read<=0;bias_read_address<=0;
        end else if(post_launch)begin
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
    // ---- KV8 clip report (display only, one line per layer) ------------------
    // [V-PERTOK-M] v_tok / v_codes are THIS layer's numbers, not running totals:
    // the V chain clears both at the first token of a layer, so the line can be
    // read directly (the old cumulative count had to be differenced by hand and
    // was reported on the wrong layer).  v_codes counts V codes that actually hit
    // |code| = 127, which is what "17.7% of the codes saturated" was measuring;
    // v_mmax is the largest per-token mantissa the layer published (255 = the
    // quantiser's MTOP, i.e. the only way a code can still clip).
    reg kv8_sld_d;
    always @(posedge clk) begin
        if(!rst_n) kv8_sld_d<=1'b0;
        else begin
            kv8_sld_d<=state_current[S_LAYER_DONE];
            if(state_current[S_LAYER_DONE] && !kv8_sld_d)
                $display("KV8_CLIP layer=%0d k=%0d v_tok=%0d v_codes=%0d v_mmax=%0d sticky=%b vsticky=%b wrerr=%b wrcode=%02h",
                         layer,kv8_k_clip_count,kv8_v_clip_count,kv8_v_clip_codes,kv8_v_m_max,
                         kv8_k_clip_sticky,kv8_v_clip_sticky,wr_error,wr_error_code);
        end
    end
endmodule
