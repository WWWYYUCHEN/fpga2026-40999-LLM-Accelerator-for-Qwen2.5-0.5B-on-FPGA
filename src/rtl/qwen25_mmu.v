`timescale 1ns / 1ps
`default_nettype none
module qwen25_mmu #(parameter AXIL_AW=16, parameter AXIL_DW=64, parameter RS_LV=0)(
    input wire clk,input wire rst_n,
    input  wire [AXIL_AW-1:0]     s_axil_awaddr,
    input  wire                   s_axil_awvalid,
    output wire                   s_axil_awready,
    input  wire [AXIL_DW-1:0]     s_axil_wdata,
    input  wire [AXIL_DW/8-1:0]   s_axil_wstrb,
    input  wire                   s_axil_wvalid,
    output wire                   s_axil_wready,
    output wire [1:0]             s_axil_bresp,
    output wire                   s_axil_bvalid,
    input  wire                   s_axil_bready,
    input  wire [AXIL_AW-1:0]     s_axil_araddr,
    input  wire                   s_axil_arvalid,
    output wire                   s_axil_arready,
    output wire [AXIL_DW-1:0]     s_axil_rdata,
    output wire [1:0]             s_axil_rresp,
    output wire                   s_axil_rvalid,
    input  wire                   s_axil_rready,

    output wire [63:0] m_axi0_araddr,
    output wire [7:0] m_axi0_arlen,
    output wire [2:0] m_axi0_arsize,
    output wire [1:0] m_axi0_arburst,
    output wire m_axi0_arvalid,
    input wire m_axi0_arready,
    output wire [3:0] m_axi0_arid,
    output wire m_axi0_arlock,
    output wire [3:0] m_axi0_arcache,
    output wire [2:0] m_axi0_arprot,
    output wire [3:0] m_axi0_arqos,
    input wire [3:0] m_axi0_rid,
    input wire [127:0] m_axi0_rdata,
    input wire [1:0] m_axi0_rresp,
    input wire m_axi0_rlast,
    input wire m_axi0_rvalid,
    output wire m_axi0_rready,
    output wire [63:0] m_axi0_awaddr,
    output wire [7:0] m_axi0_awlen,
    output wire [2:0] m_axi0_awsize,
    output wire [1:0] m_axi0_awburst,
    output wire m_axi0_awvalid,
    input wire m_axi0_awready,
    output wire [3:0] m_axi0_awid,
    output wire m_axi0_awlock,
    output wire [3:0] m_axi0_awcache,
    output wire [2:0] m_axi0_awprot,
    output wire [3:0] m_axi0_awqos,
    output wire [127:0] m_axi0_wdata,
    output wire [15:0] m_axi0_wstrb,
    output wire m_axi0_wlast,
    output wire m_axi0_wvalid,
    input wire m_axi0_wready,
    input wire [3:0] m_axi0_bid,
    input wire [1:0] m_axi0_bresp,
    input wire m_axi0_bvalid,
    output wire m_axi0_bready,
    output wire [63:0] m_axi1_araddr,
    output wire [7:0] m_axi1_arlen,
    output wire [2:0] m_axi1_arsize,
    output wire [1:0] m_axi1_arburst,
    output wire m_axi1_arvalid,
    input wire m_axi1_arready,
    output wire [3:0] m_axi1_arid,
    output wire m_axi1_arlock,
    output wire [3:0] m_axi1_arcache,
    output wire [2:0] m_axi1_arprot,
    output wire [3:0] m_axi1_arqos,
    input wire [3:0] m_axi1_rid,
    input wire [127:0] m_axi1_rdata,
    input wire [1:0] m_axi1_rresp,
    input wire m_axi1_rlast,
    input wire m_axi1_rvalid,
    output wire m_axi1_rready,
    output wire [63:0] m_axi1_awaddr,
    output wire [7:0] m_axi1_awlen,
    output wire [2:0] m_axi1_awsize,
    output wire [1:0] m_axi1_awburst,
    output wire m_axi1_awvalid,
    input wire m_axi1_awready,
    output wire [3:0] m_axi1_awid,
    output wire m_axi1_awlock,
    output wire [3:0] m_axi1_awcache,
    output wire [2:0] m_axi1_awprot,
    output wire [3:0] m_axi1_awqos,
    output wire [127:0] m_axi1_wdata,
    output wire [15:0] m_axi1_wstrb,
    output wire m_axi1_wlast,
    output wire m_axi1_wvalid,
    input wire m_axi1_wready,
    input wire [3:0] m_axi1_bid,
    input wire [1:0] m_axi1_bresp,
    input wire m_axi1_bvalid,
    output wire m_axi1_bready,
    output wire [63:0] m_axi2_araddr,
    output wire [7:0] m_axi2_arlen,
    output wire [2:0] m_axi2_arsize,
    output wire [1:0] m_axi2_arburst,
    output wire m_axi2_arvalid,
    input wire m_axi2_arready,
    output wire [3:0] m_axi2_arid,
    output wire m_axi2_arlock,
    output wire [3:0] m_axi2_arcache,
    output wire [2:0] m_axi2_arprot,
    output wire [3:0] m_axi2_arqos,
    input wire [3:0] m_axi2_rid,
    input wire [127:0] m_axi2_rdata,
    input wire [1:0] m_axi2_rresp,
    input wire m_axi2_rlast,
    input wire m_axi2_rvalid,
    output wire m_axi2_rready,
    output wire [63:0] m_axi2_awaddr,
    output wire [7:0] m_axi2_awlen,
    output wire [2:0] m_axi2_awsize,
    output wire [1:0] m_axi2_awburst,
    output wire m_axi2_awvalid,
    input wire m_axi2_awready,
    output wire [3:0] m_axi2_awid,
    output wire m_axi2_awlock,
    output wire [3:0] m_axi2_awcache,
    output wire [2:0] m_axi2_awprot,
    output wire [3:0] m_axi2_awqos,
    output wire [127:0] m_axi2_wdata,
    output wire [15:0] m_axi2_wstrb,
    output wire m_axi2_wlast,
    output wire m_axi2_wvalid,
    input wire m_axi2_wready,
    input wire [3:0] m_axi2_bid,
    input wire [1:0] m_axi2_bresp,
    input wire m_axi2_bvalid,
    output wire m_axi2_bready,
    output wire [63:0] m_axi3_araddr,
    output wire [7:0] m_axi3_arlen,
    output wire [2:0] m_axi3_arsize,
    output wire [1:0] m_axi3_arburst,
    output wire m_axi3_arvalid,
    input wire m_axi3_arready,
    output wire [3:0] m_axi3_arid,
    output wire m_axi3_arlock,
    output wire [3:0] m_axi3_arcache,
    output wire [2:0] m_axi3_arprot,
    output wire [3:0] m_axi3_arqos,
    input wire [3:0] m_axi3_rid,
    input wire [127:0] m_axi3_rdata,
    input wire [1:0] m_axi3_rresp,
    input wire m_axi3_rlast,
    input wire m_axi3_rvalid,
    output wire m_axi3_rready,
    output wire [63:0] m_axi3_awaddr,
    output wire [7:0] m_axi3_awlen,
    output wire [2:0] m_axi3_awsize,
    output wire [1:0] m_axi3_awburst,
    output wire m_axi3_awvalid,
    input wire m_axi3_awready,
    output wire [3:0] m_axi3_awid,
    output wire m_axi3_awlock,
    output wire [3:0] m_axi3_awcache,
    output wire [2:0] m_axi3_awprot,
    output wire [3:0] m_axi3_awqos,
    output wire [127:0] m_axi3_wdata,
    output wire [15:0] m_axi3_wstrb,
    output wire m_axi3_wlast,
    output wire m_axi3_wvalid,
    input wire m_axi3_wready,
    input wire [3:0] m_axi3_bid,
    input wire [1:0] m_axi3_bresp,
    input wire m_axi3_bvalid,
    output wire m_axi3_bready,
    output wire abort_req,output wire soft_reset_req,
    output wire ctrl_afu_valid,
    input wire ctrl_afu_ready,
    output wire ctrl_afu_last,
    output wire ctrl_afu_latch,
    output wire ctrl_afu_gelu_sign,
    output wire [3:0] ctrl_afu_command,
    output wire [20:0] ctrl_afu_command_4,
    output wire [26:0] ctrl_afu_sum_command_buffer,
    output wire [31:0] ctrl_afu_eps,
    output wire [127:0] ctrl_afu_x,
    output wire [127:0] ctrl_afu_y,
    output wire [127:0] ctrl_afu_meta,
    output wire [7:0] ctrl_afu_mask,
    output wire [63:0] ctrl_afu_length,
    input wire ctrl_result_valid,
    output wire ctrl_result_ready,
    input wire ctrl_result_last,
    input wire [127:0] ctrl_result_data,
    input wire [127:0] ctrl_result_aux,
    input wire [127:0] ctrl_result_meta,
    input wire [7:0] ctrl_result_mask,
    output wire ctrl_rope_req_valid,
    input wire ctrl_rope_req_ready,
    output wire ctrl_rope_req_last,
    output wire [127:0] ctrl_rope_req_x,
    output wire [127:0] ctrl_rope_req_y,
    output wire [127:0] ctrl_rope_req_meta,
    output wire [7:0] ctrl_rope_req_mask,
    input wire ctrl_rope_res_valid,
    output wire ctrl_rope_res_ready,
    input wire ctrl_rope_res_last,
    input wire [127:0] ctrl_rope_res_data,
    input wire [127:0] ctrl_rope_res_aux,
    input wire [127:0] ctrl_rope_res_meta,
    input wire [7:0] ctrl_rope_res_mask,
    input wire internal_busy,
    input wire internal_error,
    output wire ctrl_afu_abort,
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
    input wire [7:0] s_gemm_error_code
);
    wire [575:0] cfg_snapshot;wire cfg_valid;
    wire result_done;
    wire [127:0] dma_cmd,dma_meta;wire dma_cmd_valid, dma_cmd_ready;
    wire [3:0] buffer_valid,buffer_ready;
    wire [511:0] buffer_data; wire [39:0] buffer_word;
    wire [63:0] buffer_keep; wire [7:0] buffer_error;
    wire [127:0] buffer_meta;
    wire [127:0] stream_data,stream_meta;wire [15:0] stream_keep,stream_tag;wire stream_last,stream_valid,stream_ready;
    wire [3:0] stream_resource;wire [1:0] stream_error;
    wire rd_busy,rd_done,rd_error,rd_aborted;wire [7:0] rd_error_code;
    wire wr_busy,wr_done,wr_error,wr_aborted;wire [7:0] wr_error_code;
    wire wr_stream_valid,wr_stream_ready,wr_stream_last;wire [127:0] wr_stream_data,wr_stream_meta;wire [15:0] wr_stream_keep;
    wire control_busy,qkv_stage_done,control_error,lut_ready;
    wire control_result_commit;wire [17:0] control_best_token;
    wire [255:0] status_snapshot;
    assign status_snapshot={156'd0,2'd0,1'b0,control_result_commit,
        32'd0,58'd0,1'b0,lut_ready,1'b0,result_done,control_busy,!control_busy};
    qwen25_axi_lite_regs #(.AXIL_AW(AXIL_AW),.AXIL_DW(AXIL_DW),.RS_LV(RS_LV)) registers_Inst (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
        .s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb), .s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready), .s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
        .s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready), .s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp), .s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
        .cfg_valid(cfg_valid), .cfg_snapshot(cfg_snapshot), .status(status_snapshot),
        .hw_result_commit(control_result_commit), .hw_best_token_id(control_best_token), // 鏈€鍚巐ogits鍐欏叆骞舵帓绌哄悗鎻愪氦
        .best_token_id(), .result_done(result_done), .result_clear(),
        .abort_req(abort_req), .soft_reset_req(soft_reset_req), .error_clear_req(), .protocol_error()
    );
    qwen25_mmu_dma #(.RS_LV(RS_LV),.DIRECT_BUFFER_READ(1)) dma_Inst (
        .clk(clk), .rst_n(rst_n), .s_cmd(dma_cmd), .s_cmd_meta(dma_meta), .s_cmd_valid(dma_cmd_valid), .s_cmd_ready(dma_cmd_ready),
        .abort_req(1'b0), .soft_reset_req(1'b0),
        .m_axi_araddr({m_axi3_araddr,m_axi2_araddr,m_axi1_araddr,m_axi0_araddr}), .m_axi_arlen({m_axi3_arlen,m_axi2_arlen,m_axi1_arlen,m_axi0_arlen}), .m_axi_arsize({m_axi3_arsize,m_axi2_arsize,m_axi1_arsize,m_axi0_arsize}), .m_axi_arburst({m_axi3_arburst,m_axi2_arburst,m_axi1_arburst,m_axi0_arburst}), .m_axi_arvalid({m_axi3_arvalid,m_axi2_arvalid,m_axi1_arvalid,m_axi0_arvalid}), .m_axi_arready({m_axi3_arready,m_axi2_arready,m_axi1_arready,m_axi0_arready}),
        .m_axi_rid({m_axi3_rid,m_axi2_rid,m_axi1_rid,m_axi0_rid}), .m_axi_rdata({m_axi3_rdata,m_axi2_rdata,m_axi1_rdata,m_axi0_rdata}), .m_axi_rresp({m_axi3_rresp,m_axi2_rresp,m_axi1_rresp,m_axi0_rresp}), .m_axi_rlast({m_axi3_rlast,m_axi2_rlast,m_axi1_rlast,m_axi0_rlast}), .m_axi_rvalid({m_axi3_rvalid,m_axi2_rvalid,m_axi1_rvalid,m_axi0_rvalid}), .m_axi_rready({m_axi3_rready,m_axi2_rready,m_axi1_rready,m_axi0_rready}),
        .m_stream_data(stream_data), .m_stream_keep(stream_keep), .m_stream_last(stream_last), .m_stream_meta(stream_meta), .m_stream_tag(stream_tag), .m_stream_resource(stream_resource), .m_stream_error(stream_error), .m_stream_valid(stream_valid), .m_stream_ready(stream_ready),
        .m_buffer_valid(buffer_valid),.m_buffer_ready(buffer_ready),.m_buffer_data(buffer_data),
        .m_buffer_word(buffer_word),.m_buffer_keep(buffer_keep),.m_buffer_error(buffer_error),.m_buffer_meta(buffer_meta),
        .rd_busy(rd_busy), .rd_done(rd_done), .rd_error(rd_error), .rd_aborted(rd_aborted), .rd_error_code(rd_error_code), .rd_completion_tag(), .rd_completion_meta(),
        .s_stream_data(wr_stream_data), .s_stream_keep(wr_stream_keep), .s_stream_last(wr_stream_last), .s_stream_meta(wr_stream_meta), .s_stream_valid(wr_stream_valid), .s_stream_ready(wr_stream_ready), // 鍚庣画Controller鎺ュ悇buffer鐙珛DDR鍐欏洖锛屼笉鍋囬€犲啓鍛戒护
        .m_axi_awaddr({m_axi3_awaddr,m_axi2_awaddr,m_axi1_awaddr,m_axi0_awaddr}), .m_axi_awlen({m_axi3_awlen,m_axi2_awlen,m_axi1_awlen,m_axi0_awlen}), .m_axi_awsize({m_axi3_awsize,m_axi2_awsize,m_axi1_awsize,m_axi0_awsize}), .m_axi_awburst({m_axi3_awburst,m_axi2_awburst,m_axi1_awburst,m_axi0_awburst}), .m_axi_awvalid({m_axi3_awvalid,m_axi2_awvalid,m_axi1_awvalid,m_axi0_awvalid}), .m_axi_awready({m_axi3_awready,m_axi2_awready,m_axi1_awready,m_axi0_awready}),
        .m_axi_wdata({m_axi3_wdata,m_axi2_wdata,m_axi1_wdata,m_axi0_wdata}), .m_axi_wstrb({m_axi3_wstrb,m_axi2_wstrb,m_axi1_wstrb,m_axi0_wstrb}), .m_axi_wlast({m_axi3_wlast,m_axi2_wlast,m_axi1_wlast,m_axi0_wlast}), .m_axi_wvalid({m_axi3_wvalid,m_axi2_wvalid,m_axi1_wvalid,m_axi0_wvalid}), .m_axi_wready({m_axi3_wready,m_axi2_wready,m_axi1_wready,m_axi0_wready}), .m_axi_bid({m_axi3_bid,m_axi2_bid,m_axi1_bid,m_axi0_bid}), .m_axi_bresp({m_axi3_bresp,m_axi2_bresp,m_axi1_bresp,m_axi0_bresp}), .m_axi_bvalid({m_axi3_bvalid,m_axi2_bvalid,m_axi1_bvalid,m_axi0_bvalid}), .m_axi_bready({m_axi3_bready,m_axi2_bready,m_axi1_bready,m_axi0_bready}),
        .wr_busy(wr_busy), .wr_done(wr_done), .wr_error(wr_error), .wr_aborted(wr_aborted), .wr_error_code(wr_error_code), .wr_completion_tag(), .wr_completion_meta()
    );

    assign m_axi0_arid=4'd0;
    assign m_axi0_awid=4'd0;
    assign m_axi0_arlock=1'd0;
    assign m_axi0_awlock=1'd0;
    assign m_axi0_arcache=4'd2;
    assign m_axi0_awcache=4'd2;
    assign m_axi0_arprot=3'd0;
    assign m_axi0_awprot=3'd0;
    assign m_axi0_arqos=4'd0;
    assign m_axi0_awqos=4'd0;
    assign m_axi1_arid=4'd0;
    assign m_axi1_awid=4'd0;
    assign m_axi1_arlock=1'd0;
    assign m_axi1_awlock=1'd0;
    assign m_axi1_arcache=4'd2;
    assign m_axi1_awcache=4'd2;
    assign m_axi1_arprot=3'd0;
    assign m_axi1_awprot=3'd0;
    assign m_axi1_arqos=4'd0;
    assign m_axi1_awqos=4'd0;
    assign m_axi2_arid=4'd0;
    assign m_axi2_awid=4'd0;
    assign m_axi2_arlock=1'd0;
    assign m_axi2_awlock=1'd0;
    assign m_axi2_arcache=4'd2;
    assign m_axi2_awcache=4'd2;
    assign m_axi2_arprot=3'd0;
    assign m_axi2_awprot=3'd0;
    assign m_axi2_arqos=4'd0;
    assign m_axi2_awqos=4'd0;
    assign m_axi3_arid=4'd0;
    assign m_axi3_awid=4'd0;
    assign m_axi3_arlock=1'd0;
    assign m_axi3_awlock=1'd0;
    assign m_axi3_arcache=4'd2;
    assign m_axi3_awcache=4'd2;
    assign m_axi3_arprot=3'd0;
    assign m_axi3_awprot=3'd0;
    assign m_axi3_arqos=4'd0;
    assign m_axi3_awqos=4'd0;
    qwen25_qkv_controller #(.EXTERNAL_GEMM(1),.DIRECT_WEIGHT_READ(1)) controller_Inst (
        .m_gemm_cmd_valid(m_gemm_cmd_valid),
        .s_gemm_cmd_ready(s_gemm_cmd_ready),
        .m_gemm_cmd(m_gemm_cmd),
        .m_gemm_cmd_meta(m_gemm_cmd_meta),
        .m_gemm_act_valid(m_gemm_act_valid),
        .s_gemm_act_ready(s_gemm_act_ready),
        .m_gemm_act_data(m_gemm_act_data),
        .m_gemm_act_meta(m_gemm_act_meta),
        .m_gemm_weight_valid(m_gemm_weight_valid),
        .s_gemm_weight_ready(s_gemm_weight_ready),
        .m_gemm_weight_data(m_gemm_weight_data),
        .m_gemm_weight_meta(m_gemm_weight_meta),
        .m_gemm_abort(m_gemm_abort),
        .m_gemm_soft_reset(m_gemm_soft_reset),
        .s_gemm_valid(s_gemm_valid),
        .m_gemm_ready(m_gemm_ready),
        .s_gemm_bf16(s_gemm_bf16),
        .s_gemm_fp32(s_gemm_fp32),
        .s_gemm_meta(s_gemm_meta),
        .s_gemm_slice(s_gemm_slice),
        .s_gemm_row_mask(s_gemm_row_mask),
        .s_gemm_col_mask(s_gemm_col_mask),
        .s_gemm_last(s_gemm_last),
        .s_gemm_busy(s_gemm_busy),
        .s_gemm_input_done(s_gemm_input_done),
        .s_gemm_partial_done(s_gemm_partial_done),
        .s_gemm_done(s_gemm_done),
        .s_gemm_error(s_gemm_error),
        .s_gemm_aborted(s_gemm_aborted),
        .s_gemm_error_code(s_gemm_error_code),
        .clk(clk), // 绯荤粺300MHz鍚岄挓鍩?
        .rst_n(rst_n), // 鍚屾浣庢湁鏁堝浣?
        .start(cfg_valid), // AXI-Lite瀹屾暣閰嶇疆蹇収鎻愪氦
        .cfg(cfg_snapshot), // 涓嶅彲鍙樹换鍔″揩鐓?
        .abort_req(1'b0), // Reserved software control; hardware reset owns recovery.
        .soft_reset(1'b0),
        .dma_cmd_valid(dma_cmd_valid), // DMA鐪熷疄鍛戒护鏈夋晥
        .dma_cmd_ready(dma_cmd_ready), // DMA鍛戒护鎺ュ彈
        .dma_cmd(dma_cmd), // 瀹屾暣64bit鍦板潃鎻忚堪绗?
        .dma_meta(dma_meta), // DMA瀹屾暣metadata
        .dma_abort(), // 鐪熷疄DMA鎺掔┖璇锋眰
        .rd_busy(rd_busy), // AXI璇讳簨鍔″湪閫?
        .rd_done(rd_done), // AXI璇讳笌鏈湴娴佺湡瀹炴垚鍔?
        .rd_error(rd_error), // AXI璇诲け璐?
        .rd_aborted(rd_aborted), // AXI璇诲彇娑堝畬鎴?
        .rd_error_code(rd_error_code), // 璇婚敊璇師鍥?
        .direct_rd_valid(buffer_valid),.direct_rd_ready(buffer_ready),.direct_rd_data(buffer_data),
        .direct_rd_word(buffer_word),.direct_rd_keep(buffer_keep),.direct_rd_error(buffer_error),.direct_rd_meta(buffer_meta),
        .rd_valid(stream_valid), // DMA鏈湴128bit娴佹湁鏁?
        .rd_ready(stream_ready), // 鎺ュ彈128bit娴?
        .rd_data(stream_data), // DDR璇绘暟鎹?
        .rd_keep(stream_keep), // 鏈夋晥byte
        .rd_last(stream_last), // descriptor鏈媿
        .rd_meta(stream_meta), // 璇绘暟鎹韩浠?
        .rd_tag(stream_tag), // descriptor鍏宠仈
        .rd_resource(stream_resource), // 璇诲彇璧勬簮缂栧彿
        .rd_beat_error(stream_error), // AXI鏈媿閿欒
        .wr_valid(wr_stream_valid), // 閫丏MA鐨?28bit娴?
        .wr_ready(wr_stream_ready), // DMA鎺ュ彈鍐欐祦
        .wr_data(wr_stream_data), // DDR鍐欐暟鎹?
        .wr_keep(wr_stream_keep), // 鍏獴F16瀹屾暣byte
        .wr_last(wr_stream_last), // 鍐檇escriptor鏈媿
        .wr_meta(wr_stream_meta), // 鍐檇escriptor韬唤
        .wr_busy(wr_busy), // AXI鍐欎簨鍔″湪閫?
        .wr_done(wr_done), // 鐪熷疄B鍝嶅簲鎴愬姛
        .wr_error(wr_error), // 鍐欏け璐?
        .wr_aborted(wr_aborted), // 鍐欐帓绌哄彇娑?
        .wr_error_code(wr_error_code), // 鍐欏け璐ュ師鍥?
        .afu_valid(ctrl_afu_valid), // 鏈夋晥绠楁湳frame杈撳叆
        .afu_ready(ctrl_afu_ready), // 澶栭儴AFU鎺ュ彈
        .afu_command(ctrl_afu_command), // 11RMSNorm12RoPE杈圭晫
        .afu_command_4(ctrl_afu_command_4), // 闃舵鎺у埗淇濈暀
        .afu_sum_command_buffer(ctrl_afu_sum_command_buffer), // 褰掔害闃舵鎺у埗淇濈暀
        .afu_eps(ctrl_afu_eps), // FP32 RMS epsilon
        .afu_latch(ctrl_afu_latch), // 褰掔害鍙傛暟淇濇寔璇锋眰
        .afu_gelu_sign(ctrl_afu_gelu_sign), // QKV涓嶅惎鐢℅ELU
        .afu_x(ctrl_afu_x), // 鍏玪ane BF16 X
        .afu_y(ctrl_afu_y), // gamma鎴朢oPE閰嶅绗簩鍗?
        .afu_meta(ctrl_afu_meta), // 鏈夋晥琛屽瓧鍦板潃浣嶇疆韬唤
        .afu_last(ctrl_afu_last), // 琛屾垨RoPE褰撳墠鍚戦噺缁撴潫
        .afu_mask(ctrl_afu_mask), // 鏈夋晥鍏玪ane
        .afu_length(ctrl_afu_length), // 896缁碦MS鎴?4缁磆ead
        .afu_result_valid(ctrl_result_valid), // 澶栭儴瀹為檯璁＄畻缁撴灉valid
        .afu_result_ready(ctrl_result_ready), // Controller鎺ュ彈缁撴灉
        .afu_result_data(ctrl_result_data), // 绗竴缁勫叓lane缁撴灉
        .afu_result_aux(ctrl_result_aux), // RoPE绗簩缁勭粨鏋?
        .afu_result_meta(ctrl_result_meta), // 缁撴灉瀹屾暣韬唤
        .afu_result_last(ctrl_result_last), // 缁撴灉琛屾湯
        .afu_result_mask(ctrl_result_mask), // 缁撴灉lane鏈夋晥
        .rope_req_valid(ctrl_rope_req_valid),.rope_req_ready(ctrl_rope_req_ready),
        .rope_req_last(ctrl_rope_req_last),.rope_req_x(ctrl_rope_req_x),
        .rope_req_y(ctrl_rope_req_y),.rope_req_meta(ctrl_rope_req_meta),
        .rope_req_mask(ctrl_rope_req_mask),
        .rope_res_valid(ctrl_rope_res_valid),.rope_res_ready(ctrl_rope_res_ready),
        .rope_res_last(ctrl_rope_res_last),.rope_res_data(ctrl_rope_res_data),
        .rope_res_aux(ctrl_rope_res_aux),.rope_res_meta(ctrl_rope_res_meta),
        .rope_res_mask(ctrl_rope_res_mask),
        .afu_busy(internal_busy),
        .afu_error(internal_error),
        .afu_abort(ctrl_afu_abort),
        .busy(control_busy), // QKV闃舵灏氭湭瀹屾垚
        .qkv_done(qkv_stage_done), // K,V,Q鍏ㄩ儴鐪熷疄DDR鍐欏洖瀹屾垚
        .param_ready(lut_ready), // 涓夎〃鐪熷疄瀹屾垚鍚庡彧璇?
        .result_commit(control_result_commit),.best_token_id(control_best_token),
        .error(control_error), // 鎺у埗閿欒淇濇寔
        .error_code() // CSR閿欒鍘熷洜
    );
endmodule
`default_nettype wire
