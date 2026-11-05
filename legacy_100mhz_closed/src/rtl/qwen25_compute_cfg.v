`timescale 1ns / 1ps

////////////////////////////////////English///////////////////////////////////////
// Company:         Qwen25 KV260 Project
// Engineer:        Codex
// Create Date:     2026/10/07
// Design Name:     Qwen2.5 KV260 compute control
// Module Name:     qwen25_compute_cfg
// Description:     spec/compute_control/qwen25_compute_cfg_spec.md
// Simulations:     tb/tb_qwen25_axi_lite_regs.sv
// Referrences:     qwen25_kv260_accel_arch.md, qwen25_fsm_contract.md
// Dependencies:    See named module instantiations below
// Version:         V1.0
// Revision Date:   2026/10/07
// History:         2026/10/07 V1.0 Modular compute register implementation
///////////////////////////////////Chinese////////////////////////////////////////
// 版权归属:        Qwen25 KV260 项目
// 开发人员:        Codex
// 创建日期:        2026年10月07日
// 设计名称:        Qwen2.5 计算控制接口
// 模块名称:        qwen25_compute_cfg
// 模块说明:        spec/compute_control/qwen25_compute_cfg_spec.md
// 仿真工程:        tb/tb_qwen25_axi_lite_regs.sv
// 参考资料:        五份当前项目基线；不使用 module_requirements
// 依赖文件:        见下方显式命名实例
// 当前版本:        V1.0
// 修订日期:        2026年10月07日
// 修订历史:        2026年10月07日 V1.0 实现计算寄存器封装

`default_nettype none

// 计算配置存储及不可变任务快照；AXI-Lite 访问仍为单个 64 位字。
module qwen25_compute_cfg #(
	parameter AXIL_AW = 16,
	parameter RS_LV = 0
) (
	input wire clk, // 上升沿时钟，目标 300 MHz
	input wire rst_n, // 同步复位，rst_n == RS_LV 时有效
	input wire write_enable, // 一次已获许可的寄存器提交
	input wire [AXIL_AW-1:0] write_addr, // 完整提交地址
	input wire [63:0] write_data, // 软件写入数据
	input wire [7:0] write_strb, // 提交字节有效掩码
	input wire [AXIL_AW-1:0] read_addr, // 已捕获读地址
	output reg write_mapped, // 写地址属于本模块
	output reg read_mapped, // 读地址属于本模块
	output reg [63:0] read_data, // 组合读译码值，传输模块在 RD_DECODE 锁存
	input wire task_start, // 当前沿锁存配置
	output wire cfg_valid, // 快照输出单拍
	output wire [575:0] cfg_snapshot // 架构定义的完整原子配置快照
);
	wire [63:0] cfg_run_mode;
	wire [63:0] cfg_seq_len;
	wire [63:0] cfg_context_len;
	wire [63:0] cfg_token_pos;
	wire [63:0] cfg_kv_ctrl;
	wire [63:0] cfg_kv_valid_len;
	wire [63:0] cfg_rms_eps;
	wire [63:0] cfg_act_in_addr;
	wire [63:0] cfg_act_out_addr;
	wire [63:0] cfg_model_weight_base;
	wire [63:0] cfg_k_cache_base;
	wire [63:0] cfg_v_cache_base;
	wire [63:0] cfg_q_scratch_base;
	wire [63:0] cfg_p_spill_base;
	wire [63:0] cfg_axi_burst_cfg;
	reg cfg_valid_o;
	reg [575:0] cfg_snapshot_o;
	wire [575:0] next_snapshot;

	assign cfg_valid = cfg_valid_o;
	assign cfg_snapshot = cfg_snapshot_o;
	assign next_snapshot[63:0] = {11'b0, cfg_token_pos[15:0], cfg_context_len[15:0], cfg_seq_len[15:0], cfg_kv_ctrl[3:0], cfg_run_mode[0]};
	assign next_snapshot[127:64] = {24'b0, cfg_axi_burst_cfg[7:0], cfg_rms_eps[15:0], cfg_kv_valid_len[15:0]};
	assign next_snapshot[191:128] = cfg_act_in_addr;
	assign next_snapshot[255:192] = cfg_act_out_addr;
	assign next_snapshot[319:256] = cfg_model_weight_base;
	assign next_snapshot[383:320] = cfg_k_cache_base;
	assign next_snapshot[447:384] = cfg_v_cache_base;
	assign next_snapshot[511:448] = cfg_q_scratch_base;
	assign next_snapshot[575:512] = cfg_p_spill_base;

	// START 把全部计算字段及七个 DDR 地址同时锁存，后续软件写入不影响该任务。
	always @(posedge clk) begin
		if (rst_n == RS_LV) begin
			cfg_valid_o <= 1'b0;
			cfg_snapshot_o <= 576'b0;
		end else begin
			cfg_valid_o <= task_start;
			if (task_start) begin
				cfg_snapshot_o <= next_snapshot;
			end
		end
	end

	// 译码使用整个地址，不允许较宽地址高位别名到低位寄存器。
	always @(*) begin
		write_mapped = 1'b0;
		case (write_addr)
			16'h0080: begin write_mapped = 1'b1; end
			16'h0088: begin write_mapped = 1'b1; end
			16'h0090: begin write_mapped = 1'b1; end
			16'h0098: begin write_mapped = 1'b1; end
			16'h00b0: begin write_mapped = 1'b1; end
			16'h00b8: begin write_mapped = 1'b1; end
			16'h00d0: begin write_mapped = 1'b1; end
			16'h0100: begin write_mapped = 1'b1; end
			16'h0108: begin write_mapped = 1'b1; end
			16'h0110: begin write_mapped = 1'b1; end
			16'h0118: begin write_mapped = 1'b1; end
			16'h0120: begin write_mapped = 1'b1; end
			16'h0128: begin write_mapped = 1'b1; end
			16'h0130: begin write_mapped = 1'b1; end
			16'h0140: begin write_mapped = 1'b1; end
			default: begin write_mapped = 1'b0; end
		endcase
		read_data = 64'b0;
		read_mapped = 1'b1;
		case (read_addr)
			16'h0080: begin read_data = cfg_run_mode; end
			16'h0088: begin read_data = cfg_seq_len; end
			16'h0090: begin read_data = cfg_context_len; end
			16'h0098: begin read_data = cfg_token_pos; end
			16'h00b0: begin read_data = cfg_kv_ctrl; end
			16'h00b8: begin read_data = cfg_kv_valid_len; end
			16'h00d0: begin read_data = cfg_rms_eps; end
			16'h0100: begin read_data = cfg_act_in_addr; end
			16'h0108: begin read_data = cfg_act_out_addr; end
			16'h0110: begin read_data = cfg_model_weight_base; end
			16'h0118: begin read_data = cfg_k_cache_base; end
			16'h0120: begin read_data = cfg_v_cache_base; end
			16'h0128: begin read_data = cfg_q_scratch_base; end
			16'h0130: begin read_data = cfg_p_spill_base; end
			16'h0140: begin read_data = cfg_axi_burst_cfg; end
			default: begin read_mapped = 1'b0; end
		endcase
	end

	// RUN_MODE 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h0000000000000001), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_run_mode_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0080)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_run_mode)
	);

	// SEQ_LEN 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h000000000000ffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_seq_len_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0088)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_seq_len)
	);

	// CONTEXT_LEN 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h000000000000ffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_context_len_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0090)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_context_len)
	);

	// TOKEN_POS 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h000000000000ffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_token_pos_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0098)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_token_pos)
	);

	// KV_CTRL 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h000000000000000f), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_kv_ctrl_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h00b0)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_kv_ctrl)
	);

	// KV_VALID_LEN 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h000000000000ffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_kv_valid_len_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h00b8)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_kv_valid_len)
	);

	// RMS_EPS 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h000000000000ffff), .RESET_VALUE(64'h0000000000003586), .RS_LV(RS_LV)
	) cfg_rms_eps_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h00d0)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_rms_eps)
	);

	// ACT_IN_ADDR 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_act_in_addr_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0100)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_act_in_addr)
	);

	// ACT_OUT_ADDR 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_act_out_addr_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0108)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_act_out_addr)
	);

	// MODEL_WEIGHT_BASE 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_model_weight_base_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0110)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_model_weight_base)
	);

	// K_CACHE_BASE 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_k_cache_base_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0118)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_k_cache_base)
	);

	// V_CACHE_BASE 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_v_cache_base_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0120)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_v_cache_base)
	);

	// Q_SCRATCH_BASE 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_q_scratch_base_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0128)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_q_scratch_base)
	);

	// P_SPILL_BASE 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'hffffffffffffffff), .RESET_VALUE(64'h0000000000000000), .RS_LV(RS_LV)
	) cfg_p_spill_base_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0130)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_p_spill_base)
	);

	// AXI_BURST_CFG 的有效位与复位值来自当前计算寄存器契约。
	qwen25_cfg_word #(
		.WRITE_MASK(64'h00000000000000ff), .RESET_VALUE(64'h000000000000000f), .RS_LV(RS_LV)
	) cfg_axi_burst_cfg_word (
		.clk(clk), .rst_n(rst_n),
		.write_enable(write_enable && (write_addr == 16'h0140)),
		.write_data(write_data), .write_strb(write_strb), .value(cfg_axi_burst_cfg)
	);

endmodule

`default_nettype wire
