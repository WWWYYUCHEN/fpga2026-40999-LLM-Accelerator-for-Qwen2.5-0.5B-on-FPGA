`timescale 1ns / 1ps

////////////////////////////////////English///////////////////////////////////////
// Company:         Qwen25 KV260 Project
// Engineer:        Codex
// Create Date:     2026/10/07
// Design Name:     Qwen2.5 KV260 compute control
// Module Name:     qwen25_axi_lite_regs
// Description:     spec/compute_control/qwen25_axi_lite_regs_spec.md
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
// 模块名称:        qwen25_axi_lite_regs
// 模块说明:        spec/compute_control/qwen25_axi_lite_regs_spec.md
// 仿真工程:        tb/tb_qwen25_axi_lite_regs.sv
// 参考资料:        五份当前项目基线；不使用 module_requirements
// 依赖文件:        见下方显式命名实例
// 当前版本:        V1.0
// 修订日期:        2026年10月07日
// 修订历史:        2026年10月07日 V1.0 实现计算寄存器封装

`default_nettype none

// 计算寄存器顶层：AXI 传输、配置快照、结果/中断分别封装。
// 宽度固定为 64 位 AXI-Lite、576 位任务快照；完整 DDR 地址原子锁存。
module qwen25_axi_lite_regs #(
	parameter AXIL_AW = 16,
	parameter AXIL_DW = 64,
	parameter RS_LV = 0
) (
	input wire clk, // 上升沿时钟，目标 300 MHz
	input wire rst_n, // 同步复位，rst_n == RS_LV 时有效
	input wire [AXIL_AW-1:0] s_axil_awaddr, // AXI 写地址，8-byte 对齐
	input wire s_axil_awvalid, // AW 独立请求有效
	output wire s_axil_awready, // 写地址槽可接收
	input wire [AXIL_DW-1:0] s_axil_wdata, // AXI 64-bit 软件写值
	input wire [AXIL_DW/8-1:0] s_axil_wstrb, // 八个独立写字节通道
	input wire s_axil_wvalid, // W 独立请求有效
	output wire s_axil_wready, // 写数据槽可接收
	output wire [1:0] s_axil_bresp, // 写响应 OKAY/SLVERR
	output wire s_axil_bvalid, // 已锁存写响应
	input wire s_axil_bready, // 主机接受写响应
	input wire [AXIL_AW-1:0] s_axil_araddr, // AXI 读地址，完整高位参与匹配
	input wire s_axil_arvalid, // 读地址请求有效
	output wire s_axil_arready, // 读地址槽可接收
	output wire [AXIL_DW-1:0] s_axil_rdata, // 锁存的读返回值
	output wire [1:0] s_axil_rresp, // 读响应 OKAY/SLVERR
	output wire s_axil_rvalid, // 读响应有效且可保持
	input wire s_axil_rready, // 主机接受读响应
	output wire cfg_valid, // 快照输出单拍
	output wire [575:0] cfg_snapshot, // 架构定义的完整原子配置快照
	input wire [255:0] status, // 低位运行状态、95:64 错误码、99:96 IRQ 事件
	input wire hw_result_commit, // Controller 提交单拍
	input wire [17:0] hw_best_token_id, // 提交时稳定的硬件 token
	output wire [17:0] best_token_id, // 硬件捕获的结果存储
	output wire result_done, // sticky 结果可读电平
	output wire result_clear, // 已接受控制写产生的清除事件
	output wire abort_req, // 要求停止新请求并排空在途事务
	output wire soft_reset_req, // 软件复位事件，优先于 START/ABORT
	output wire error_clear_req, // 软件清除错误事件
	output wire irq, // IRQ_EN 与 mask/status 的组合中断电平
	output wire protocol_error
);
	wire csr_write_valid, csr_read_valid;
	wire [AXIL_AW-1:0] csr_write_addr, csr_read_addr;
	wire [63:0] csr_write_data, csr_read_data;
	wire [7:0] csr_write_strb;
	wire [1:0] csr_write_resp, csr_read_resp;
	wire cfg_write_mapped, cfg_read_mapped;
	wire result_write_mapped, result_read_mapped;
	wire result_read_allowed, start_while_busy;
	wire [63:0] cfg_read_data, result_read_data;
	wire write_allowed, accepted_write, task_start;

	// 对正在执行的任务禁止配置更新；软件取消位保留但无执行作用。
	assign write_allowed = (cfg_write_mapped && !status[1]) ||
						   (result_write_mapped && !start_while_busy);
	assign csr_write_resp = write_allowed ? 2'b00 : 2'b10;
	assign accepted_write = csr_write_valid && write_allowed;
	assign csr_read_data = cfg_read_mapped ? cfg_read_data : result_read_data;
	assign csr_read_resp = ((cfg_read_mapped || result_read_mapped) && result_read_allowed) ? 2'b00 : 2'b10;

	qwen25_axil_transport #(.AXIL_AW(AXIL_AW), .RS_LV(RS_LV)) transport_Inst (
		.clk(clk), .rst_n(rst_n),
		.s_axil_awaddr(s_axil_awaddr), .s_axil_awvalid(s_axil_awvalid), .s_axil_awready(s_axil_awready),
		.s_axil_wdata(s_axil_wdata), .s_axil_wstrb(s_axil_wstrb),
		.s_axil_wvalid(s_axil_wvalid), .s_axil_wready(s_axil_wready),
		.s_axil_bresp(s_axil_bresp), .s_axil_bvalid(s_axil_bvalid), .s_axil_bready(s_axil_bready),
		.s_axil_araddr(s_axil_araddr), .s_axil_arvalid(s_axil_arvalid), .s_axil_arready(s_axil_arready),
		.s_axil_rdata(s_axil_rdata), .s_axil_rresp(s_axil_rresp),
		.s_axil_rvalid(s_axil_rvalid), .s_axil_rready(s_axil_rready),
		.csr_write_valid(csr_write_valid), .csr_write_addr(csr_write_addr),
		.csr_write_data(csr_write_data), .csr_write_strb(csr_write_strb), .csr_write_resp(csr_write_resp),
		.csr_read_valid(csr_read_valid), .csr_read_addr(csr_read_addr),
		.csr_read_data(csr_read_data), .csr_read_resp(csr_read_resp), .protocol_error(protocol_error)
	);

	qwen25_compute_cfg #(.AXIL_AW(AXIL_AW), .RS_LV(RS_LV)) compute_cfg_Inst (
		.clk(clk), .rst_n(rst_n), .write_enable(accepted_write),
		.write_addr(csr_write_addr), .write_data(csr_write_data), .write_strb(csr_write_strb),
		.read_addr(csr_read_addr), .write_mapped(cfg_write_mapped), .read_mapped(cfg_read_mapped),
		.read_data(cfg_read_data), .task_start(task_start), .cfg_valid(cfg_valid), .cfg_snapshot(cfg_snapshot)
	);

	qwen25_result_regs #(.AXIL_AW(AXIL_AW), .RS_LV(RS_LV)) result_regs_Inst (
		.clk(clk), .rst_n(rst_n), .write_enable(accepted_write),
		.write_addr(csr_write_addr), .write_data(csr_write_data), .write_strb(csr_write_strb),
		.read_addr(csr_read_addr), .write_mapped(result_write_mapped), .read_mapped(result_read_mapped),
		.read_data(result_read_data), .read_allowed(result_read_allowed), .start_while_busy(start_while_busy),
		.status(status), .hw_result_commit(hw_result_commit), .hw_best_token_id(hw_best_token_id),
		.best_token_id(best_token_id), .result_done(result_done), .task_start(task_start),
		.result_clear(result_clear), .abort_req(abort_req), .soft_reset_req(soft_reset_req),
		.error_clear_req(error_clear_req), .irq(irq)
	);
endmodule

`default_nettype wire
