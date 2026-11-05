`timescale 1ns / 1ps

////////////////////////////////////English///////////////////////////////////////
// Company:         Qwen25 KV260 Project
// Engineer:        Codex
// Create Date:     2026/10/07
// Design Name:     Qwen2.5 KV260 compute control
// Module Name:     qwen25_result_regs
// Description:     spec/compute_control/qwen25_result_regs_spec.md
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
// 模块名称:        qwen25_result_regs
// 模块说明:        spec/compute_control/qwen25_result_regs_spec.md
// 仿真工程:        tb/tb_qwen25_axi_lite_regs.sv
// 参考资料:        五份当前项目基线；不使用 module_requirements
// 依赖文件:        见下方显式命名实例
// 当前版本:        V1.0
// 修订日期:        2026年10月07日
// 修订历史:        2026年10月07日 V1.0 实现计算寄存器封装

`default_nettype none

// 硬件结果、控制事件和中断状态，软件不能写入 token 存储。
module qwen25_result_regs #(
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
	output wire read_allowed, // 结果读取权限
	output wire start_while_busy, // 纯 START 在 BUSY 时禁止
	input wire [255:0] status, // 低位运行状态、95:64 错误码、99:96 IRQ 事件
	input wire hw_result_commit, // Controller 提交单拍
	input wire [17:0] hw_best_token_id, // 提交时稳定的硬件 token
	output wire [17:0] best_token_id, // 硬件捕获的结果存储
	output wire result_done, // sticky 结果可读电平
	output wire task_start, // 获准 START 提交事件
	output wire result_clear, // 已接受控制写产生的清除事件
	output wire abort_req, // 要求停止新请求并排空在途事务
	output wire soft_reset_req, // 软件复位事件，优先于 START/ABORT
	output wire error_clear_req, // 软件清除错误事件
	output wire irq // IRQ_EN 与 mask/status 的组合中断电平
);
	reg [17:0] best_token_o;
	reg result_valid_o, irq_enable;
	reg [3:0] irq_status, irq_mask;
    reg result_clear_o;
	wire [7:0] control_bits;
	wire control_select, clear_event;
	wire [3:0] irq_ack;
	wire [3:0] irq_events;

	// 控制字段在最低字节；其它七个字节不能触发命令。
	assign control_bits = write_strb[0] ? write_data[7:0] : 8'b0;
	assign control_select = write_enable && (write_addr == 16'h0010);
	assign task_start = control_select && control_bits[0] && !control_bits[1] && !control_bits[2];
    assign clear_event = control_select && (control_bits[0] || control_bits[3]);
	assign start_while_busy = (write_addr == 16'h0010) && control_bits[0] &&
							  !control_bits[1] && !control_bits[2] && status[1];
	assign irq_ack = (write_enable && (write_addr == 16'h0020)) ? control_bits[3:0] : 4'b0;
	assign irq_events = status[99:96];
	assign read_allowed = (read_addr != 16'h01f8) || result_valid_o;
	assign best_token_id = best_token_o;
	assign result_done = result_valid_o;
	assign result_clear = result_clear_o;
    assign abort_req = 1'b0;
    assign soft_reset_req = 1'b0;
    assign error_clear_req = 1'b0;
	assign irq = irq_enable && (|(irq_status & irq_mask));

	// 只保留运行协议所需状态；ERROR_CODE 与 STATUS 使用不同位段。
	always @(*) begin
		write_mapped = 1'b1;
		case (write_addr)
			16'h0010, 16'h0020, 16'h0028, 16'h0038: begin
				write_mapped = 1'b1;
			end
			default: begin
				write_mapped = 1'b0;
			end
		endcase
		read_data = 64'b0;
		read_mapped = 1'b1;
		case (read_addr)
			16'h0000: begin read_data = 64'h515732354b563236; end
			16'h0008: begin read_data = 64'h0000000000010000; end
			16'h0010: begin read_data = {59'b0, irq_enable, 4'b0}; end
			16'h0018: begin read_data = {57'b0, result_valid_o, status[5:0]}; end
			16'h0020: begin read_data = {60'b0, irq_status}; end
			16'h0028: begin read_data = {60'b0, irq_mask}; end
			16'h0030: begin read_data = {32'b0, status[95:64]}; end
			16'h01f8: begin
				read_data = {46'b0, best_token_o};
			end
			default: begin read_mapped = 1'b0; end
		endcase
	end

	always @(posedge clk) begin
		if (rst_n == RS_LV) begin
			best_token_o <= 18'b0;
			result_valid_o <= 1'b0;
			irq_enable <= 1'b0;
			irq_status <= 4'b0;
			irq_mask <= 4'b0;
			result_clear_o <= 1'b0;
		end else begin
			result_clear_o <= clear_event;

			// W1C 与新事件同拍时，新事件保留；DONE/BUSY 状态电平不会重复置位。
			irq_status <= (irq_status & ~irq_ack) | irq_events;
            if (control_select && write_strb[0]) begin
				irq_enable <= irq_enable | control_bits[4];
			end
			if (write_enable && (write_addr == 16'h0028) && write_strb[0]) begin
				irq_mask <= write_data[3:0];
			end
			// 清除优先于同周期提交，不能暴露上一任务的 token。
			if (clear_event) begin
				best_token_o <= 18'b0;
				result_valid_o <= 1'b0;
			end else if (hw_result_commit) begin
				best_token_o <= hw_best_token_id;
				result_valid_o <= 1'b1;
			end
		end
	end
endmodule

`default_nettype wire
