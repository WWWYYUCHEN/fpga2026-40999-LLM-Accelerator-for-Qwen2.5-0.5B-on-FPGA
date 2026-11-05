`timescale 1ns / 1ps

////////////////////////////////////English///////////////////////////////////////
// Company:         Qwen25 KV260 Project
// Engineer:        Codex
// Create Date:     2026/10/07
// Design Name:     Qwen2.5 KV260 compute control
// Module Name:     qwen25_cfg_word
// Description:     spec/compute_control/qwen25_cfg_word_spec.md
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
// 模块名称:        qwen25_cfg_word
// 模块说明:        spec/compute_control/qwen25_cfg_word_spec.md
// 仿真工程:        tb/tb_qwen25_axi_lite_regs.sv
// 参考资料:        五份当前项目基线；不使用 module_requirements
// 依赖文件:        见下方显式命名实例
// 当前版本:        V1.0
// 修订日期:        2026年10月07日
// 修订历史:        2026年10月07日 V1.0 实现计算寄存器封装

`default_nettype none

// 一个按字节更新的计算配置字；保留位始终为零，综合可裁剪无效位。
module qwen25_cfg_word #(
	parameter [63:0] WRITE_MASK = 64'hffffffffffffffff,
	parameter [63:0] RESET_VALUE = 64'b0,
	parameter RS_LV = 0
) (
	input wire clk, // 上升沿时钟，目标 300 MHz
	input wire rst_n, // 同步复位，rst_n == RS_LV 时有效
	input wire write_enable, // 配置字提交许可
	input wire [63:0] write_data, // 64-bit 配置写值
	input wire [7:0] write_strb, // 字节掩码
	output wire [63:0] value // 配置字值，WRITE_MASK 外为零
);
	reg [63:0] value_o;
	integer byte_lane;
	assign value = value_o;

	always @(posedge clk) begin
		if (rst_n == RS_LV) begin
			value_o <= RESET_VALUE & WRITE_MASK;
		end else begin
			for (byte_lane = 0; byte_lane < 8; byte_lane = byte_lane + 1) begin
				if (write_enable && write_strb[byte_lane]) begin
					value_o[byte_lane*8 +: 8] <= write_data[byte_lane*8 +: 8] & WRITE_MASK[byte_lane*8 +: 8];
				end
			end
		end
	end
endmodule

`default_nettype wire
