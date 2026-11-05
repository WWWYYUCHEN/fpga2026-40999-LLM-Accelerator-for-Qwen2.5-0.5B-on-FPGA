`timescale 1ns / 1ps

////////////////////////////////////English///////////////////////////////////////
// Company:         Qwen25 KV260 Project
// Engineer:        Codex
// Create Date:     2026/10/07
// Design Name:     Qwen2.5 KV260 compute control
// Module Name:     qwen25_axil_transport
// Description:     spec/compute_control/qwen25_axil_transport_spec.md
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
// 模块名称:        qwen25_axil_transport
// 模块说明:        spec/compute_control/qwen25_axil_transport_spec.md
// 仿真工程:        tb/tb_qwen25_axi_lite_regs.sv
// 参考资料:        五份当前项目基线；不使用 module_requirements
// 依赖文件:        见下方显式命名实例
// 当前版本:        V1.0
// 修订日期:        2026年10月07日
// 修订历史:        2026年10月07日 V1.0 实现计算寄存器封装

`default_nettype none

// AXI-Lite 通道封装：独立保存 AW/W，内部寄存器请求始终被译码器接收。
// 每个方向最多一个在途访问，响应及其数据保持到主机完成握手。
module qwen25_axil_transport #(
	parameter AXIL_AW = 16,
	parameter RS_LV = 0
) (
	input wire clk, // 上升沿时钟，目标 300 MHz
	input wire rst_n, // 同步复位，rst_n == RS_LV 时有效
	input wire [AXIL_AW-1:0] s_axil_awaddr, // AXI 写地址，8-byte 对齐
	input wire s_axil_awvalid, // AW 独立请求有效
	output wire s_axil_awready, // 写地址槽可接收
	input wire [63:0] s_axil_wdata, // AXI 64-bit 软件写值
	input wire [7:0] s_axil_wstrb, // 八个独立写字节通道
	input wire s_axil_wvalid, // W 独立请求有效
	output wire s_axil_wready, // 写数据槽可接收
	output wire [1:0] s_axil_bresp, // 写响应 OKAY/SLVERR
	output wire s_axil_bvalid, // 已锁存写响应
	input wire s_axil_bready, // 主机接受写响应
	input wire [AXIL_AW-1:0] s_axil_araddr, // AXI 读地址，完整高位参与匹配
	input wire s_axil_arvalid, // 读地址请求有效
	output wire s_axil_arready, // 读地址槽可接收
	output wire [63:0] s_axil_rdata, // 锁存的读返回值
	output wire [1:0] s_axil_rresp, // 读响应 OKAY/SLVERR
	output wire s_axil_rvalid, // 读响应有效且可保持
	input wire s_axil_rready, // 主机接受读响应
	output wire csr_write_valid, // WR_DECODE 中的一次提交
	output wire [AXIL_AW-1:0] csr_write_addr, // 保存的 AW 地址
	output wire [63:0] csr_write_data, // 保存的 W 数据
	output wire [7:0] csr_write_strb, // 保存的 W 字节掩码
	input wire [1:0] csr_write_resp, // 总是就绪的译码器响应
	output wire csr_read_valid, // RD_DECODE 读采样阶段
	output wire [AXIL_AW-1:0] csr_read_addr, // 保存的 AR 地址
	input wire [63:0] csr_read_data, // 译码器组合读值
	input wire [1:0] csr_read_resp, // 译码器读权限结果
	output wire protocol_error
);
	localparam [1:0] WR_COLLECT = 2'd0, WR_DECODE = 2'd1, WR_BRESP = 2'd2;
	localparam [1:0] RD_IDLE = 2'd0, RD_DECODE = 2'd1, RD_RRESP = 2'd2;
	reg [1:0] write_state, write_next;
	reg [1:0] read_state, read_next;
	reg aw_pending, w_pending;
	reg [AXIL_AW-1:0] awaddr_reg, araddr_reg;
	reg [63:0] wdata_reg, rdata_o;
	reg [7:0] wstrb_reg;
	reg [1:0] bresp_o, rresp_o;
	reg bvalid_o, rvalid_o;
	reg reg_available;

    assign s_axil_awready = reg_available && (write_state == WR_COLLECT) && !aw_pending;
    assign s_axil_wready = reg_available && (write_state == WR_COLLECT) && !w_pending;
    assign s_axil_arready = reg_available && (read_state == RD_IDLE);
	assign s_axil_bresp = bresp_o;
	assign s_axil_bvalid = bvalid_o;
	assign s_axil_rdata = rdata_o;
	assign s_axil_rresp = rresp_o;
	assign s_axil_rvalid = rvalid_o;
    assign csr_write_valid = (write_state == WR_DECODE);
	assign csr_write_addr = awaddr_reg;
	assign csr_write_data = wdata_reg;
	assign csr_write_strb = wstrb_reg;
    assign csr_read_valid = (read_state == RD_DECODE);
	assign csr_read_addr = araddr_reg;
    assign protocol_error = 1'b0;

	// 写通道只在两个独立捕获均完成后进入一次译码。
	always @(*) begin
		write_next = write_state;
		case (write_state)
			WR_COLLECT: begin
				if (aw_pending && w_pending) begin
					write_next = WR_DECODE;
				end
			end
			WR_DECODE: begin
				write_next = WR_BRESP;
			end
			WR_BRESP: begin
				if (s_axil_bready) begin
					write_next = WR_COLLECT;
				end
			end
			default: begin
				write_next = WR_COLLECT;
			end
		endcase
	end

	// 读地址先寄存，再一次性锁存数据和响应。
	always @(*) begin
		read_next = read_state;
		case (read_state)
			RD_IDLE: begin
				if (s_axil_arvalid && s_axil_arready) begin
					read_next = RD_DECODE;
				end
			end
			RD_DECODE: begin
				read_next = RD_RRESP;
			end
			RD_RRESP: begin
				if (s_axil_rready) begin
					read_next = RD_IDLE;
				end
			end
			default: begin
				read_next = RD_IDLE;
			end
		endcase
	end

	// 状态寄存与数据/响应处理独立。
	always @(posedge clk) begin
		if (rst_n == RS_LV) begin
			write_state <= WR_COLLECT;
			read_state <= RD_IDLE;
		end else begin
			write_state <= write_next;
			read_state <= read_next;
		end
	end

	always @(posedge clk) begin
		if (rst_n == RS_LV) begin
			reg_available <= 1'b0;
			aw_pending <= 1'b0;
			w_pending <= 1'b0;
			awaddr_reg <= {AXIL_AW{1'b0}};
			araddr_reg <= {AXIL_AW{1'b0}};
			wdata_reg <= 64'b0;
			wstrb_reg <= 8'b0;
			bresp_o <= 2'b00;
			bvalid_o <= 1'b0;
			rdata_o <= 64'b0;
			rresp_o <= 2'b00;
			rvalid_o <= 1'b0;
		end else begin
			reg_available <= 1'b1;
			if (s_axil_awvalid && s_axil_awready) begin
				aw_pending <= 1'b1;
				awaddr_reg <= s_axil_awaddr;
			end
			if (s_axil_wvalid && s_axil_wready) begin
				w_pending <= 1'b1;
				wdata_reg <= s_axil_wdata;
				wstrb_reg <= s_axil_wstrb;
			end
			if (write_state == WR_DECODE) begin
				aw_pending <= 1'b0;
				w_pending <= 1'b0;
				bresp_o <= csr_write_resp;
				bvalid_o <= 1'b1;
			end else if (bvalid_o && s_axil_bready) begin
				bvalid_o <= 1'b0;
			end
			if (s_axil_arvalid && s_axil_arready) begin
				araddr_reg <= s_axil_araddr;
			end
			if (read_state == RD_DECODE) begin
				rdata_o <= csr_read_data;
				rresp_o <= csr_read_resp;
				rvalid_o <= 1'b1;
			end else if (rvalid_o && s_axil_rready) begin
				rvalid_o <= 1'b0;
			end
		end
	end
endmodule

`default_nettype wire
