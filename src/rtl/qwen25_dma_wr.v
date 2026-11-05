`timescale 1ns / 1ps
`default_nettype none

// 写通道先缓冲完整 burst；AW后发完W并等B，仅硬复位恢复错误。
module qwen25_dma_wr #(
    parameter RS_LV = 0
) (
    input wire clk,
    input wire rst_n,
    input wire s_cmd_valid,
    output wire s_cmd_ready,
    input wire [127:0] s_cmd,
    input wire [127:0] s_cmd_meta,
    input wire abort_req,
    input wire soft_reset_req,
    input wire [127:0] s_stream_data,
    input wire [15:0] s_stream_keep,
    input wire s_stream_last,
    input wire [127:0] s_stream_meta,
    input wire s_stream_valid,
    output wire s_stream_ready,
    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [127:0] m_axi_wdata,
    output wire [15:0] m_axi_wstrb,
    output wire m_axi_wlast,
    output wire m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready,
    output wire busy,
    output wire done,
    output wire error,
    output wire aborted,
    output wire [7:0] error_code,
    output wire [15:0] completion_tag,
    output wire [127:0] completion_meta
);
    localparam [3:0] ST_IDLE = 4'd0, ST_CHECK = 4'd1, ST_PLAN = 4'd2,
        ST_COLLECT = 4'd3, ST_PRIME = 4'd4, ST_AW = 4'd5,
        ST_WDATA = 4'd6, ST_BRESP = 4'd7, ST_NEXT = 4'd8;
    reg [3:0] state_current, state_next;
    reg [63:0] reg_address;
    reg [23:0] reg_remaining;
    reg [7:0] reg_limit;
    reg [15:0] reg_tag;
    reg [127:0] reg_metadata;
    reg [8:0] reg_burst_beats, cnt_collected, cnt_sent;
    reg [12:0] reg_burst_bytes;
    reg flag_done_o;
    reg reg_available;
    (* ram_style = "block" *) reg [143:0] burst_memory [0:255];
    reg [143:0] reg_memory_data;
    wire [8:0] planned_beats;
    wire [12:0] planned_bytes;
    wire source_fire, send_fire, collect_last, expected_last, send_last;
    wire [15:0] expected_keep, tail_keep;
    wire [7:0] memory_read_address;

    qwen25_dma_burst burst_plan_Inst (
        .address(reg_address), .remaining_bytes(reg_remaining), .maximum_len(reg_limit),
        .burst_beats(planned_beats), .burst_bytes(planned_bytes)
    );

    assign s_cmd_ready = reg_available && (state_current == ST_IDLE);
    assign s_stream_ready = (state_current == ST_COLLECT);
    assign source_fire = s_stream_valid && s_stream_ready;
    assign collect_last = (cnt_collected == reg_burst_beats - 9'd1);
    assign expected_last = collect_last && (reg_remaining == reg_burst_bytes);
    assign tail_keep = 16'hffff >> (5'd16 - {1'b0, reg_burst_bytes[3:0]});
    assign expected_keep = (collect_last && (reg_burst_bytes[3:0] != 0)) ? tail_keep : 16'hffff;
    assign m_axi_awaddr = reg_address;
    assign m_axi_awlen = reg_burst_beats[7:0] - 8'd1;
    assign m_axi_awsize = 3'd4;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = (state_current == ST_AW);
    assign m_axi_wdata = reg_memory_data[127:0];
    assign m_axi_wstrb = reg_memory_data[143:128];
    assign send_last = (cnt_sent == reg_burst_beats - 9'd1);
    assign m_axi_wlast = send_last;
    assign m_axi_wvalid = (state_current == ST_WDATA);
    assign send_fire = m_axi_wvalid && m_axi_wready;
    assign m_axi_bready = (state_current == ST_BRESP);
    assign memory_read_address = (state_current == ST_PRIME) ? 8'b0 : cnt_sent[7:0] + 8'd1;
    assign busy = state_current != ST_IDLE;
    assign done = flag_done_o;
    assign error = 1'b0;
    assign aborted = 1'b0;
    assign error_code = 8'd0;
    assign completion_tag = reg_tag;
    assign completion_meta = reg_metadata;

    // BRAM 不复位；只有有效 COLLECT 写入，只有 PRIME/已接受 W 更新读寄存器。
    always @(posedge clk) begin
        if (source_fire && (rst_n != RS_LV)) begin
            burst_memory[cnt_collected[7:0]] <= {s_stream_keep, s_stream_data};
        end
        if ((state_current == ST_PRIME) || (send_fire && !send_last)) begin
            reg_memory_data <= burst_memory[memory_read_address];
        end
    end

    always @(*) begin
        state_next = state_current;
        case (state_current)
            ST_IDLE: begin
                if (s_cmd_valid && s_cmd_ready) begin state_next = ST_CHECK; end
            end
            ST_CHECK: begin state_next = ST_PLAN; end
            ST_PLAN: begin
                state_next = ST_COLLECT;
            end
            ST_COLLECT: begin
                if (source_fire && collect_last) begin state_next = ST_PRIME; end
            end
            ST_PRIME: begin
                state_next = ST_AW;
            end
            ST_AW: begin
                if (m_axi_awready) begin state_next = ST_WDATA; end
            end
            ST_WDATA: begin
                if (send_fire && send_last) begin state_next = ST_BRESP; end
            end
            ST_BRESP: begin
                if (m_axi_bvalid) begin state_next = ST_NEXT; end
            end
            ST_NEXT: begin
                if (reg_remaining == reg_burst_bytes) begin state_next = ST_IDLE; end
                else begin state_next = ST_PLAN; end
            end
            default: begin state_next = ST_IDLE; end
        endcase
    end

    always @(posedge clk) begin
        if (rst_n == RS_LV) begin state_current <= ST_IDLE; end
        else begin state_current <= state_next; end
    end

    always @(posedge clk) begin
        if (rst_n == RS_LV) begin
            reg_available <= 1'b0;
            reg_address <= 0;
            reg_remaining <= 0;
            reg_limit <= 0;
            reg_tag <= 0;
            reg_metadata <= 0;
            reg_burst_beats <= 0;
            reg_burst_bytes <= 0;
            cnt_collected <= 0;
            cnt_sent <= 0;
            flag_done_o <= 0;
        end else begin
            reg_available <= 1'b1;
            flag_done_o <= 0;
            case (state_current)
                ST_IDLE: begin
                    if (s_cmd_valid && s_cmd_ready) begin
                        reg_address <= s_cmd[63:0];
                        reg_remaining <= s_cmd[87:64];
                        reg_limit <= s_cmd[95:88];
                        reg_tag <= s_cmd[111:96];
                        reg_metadata <= s_cmd_meta;
                        reg_burst_bytes <= 0;
                    end
                end
                ST_CHECK: begin end
                ST_PLAN: begin
                    reg_burst_beats <= planned_beats;
                    reg_burst_bytes <= planned_bytes;
                    cnt_collected <= 0;
                    cnt_sent <= 0;
                end
                ST_COLLECT: begin
                    if (source_fire) begin
                        cnt_collected <= cnt_collected + 9'd1;
                    end
                end
                ST_WDATA: begin
                    if (send_fire) begin cnt_sent <= cnt_sent + 9'd1; end
                end
                ST_BRESP: begin end
                ST_NEXT: begin
                    if (reg_remaining == reg_burst_bytes) begin flag_done_o <= 1'b1; end
                    else begin
                        reg_remaining <= reg_remaining - reg_burst_bytes;
                        reg_address <= reg_address + {51'b0, reg_burst_beats, 4'b0};
                    end
                end
                default: begin
                    begin end
                end
            endcase
        end
    end
endmodule

`default_nettype wire
