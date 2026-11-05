`timescale 1ns / 1ps
`default_nettype none

// 独立 DDR 读引擎：一笔 burst 在途，一个弹性输出槽；仅硬复位恢复错误。
module qwen25_dma_rd #(
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
    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [3:0] m_axi_rid,
    input wire [127:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast,
    input wire m_axi_rvalid,
    output wire m_axi_rready,
    output wire [127:0] m_stream_data,
    output wire [15:0] m_stream_keep,
    output wire m_stream_last,
    output wire [127:0] m_stream_meta,
    output wire [15:0] m_stream_tag,
    output wire [3:0] m_stream_resource,
    output wire [1:0] m_stream_error,
    output wire m_stream_valid,
    input wire m_stream_ready,
    output wire busy,
    output wire done,
    output wire error,
    output wire aborted,
    output wire [7:0] error_code,
    output wire [15:0] completion_tag,
    output wire [127:0] completion_meta
);
    localparam [3:0] ST_IDLE = 4'd0, ST_CHECK = 4'd1, ST_PLAN = 4'd2,
        ST_AR = 4'd3, ST_RDATA = 4'd4, ST_COMMIT = 4'd5, ST_NEXT = 4'd6;
    reg [3:0] state_current, state_next;
    reg [63:0] reg_address;
    reg [23:0] reg_remaining;
    reg [7:0] reg_limit;
    reg [15:0] reg_tag;
    reg [3:0] reg_resource;
    reg [127:0] reg_metadata;
    reg [8:0] reg_burst_beats, cnt_received;
    reg [12:0] reg_burst_bytes;
    reg flag_done_o;
    reg reg_available;
    reg [127:0] reg_stream_data_o;
    reg [15:0] reg_stream_keep_o;
    reg flag_stream_valid_o, flag_stream_last_o;
    wire [8:0] planned_beats;
    wire [12:0] planned_bytes;
    wire stream_slot, read_fire, expected_beat, expected_last, descriptor_last;
    wire [15:0] tail_keep;

    qwen25_dma_burst burst_plan_Inst (
        .address(reg_address), .remaining_bytes(reg_remaining), .maximum_len(reg_limit),
        .burst_beats(planned_beats), .burst_bytes(planned_bytes)
    );

    assign s_cmd_ready = reg_available && (state_current == ST_IDLE);
    assign m_axi_araddr = reg_address;
    assign m_axi_arlen = reg_burst_beats[7:0] - 8'd1;
    assign m_axi_arsize = 3'd4;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arvalid = (state_current == ST_AR);
    assign expected_beat = (cnt_received < reg_burst_beats);
    assign expected_last = (cnt_received == reg_burst_beats - 9'd1);
    assign descriptor_last = (reg_remaining == reg_burst_bytes) && expected_last;
    assign stream_slot = !flag_stream_valid_o || m_stream_ready;
    // 晚 RLAST 的额外拍只排空；之前呈现的数据仍必须稳定到消费者接收。
    assign m_axi_rready = (state_current == ST_RDATA) && (stream_slot || !expected_beat);
    assign read_fire = m_axi_rvalid && m_axi_rready;
    assign tail_keep = 16'hffff >> (5'd16 - {1'b0, reg_burst_bytes[3:0]});
    assign m_stream_data = reg_stream_data_o;
    assign m_stream_keep = reg_stream_keep_o;
    assign m_stream_last = flag_stream_last_o;
    assign m_stream_meta = reg_metadata;
    assign m_stream_tag = reg_tag;
    assign m_stream_resource = reg_resource;
    assign m_stream_error = 2'b00;
    assign m_stream_valid = flag_stream_valid_o;
    assign busy = state_current != ST_IDLE;
    assign done = flag_done_o;
    assign error = 1'b0;
    assign aborted = 1'b0;
    assign error_code = 8'd0;
    assign completion_tag = reg_tag;
    assign completion_meta = reg_metadata;

    always @(*) begin
        state_next = state_current;
        case (state_current)
            ST_IDLE: begin
                if (s_cmd_valid && s_cmd_ready) begin state_next = ST_CHECK; end
            end
            ST_CHECK: begin state_next = ST_PLAN; end
            ST_PLAN: begin
                state_next = ST_AR;
            end
            ST_AR: begin
                if (m_axi_arready) begin state_next = ST_RDATA; end
            end
            ST_RDATA: begin
                if (read_fire && m_axi_rlast) begin state_next = ST_COMMIT; end
            end
            ST_COMMIT: begin
                if (stream_slot) begin state_next = ST_NEXT; end
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
            reg_resource <= 0;
            reg_metadata <= 0;
            reg_burst_beats <= 0;
            reg_burst_bytes <= 0;
            cnt_received <= 0;
            flag_done_o <= 0;
            flag_stream_valid_o <= 0;
            flag_stream_last_o <= 0;
            reg_stream_data_o <= 0;
            reg_stream_keep_o <= 0;
        end else begin
            reg_available <= 1'b1;
            flag_done_o <= 0;
            if (flag_stream_valid_o && m_stream_ready) begin flag_stream_valid_o <= 1'b0; end

            case (state_current)
                ST_IDLE: begin
                    if (s_cmd_valid && s_cmd_ready) begin
                        reg_address <= s_cmd[63:0];
                        reg_remaining <= s_cmd[87:64];
                        reg_limit <= s_cmd[95:88];
                        reg_tag <= s_cmd[111:96];
                        reg_resource <= s_cmd[115:112];
                        reg_metadata <= s_cmd_meta;
                        reg_burst_bytes <= 0;
                    end
                end
                ST_CHECK: begin end
                ST_PLAN: begin
                    reg_burst_beats <= planned_beats;
                    reg_burst_bytes <= planned_bytes;
                    cnt_received <= 0;
                end
                ST_RDATA: begin
                    if (read_fire) begin
                        if (expected_beat) begin
                            cnt_received <= cnt_received + 9'd1;
                            reg_stream_data_o <= m_axi_rdata;
                            reg_stream_keep_o <= (expected_last && (reg_burst_bytes[3:0] != 0)) ? tail_keep : 16'hffff;
                            flag_stream_last_o <= descriptor_last;
                            flag_stream_valid_o <= 1'b1;
                        end
                    end
                end
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
