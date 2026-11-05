`timescale 1ns / 1ps
`default_nettype none

// Write engine: descriptor = {addr, byte count (24b), max burst len}.
// Splits internally into 16-beat (256B) INCR bursts that never cross a 4KB
// boundary, keeps up to AW_PIPE AWs in flight, and forwards W beats straight
// from the local stream (strictly in AW order, no interleaving).
// A descriptor frees the issuing slot as soon as its last W beat is sent.
module qwen25_dma_wr #(
    parameter RS_LV = 0,
    parameter AW_PIPE = 4
) (
    input  wire         clk,
    input  wire         rst_n,

    input  wire [127:0] s_cmd,
    input  wire [127:0] s_cmd_meta,
    input  wire         s_cmd_valid,
    output wire         s_cmd_ready,

    input  wire         abort_req,
    input  wire         soft_reset_req,

    input  wire [127:0] s_stream_data,
    input  wire [15:0]  s_stream_keep,
    input  wire         s_stream_last,
    input  wire [127:0] s_stream_meta,
    input  wire         s_stream_valid,
    output wire         s_stream_ready,

    output wire [63:0]  m_axi_awaddr,
    output wire [7:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire         m_axi_awvalid,
    input  wire         m_axi_awready,

    output wire [127:0] m_axi_wdata,
    output wire [15:0]  m_axi_wstrb,
    output wire         m_axi_wlast,
    output wire         m_axi_wvalid,
    input  wire         m_axi_wready,

    input  wire [3:0]   m_axi_bid,
    input  wire [1:0]   m_axi_bresp,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready,

    output wire         busy,
    output wire         done,
    output wire         error,
    output wire         aborted,
    output wire [7:0]   error_code,
    output wire [15:0]  completion_tag,
    output wire [127:0] completion_meta
);
    localparam integer QD = 2;

    reg [63:0]  q_base   [0:QD-1];
    reg [23:0]  q_bytes  [0:QD-1];
    reg [7:0]   q_max    [0:QD-1];
    reg [15:0]  q_tag    [0:QD-1];
    reg [127:0] q_meta   [0:QD-1];
    reg [0:0]   q_wr, q_rd;
    reg [1:0]   q_count;

    reg         cur_valid;
    reg [63:0]  cur_base;
    reg [7:0]   cur_max;
    reg [15:0]  cur_tag;
    reg [127:0] cur_meta;
    reg [63:0]  aw_addr;
    reg [16:0]  aw_left;       // beats not yet covered by an AW
    reg [16:0]  w_left;        // beats not yet sent on W
    reg [4:0]   w_beat;
    reg [4:0]   aw_out;        // AWs without B

    reg         flag_done_o, flag_error_o;
    reg [7:0]   reg_error_code;
    reg         reg_available;

    wire [23:0] cmd_bytes = s_cmd[87:64];
    wire [7:0]  cmd_max   = s_cmd[95:88];

    wire cmd_fire = s_cmd_valid && s_cmd_ready;
    wire aw_fire  = m_axi_awvalid && m_axi_awready;
    wire w_fire   = m_axi_wvalid && m_axi_wready;
    wire b_fire   = m_axi_bvalid && m_axi_bready;

    wire q_push = cmd_fire && (cmd_bytes != 24'd0) && (s_cmd[3:0] == 4'd0);
    wire q_pop  = (!cur_valid) && (q_count != 2'd0);

    wire [8:0] b_max    = (cur_max == 8'd255) ? 9'd16 : {4'd0, cur_max} + 9'd1;
    wire [8:0] b_page   = 9'd256 - {4'd0, aw_addr[11:4]};
    wire [8:0] b_left   = (aw_left > 17'd16) ? 9'd16 : {4'd0, aw_left[4:0]};
    wire [8:0] b_min    = (b_left < b_page) ? ((b_left < b_max) ? b_left : b_max)
                                            : ((b_page < b_max) ? b_page : b_max);
    wire [4:0] aw_beats = (b_min > 9'd16) ? 5'd16 : b_min[4:0];

    wire [4:0] w_beats     = (w_left > 17'd16) ? 5'd16 : w_left[4:0];
    wire       w_last_beat = (w_beat + 5'd1 == w_beats);

    wire [4:0] aw_room = (aw_out < AW_PIPE[4:0]);
    wire       aw_need = cur_valid && (aw_left != 17'd0);
    wire       w_ready_ctx = cur_valid && (w_left != 17'd0);

    assign s_cmd_ready   = reg_available && (q_count < 2'd2);
    assign m_axi_awaddr  = aw_addr;
    assign m_axi_awlen   = aw_beats - 5'd1;
    assign m_axi_awsize  = 3'd4;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = aw_need && aw_room;

    assign s_stream_ready = w_ready_ctx && m_axi_wready;
    assign m_axi_wvalid   = s_stream_valid && w_ready_ctx;
    assign m_axi_wdata    = s_stream_data;
    assign m_axi_wstrb    = s_stream_keep;
    assign m_axi_wlast    = w_last_beat;

    // BREADY must stay high while any AW is outstanding: a pipelined B response
    // can arrive after the last W beat, and if it is not accepted aw_out never
    // decrements and the completion pulse is lost.
    assign m_axi_bready   = (aw_out != 5'd0) || cur_valid || (q_count != 2'd0);

    assign busy            = cur_valid || (q_count != 0);
    assign done            = flag_done_o;
    assign error           = flag_error_o;
    assign aborted         = 1'b0;
    assign error_code      = reg_error_code;
    assign completion_tag  = cur_tag;
    assign completion_meta = cur_meta;

    always @(posedge clk) begin
        if (rst_n == RS_LV) begin
            q_wr <= 0; q_rd <= 0; q_count <= 0;
            cur_valid <= 0; cur_base <= 0; cur_max <= 8'd255;
            cur_tag <= 0; cur_meta <= 0;
            aw_addr <= 0; aw_left <= 0; w_left <= 0; w_beat <= 0; aw_out <= 0;
            flag_done_o <= 0; flag_error_o <= 0; reg_error_code <= 0;
            reg_available <= 0;
        end else begin
            reg_available <= 1'b1;
            flag_done_o   <= 1'b0;

            if (cmd_fire && !q_push) begin
                flag_error_o   <= 1'b1;
                reg_error_code <= 8'h21;
            end

            if (q_push) begin
                q_base[q_wr]  <= s_cmd[63:0];
                q_bytes[q_wr] <= cmd_bytes;
                q_max[q_wr]   <= cmd_max;
                q_tag[q_wr]   <= s_cmd[111:96];
                q_meta[q_wr]  <= s_cmd_meta;
                q_wr          <= q_wr + 1'b1;
            end

            if (q_pop) begin
                cur_valid <= 1'b1;
                cur_base  <= q_base[q_rd];
                cur_max   <= q_max[q_rd];
                cur_tag   <= q_tag[q_rd];
                cur_meta  <= q_meta[q_rd];
                aw_addr   <= q_base[q_rd];
                aw_left   <= {9'd0, q_bytes[q_rd][23:4]} + (|q_bytes[q_rd][3:0] ? 17'd1 : 17'd0);
                w_left    <= {9'd0, q_bytes[q_rd][23:4]} + (|q_bytes[q_rd][3:0] ? 17'd1 : 17'd0);
                w_beat    <= 5'd0;
                q_rd      <= q_rd + 1'b1;
            end

            if (aw_fire) begin
                aw_addr <= aw_addr + {55'd0, aw_beats, 4'd0};
                aw_left <= aw_left - {12'd0, aw_beats};
                aw_out  <= aw_out + 5'd1;
            end

            if (w_fire) begin
                if (w_last_beat) begin
                    w_beat <= 5'd0;
                    w_left <= (w_left >= {12'd0, w_beats}) ? (w_left - {12'd0, w_beats}) : 17'd0;
                    if (w_left <= 17'd16) cur_valid <= 1'b0;   // last beat of descriptor
                end else begin
                    w_beat <= w_beat + 5'd1;
                end
            end

            if (b_fire) begin
                aw_out <= aw_out - 5'd1;
                if (m_axi_bid != 4'd0) begin
                    flag_error_o   <= 1'b1;
                    reg_error_code <= 8'h32;
                end
                if (m_axi_bresp != 2'b00) begin
                    flag_error_o   <= 1'b1;
                    reg_error_code <= 8'h31;
                end
                if ((aw_out == 5'd1) && (w_left == 17'd0) && !cur_valid) flag_done_o <= 1'b1;
            end

            case ({q_push, q_pop})
                2'b10:   q_count <= q_count + 2'd1;
                2'b01:   q_count <= q_count - 2'd1;
                default: q_count <= q_count;
            endcase
        end
    end
endmodule

`default_nettype wire
