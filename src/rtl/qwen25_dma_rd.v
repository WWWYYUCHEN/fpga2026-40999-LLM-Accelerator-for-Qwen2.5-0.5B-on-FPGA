`timescale 1ns / 1ps
`default_nettype none

// Read engine: descriptor = {addr, byte count (24b), max burst len}.
// The engine splits the descriptor internally into AXI INCR bursts of at most
// 16 beats (256B) that never cross a 4KB boundary, and keeps up to OUTSTANDING
// ARs in flight (single ID, responses in AR order).
// Address only needs 16B alignment; the length may be any multiple of 16B.
module qwen25_dma_rd #(
    parameter RS_LV = 0,
    parameter OUTSTANDING = 8
) (
    input  wire         clk,
    input  wire         rst_n,

    input  wire [127:0] s_cmd,
    input  wire [127:0] s_cmd_meta,
    input  wire         s_cmd_valid,
    output wire         s_cmd_ready,

    input  wire         abort_req,
    input  wire         soft_reset_req,

    output wire [63:0]  m_axi_araddr,
    output wire [7:0]   m_axi_arlen,
    output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst,
    output wire         m_axi_arvalid,
    input  wire         m_axi_arready,
    input  wire [3:0]   m_axi_rid,
    input  wire [127:0] m_axi_rdata,
    input  wire [1:0]   m_axi_rresp,
    input  wire         m_axi_rlast,
    input  wire         m_axi_rvalid,
    output wire         m_axi_rready,

    output wire [127:0] m_stream_data,
    output wire [15:0]  m_stream_keep,
    output wire         m_stream_last,
    output wire [127:0] m_stream_meta,
    output wire [15:0]  m_stream_tag,
    output wire [3:0]   m_stream_resource,
    output wire [1:0]   m_stream_error,
    output wire         m_stream_valid,
    input  wire         m_stream_ready,

    output wire         busy,
    output wire         done,
    output wire         error,
    output wire         aborted,
    output wire [7:0]   error_code,
    output wire [15:0]  completion_tag,
    output wire [127:0] completion_meta
);
    localparam [1:0] ST_IDLE = 2'd0, ST_RUN = 2'd1;

    reg [1:0]   state_current, state_next;

    reg [63:0]  rx_base;
    reg [15:0]  rx_tag;
    reg [3:0]   rx_res;
    reg [127:0] rx_meta;
    reg [7:0]   rx_max;

    reg [63:0]  ar_addr;        // address of the next burst to issue
    reg [63:0]  rx_addr;        // address of the burst currently being received
    reg [16:0]  ar_left;        // beats not yet issued
    reg [16:0]  rx_left;        // beats not yet received
    reg [4:0]   rx_beat;        // beat index inside the current burst
    reg [4:0]   ar_out;         // ARs issued but not completed

    reg         flag_done_o, flag_error_o;
    reg [7:0]   reg_error_code;
    reg         reg_available;

    reg [127:0] reg_stream_data;
    reg         reg_stream_last;
    reg         reg_stream_valid;

    wire [23:0] cmd_bytes = s_cmd[87:64];
    wire [7:0]  cmd_max   = s_cmd[95:88];
    wire [16:0] cmd_beats = {9'd0, cmd_bytes[23:4]} + (|cmd_bytes[3:0] ? 17'd1 : 17'd0);

    wire cmd_fire = s_cmd_valid && s_cmd_ready;
    wire ar_fire  = m_axi_arvalid && m_axi_arready;
    wire r_fire   = m_axi_rvalid && m_axi_rready;

    // beats of the burst currently being issued
    // plan in 9 bits: up to 256 beats to the 4KB page end, then cap at 16
    wire [8:0] b_max     = (rx_max == 8'd255) ? 9'd16 : {4'd0, rx_max} + 9'd1;
    wire [8:0] b_page    = 9'd256 - {4'd0, ar_addr[11:4]};   // beats to the 4KB page end
    wire [8:0] b_left    = (ar_left > 17'd16) ? 9'd16 : {4'd0, ar_left[4:0]};
    wire [8:0] b_min     = (b_left < b_page) ? ((b_left < b_max) ? b_left : b_max)
                                             : ((b_page < b_max) ? b_page : b_max);
    wire [4:0] ar_beats  = (b_min > 9'd16) ? 5'd16 : b_min[4:0];

    // beats of the burst currently being received: must mirror the AR planner
    wire [8:0] rx_page  = 9'd256 - {4'd0, rx_addr[11:4]};
    wire [8:0] rx_mx    = (rx_max == 8'd255) ? 9'd16 : {4'd0, rx_max} + 9'd1;
    wire [8:0] rx_cap   = (rx_left > 17'd16) ? 9'd16 : {4'd0, rx_left[4:0]};
    wire [8:0] rx_cap2  = (rx_cap < rx_mx) ? rx_cap : rx_mx;
    wire [8:0] rx_min   = (rx_cap2 < rx_page) ? rx_cap2 : rx_page;
    wire [4:0] rx_beats = (rx_min > 9'd16) ? 5'd16 : rx_min[4:0];
    wire       rx_last_beat = (rx_beat + 5'd1 == rx_beats);

    wire [4:0] ar_room   = (ar_out < OUTSTANDING[4:0]);
    wire       ar_need   = (ar_left != 17'd0);
    wire       rx_busy   = (rx_left != 17'd0);

    wire stream_room = !reg_stream_valid || m_stream_ready;

    assign s_cmd_ready   = reg_available && (state_current == ST_IDLE);
    assign m_axi_arvalid = (state_current == ST_RUN) && ar_need && ar_room;
    assign m_axi_araddr  = ar_addr;
    assign m_axi_arlen   = ar_beats - 5'd1;
    assign m_axi_arsize  = 3'd4;
    assign m_axi_arburst = 2'b01;
    assign m_axi_rready  = (state_current == ST_RUN) && rx_busy && stream_room;

    assign m_stream_data     = reg_stream_data;
    assign m_stream_keep     = 16'hffff;
    assign m_stream_last     = reg_stream_last;
    assign m_stream_meta     = rx_meta;
    assign m_stream_tag      = rx_tag;
    assign m_stream_resource = rx_res;
    assign m_stream_error    = 2'b00;
    assign m_stream_valid    = reg_stream_valid;

    assign busy            = (state_current != ST_IDLE);
    assign done            = flag_done_o;
    assign error           = flag_error_o;
    assign aborted         = 1'b0;
    assign error_code      = reg_error_code;
    assign completion_tag  = rx_tag;
    assign completion_meta = rx_meta;

    always @(*) begin
        state_next = state_current;
        case (state_current)
            ST_IDLE: if (cmd_fire) state_next = ST_RUN;
            ST_RUN:  if (!rx_busy && !reg_stream_valid && (ar_left == 17'd0) &&
                         (ar_out == 5'd0)) state_next = ST_IDLE;
            default: state_next = ST_IDLE;
        endcase
    end

    always @(posedge clk) begin
        if (rst_n == RS_LV) state_current <= ST_IDLE;
        else                state_current <= state_next;
    end

    always @(posedge clk) begin
        if (rst_n == RS_LV) begin
            reg_available    <= 1'b0;
            rx_base          <= 64'd0;
            rx_tag           <= 16'd0;
            rx_res           <= 4'd0;
            rx_meta          <= 128'd0;
            rx_max           <= 8'd255;
            ar_addr          <= 64'd0;
            rx_addr          <= 64'd0;
            ar_left          <= 17'd0;
            rx_left          <= 17'd0;
            rx_beat          <= 5'd0;
            ar_out           <= 5'd0;
            flag_done_o      <= 1'b0;
            flag_error_o     <= 1'b0;
            reg_error_code   <= 8'd0;
            reg_stream_data  <= 128'd0;
            reg_stream_last  <= 1'b0;
            reg_stream_valid <= 1'b0;
        end else begin
            reg_available <= 1'b1;
            flag_done_o   <= 1'b0;

            if (state_current == ST_IDLE && cmd_fire) begin
                rx_base  <= s_cmd[63:0];
                rx_tag   <= s_cmd[111:96];
                rx_res   <= s_cmd[115:112];
                rx_meta  <= s_cmd_meta;
                rx_max   <= cmd_max;
                ar_addr  <= s_cmd[63:0];
                rx_addr  <= s_cmd[63:0];
                ar_left  <= cmd_beats;
                rx_left  <= cmd_beats;
                rx_beat  <= 5'd0;
                ar_out   <= 5'd0;
                flag_error_o   <= 1'b0;
                reg_error_code <= 8'd0;
                if (cmd_bytes == 24'd0 || s_cmd[3:0] != 4'd0) begin
                    flag_error_o   <= 1'b1;
                    reg_error_code <= 8'h21;   // empty descriptor or unaligned address
                    ar_left        <= 17'd0;
                    rx_left        <= 17'd0;
                end
            end

            if (ar_fire) begin
                ar_addr <= ar_addr + {55'd0, ar_beats, 4'd0};
                ar_left <= ar_left - {12'd0, ar_beats};
                ar_out  <= ar_out + 5'd1;
            end

            if (r_fire) begin
                reg_stream_data  <= m_axi_rdata;
                reg_stream_valid <= 1'b1;
                reg_stream_last  <= rx_last_beat && (rx_left <= 17'd16);
                if (rx_last_beat) begin
                    rx_beat <= 5'd0;
                    rx_addr <= rx_addr + {55'd0, rx_beats, 4'd0};
                    rx_left <= (rx_left >= {12'd0, rx_beats}) ? (rx_left - {12'd0, rx_beats}) : 17'd0;
                    ar_out  <= ar_out - 5'd1;
                end else begin
                    rx_beat <= rx_beat + 5'd1;
                end
            end else if (reg_stream_valid && m_stream_ready) begin
                reg_stream_valid <= 1'b0;
            end

            if ((state_current == ST_RUN) && !rx_busy && !reg_stream_valid &&
                (ar_left == 17'd0) && (ar_out == 5'd0))
                flag_done_o <= 1'b1;
        end
    end
endmodule

`default_nettype wire
