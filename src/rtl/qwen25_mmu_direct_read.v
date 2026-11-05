`timescale 1ns / 1ps
`default_nettype none

// 4-lane direct read router with a command queue.
//   resource == 1 -> weight read: all four lanes, fixed 4KB lane stride
//   otherwise     -> single lane, round-robin over the four lanes
// Commands are queued first and dispatched from the queue head, so the lane
// command word is a registered value and never races with a state transition.
// Every descriptor is completed per lane; the router retires it once all lanes
// it owns have finished.
module qwen25_mmu_direct_read #(
    parameter RS_LV = 0
) (
    input  wire         clk,
    input  wire         rst_n,

    input  wire [127:0] s_cmd,
    input  wire [127:0] s_cmd_meta,
    input  wire         s_cmd_valid,
    output wire         s_cmd_ready,

    input  wire         abort_req,
    input  wire         soft_reset_req,

    output wire [255:0] m_axi_araddr,
    output wire [31:0]  m_axi_arlen,
    output wire [11:0]  m_axi_arsize,
    output wire [7:0]   m_axi_arburst,
    output wire [3:0]   m_axi_arvalid,
    input  wire [3:0]   m_axi_arready,
    input  wire [15:0]  m_axi_rid,
    input  wire [511:0] m_axi_rdata,
    input  wire [7:0]   m_axi_rresp,
    input  wire [3:0]   m_axi_rlast,
    input  wire [3:0]   m_axi_rvalid,
    output wire [3:0]   m_axi_rready,

    output wire [127:0] m_stream_data,
    output wire [15:0]  m_stream_keep,
    output wire         m_stream_last,
    output wire [127:0] m_stream_meta,
    output wire [15:0]  m_stream_tag,
    output wire [3:0]   m_stream_resource,
    output wire [1:0]   m_stream_error,
    output wire         m_stream_valid,
    input  wire         m_stream_ready,

    output wire [3:0]   m_buffer_valid,
    input  wire [3:0]   m_buffer_ready,
    output wire [511:0] m_buffer_data,
    output wire [39:0]  m_buffer_word,
    output wire [63:0]  m_buffer_keep,
    output wire [7:0]   m_buffer_error,
    output wire [127:0] m_buffer_meta,

    output wire         busy,
    output wire         done,
    output wire         error,
    output wire         aborted,
    output wire [7:0]   error_code,
    output wire [15:0]  completion_tag,
    output wire [127:0] completion_meta
);
    localparam integer QD = 2;

    // ---------------------------------------------------------------- command queue
    reg [127:0] f_cmd  [0:QD-1];
    reg [127:0] f_meta [0:QD-1];
    reg [0:0]   f_wr, f_rd;
    reg [1:0]   f_count;

    // ---------------------------------------------------------------- in-flight table
    reg [3:0]   t_mask [0:QD-1];
    reg [2:0]   t_rem  [0:QD-1];
    reg [15:0]  t_tag  [0:QD-1];
    reg [127:0] t_meta [0:QD-1];
    reg [3:0]   t_res  [0:QD-1];
    reg [0:0]   t_wr, t_rd;
    reg [1:0]   t_count;

    reg [3:0]   lane_busy;
    reg [3:0]   lane_w;
    reg [1:0]   lane_owner [0:3];
    reg [1:0]   serial_port;          // next lane for a single-lane descriptor
    reg [1:0]   t_lane [0:QD-1];      // lane that owns each in-flight entry

    reg         flag_done_o;

    wire cmd_fire = s_cmd_valid && s_cmd_ready;

    wire [3:0] lane_done;
    wire [3:0] lane_ready;
    wire [3:0] lane_valid, lane_last;
    wire [511:0] lane_data;
    wire [63:0]  lane_keep;

    reg [8:0]    word_count [0:3];

    // ---------------------------------------------------------------- dispatch
    // The stream port always belongs to the oldest in-flight descriptor (t_rd),
    // never to the descriptor that most recently dispatched: with a two-deep
    // queue the head can advance while the previous descriptor is still
    // streaming its beats.
    wire [1:0]  stream_lane = t_lane[t_rd];
    wire [3:0]  head_res   = f_cmd[f_rd][115:112];
    wire [3:0]  head_mask  = (head_res == 4'd1) ? 4'b1111 : (4'b0001 << serial_port);
    wire [23:0] head_bytes = f_cmd[f_rd][87:64];
    wire [23:0] weight_lane_bytes = (head_bytes == 24'd8192) ? 24'd2048 : 24'd4096;
    wire        lane_free  = ((head_mask & lane_busy) == 4'b0000);
    wire        lanes_rdy  = ((head_mask & ~lane_ready) == 4'b0000);
    wire        disp       = (f_count != 2'd0) && lane_free && lanes_rdy && (t_count < QD[1:0]);

    assign s_cmd_ready = (f_count < QD[1:0]);

    genvar port;
    generate for (port = 0; port < 4; port = port + 1) begin : gen_reader
        wire lane_start = (f_count != 2'd0) && head_mask[port] && lane_free && (t_count < QD[1:0]);
        wire [127:0] lane_cmd = (head_res == 4'd1)
            ? {f_cmd[f_rd][127:88], weight_lane_bytes, f_cmd[f_rd][63:0] + port * 64'd4096}
            : f_cmd[f_rd];
        // One stream port, one owner: only the lane of the oldest in-flight
        // descriptor may advance.  Without this gate every non-weight lane sees
        // m_stream_ready, so an overlapping descriptor's beats are silently
        // consumed while the mux is pointing at another lane.
        wire stream_sel   = (t_count != 2'd0) && (stream_lane == port[1:0]);
        wire stream_ready = lane_w[port] ? m_buffer_ready[port] : (m_stream_ready && stream_sel);

        qwen25_dma_rd #(.RS_LV(RS_LV)) reader_Inst (
            .clk(clk), .rst_n(rst_n),
            .s_cmd(lane_cmd), .s_cmd_meta(f_meta[f_rd]),
            .s_cmd_valid(lane_start), .s_cmd_ready(lane_ready[port]),
            .abort_req(abort_req), .soft_reset_req(soft_reset_req),
            .m_axi_araddr(m_axi_araddr[port*64 +: 64]),
            .m_axi_arlen(m_axi_arlen[port*8 +: 8]),
            .m_axi_arsize(m_axi_arsize[port*3 +: 3]),
            .m_axi_arburst(m_axi_arburst[port*2 +: 2]),
            .m_axi_arvalid(m_axi_arvalid[port]), .m_axi_arready(m_axi_arready[port]),
            .m_axi_rid(m_axi_rid[port*4 +: 4]),
            .m_axi_rdata(m_axi_rdata[port*128 +: 128]),
            .m_axi_rresp(m_axi_rresp[port*2 +: 2]),
            .m_axi_rlast(m_axi_rlast[port]), .m_axi_rvalid(m_axi_rvalid[port]),
            .m_axi_rready(m_axi_rready[port]),
            .m_stream_data(lane_data[port*128 +: 128]),
            .m_stream_keep(lane_keep[port*16 +: 16]),
            .m_stream_last(lane_last[port]), .m_stream_meta(),
            .m_stream_tag(), .m_stream_resource(), .m_stream_error(),
            .m_stream_valid(lane_valid[port]), .m_stream_ready(stream_ready),
            .busy(), .done(lane_done[port]),
            .error(), .aborted(), .error_code(),
            .completion_tag(), .completion_meta()
        );
    end endgenerate

    // ---------------------------------------------------------------- outputs
    assign m_stream_data     = lane_data[stream_lane*128 +: 128];
    assign m_stream_keep     = lane_keep[stream_lane*16 +: 16];
    assign m_stream_last     = lane_last[stream_lane];
    assign m_stream_meta     = t_meta[t_rd];
    assign m_stream_tag      = t_tag[t_rd];
    assign m_stream_resource = t_res[t_rd];
    assign m_stream_error    = 2'b00;
    assign m_stream_valid    = lane_valid[stream_lane] & ~lane_w[stream_lane];

    assign m_buffer_valid = lane_valid & lane_w;
    assign m_buffer_data  = lane_data;
    assign m_buffer_keep  = lane_keep;
    assign m_buffer_error = 8'd0;
    assign m_buffer_meta  = t_meta[t_rd];

    genvar w;
    generate for (w = 0; w < 4; w = w + 1) begin : gen_word
        assign m_buffer_word[w*10 +: 10] = {word_count[w][7:0], w[1:0]};
    end endgenerate

    assign busy            = (f_count != 0) || (t_count != 0) || (lane_busy != 4'd0);
    assign done            = flag_done_o;
    assign error           = 1'b0;
    assign aborted         = 1'b0;
    assign error_code      = 8'd0;
    assign completion_tag  = t_tag[t_rd];
    assign completion_meta = t_meta[t_rd];

    // Number of lanes belonging to each in-flight entry that report done in
    // this cycle.  All four weight lanes can finish on the same edge.
    reg [2:0] done_cnt [0:QD-1];
    integer d;
    always @(*) begin
        for (d = 0; d < QD; d = d + 1) done_cnt[d] = 3'd0;
        for (d = 0; d < 4; d = d + 1)
            if (lane_done[d]) done_cnt[lane_owner[d]] = done_cnt[lane_owner[d]] + 3'd1;
    end

    integer i;
    always @(posedge clk) begin
        if (rst_n == RS_LV) begin
            f_wr <= 0; f_rd <= 0; f_count <= 0;
            t_wr <= 0; t_rd <= 0; t_count <= 0;
            lane_busy <= 0; lane_w <= 0; serial_port <= 0;
            flag_done_o <= 0;
            for (i = 0; i < 4; i = i + 1) begin
                lane_owner[i] <= 0;
                word_count[i] <= 0;
            end
            for (i = 0; i < QD; i = i + 1) begin
                t_mask[i] <= 0; t_rem[i] <= 0;
                t_lane[i] <= 0;
            end
        end else begin
            flag_done_o <= 1'b0;

            if (cmd_fire) begin
                f_cmd[f_wr]  <= s_cmd;
                f_meta[f_wr] <= s_cmd_meta;
                f_wr         <= f_wr + 1'b1;
            end

            if (disp) begin
                f_rd         <= f_rd + 1'b1;
                t_mask[t_wr] <= head_mask;
                t_rem[t_wr]  <= {1'b0, head_mask[0]} + {1'b0, head_mask[1]} +
                                {1'b0, head_mask[2]} + {1'b0, head_mask[3]};
                t_tag[t_wr]  <= f_cmd[f_rd][111:96];
                t_meta[t_wr] <= f_meta[f_rd];
                t_res[t_wr]  <= head_res;
                t_lane[t_wr] <= serial_port;
                t_wr         <= t_wr + 1'b1;
                lane_busy    <= lane_busy | head_mask;
                for (i = 0; i < 4; i = i + 1)
                    if (head_mask[i]) lane_w[i] <= (head_res == 4'd1);
                for (i = 0; i < 4; i = i + 1)
                    if (head_mask[i]) lane_owner[i] <= t_wr;
                if (head_res != 4'd1) begin
                    serial_port <= serial_port + 1'b1;
                end
            end

            for (i = 0; i < 4; i = i + 1) begin
                if (lane_done[i]) begin
                    lane_busy[i] <= 1'b0;
                    lane_w[i] <= 1'b0;
                    word_count[i] <= 9'd0;
                end else if (lane_valid[i] && lane_w[i] && m_buffer_ready[i]) begin
                    word_count[i] <= word_count[i] + 9'd1;
                end
            end

            // Completion bookkeeping must count ALL lanes that finish in one
            // cycle.  Decrementing inside the per-lane loop collapses to a
            // single decrement when several lanes share the same in-flight
            // entry (the weight case, all four lanes), so t_rem never reaches
            // zero and the descriptor is never retired.
            for (i = 0; i < QD; i = i + 1)
                if (done_cnt[i] != 3'd0) t_rem[i] <= t_rem[i] - done_cnt[i];

            if ((t_count != 2'd0) && (t_rem[t_rd] == 3'd0)) begin
                t_rd        <= t_rd + 1'b1;
                flag_done_o <= 1'b1;
            end

            case ({cmd_fire, disp})
                2'b10:   f_count <= f_count + 2'd1;
                2'b01:   f_count <= f_count - 2'd1;
                default: f_count <= f_count;
            endcase

            case ({disp, (t_count != 2'd0) && (t_rem[t_rd] == 3'd0)})
                2'b10:   t_count <= t_count + 2'd1;
                2'b01:   t_count <= t_count - 2'd1;
                default: t_count <= t_count;
            endcase
        end
    end
endmodule

`default_nettype wire
