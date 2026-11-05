`timescale 1ns / 1ps
`default_nettype none
// 按资源直接路由AXI返回。Weight四路独立地址，不排序、不缓存完整burst。
module qwen25_mmu_direct_read #(parameter RS_LV=0) (
    input wire clk, input wire rst_n,
    input wire [127:0] s_cmd, input wire [127:0] s_cmd_meta,
    input wire s_cmd_valid, output wire s_cmd_ready,
    input wire abort_req, input wire soft_reset_req,
    output wire [255:0] m_axi_araddr, output wire [31:0] m_axi_arlen,
    output wire [11:0] m_axi_arsize, output wire [7:0] m_axi_arburst,
    output wire [3:0] m_axi_arvalid, input wire [3:0] m_axi_arready,
    input wire [15:0] m_axi_rid, input wire [511:0] m_axi_rdata,
    input wire [7:0] m_axi_rresp, input wire [3:0] m_axi_rlast,
    input wire [3:0] m_axi_rvalid, output wire [3:0] m_axi_rready,
    output wire [127:0] m_stream_data, output wire [15:0] m_stream_keep,
    output wire m_stream_last, output wire [127:0] m_stream_meta,
    output wire [15:0] m_stream_tag, output wire [3:0] m_stream_resource,
    output wire [1:0] m_stream_error, output wire m_stream_valid,
    input wire m_stream_ready,
    output wire [3:0] m_buffer_valid, input wire [3:0] m_buffer_ready,
    output wire [511:0] m_buffer_data, output wire [39:0] m_buffer_word,
    output wire [63:0] m_buffer_keep, output wire [7:0] m_buffer_error,
    output wire [127:0] m_buffer_meta,
    output wire busy, output wire done, output wire error, output wire aborted,
    output wire [7:0] error_code, output wire [15:0] completion_tag,
    output wire [127:0] completion_meta
);
    localparam [1:0] ST_IDLE=0, ST_RUN=1;
    reg [1:0] state_current, state_next;
    reg [127:0] command_reg, meta_reg;
    reg direct_weight;
    reg [1:0] serial_port, next_serial_port;
    reg [3:0] started, terminated;
    reg [8:0] word_count [0:3];
    reg done_reg;
    wire [3:0] lane_ready, lane_busy, lane_done, lane_aborted;
    wire [3:0] lane_valid, lane_last;
    wire [511:0] lane_data;
    wire [63:0] lane_keep;
    wire [3:0] target_mask=direct_weight ? 4'b1111 : (4'b0001<<serial_port);
    wire complete=(terminated & target_mask)==target_mask && started==target_mask;
    wire quiet=lane_busy==0 && lane_valid==0;
    integer index;
    assign s_cmd_ready=state_current==ST_IDLE;
    assign busy=state_current==ST_RUN;
    assign done=done_reg;
    assign error=1'b0;
    assign aborted=1'b0;
    assign error_code=8'd0;
    assign completion_tag=command_reg[111:96];
    assign completion_meta=meta_reg;
    assign m_stream_data=lane_data[serial_port*128+:128];
    assign m_stream_keep=lane_keep[serial_port*16+:16];
    assign m_stream_error=2'b00;
    assign m_stream_last=lane_last[serial_port];
    assign m_stream_meta=meta_reg;
    assign m_stream_tag=command_reg[111:96];
    assign m_stream_resource=command_reg[115:112];
    assign m_stream_valid=lane_valid[serial_port] && !direct_weight;
    assign m_buffer_valid=lane_valid & {4{direct_weight}};
    assign m_buffer_data=lane_data;
    assign m_buffer_keep=lane_keep;
    assign m_buffer_error=8'd0;
    assign m_buffer_meta=meta_reg;
    genvar port;
    generate for(port=0;port<4;port=port+1)begin:gen_reader
        wire begin_lane=state_current==ST_RUN && target_mask[port] && !started[port];
        wire stream_ready=direct_weight ? m_buffer_ready[port] : m_stream_ready;
        // Keep the original 128-K DDR tile layout. A 64-K request addresses
        // one 2048-byte half of each original 4096-byte column-lane opening.
        wire [23:0] weight_lane_bytes=command_reg[87:64]==24'd8192 ? 24'd2048 : 24'd4096;
        wire [127:0] lane_cmd=direct_weight ? {command_reg[127:88],weight_lane_bytes,command_reg[63:0]+port*64'd4096} : command_reg;
        wire [127:0] unused_meta, unused_completion_meta;
        wire [15:0] unused_tag, unused_completion_tag;
        wire [3:0] unused_resource;
        // Canonical BF16 write word: K row, half-row, independent column opening.
        assign m_buffer_word[port*10+:10]={word_count[port][7:0],port[1:0]};
        qwen25_dma_rd #(.RS_LV(RS_LV)) reader_Inst(
            .clk(clk),.rst_n(rst_n),.s_cmd(lane_cmd),.s_cmd_meta(meta_reg),
            .s_cmd_valid(begin_lane),.s_cmd_ready(lane_ready[port]),
            .abort_req(1'b0),.soft_reset_req(1'b0),
            .m_axi_araddr(m_axi_araddr[port*64+:64]),.m_axi_arlen(m_axi_arlen[port*8+:8]),
            .m_axi_arsize(m_axi_arsize[port*3+:3]),.m_axi_arburst(m_axi_arburst[port*2+:2]),
            .m_axi_arvalid(m_axi_arvalid[port]),.m_axi_arready(m_axi_arready[port]),
            .m_axi_rid(m_axi_rid[port*4+:4]),.m_axi_rdata(m_axi_rdata[port*128+:128]),
            .m_axi_rresp(m_axi_rresp[port*2+:2]),.m_axi_rlast(m_axi_rlast[port]),
            .m_axi_rvalid(m_axi_rvalid[port]),.m_axi_rready(m_axi_rready[port]),
            .m_stream_data(lane_data[port*128+:128]),.m_stream_keep(lane_keep[port*16+:16]),
            .m_stream_error(),.m_stream_valid(lane_valid[port]),
            .m_stream_ready(stream_ready),.m_stream_last(lane_last[port]),.m_stream_meta(unused_meta),
            .m_stream_tag(unused_tag),.m_stream_resource(unused_resource),
            .busy(lane_busy[port]),.done(lane_done[port]),.error(),
            .aborted(lane_aborted[port]),.error_code(),
            .completion_tag(unused_completion_tag),.completion_meta(unused_completion_meta));
        always @(posedge clk)begin
            if(rst_n==RS_LV)word_count[port]<=0;
            else if(state_current==ST_IDLE)word_count[port]<=0;
            else if(lane_valid[port] && stream_ready)word_count[port]<=word_count[port]+1'b1;
        end
    end endgenerate
    always @(*)begin
        state_next=state_current;
        case(state_current)
            ST_IDLE:if(s_cmd_valid && s_cmd_ready)begin
                state_next=ST_RUN;
            end
            ST_RUN:if(complete && quiet)state_next=ST_IDLE;
            default:state_next=ST_IDLE;
        endcase
    end
    always @(posedge clk)begin
        if(rst_n==RS_LV)state_current<=ST_IDLE;else state_current<=state_next;
    end
    always @(posedge clk)begin
        if(rst_n==RS_LV)begin
            command_reg<=0;meta_reg<=0;direct_weight<=0;serial_port<=0;next_serial_port<=0;
            started<=0;terminated<=0;
            done_reg<=0;
        end else begin
            done_reg<=0;
            if(state_current==ST_IDLE)begin
                if(s_cmd_valid && s_cmd_ready)begin
                    command_reg<=s_cmd;meta_reg<=s_cmd_meta;direct_weight<=s_cmd[115:112]==1;
                    serial_port<=next_serial_port;next_serial_port<=next_serial_port+1'b1;
                    started<=0;terminated<=0;
                end
            end
            if(state_current==ST_RUN)begin
                for(index=0;index<4;index=index+1)begin
                    if(target_mask[index] && !started[index] && lane_ready[index])started[index]<=1;
                    if(lane_done[index] || lane_aborted[index])terminated[index]<=1;
                end
                if(state_next==ST_IDLE)done_reg<=1;
            end
        end
    end
endmodule
`default_nettype wire
