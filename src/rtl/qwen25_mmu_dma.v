`timescale 1ns / 1ps
`default_nettype none

// MMU transport: four concurrent read bursts, ordered local stream, independent writes.
// Each write descriptor selects a whole engine; W is never interleaved between AXI ports.
module qwen25_mmu_dma #(parameter RS_LV=0, parameter DIRECT_BUFFER_READ=0)(
    input wire clk,input wire rst_n,
    input wire [127:0] s_cmd,input wire [127:0] s_cmd_meta,
    input wire s_cmd_valid,output wire s_cmd_ready,
    input wire abort_req,input wire soft_reset_req,
    output wire [255:0] m_axi_araddr,output wire [31:0] m_axi_arlen,
    output wire [11:0] m_axi_arsize,output wire [7:0] m_axi_arburst,
    output wire [3:0] m_axi_arvalid,input wire [3:0] m_axi_arready,
    input wire [15:0] m_axi_rid,input wire [511:0] m_axi_rdata,
    input wire [7:0] m_axi_rresp,input wire [3:0] m_axi_rlast,
    input wire [3:0] m_axi_rvalid,output wire [3:0] m_axi_rready,
    output wire [127:0] m_stream_data,output wire [15:0] m_stream_keep,
    output wire m_stream_last,output wire [127:0] m_stream_meta,
    output wire [15:0] m_stream_tag,output wire [3:0] m_stream_resource,
    output wire [1:0] m_stream_error,output wire m_stream_valid,input wire m_stream_ready,
    output wire [3:0] m_buffer_valid, input wire [3:0] m_buffer_ready,
    output wire [511:0] m_buffer_data, output wire [39:0] m_buffer_word,
    output wire [63:0] m_buffer_keep, output wire [7:0] m_buffer_error,
    output wire [127:0] m_buffer_meta,
    output wire rd_busy,output wire rd_done,output wire rd_error,output wire rd_aborted,
    output wire [7:0] rd_error_code,output wire [15:0] rd_completion_tag,
    output wire [127:0] rd_completion_meta,
    input wire [127:0] s_stream_data,input wire [15:0] s_stream_keep,
    input wire s_stream_last,input wire [127:0] s_stream_meta,
    input wire s_stream_valid,output wire s_stream_ready,
    output wire [255:0] m_axi_awaddr,output wire [31:0] m_axi_awlen,
    output wire [11:0] m_axi_awsize,output wire [7:0] m_axi_awburst,
    output wire [3:0] m_axi_awvalid,input wire [3:0] m_axi_awready,
    output wire [511:0] m_axi_wdata,output wire [63:0] m_axi_wstrb,
    output wire [3:0] m_axi_wlast,output wire [3:0] m_axi_wvalid,
    input wire [3:0] m_axi_wready,input wire [7:0] m_axi_bresp,input wire [15:0] m_axi_bid,
    input wire [3:0] m_axi_bvalid,output wire [3:0] m_axi_bready,
    output wire wr_busy,output wire wr_done,output wire wr_error,output wire wr_aborted,
    output wire [7:0] wr_error_code,output wire [15:0] wr_completion_tag,
    output wire [127:0] wr_completion_meta
);
    wire rd_cmd_ready;
    wire wr_cmd_ready,wr_stream_ready,wr_engine_busy;
    wire [63:0] write_address;
    wire [7:0] write_length;
    wire [2:0] write_size;
    wire [1:0] write_burst;
    wire write_address_valid,write_data_valid,write_data_last,write_response_ready;
    wire [127:0] write_data;
    wire [15:0] write_strobe;
    reg [1:0] reg_write_port,reg_write_next;
    reg flag_write_active;
    // A write command is only in flight if the writer will queue it: qwen25_dma_wr
    // drops 0-byte / non-16B-aligned commands (sticky 0x21) and queues nothing, so
    // asserting flag_write_active on that handshake would latch the write port off
    // forever (wr_done never comes, every later write is refused).  The writer keeps
    // 0x21 as the alarm; refusing the handshake here can never lose the port.
    wire wr_cmd_ok = (s_cmd[87:64] != 24'd0) && (s_cmd[3:0] == 4'd0);
    wire write_command_fire=s_cmd_valid && s_cmd_ready && s_cmd[118] && wr_cmd_ok;
    assign s_cmd_ready=s_cmd[118] ? (!flag_write_active && wr_cmd_ready && wr_cmd_ok) : rd_cmd_ready;
    assign s_stream_ready=flag_write_active && wr_stream_ready;
    assign wr_busy=flag_write_active || wr_engine_busy;
    always @(posedge clk)begin
        if(rst_n==RS_LV)begin reg_write_port<=0;reg_write_next<=0;flag_write_active<=0;end
        else begin
            if(write_command_fire)begin
                reg_write_port<=reg_write_next;reg_write_next<=reg_write_next+1'b1;flag_write_active<=1;
            end
            if(wr_done)flag_write_active<=0;
        end
    end
    generate if(DIRECT_BUFFER_READ!=0) begin: gen_direct
    qwen25_mmu_direct_read #(.RS_LV(RS_LV)) read_Inst(
        .clk(clk),.rst_n(rst_n),.s_cmd(s_cmd),.s_cmd_meta(s_cmd_meta),
        .s_cmd_valid(s_cmd_valid && !s_cmd[118]),.s_cmd_ready(rd_cmd_ready),
        .abort_req(abort_req),.soft_reset_req(soft_reset_req),
        .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready),
        .m_stream_data(m_stream_data),.m_stream_keep(m_stream_keep),.m_stream_last(m_stream_last),
        .m_stream_meta(m_stream_meta),.m_stream_tag(m_stream_tag),.m_stream_resource(m_stream_resource),
        .m_stream_error(m_stream_error),.m_stream_valid(m_stream_valid),.m_stream_ready(m_stream_ready),
        .m_buffer_valid(m_buffer_valid),.m_buffer_ready(m_buffer_ready),.m_buffer_data(m_buffer_data),
        .m_buffer_word(m_buffer_word),.m_buffer_keep(m_buffer_keep),.m_buffer_error(m_buffer_error),.m_buffer_meta(m_buffer_meta),
        .busy(rd_busy),.done(rd_done),.error(rd_error),.aborted(rd_aborted),.error_code(rd_error_code),
        .completion_tag(rd_completion_tag),.completion_meta(rd_completion_meta));
    end else begin: gen_ordered
    assign m_buffer_valid=0; assign m_buffer_data=0; assign m_buffer_word=0;
    assign m_buffer_keep=0; assign m_buffer_error=0; assign m_buffer_meta=0;
    qwen25_mmu_read #(.RS_LV(RS_LV)) read_Inst(
        .clk(clk),.rst_n(rst_n),.s_cmd(s_cmd),.s_cmd_meta(s_cmd_meta),
        .s_cmd_valid(s_cmd_valid && !s_cmd[118]),.s_cmd_ready(rd_cmd_ready),
        .abort_req(abort_req),.soft_reset_req(soft_reset_req),
        .m_axi_araddr(m_axi_araddr),.m_axi_arlen(m_axi_arlen),.m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),.m_axi_arvalid(m_axi_arvalid),.m_axi_arready(m_axi_arready),
        .m_axi_rid(m_axi_rid),.m_axi_rdata(m_axi_rdata),.m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast),.m_axi_rvalid(m_axi_rvalid),.m_axi_rready(m_axi_rready),
        .m_stream_data(m_stream_data),.m_stream_keep(m_stream_keep),.m_stream_last(m_stream_last),
        .m_stream_meta(m_stream_meta),.m_stream_tag(m_stream_tag),.m_stream_resource(m_stream_resource),
        .m_stream_error(m_stream_error),.m_stream_valid(m_stream_valid),.m_stream_ready(m_stream_ready),
        .busy(rd_busy),.done(rd_done),.error(rd_error),.aborted(rd_aborted),.error_code(rd_error_code),
        .completion_tag(rd_completion_tag),.completion_meta(rd_completion_meta));
    end endgenerate
    // One local 128-bit producer requires one burst RAM; route it to all four masters.
    qwen25_dma_wr #(.RS_LV(RS_LV)) write_Inst(
            .clk(clk),.rst_n(rst_n),.s_cmd(s_cmd),.s_cmd_meta(s_cmd_meta),
            .s_cmd_valid(write_command_fire),.s_cmd_ready(wr_cmd_ready),
            .abort_req(abort_req),.soft_reset_req(soft_reset_req),
            .s_stream_data(s_stream_data),.s_stream_keep(s_stream_keep),.s_stream_last(s_stream_last),
            .s_stream_meta(s_stream_meta),.s_stream_valid(s_stream_valid && flag_write_active),
            .s_stream_ready(wr_stream_ready),
            .m_axi_awaddr(write_address),.m_axi_awlen(write_length),
            .m_axi_awsize(write_size),.m_axi_awburst(write_burst),
            .m_axi_awvalid(write_address_valid),.m_axi_awready(m_axi_awready[reg_write_port]),
            .m_axi_wdata(write_data),.m_axi_wstrb(write_strobe),
            .m_axi_wlast(write_data_last),.m_axi_wvalid(write_data_valid),.m_axi_wready(m_axi_wready[reg_write_port]),
            .m_axi_bid(m_axi_bid[reg_write_port*4+:4]),.m_axi_bresp(m_axi_bresp[reg_write_port*2+:2]),.m_axi_bvalid(m_axi_bvalid[reg_write_port]),.m_axi_bready(write_response_ready),
            .busy(wr_engine_busy),.done(wr_done),.error(wr_error),
            .aborted(wr_aborted),.error_code(wr_error_code),
            .completion_tag(wr_completion_tag),.completion_meta(wr_completion_meta));
    genvar port;
    generate for(port=0;port<4;port=port+1)begin:gen_write_route
        assign m_axi_awaddr[port*64+:64]=write_address;
        assign m_axi_awlen[port*8+:8]=write_length;
        assign m_axi_awsize[port*3+:3]=write_size;
        assign m_axi_awburst[port*2+:2]=write_burst;
        assign m_axi_awvalid[port]=write_address_valid && reg_write_port==port;
        assign m_axi_wdata[port*128+:128]=write_data;
        assign m_axi_wstrb[port*16+:16]=write_strobe;
        assign m_axi_wlast[port]=write_data_last;
        assign m_axi_wvalid[port]=write_data_valid && reg_write_port==port;
        assign m_axi_bready[port]=write_response_ready && reg_write_port==port;
    end endgenerate
endmodule
`default_nettype wire
