`timescale 1ns / 1ps
// Gamma table: two independent 64-bit words form one 128-bit vector.
// [P16K] 64-bit x 5488 x2 = 702 Kbit 的 norm 系数表，读带宽每拍 1 字，放 BRAM 完全够；
//        改前 URAM 是 100% 硬阻塞资源，这张小表占着 4 块 URAM288，改到 BRAM 后
//        全芯片 URAM 52→48，BRAM36 仍有余量（89→~109 / 144）。端口/延迟/行为不变。
module qwen25_fixed_gamma (
    input wire clk,
    input wire rst_n,
    input wire load_start,
    input wire load_valid,
    output wire load_ready,
    input wire [127:0] load_data,
    input wire load_last,
    output wire load_done,
    output wire loaded,
    input wire read_en,
    input wire [12:0] read_addr,
    output wire read_valid,
    output wire [127:0] read_data
);
    localparam integer DEPTH=5488;
    reg [12:0] write_addr;
    reg loaded_reg,done_reg;
    reg [3:0] read_valid_pipe;
    reg read_request;
    reg [12:0] read_address;
    reg write_pending,write_pending_last;
    reg [12:0] write_pending_address;
    reg [127:0] write_pending_data;
    wire [63:0] read0,read1;
    wire write_accept=load_valid && load_ready;

    wire final_write_pending=write_pending && write_pending_last;
    wire restart=load_start && !loaded_reg && !final_write_pending;
    wire write_commit=write_pending && !restart;
    assign load_ready=!loaded_reg && !load_start && !final_write_pending;
    assign load_done=done_reg;
    assign loaded=loaded_reg;
    assign read_valid=read_valid_pipe[3];
    assign read_data={read1,read0};

    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(13),.C_STYLE("block"),.C_READ_LATENCY(3)) mem0_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(write_commit),.i_wr_addr(write_pending_address),.i_wr_data(write_pending_data[63:0]),
        .i_rd_en(read_request),.i_rd_addr(read_address),.o_rd_data(read0)
    );
    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(13),.C_STYLE("block"),.C_READ_LATENCY(3)) mem1_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(write_commit),.i_wr_addr(write_pending_address),.i_wr_data(write_pending_data[127:64]),
        .i_rd_en(read_request),.i_rd_addr(read_address),.o_rd_data(read1)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            write_addr<=0;
            loaded_reg<=0;
            done_reg<=0;
            read_valid_pipe<=0;read_request<=0;write_pending<=0;write_pending_last<=0;
        end else begin
            done_reg<=0;
            read_valid_pipe<={read_valid_pipe[2:0],read_en};read_request<=read_en;
            if(read_en)read_address<=read_addr;
            write_pending<=write_accept;
            if(write_accept)begin
                write_pending_address<=write_addr;write_pending_data<=load_data;write_pending_last<=load_last;
            end
            if(write_commit && write_pending_last)begin loaded_reg<=1;done_reg<=1;end
            if(restart) begin
                write_addr<=0;
                write_pending<=0;
            end else if(write_accept) begin
                write_addr<=write_addr+1'b1;
            end
        end
    end
endmodule
