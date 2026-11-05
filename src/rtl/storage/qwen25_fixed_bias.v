`timescale 1ns / 1ps
// Bias table: two independent 64-bit BRAM words form one 128-bit vector.
module qwen25_fixed_bias (
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
    input wire [11:0] read_addr,
    output wire read_valid,
    output wire [127:0] read_data
);
    localparam integer DEPTH=3456;
    reg [11:0] write_addr;
    reg loaded_reg,done_reg,read_valid_reg;
    wire [63:0] read0,read1;
    wire write_accept=load_valid && load_ready;

    assign load_ready=rst_n && !loaded_reg && !load_start;
    assign load_done=done_reg;
    assign loaded=loaded_reg;
    assign read_valid=read_valid_reg;
    assign read_data={read1,read0};

    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(12),.C_STYLE("block")) mem0_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(write_accept),.i_wr_addr(write_addr),.i_wr_data(load_data[63:0]),
        .i_rd_en(read_en),.i_rd_addr(read_addr),.o_rd_data(read0)
    );
    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(12),.C_STYLE("block")) mem1_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(write_accept),.i_wr_addr(write_addr),.i_wr_data(load_data[127:64]),
        .i_rd_en(read_en),.i_rd_addr(read_addr),.o_rd_data(read1)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            write_addr<=0;
            loaded_reg<=0;
            done_reg<=0;
            read_valid_reg<=0;
        end else begin
            done_reg<=0;
            read_valid_reg<=read_en;
            if(load_start && !loaded_reg) begin
                write_addr<=0;
            end else if(write_accept) begin
                write_addr<=write_addr+1'b1;
                if(load_last) begin
                    loaded_reg<=1;
                    done_reg<=1;
                end
            end
        end
    end
endmodule
