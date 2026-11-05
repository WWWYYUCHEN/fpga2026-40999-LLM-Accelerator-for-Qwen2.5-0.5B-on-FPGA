`timescale 1ns / 1ps
// Scale table: four independent 64-bit URAM words form one aligned 256-bit vector.
module qwen25_fixed_scale (
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
    input wire [14:0] read_addr,
    output wire read_valid,
    output wire [255:0] read_data
);
    localparam integer DEPTH=28504;
    reg [127:0] pending_data;
    reg pending_valid;
    reg [14:0] write_addr;
    reg reg_write_valid,reg_write_last;
    reg [14:0] reg_write_addr;
    reg [255:0] reg_write_data;
    reg loaded_reg,done_reg;
    reg [14:0] reg_read_addr;
    reg [3:0] reg_read_valid;
    wire accept=load_valid && load_ready;
    wire write_pair=accept && pending_valid;
    wire [63:0] read0,read1,read2,read3;
    assign load_ready=rst_n && !loaded_reg && !reg_write_last && !load_start;
    assign load_done=done_reg;
    assign loaded=loaded_reg;
    assign read_valid=reg_read_valid[3];
    assign read_data={read3,read2,read1,read0};
    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(15),.C_STYLE("ultra"),.C_READ_LATENCY(3)) mem0_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(reg_write_valid),.i_wr_addr(reg_write_addr),.i_wr_data(reg_write_data[63:0]),
        .i_rd_en(reg_read_valid[0]),.i_rd_addr(reg_read_addr),.o_rd_data(read0)
    );
    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(15),.C_STYLE("ultra"),.C_READ_LATENCY(3)) mem1_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(reg_write_valid),.i_wr_addr(reg_write_addr),.i_wr_data(reg_write_data[127:64]),
        .i_rd_en(reg_read_valid[0]),.i_rd_addr(reg_read_addr),.o_rd_data(read1)
    );
    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(15),.C_STYLE("ultra"),.C_READ_LATENCY(3)) mem2_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(reg_write_valid),.i_wr_addr(reg_write_addr),.i_wr_data(reg_write_data[191:128]),
        .i_rd_en(reg_read_valid[0]),.i_rd_addr(reg_read_addr),.o_rd_data(read2)
    );
    qwen25_param_mem64 #(.C_DEPTH(DEPTH),.C_ADDR_W(15),.C_STYLE("ultra"),.C_READ_LATENCY(3)) mem3_Inst (
        .i_clk(clk),.i_rstn(rst_n),
        .i_wr_en(reg_write_valid),.i_wr_addr(reg_write_addr),.i_wr_data(reg_write_data[255:192]),
        .i_rd_en(reg_read_valid[0]),.i_rd_addr(reg_read_addr),.o_rd_data(read3)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            pending_valid<=0;
            write_addr<=0;
            loaded_reg<=0;
            done_reg<=0;
            reg_read_valid<=0;
            reg_write_valid<=0;
            reg_write_last<=0;
        end else begin
            done_reg<=0;
            // Cut stream validation/address generation before the URAM cascade write port.
            reg_write_valid<=write_pair;
            reg_write_last<=write_pair && load_last;
            if(write_pair) begin
                reg_write_addr<=write_addr;
                reg_write_data<={load_data,pending_data};
            end
            if(reg_write_valid && reg_write_last) begin
                loaded_reg<=1;
                done_reg<=1;
            end
            reg_read_valid<={reg_read_valid[2:0],read_en};
            if(read_en) reg_read_addr<=read_addr;
            if(load_start && !loaded_reg) begin
                pending_valid<=0;
                write_addr<=0;
            end else if(accept && !pending_valid) begin
                pending_data<=load_data;
                pending_valid<=1;
            end else if(write_pair) begin
                pending_valid<=0;
                write_addr<=write_addr+1'b1;
            end
        end
    end
endmodule
