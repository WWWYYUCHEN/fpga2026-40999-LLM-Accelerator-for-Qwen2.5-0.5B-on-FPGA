`timescale 1ns/1ps
// Simulation contract for the common-clock, 64-bit read-first parameter RAMs.
module xpm_memory_sdpram #(
    parameter integer MEMORY_SIZE=65536,
    parameter MEMORY_PRIMITIVE="ultra",
    parameter CLOCKING_MODE="common_clock",
    parameter MEMORY_INIT_FILE="none",
    parameter integer USE_MEM_INIT=0,
    parameter integer WRITE_DATA_WIDTH_A=64,
    parameter integer BYTE_WRITE_WIDTH_A=64,
    parameter integer ADDR_WIDTH_A=10,
    parameter integer READ_DATA_WIDTH_B=64,
    parameter integer ADDR_WIDTH_B=10,
    parameter integer READ_LATENCY_B=1,
    parameter WRITE_MODE_B="read_first",
    parameter READ_RESET_VALUE_B="0",
    parameter ECC_MODE="no_ecc"
) (
    input wire sleep,clka,ena,
    input wire [WRITE_DATA_WIDTH_A/BYTE_WRITE_WIDTH_A-1:0] wea,
    input wire [ADDR_WIDTH_A-1:0] addra,
    input wire [WRITE_DATA_WIDTH_A-1:0] dina,
    input wire injectsbiterra,injectdbiterra,
    input wire clkb,rstb,enb,regceb,
    input wire [ADDR_WIDTH_B-1:0] addrb,
    output wire [READ_DATA_WIDTH_B-1:0] doutb,
    output wire sbiterrb,dbiterrb
);
    localparam integer DEPTH=MEMORY_SIZE/WRITE_DATA_WIDTH_A;
    reg [WRITE_DATA_WIDTH_A-1:0] memory[0:DEPTH-1];
    reg [READ_DATA_WIDTH_B-1:0] read_pipe[0:READ_LATENCY_B-1];
    reg [READ_LATENCY_B-1:0] enable_pipe=0;
    initial begin
        if(CLOCKING_MODE!="common_clock" || WRITE_DATA_WIDTH_A!=64 ||
           READ_DATA_WIDTH_B!=64 || BYTE_WRITE_WIDTH_A!=64 || READ_LATENCY_B<1 ||
           WRITE_MODE_B!="read_first" || ECC_MODE!="no_ecc" || USE_MEM_INIT!=0)
            $fatal(1,"Unsupported XPM configuration in Verilator parameter-memory model");
    end
    assign doutb=read_pipe[READ_LATENCY_B-1];
    assign sbiterrb=1'b0;
    assign dbiterrb=1'b0;
    always @(posedge clka)begin
        if(ena && !sleep && wea[0])begin
            if(addra>=DEPTH)$fatal(1,"XPM write address out of bounds");
            memory[addra]<=dina;
        end
    end
    always @(posedge clkb)begin
        if(rstb)begin
            enable_pipe<=0;
            for(integer stage=0;stage<READ_LATENCY_B;stage=stage+1)read_pipe[stage]<=0;
        end else if(!sleep)begin
            enable_pipe[0]<=enb;
            if(enb)begin
                if(addrb>=DEPTH)$fatal(1,"XPM read address out of bounds");
                read_pipe[0]<=memory[addrb];
            end
            for(integer stage=1;stage<READ_LATENCY_B;stage=stage+1)begin
                enable_pipe[stage]<=enable_pipe[stage-1];
                if(enable_pipe[stage-1] && (stage!=READ_LATENCY_B-1 || regceb))
                    read_pipe[stage]<=read_pipe[stage-1];
            end
        end
    end
endmodule
