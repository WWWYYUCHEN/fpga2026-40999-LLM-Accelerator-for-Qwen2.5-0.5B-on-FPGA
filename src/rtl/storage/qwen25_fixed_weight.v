`timescale 1ns / 1ps
// Two 64x128 BF16 banks occupy the low and high address halves of one RAM set.
// A wide read returns the adjacent even/odd 1024-bit words for both GEMM lanes.
// Eight 256-bit slices allow each dequantization lane to write independently.
module qwen25_fixed_weight # (parameter DIRECT_PORTS=0) (
    input wire clk, input wire rst_n,
    input wire fill_done, input wire fill_bank,
    input wire release_bank, input wire release_en, output wire [1:0] full,
    input wire load_valid, output wire load_ready, input wire load_bank,
    input wire [9:0] load_word, input wire [255:0] load_data,
    input wire [3:0] direct_valid, output wire [3:0] direct_ready,
    input wire direct_bank, input wire [31:0] direct_word,
    input wire [1023:0] direct_data,
    input wire gemm_rd_en, input wire gemm_bank,
    input wire [7:0] gemm_word, output wire gemm_valid,
    output wire [1023:0] gemm_q,
    input wire gemm_rd_en_pair, input wire gemm_bank_pair,
    input wire [7:0] gemm_word_pair, output wire gemm_valid_pair,
    output wire [1023:0] gemm_q_pair
);
    reg [1:0] full_bits;
	localparam [1:0] W_IDLE=2'b00,W_PING=2'b01,W_PONG=2'b10;
	reg [1:0] write_state,write_next;
	wire request_bank=(DIRECT_PORTS!=0 && |direct_valid) ? direct_bank : load_bank;
	wire write_request=load_valid || (DIRECT_PORTS!=0 && |direct_valid);
	wire write_active=write_state!=W_IDLE;
	wire write_bank=write_state==W_PONG;
    reg read_valid, read_valid_pair;
    reg read_half;
    wire [2047:0] read_data;
    wire [6:0] read_address = gemm_rd_en ?
        {gemm_bank,gemm_word[6:1]} : {gemm_bank_pair,gemm_word_pair[6:1]};
    genvar half, lane;

    assign full=full_bits;
    assign load_ready=write_active && load_bank==write_bank;
    assign direct_ready={4{write_active && direct_bank==write_bank}};
    assign gemm_valid=read_valid;
    assign gemm_valid_pair=read_valid_pair;
    assign gemm_q=read_half ? read_data[2047:1024] : read_data[1023:0];
    assign gemm_q_pair=read_data[2047:1024];

    // The first offered word reserves an empty address half. Ownership remains
    // fixed while independently stalled dequantization lanes finish that bank.
	always @(*) begin
		write_next=write_state;
		case(write_state)
			W_IDLE:if(write_request && !full_bits[request_bank])begin
				write_next=request_bank ? W_PONG : W_PING;
			end
			W_PING,W_PONG:if(fill_done && fill_bank==write_bank)begin
				write_next=W_IDLE;
			end
			default:begin
				write_next=W_IDLE;
			end
		endcase
	end
	always @(posedge clk) begin
		if(!rst_n)begin
			write_state<=W_IDLE;
		end else begin
			write_state<=write_next;
		end
	end

    // Each slice has one write port and one read port. The address MSB selects
    // ping/pong; the remaining six bits select one of 64 complete BF16 rows.
    // Concurrent reads and writes must target different logical banks.
    // Simultaneous read requests must select the same bank and complete row.
    // The linear controller supplies an even word followed by its odd neighbor.
    generate
        for(half=0; half<2; half=half+1) begin: weight_half
            for(lane=0; lane<4; lane=lane+1) begin: weight_slice
                (* ram_style="block" *) reg [255:0] memory [0:127];
                reg [255:0] read_q;
                wire legacy_write = rst_n && load_valid && load_ready &&
                    load_word[2]==half && load_word[1:0]==lane;
                wire direct_write = rst_n && DIRECT_PORTS!=0 &&
                    direct_valid[lane] && direct_ready[lane] && direct_word[lane*8]==half;
                wire write_enable = legacy_write || direct_write;
                wire [6:0] write_address = direct_write ?
                    {write_bank,direct_word[lane*8+1+:6]} : {write_bank,load_word[8:3]};
                wire [255:0] write_data = direct_write ? direct_data[lane*256+:256] : load_data;

                assign read_data[half*1024+lane*256+:256]=read_q;
                // No write-side readback: Vivado can use the 72-bit SDP mode.
                always @(posedge clk) begin
                    if(write_enable)memory[write_address]<=write_data;
                end
                always @(posedge clk) begin
                    read_q<=memory[read_address];
                end
            end
        end
    endgenerate

    always @(posedge clk) begin
        if(!rst_n) begin
            full_bits<=0;
            read_valid<=0;read_valid_pair<=0;
            read_half<=0;
        end else begin
            if(release_en)full_bits[release_bank]<=0;
            if(fill_done)full_bits[fill_bank]<=1;
            read_valid<=gemm_rd_en;
            read_valid_pair<=gemm_rd_en_pair;
            if(gemm_rd_en)read_half<=gemm_word[0];
        end
    end
endmodule
