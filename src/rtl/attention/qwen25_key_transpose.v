`timescale 1ns / 1ps
`default_nettype none
// Sixteen token rows of eight dimensions become eight dimension rows of sixteen tokens.
// The 256-byte staging block feeds the existing 256-bit Weight write port.
module qwen25_key_transpose (
    input wire i_clk,
    input wire i_rstn,
    input wire i_flush,
    input wire i_valid,
    output wire o_ready,
    input wire [127:0] i_data,
    output wire o_valid,
    input wire i_ready,
    output wire [255:0] o_data,
    output wire [2:0] o_dimension,
    output wire o_busy
);
    reg [127:0] reg_rows [0:15]; // Data registers are not reset.
    reg [3:0] cnt_token;
    reg [2:0] cnt_dimension;
    reg flag_output;
    genvar token;
    assign o_ready=!i_flush && !flag_output;
    assign o_valid=!i_flush && flag_output;
    assign o_dimension=cnt_dimension;
    assign o_busy=flag_output || cnt_token!=0;
    generate
        for(token=0;token<16;token=token+1)begin:gen_token
            assign o_data[token*16 +: 16]=reg_rows[token][cnt_dimension*16 +: 16];
        end
    endgenerate
    always @(posedge i_clk)begin
        if(!i_rstn)begin
            cnt_token<=0;cnt_dimension<=0;flag_output<=0;
        end else if(i_flush)begin
            cnt_token<=0;cnt_dimension<=0;flag_output<=0;
        end else begin
            if(i_valid && o_ready)begin
                reg_rows[cnt_token]<=i_data;
                cnt_token<=cnt_token+1'b1;
                if(cnt_token==15)begin flag_output<=1;cnt_dimension<=0;end
            end
            if(o_valid && i_ready)begin
                cnt_dimension<=cnt_dimension+1'b1;
                if(cnt_dimension==7)begin flag_output<=0;cnt_token<=0;end
            end
        end
    end
endmodule
`default_nettype wire
