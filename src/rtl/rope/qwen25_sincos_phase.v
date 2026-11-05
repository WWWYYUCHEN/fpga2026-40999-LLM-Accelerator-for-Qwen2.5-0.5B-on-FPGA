`timescale 1ns / 1ps
`default_nettype none

// Binary32 radians -> signed Q3.45 scaled radians, modulo two turns.
// The coefficient ROM retains 80 fractional bits for every normal exponent.
module qwen25_sincos_phase (
    input wire i_clk,
    input wire i_rstn,
    input wire i_valid,
    input wire [31:0] i_angle,
    output wire o_valid,
    output wire [47:0] o_phase,
    output wire [64:0] o_bypass
);
    wire [80:0] coefficient;
    reg [23:0] mantissa;
    reg [23:0] mantissa_pipe;
    reg [80:0] coefficient_pipe;
    reg [6:0] valid_pipe, sign_pipe;
    reg [64:0] bypass_pipe[0:6];
    (* use_dsp = "yes" *) reg [40:0] products[0:4];
    reg [57:0] low_pair, middle_pair;
    reg [40:0] high_pair, high_delayed;
    reg [91:0] lower_sum;
    reg [104:0] phase_product;
    reg [47:0] phase_result;
    wire [45:0] phase_truncated=phase_product[80:35];
    wire round_up=phase_product[34] && (|phase_product[33:0] || phase_product[35]);
    wire [45:0] phase_rounded=phase_truncated+round_up;
    wire [47:0] phase_signed={{2{phase_rounded[45]}},phase_rounded};
    wire input_tiny=i_angle[30:23]<8'd114;
    wire input_nonfinite=i_angle[30:23]==8'hff;
    wire [31:0] tiny_sine=i_angle[30:23]==0 ? {i_angle[31],31'd0} : i_angle;
    assign o_valid=valid_pipe[6];
    assign o_phase=phase_result;
    assign o_bypass=bypass_pipe[6];

    qwen25_sincos_coeff coefficient_Inst(.i_clk(i_clk),.i_enable(i_valid),
        .i_exponent(i_angle[30:23]),.o_coefficient(coefficient));
    genvar limb;
    generate
        for(limb=0;limb<4;limb=limb+1)begin:gen_multiply
            always @(posedge i_clk)begin
                if(valid_pipe[1])products[limb]<=mantissa_pipe*coefficient_pipe[limb*17+:17];
            end
        end
    endgenerate
    integer stage;
    always @(posedge i_clk)begin
        if(!i_rstn)begin valid_pipe<=0;sign_pipe<=0;end
        else begin
            valid_pipe<={valid_pipe[5:0],i_valid};
            sign_pipe<={sign_pipe[5:0],i_angle[31]};
        end
        if(i_valid)begin
            mantissa<={1'b1,i_angle[22:0]};
            bypass_pipe[0]<={input_tiny || input_nonfinite,
                input_nonfinite ? 32'h7fc00000 : tiny_sine,
                input_nonfinite ? 32'h7fc00000 : 32'h3f800000};
        end
        for(stage=1;stage<7;stage=stage+1)begin
            if(valid_pipe[stage-1])bypass_pipe[stage]<=bypass_pipe[stage-1];
        end
        if(valid_pipe[0])begin
            mantissa_pipe<=mantissa;
            coefficient_pipe<=coefficient;
        end
        if(valid_pipe[1])products[4]<=mantissa_pipe*{4'd0,coefficient_pipe[80:68]};
        if(valid_pipe[2])begin
            low_pair<={products[1],17'd0}+{17'd0,products[0]};
            middle_pair<={products[3],17'd0}+{17'd0,products[2]};
            high_pair<=products[4];
        end
        if(valid_pipe[3])begin
            lower_sum<={middle_pair,34'd0}+{34'd0,low_pair};
            high_delayed<=high_pair;
        end
        if(valid_pipe[4])phase_product<={high_delayed,68'd0}+{13'd0,lower_sum};
        if(valid_pipe[5])phase_result<=bypass_pipe[5][64] ? 48'd0 :
            (sign_pipe[5] ? -phase_signed : phase_signed);
    end
endmodule
`default_nettype wire
