`timescale 1ns / 1ps
`default_nettype none
module qwen25_afu_service (
    input wire clk,
    input wire rst_n,
    input wire abort_req,
    input wire soft_reset_req,
    input wire ctrl_afu_valid,
    output wire ctrl_afu_ready,
    input wire ctrl_afu_last,
    input wire ctrl_afu_latch,
    input wire ctrl_afu_gelu_sign,
    input wire [3:0] ctrl_afu_command,
    input wire [20:0] ctrl_afu_command_4,
    input wire [26:0] ctrl_afu_sum_command_buffer,
    input wire [31:0] ctrl_afu_eps,
    input wire [127:0] ctrl_afu_x,
    input wire [127:0] ctrl_afu_y,
    input wire [127:0] ctrl_afu_meta,
    input wire [7:0] ctrl_afu_mask,
    input wire [63:0] ctrl_afu_length,
    output wire ctrl_result_valid,
    input wire ctrl_result_ready,
    output wire ctrl_result_last,
    output wire [127:0] ctrl_result_data,
    output wire [127:0] ctrl_result_aux,
    output wire [127:0] ctrl_result_meta,
    output wire [7:0] ctrl_result_mask,
    output wire internal_busy,
    output wire internal_error,
    input wire ctrl_afu_abort
);
    reg rope_active;
    (* DONT_TOUCH = "yes" *) reg [3:0] rope_result_data_copies;
    (* DONT_TOUCH = "yes" *) reg rope_result_meta_copy,rope_result_control_copy;
    wire rope_ready,rope_valid,rope_last,rope_busy;
    wire [127:0] rope_data,rope_aux,rope_meta; wire [7:0] rope_mask;
    reg [1:0] rope_reset_release;
    wire rope_reset_n=rst_n && (&rope_reset_release);
    reg [1:0] afu_reset_release; // Vendor FP IP requires at least two clock edges in reset.

    wire afu_reset_n=rst_n && (&afu_reset_release);
    wire internal_ready,internal_valid,internal_last;
    wire scalar_core_busy;
    wire [127:0] internal_data,internal_meta;wire [7:0] internal_mask;
    // One elastic response stage keeps all payload and sidebands aligned.
    // Only validity is reset; a stalled response holds every payload bit.
    reg result_pending;
    reg result_last;
    reg [127:0] result_data,result_aux,result_meta;
    reg [7:0] result_mask;
    wire result_stage_ready=!result_pending || ctrl_result_ready;
    wire selected_result_valid=rope_result_control_copy ? rope_valid : internal_valid;
    wire selected_result_last=rope_result_control_copy ? rope_last : internal_last;
    wire [127:0] selected_result_data,selected_result_aux;
    wire [127:0] selected_result_meta=rope_result_meta_copy ? rope_meta : internal_meta;
    wire [7:0] selected_result_mask=rope_result_control_copy ? rope_mask : internal_mask;
    wire result_capture=selected_result_valid && result_stage_ready;
    // All arithmetic requests cross one elastic register before service selection.
    // Data has no reset: request_pending is the sole ownership/validity bit.
    reg request_pending;
    reg [3:0] request_command;
    reg [1:0] request_phase;
    reg [31:0] request_eps;
    reg [63:0] request_length;
    reg [127:0] request_x,request_y,request_meta;
    reg request_last;
    reg [7:0] request_mask;
    wire request_rope=request_command==12;
    // A service may change only after the previous service has drained all responses.
    wire scalar_drained=!scalar_core_busy && !internal_valid;
    wire rope_drained=!rope_busy && !rope_valid;
    wire backend_ready=request_rope ? (rope_ready && scalar_drained) :
        (internal_ready && rope_drained);
    wire request_fire=ctrl_afu_valid && ctrl_afu_ready;
    wire backend_fire=request_pending && backend_ready;
    always @(posedge clk)begin
        if(!rst_n)begin
            rope_result_data_copies<=4'd0;
            rope_result_meta_copy<=1'b0;
            rope_result_control_copy<=1'b0;
        end else if(backend_fire)begin
            rope_result_data_copies<={4{request_rope}};
            rope_result_meta_copy<=request_rope;
            rope_result_control_copy<=request_rope;
        end
    end
    always @(posedge clk) begin
        if(!rst_n)begin rope_reset_release<=2'b00;end
        else begin rope_reset_release<={rope_reset_release[0],1'b1};end
    end
    always @(posedge clk) begin
        if(!rst_n)begin afu_reset_release<=2'b00;end
        else begin afu_reset_release<={afu_reset_release[0],1'b1};end
    end
    always @(posedge clk) begin
        if(!rst_n)begin rope_active<=0;end
        else if(backend_fire)begin
            rope_active<=request_rope;
        end
    end
    // Consume and refill on the same edge to retain one request per cycle.
    always @(posedge clk) begin
        if(!afu_reset_n)begin request_pending<=1'b0;end
        else if(request_fire)begin request_pending<=1'b1;end
        else if(backend_ready)begin request_pending<=1'b0;end
    end
    always @(posedge clk) begin
        if(request_fire)begin
            request_command<=ctrl_afu_command;
            request_phase<=ctrl_afu_command_4[1:0];
            request_eps<=ctrl_afu_eps;
            request_length<=ctrl_afu_length;
            request_x<=ctrl_afu_x;
            request_y<=ctrl_afu_y;
            request_meta<=ctrl_afu_meta;
            request_last<=ctrl_afu_last;
            request_mask<=ctrl_afu_mask;
        end
    end
    // Ready no longer contains the current controller command/length decode cone.
    assign internal_error=1'b0;
    assign ctrl_afu_ready=afu_reset_n && (!request_pending || backend_ready);
    assign internal_busy=scalar_core_busy || request_pending || result_pending;
    assign ctrl_result_valid=result_pending;
    assign ctrl_result_data=result_data;
    assign ctrl_result_aux=result_aux;
    assign ctrl_result_meta=result_meta;
    assign ctrl_result_last=result_last;
    assign ctrl_result_mask=result_mask;
    always @(posedge clk)begin
        if(!afu_reset_n)result_pending<=1'b0;
        else if(result_stage_ready)result_pending<=selected_result_valid;
    end
    always @(posedge clk)begin
        if(result_capture)begin
            result_data<=selected_result_data;
            result_aux<=selected_result_aux;
            result_meta<=selected_result_meta;
            result_last<=selected_result_last;
            result_mask<=selected_result_mask;
        end
    end
    genvar result_slice;
    generate for(result_slice=0;result_slice<4;result_slice=result_slice+1)begin:gen_result_service_slice
        assign selected_result_data[result_slice*32+:32]=rope_result_data_copies[result_slice] ?
            rope_data[result_slice*32+:32] : internal_data[result_slice*32+:32];
        assign selected_result_aux[result_slice*32+:32]=rope_result_data_copies[result_slice] ?
            rope_aux[result_slice*32+:32] : 32'd0;
    end endgenerate
    // Auxiliary data follows the matching result-data copy.
    // Compute services run to completion; recovery uses the common hardware reset.
    qwen25_rope_vector rope_Inst (
        .i_clk(clk),.i_rstn(rope_reset_n),
        .i_valid(request_pending && request_rope && scalar_drained && afu_reset_n),.o_ready(rope_ready),
        .i_x(request_x),.i_y(request_y),.i_meta(request_meta),.i_last(request_last),.i_mask(request_mask),
        .o_valid(rope_valid),.i_ready(result_stage_ready && rope_active),
        .o_data(rope_data),.o_aux(rope_aux),.o_meta(rope_meta),.o_last(rope_last),.o_mask(rope_mask),.o_busy(rope_busy)
    );
    generate
        if(1)begin:gen_internal_afu
            qwen25_afu afu_Inst (
                .i_clk(clk),.i_rstn(afu_reset_n),
                .i_valid(request_pending && !request_rope && rope_drained && afu_reset_n),.o_ready(internal_ready),
                .i_command(request_command),.i_phase(request_phase),.i_eps(request_eps),.i_length(request_length),
                .i_x(request_x),.i_y(request_y),.i_meta(request_meta),.i_last(request_last),.i_mask(request_mask),
                .o_valid(internal_valid),.i_ready(result_stage_ready && !rope_active),
                .o_data(internal_data),.o_meta(internal_meta),.o_last(internal_last),.o_mask(internal_mask),
                .o_busy(scalar_core_busy),.o_error()
            );
        end
    endgenerate

endmodule
`default_nettype wire
