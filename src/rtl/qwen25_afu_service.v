`timescale 1ns / 1ps
`default_nettype none
// ---------------------------------------------------------------------------
// Split AFU service.
//
// The original module served the scalar AFU and the RoPE vector engine through
// ONE request port and ONE result port with hard mutual exclusion:
//     scalar_drained = !scalar_core_busy && !internal_valid
//     rope_drained   = !rope_busy && !rope_valid
//     backend_ready  = request_rope ? (rope_ready && scalar_drained)
//                                   : (internal_ready && rope_drained)
// so a RoPE frame could not start until the scalar core and its result stage
// were empty, and an AFU frame could not start until the RoPE engine was
// drained.  Both services also shared one elastic result stage.
//
// This version gives each service its own request elastic stage, its own
// backend and its own result elastic stage.  Neither side observes the other:
//   * ctrl_afu_* / ctrl_result_*       : scalar AFU (unchanged semantics)
//   * rope_req_* / rope_res_*          : RoPE vector engine
// A stalled result on one channel can therefore never stall the other.
// ---------------------------------------------------------------------------
module qwen25_afu_service (
    input wire clk,
    input wire rst_n,
    input wire abort_req,
    input wire soft_reset_req,
    // Scalar AFU request channel (kept bit-for-bit compatible).
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
    // Scalar AFU result channel.
    output wire ctrl_result_valid,
    input wire ctrl_result_ready,
    output wire ctrl_result_last,
    output wire [127:0] ctrl_result_data,
    output wire [127:0] ctrl_result_aux,
    output wire [127:0] ctrl_result_meta,
    output wire [7:0] ctrl_result_mask,
    // Dedicated RoPE request channel.
    input wire rope_req_valid,
    output wire rope_req_ready,
    input wire rope_req_last,
    input wire [127:0] rope_req_x,
    input wire [127:0] rope_req_y,
    input wire [127:0] rope_req_meta,
    input wire [7:0] rope_req_mask,
    // Dedicated RoPE result channel.
    output wire rope_res_valid,
    input wire rope_res_ready,
    output wire rope_res_last,
    output wire [127:0] rope_res_data,
    output wire [127:0] rope_res_aux,
    output wire [127:0] rope_res_meta,
    output wire [7:0] rope_res_mask,
    output wire internal_busy,
    output wire internal_error,
    input wire ctrl_afu_abort
);
    // Both engines keep the original two-edge reset release.
    reg [1:0] rope_reset_release;
    wire rope_reset_n=rst_n && (&rope_reset_release);
    reg [1:0] afu_reset_release;
    wire afu_reset_n=rst_n && (&afu_reset_release);

    // ---- Scalar AFU: private request elastic stage ------------------------
    // Identical to the historic request register; it now only carries scalar
    // requests, so a pending RoPE frame never blocks ctrl_afu_ready.
    reg afu_request_pending;
    reg [3:0] afu_request_command;
    reg [1:0] afu_request_phase;
    reg [31:0] afu_request_eps;
    reg [63:0] afu_request_length;
    reg [127:0] afu_request_x,afu_request_y,afu_request_meta;
    reg afu_request_last;
    reg [7:0] afu_request_mask;
    wire afu_core_ready,afu_core_valid,afu_core_last,afu_core_busy;
    wire [127:0] afu_core_data,afu_core_meta;
    wire [7:0] afu_core_mask;
    wire afu_request_fire=ctrl_afu_valid && ctrl_afu_ready;

    // ---- Scalar AFU: private result elastic stage -------------------------
    // Only validity is reset; a stalled response holds every payload bit.
    reg afu_result_pending;
    reg afu_result_last;
    reg [127:0] afu_result_data,afu_result_meta;
    reg [7:0] afu_result_mask;
    wire afu_result_stage_ready=!afu_result_pending || ctrl_result_ready;
    wire afu_result_capture=afu_core_valid && afu_result_stage_ready;

    // ---- RoPE: private request elastic stage ------------------------------
    reg rope_request_pending;
    reg [127:0] rope_request_x,rope_request_y,rope_request_meta;
    reg rope_request_last;
    reg [7:0] rope_request_mask;
    wire rope_engine_ready, rope_engine_valid, rope_engine_last;
    wire [127:0] rope_engine_data,rope_engine_aux,rope_engine_meta;
    wire [7:0] rope_engine_mask;
    wire rope_request_fire=rope_req_valid && rope_req_ready;

    // ---- RoPE: private result elastic stage --------------------------------
    reg rope_result_pending;
    reg rope_result_last;
    reg [127:0] rope_result_data,rope_result_aux,rope_result_meta;
    reg [7:0] rope_result_mask;
    wire rope_result_stage_ready=!rope_result_pending || rope_res_ready;
    wire rope_result_capture=rope_engine_valid && rope_result_stage_ready;

    // Each side is reset independently; both releases last two clock edges.
    always @(posedge clk) begin
        if(!rst_n)begin rope_reset_release<=2'b00;end
        else begin rope_reset_release<={rope_reset_release[0],1'b1};end
    end
    always @(posedge clk) begin
        if(!rst_n)begin afu_reset_release<=2'b00;end
        else begin afu_reset_release<={afu_reset_release[0],1'b1};end
    end

    // Scalar AFU request stage: consume and refill on the same edge to retain
    // one request per cycle.  Ready no longer contains a backend decode cone.
    always @(posedge clk) begin
        if(!afu_reset_n)begin afu_request_pending<=1'b0;end
        else if(afu_request_fire)begin afu_request_pending<=1'b1;end
        else if(afu_core_ready)begin afu_request_pending<=1'b0;end
    end
    always @(posedge clk) begin
        if(afu_request_fire)begin
            afu_request_command<=ctrl_afu_command;
            afu_request_phase<=ctrl_afu_command_4[1:0];
            afu_request_eps<=ctrl_afu_eps;
            afu_request_length<=ctrl_afu_length;
            afu_request_x<=ctrl_afu_x;
            afu_request_y<=ctrl_afu_y;
            afu_request_meta<=ctrl_afu_meta;
            afu_request_last<=ctrl_afu_last;
            afu_request_mask<=ctrl_afu_mask;
        end
    end
    // Scalar AFU result stage: capture only when the stage can retire.
    always @(posedge clk)begin
        if(!afu_reset_n)afu_result_pending<=1'b0;
        else if(afu_result_stage_ready)afu_result_pending<=afu_core_valid;
    end
    always @(posedge clk)begin
        if(afu_result_capture)begin
            afu_result_data<=afu_core_data;
            afu_result_meta<=afu_core_meta;
            afu_result_last<=afu_core_last;
            afu_result_mask<=afu_core_mask;
        end
    end

    // RoPE request stage: same construct, sized to the RoPE payload set
    // (the engine consumes x/y/meta/last/mask only).
    always @(posedge clk) begin
        if(!rope_reset_n)begin rope_request_pending<=1'b0;end
        else if(rope_request_fire)begin rope_request_pending<=1'b1;end
        else if(rope_engine_ready)begin rope_request_pending<=1'b0;end
    end
    always @(posedge clk) begin
        if(rope_request_fire)begin
            rope_request_x<=rope_req_x;
            rope_request_y<=rope_req_y;
            rope_request_meta<=rope_req_meta;
            rope_request_last<=rope_req_last;
            rope_request_mask<=rope_req_mask;
        end
    end
    // RoPE result stage.
    always @(posedge clk)begin
        if(!rope_reset_n)rope_result_pending<=1'b0;
        else if(rope_result_stage_ready)rope_result_pending<=rope_engine_valid;
    end
    always @(posedge clk)begin
        if(rope_result_capture)begin
            rope_result_data<=rope_engine_data;
            rope_result_aux<=rope_engine_aux;
            rope_result_meta<=rope_engine_meta;
            rope_result_last<=rope_engine_last;
            rope_result_mask<=rope_engine_mask;
        end
    end

    assign internal_error=1'b0;
    // internal_busy now describes the scalar AFU channel only.  The controller
    // tracks the RoPE channel with its own rope_outstanding flag.
    assign internal_busy=afu_core_busy || afu_request_pending || afu_result_pending;
    assign ctrl_afu_ready=(!afu_request_pending || afu_core_ready);
    assign rope_req_ready=(!rope_request_pending || rope_engine_ready);

    assign ctrl_result_valid=afu_result_pending;
    assign ctrl_result_data=afu_result_data;
    // The scalar AFU never emits an auxiliary vector; the historic shared mux
    // drove 128 zero bits for every AFU result and this keeps that contract.
    assign ctrl_result_aux=128'd0;
    assign ctrl_result_meta=afu_result_meta;
    assign ctrl_result_last=afu_result_last;
    assign ctrl_result_mask=afu_result_mask;

    assign rope_res_valid=rope_result_pending;
    assign rope_res_data=rope_result_data;
    assign rope_res_aux=rope_result_aux;
    assign rope_res_meta=rope_result_meta;
    assign rope_res_last=rope_result_last;
    assign rope_res_mask=rope_result_mask;

    // Compute services run to completion; recovery uses the common hardware reset.
    qwen25_rope_vector rope_Inst (
        .i_clk(clk),.i_rstn(rope_reset_n),
        .i_valid(rope_request_pending && rope_reset_n),.o_ready(rope_engine_ready),
        .i_x(rope_request_x),.i_y(rope_request_y),.i_meta(rope_request_meta),
        .i_last(rope_request_last),.i_mask(rope_request_mask),
        .o_valid(rope_engine_valid),.i_ready(rope_result_stage_ready),
        .o_data(rope_engine_data),.o_aux(rope_engine_aux),.o_meta(rope_engine_meta),
        .o_last(rope_engine_last),.o_mask(rope_engine_mask),.o_busy()
    );
    generate
        if(1)begin:gen_internal_afu
            qwen25_afu afu_Inst (
                .i_clk(clk),.i_rstn(afu_reset_n),
                .i_valid(afu_request_pending && afu_reset_n),.o_ready(afu_core_ready),
                .i_command(afu_request_command),.i_phase(afu_request_phase),.i_eps(afu_request_eps),
                .i_length(afu_request_length),
                .i_x(afu_request_x),.i_y(afu_request_y),.i_meta(afu_request_meta),
                .i_last(afu_request_last),.i_mask(afu_request_mask),
                .o_valid(afu_core_valid),.i_ready(afu_result_stage_ready),
                .o_data(afu_core_data),.o_meta(afu_core_meta),.o_last(afu_core_last),
                .o_mask(afu_core_mask),.o_busy(afu_core_busy),.o_error()
            );
        end
    endgenerate

endmodule
`default_nettype wire
