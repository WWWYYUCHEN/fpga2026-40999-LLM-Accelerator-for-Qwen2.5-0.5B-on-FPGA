`timescale 1ns / 1ps
// Eight physical ADDs serve either eight independent pairs or a 4/2/1 tree and feedback.
module qwen25_afu_add_bank (
    input wire i_clk,
    input wire i_rstn,
    input wire i_start,
    input wire i_reduce,
    input wire i_valid,
    output wire o_ready,
    input wire [255:0] i_a,
    input wire [255:0] i_b,
    input wire i_finish,
    output wire o_vector_valid,
    output wire [255:0] o_vector,
    output wire o_sum_valid,
    output wire [31:0] o_sum,
    output wire o_busy
);
    localparam [1:0] ST_ACCEPT = 0, ST_DRAIN = 1, ST_FINAL = 2, ST_WAIT = 3;
    reg [1:0] state_current, state_next;
    reg [31:0] reg_partial [0:7];
    reg [2:0] cnt_slot;
    reg [2:0] reg_slot_pipe [0:7];
    reg [12:0] cnt_reduce_pending;
    reg [4:0] cnt_vector_pending;
    reg [255:0] dec_a, dec_b;
    reg [7:0] dec_valid;
    wire [255:0] wire_result;
    wire [7:0] wire_valid;
    wire flag_accept = i_valid && o_ready;
    wire flag_reduce_accept = flag_accept && i_reduce;
    wire flag_vector_accept = flag_accept && !i_reduce;
    wire flag_feedback = i_reduce && wire_valid[6] &&
                         (state_current == ST_ACCEPT || state_current == ST_DRAIN);
    wire flag_feedback_retire = i_reduce && wire_valid[7] &&
                                (state_current == ST_ACCEPT || state_current == ST_DRAIN);
    wire [31:0] wire_previous = flag_feedback_retire && reg_slot_pipe[7] == cnt_slot ?
                               wire_result[255:224] : reg_partial[cnt_slot];
    integer slot_index;
    genvar adder;
    assign o_ready = state_current == ST_ACCEPT && !i_start;
    assign o_vector_valid = !i_reduce && (&wire_valid);
    assign o_vector = wire_result;
    assign o_sum_valid = state_current == ST_WAIT && wire_valid[6];
    assign o_sum = wire_result[223:192];
    assign o_busy = cnt_reduce_pending != 0 || cnt_vector_pending != 0 || state_current != ST_ACCEPT;

    always @(*) begin
        dec_a = i_a;
        dec_b = i_b;
        dec_valid = {8{flag_vector_accept}};
        if (i_reduce) begin
            dec_a = 0;
            dec_b = 0;
            dec_valid = 0;
            for (slot_index = 0; slot_index < 4; slot_index = slot_index + 1) begin
                dec_a[slot_index*32 +: 32] = state_current == ST_FINAL ?
                    reg_partial[slot_index*2] : i_a[slot_index*64 +: 32];
                dec_b[slot_index*32 +: 32] = state_current == ST_FINAL ?
                    reg_partial[slot_index*2+1] : i_a[slot_index*64+32 +: 32];
                dec_valid[slot_index] = flag_reduce_accept || state_current == ST_FINAL;
            end
            dec_a[159:128] = wire_result[31:0];
            dec_b[159:128] = wire_result[63:32];
            dec_valid[4] = wire_valid[0] && wire_valid[1];
            dec_a[191:160] = wire_result[95:64];
            dec_b[191:160] = wire_result[127:96];
            dec_valid[5] = wire_valid[2] && wire_valid[3];
            dec_a[223:192] = wire_result[159:128];
            dec_b[223:192] = wire_result[191:160];
            dec_valid[6] = wire_valid[4] && wire_valid[5];
            dec_a[255:224] = wire_previous;
            dec_b[255:224] = wire_result[223:192];
            dec_valid[7] = flag_feedback;
        end
    end
    generate
        for (adder = 0; adder < 8; adder = adder + 1) begin : gen_add
            qwen25_afu_add arithmetic_Inst (
                .i_clk(i_clk), .i_rstn(i_rstn), .i_clear(i_start),
                .i_valid(dec_valid[adder]), .i_a(dec_a[adder*32 +: 32]),
                .i_b(dec_b[adder*32 +: 32]),
                .o_valid(wire_valid[adder]), .o_result(wire_result[adder*32 +: 32])
            );
        end
    endgenerate
    always @(*) begin
        state_next = state_current;
        case (state_current)
            ST_ACCEPT: begin
                if (i_finish && i_reduce) begin
                    state_next = ST_DRAIN;
                end
            end
            ST_DRAIN: begin
                if (cnt_reduce_pending == 0) begin
                    state_next = ST_FINAL;
                end
            end
            ST_FINAL: begin
                state_next = ST_WAIT;
            end
            ST_WAIT: begin
                if (o_sum_valid) begin
                    state_next = ST_ACCEPT;
                end
            end
            default: begin
                state_next = ST_ACCEPT;
            end
        endcase
        if (i_start) begin
            state_next = ST_ACCEPT;
        end
    end
    always @(posedge i_clk) begin
        if (!i_rstn) begin
            state_current <= ST_ACCEPT;
        end else begin
            state_current <= state_next;
        end
    end
    integer partial_index;
    always @(posedge i_clk) begin
        if (!i_rstn || i_start) begin
            cnt_slot <= 0;
            cnt_reduce_pending <= 0;
            cnt_vector_pending <= 0;
            for (partial_index = 0; partial_index < 8; partial_index = partial_index + 1) begin
                reg_partial[partial_index] <= 0;
                reg_slot_pipe[partial_index] <= 0;
            end
        end else begin
            case ({flag_reduce_accept, flag_feedback_retire})
                2'b10: begin
                    cnt_reduce_pending <= cnt_reduce_pending + 1'b1;
                end
                2'b01: begin
                    cnt_reduce_pending <= cnt_reduce_pending - 1'b1;
                end
                default: begin
                    cnt_reduce_pending <= cnt_reduce_pending;
                end
            endcase
            case ({flag_vector_accept, o_vector_valid})
                2'b10: begin
                    cnt_vector_pending <= cnt_vector_pending + 1'b1;
                end
                2'b01: begin
                    cnt_vector_pending <= cnt_vector_pending - 1'b1;
                end
                default: begin
                    cnt_vector_pending <= cnt_vector_pending;
                end
            endcase
            if (flag_feedback) begin
                cnt_slot <= cnt_slot + 1'b1;
            end
            reg_slot_pipe[0] <= cnt_slot;
            for (partial_index = 1; partial_index < 8; partial_index = partial_index + 1) begin
                reg_slot_pipe[partial_index] <= reg_slot_pipe[partial_index-1];
            end
            if (flag_feedback_retire) begin
                reg_partial[reg_slot_pipe[7]] <= wire_result[255:224];
            end
        end
    end
endmodule
