`timescale 1ns / 1ps
`default_nettype none

// Sixteen independent 64x32 LUTRAM columns retain the 8x128 FP32 partial tile.
// 中间和保存在GEMM内部LUTRAM；真实加法器valid驱动写回，不占BRAM/URAM。
module qwen_gemm_partial #(parameter integer COL = 16) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,
    input  wire         update_valid,
    input  wire         first_k,
    input  wire [2:0]   update_slot,
    input  wire [2:0]   update_row,
    input  wire [COL*32-1:0] update_data,
    output wire         update_done,
    output wire         busy,
    output wire         protocol_error,
    input  wire         read_enable,
    input  wire [5:0]   read_address,
    output wire [COL*32-1:0] read_data,
    output wire         read_valid
);
    wire select_valid = update_valid || read_enable;
    wire [2:0] select_slot = update_valid ? update_slot : read_address[5:3];
    wire [2:0] select_row = update_valid ? update_row : read_address[2:0];
    logic [1:0] reg_valid, reg_update, reg_first;
    logic [2:0] reg_slot [0:1], reg_row [0:1];
    logic [COL*32-1:0] reg_sum [0:1];
    wire [COL*32-1:0] add_results;
    wire [COL-1:0] add_output_valid;

    // 16列同拍，一个共享地址FIFO对应最终加法器真实完成事件。
    (* ram_style = "distributed" *) logic [5:0] addition_identity [0:15];
    logic [3:0] identity_write_pointer, identity_read_pointer;
    logic [4:0] pending_additions;
    logic [6:0] pending_updates;
    wire add_launch = reg_valid[1] && reg_update[1] && !reg_first[1];
    wire first_write = reg_valid[1] && reg_update[1] && reg_first[1];
    wire accumulate_write = (&add_output_valid) && (pending_additions != 0);
    wire write_fire = first_write || accumulate_write;
    wire [COL*32-1:0] first_results;
    wire [COL*32-1:0] write_data = first_write ? first_results : add_results;
    wire [5:0] write_address = first_write ? {reg_slot[1],reg_row[1]} : addition_identity[identity_read_pointer];
    logic reg_write_valid;
    logic [5:0] reg_write_address;
    logic [COL*32-1:0] reg_write_data;
    wire [COL-1:0] write_committed;
    wire commit_fire = &write_committed;

    assign update_done = commit_fire;
    assign busy = pending_updates != 0;
    assign protocol_error = 1'b0;
    assign read_valid = reg_valid[1] && !reg_update[1];
    for (genvar col = 0; col < COL; col++) begin : gen_column_accumulator
        // 首块树结果直接写，±0规范为+0，保持原+0初值加法的数值合同。
        assign first_results[col*32 +: 32] =
            (reg_sum[1][col*32 +: 31] == 0) ? 32'b0 : reg_sum[1][col*32 +: 32];
             qwen_gemm_acc_column column_acc_Inst (
            .clk(clk), .rst_n(rst_n), .clear(clear), .select_valid(select_valid),
            .select_slot(select_slot), .select_row(select_row),
            .selected_value(read_data[col*32 +: 32]),
            .add_valid(add_launch), .add_value(reg_sum[1][col*32 +: 32]),
            .add_result(add_results[col*32 +: 32]), .add_output_valid(add_output_valid[col]),
            .write_enable(reg_write_valid), .write_slot(reg_write_address[5:3]), .write_row(reg_write_address[2:0]),
            .write_value(reg_write_data[col*32 +: 32]), .write_committed(write_committed[col])
        );
    end

    always_ff @(posedge clk) begin
        if (!rst_n ) begin
            reg_valid <= 0; reg_update <= 0; reg_first <= 0;
            identity_write_pointer <= 0; identity_read_pointer <= 0;
            pending_additions <= 0; pending_updates <= 0;
            reg_write_valid <= 0;
        end else begin
            reg_write_valid <= write_fire;
            if (write_fire) begin
                reg_write_address <= write_address; reg_write_data <= write_data;
            end
            reg_valid <= {reg_valid[0],select_valid};
            reg_update <= {reg_update[0],update_valid};
            reg_first <= {reg_first[0],first_k};
            if (select_valid) begin
                reg_slot[0] <= select_slot; reg_row[0] <= select_row;
                reg_sum[0] <= update_data;
            end
            if (reg_valid[0]) begin
                reg_slot[1] <= reg_slot[0]; reg_row[1] <= reg_row[0];
                reg_sum[1] <= reg_sum[0];
            end
            if (add_launch) begin
                addition_identity[identity_write_pointer] <= {reg_slot[1],reg_row[1]};
                identity_write_pointer <= identity_write_pointer + 4'd1;
            end
            if (accumulate_write) identity_read_pointer <= identity_read_pointer + 4'd1;
            case ({add_launch,accumulate_write})
                2'b10: pending_additions <= pending_additions + 5'd1;
                2'b01: pending_additions <= pending_additions - 5'd1;
                default: begin end
            endcase
            case ({update_valid,commit_fire})
                2'b10: pending_updates <= pending_updates + 7'd1;
                2'b01: pending_updates <= pending_updates - 7'd1;
                default: begin end
            endcase
        end
    end
endmodule

`default_nettype wire
