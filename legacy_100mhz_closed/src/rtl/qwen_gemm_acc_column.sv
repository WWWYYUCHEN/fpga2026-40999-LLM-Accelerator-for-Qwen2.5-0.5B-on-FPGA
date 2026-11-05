`timescale 1ns / 1ps
`default_nettype none

// One physical column stores eight slots of eight FP32 rows in 64x32 LUTRAM.
// Delay the read address, then sample RAM feedback at the final adder input edge.
module qwen_gemm_acc_column (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,
    input  wire         select_valid,
    input  wire [2:0]   select_slot,
    input  wire [2:0]   select_row,
    output wire [31:0]  selected_value,
    input  wire         add_valid,
    input  wire [31:0]  add_value,
    output wire [31:0]  add_result,
    output wire         add_output_valid,
    input  wire         write_enable,
    input  wire [2:0]   write_slot,
    input  wire [2:0]   write_row,
    input  wire [31:0]  write_value,
    output wire         write_committed
);
    // 地址{slot,row}对应物理列col+16*slot中的row；只复位事务有效位。
    // 单写口、无数据复位、异步读，符合分布式RAM推断模板。
    (* ram_style = "distributed" *) logic [31:0] reg_acc_values [0:63];
    wire [5:0] select_address = {select_slot, select_row};
    logic [5:0] reg_select_address;
    logic reg_select_valid;
    logic [5:0] reg_selected_address;
    logic [5:0] reg_write_address;
    logic [31:0] reg_write_value;
    logic reg_write_valid;

    // Write-through covers a RAM commit on the same edge as the adder input sample.
    wire feedback_forward = reg_write_valid && reg_write_address == reg_selected_address;
    assign selected_value = feedback_forward ? reg_write_value : reg_acc_values[reg_selected_address];
    assign write_committed = reg_write_valid;
    // 本列写地址和数据先寄存；下一拍落入RAM，保持原写提交时刻。
    always_ff @(posedge clk) begin
        if (!rst_n || clear) reg_write_valid <= 1'b0;
        else reg_write_valid <= write_enable;
        if (write_enable) begin
            reg_write_address <= {write_slot, write_row};
            reg_write_value <= write_value;
        end
    end
    always_ff @(posedge clk) begin
        if (reg_write_valid) begin
            reg_acc_values[reg_write_address] <= reg_write_value;
        end
    end

    // 只延迟地址，不提前锁存RAM数值；第二拍地址与partial-update数据对齐。
    always_ff @(posedge clk) begin
        if (!rst_n || clear) reg_select_valid <= 1'b0;
        else reg_select_valid <= select_valid;
        if (select_valid) begin
            reg_select_address <= select_address;
        end
        if (reg_select_valid) begin
            reg_selected_address <= reg_select_address;
        end
    end
    qwen_fp32_add_valid column_accumulate_Inst (
        .clk(clk), .rst_n(rst_n), .clear(clear), .input_valid(add_valid),
        .a(selected_value), .b(add_value),
        .result(add_result), .output_valid(add_output_valid)
    );
endmodule

`default_nettype wire
