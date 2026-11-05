`timescale 1ns / 1ps
`default_nettype none

// Weight-stationary PE: two independent banks of eight BF16 slots.
// 权重逐行装载；activation同拍广播，权重数据没有reset/flush。
module qwen_gemm_pe (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         clear,

    input  wire         weight_write,
    input  wire         weight_write_bank,
    input  wire         compute_bank,
    input  wire [127:0]  weight_write_data,

    input  wire [15:0]   activation_in,
    input  wire         token_valid_in,
    input  wire         advance_weight_in,
    output wire [31:0]   product,
    output wire         product_valid
);
    // 每个bank八槽FF；装载bank与计算bank独立选择。
    (* keep = "true", ram_style = "registers", shreg_extract = "no", extract_enable = "yes" *) logic [15:0] reg_weights [0:15];
    logic [2:0] reg_weight_index [0:1];

    // Loading the idle bank must not suppress active-bank slot advancement.
    for (genvar bank = 0; bank < 2; bank++) begin : gen_bank_index
        always_ff @(posedge clk) begin
            if (!rst_n ) begin
                reg_weight_index[bank] <= 3'd0;
            end else if (weight_write && weight_write_bank == bank) begin
                reg_weight_index[bank] <= 3'd0;
            end else if (token_valid_in && advance_weight_in && compute_bank == bank) begin
                reg_weight_index[bank] <= reg_weight_index[bank] + 3'd1;
            end
        end
    end

    for (genvar weight_slot = 0; weight_slot < 8; weight_slot++) begin : gen_weight_slot
        // Slots remain stationary; only a weight-load handshake writes their FFs.
        always_ff @(posedge clk) begin
            if (weight_write && !weight_write_bank)
                reg_weights[weight_slot] <= weight_write_data[weight_slot*16 +: 16];
            if (weight_write && weight_write_bank)
                reg_weights[8+weight_slot] <= weight_write_data[weight_slot*16 +: 16];
        end
    end

    wire [255:0] packed_weights;
    wire [15:0] selected_weight;
    for(genvar mux_slot=0;mux_slot<16;mux_slot++)begin:gen_mux_slot
        assign packed_weights[mux_slot*16+:16]=reg_weights[mux_slot];
    end
    qwen_pe_weight_select select_Inst(.i_weights(packed_weights),.i_bank(compute_bank),
        .i_slot(reg_weight_index[compute_bank]),.o_weight(selected_weight));
    // Operand and result data are direct; output_valid controls consumption.
    qwen_bf16_mul_valid multiply_Inst (
        .clk(clk), .rst_n(rst_n), .clear(clear), .input_valid(token_valid_in),
        .a(activation_in), .b(selected_weight),
        .result(product), .output_valid(product_valid)
    );
endmodule

`default_nettype wire

