`timescale 1ns / 1ps
`default_nettype none

// 8 K-rows × 16 columns: one streamed weight row per handshake, activation broadcast.
// 每个K行activation同时连接16列；权重八行装载完整后才能注入计算。
module qwen_gemm_array #(parameter integer DISTRIBUTED_COMPUTE_BANK=0) (
    input wire clk,
    input wire rst_n,
    input wire clear,
    input wire s_weight_load_valid,
    output wire s_weight_load_ready,
    input wire [2047:0] s_weight_load_data,
    input wire [2:0] s_weight_load_row,
    input wire s_weight_load_bank,
    input wire s_compute_bank,
    input wire [7:0] s_compute_bank_rows,
    input wire s_compute_release,
    input wire s_valid,
    output wire s_ready,
    input wire [127:0] s_activation,
    input wire s_weight_advance,
    input wire [159:0] s_meta,
    output wire m_valid,
    input wire m_ready,
    output wire [511:0] m_data,
    output wire [159:0] m_meta,
    output wire busy,
    output wire protocol_error
);
    localparam integer FIFO_DEPTH = 64;
    wire [511:0] aligned_results;
    wire [15:0] aligned_valid;
    logic [1:0] reg_weights_loaded;
    logic reg_available;
    // 顺序身份FIFO与输出FIFO分离。真实aligned_valid弹出身份，不以固定周期判完成。
    (* ram_style = "distributed" *) logic [159:0] pending_identity [0:63];
    logic [5:0] identity_write_pointer, identity_read_pointer;
    logic [6:0] pending_count;
    (* ram_style = "distributed" *) logic [671:0] output_fifo [0:63];
    logic [5:0] read_pointer, write_pointer;
    logic [6:0] occupancy, credits;
    (* max_fanout = 64 *) wire load_fire = s_weight_load_valid && s_weight_load_ready;
    wire accept = s_valid && s_ready;
    wire retire = m_valid && m_ready;
    wire produce = (&aligned_valid) && (pending_count != 0);

    // 乘法接受沿已锁存旧操作数；输入停止后可覆盖权重，在途结果继续输出。
    assign s_weight_load_ready = reg_available &&
        (!s_valid || s_weight_load_bank!=s_compute_bank);
    assign s_ready = reg_available && reg_weights_loaded[s_compute_bank] && (credits < FIFO_DEPTH) &&
        (!s_weight_load_valid || s_weight_load_bank!=s_compute_bank);
    assign m_valid = (occupancy != 0);
    assign {m_meta, m_data} = output_fifo[read_pointer];
    assign busy = (credits != 0);
    assign protocol_error = 1'b0;

    // 每拍只装一个K行的16个PE，乘法输入同拍广播，无列间hop/deskew。
    // K-tail operands are zero-padded upstream; arithmetic data is never valid-masked.
    wire [31:0] products [0:7][0:15];
    wire product_valid [0:7][0:15];

    for (genvar k = 0; k < 8; k++) begin : gen_k_row
        (* max_fanout = 128 *) wire row_write = load_fire && (s_weight_load_row == k);
        for (genvar col = 0; col < 16; col++) begin : gen_column
            wire [127:0] load_slots;
            for (genvar slot = 0; slot < 8; slot++) begin : gen_load_slot
                // 本拍W[row,col+16*slot]；只有row_write行的寄存器捕获。
                assign load_slots[slot*16 +: 16] = s_weight_load_data[(col+16*slot)*16 +: 16];
            end
            (* keep_hierarchy = "yes" *) qwen_gemm_pe pe_Inst (
                .clk(clk), .rst_n(rst_n), .clear(clear), .weight_write(row_write),
                .weight_write_bank(s_weight_load_bank),
                .compute_bank(DISTRIBUTED_COMPUTE_BANK!=0 ? s_compute_bank_rows[k] : s_compute_bank),
                .weight_write_data(load_slots),
                .activation_in(s_activation[k*16 +: 16]), .token_valid_in(accept),
                .advance_weight_in(s_weight_advance),
                .product(products[k][col]), .product_valid(product_valid[k][col])
            );
        end
    end

    for (genvar col = 0; col < 16; col++) begin : gen_reduction
        wire [255:0] column_products;
        wire [31:0] column_sum;
        wire column_valid;
        wire [7:0] column_product_valid;
        for (genvar k = 0; k < 8; k++) begin : gen_tree_input
            assign column_products[k*32 +: 32] = products[k][col];
            assign column_product_valid[k] = product_valid[k][col];
        end
        qwen_fp32_sum8 sum_tree_Inst (
            .clk(clk), .rst_n(rst_n), .clear(clear), .input_valid(&column_product_valid), .values(column_products),
            .result(column_sum), .output_valid(column_valid)
        );
        assign aligned_results[col*32 +: 32] = column_sum;
        assign aligned_valid[col] = column_valid;
    end

    // 数值流水无ready，通过先预留FIFO credit处理任意长输出停顿。
    always_ff @(posedge clk) begin
        if (!rst_n || clear) begin
            identity_write_pointer <= 0; identity_read_pointer <= 0; pending_count <= 0;
            read_pointer <= 0; write_pointer <= 0;
            occupancy <= 0; credits <= 0;
            reg_weights_loaded <= 0;
            reg_available <= 0;
        end else begin
            reg_available <= 1;
            if (accept) begin
                pending_identity[identity_write_pointer] <= s_meta;
                identity_write_pointer <= identity_write_pointer + 6'd1;
            end
            if (produce) identity_read_pointer <= identity_read_pointer + 6'd1;
            case ({accept, produce})
                2'b10: pending_count <= pending_count + 7'd1;
                2'b01: pending_count <= pending_count - 7'd1;
                default: begin end
            endcase
            if (load_fire && s_weight_load_row == 0) reg_weights_loaded[s_weight_load_bank] <= 1'b0;
            if (load_fire && s_weight_load_row == 7) reg_weights_loaded[s_weight_load_bank] <= 1'b1;
            if (s_compute_release) reg_weights_loaded[s_compute_bank] <= 1'b0;
            case ({accept, retire})
                2'b10: credits <= credits + 7'd1;
                2'b01: credits <= credits - 7'd1;
                default: begin end
            endcase
            case ({produce, retire})
                2'b10: occupancy <= occupancy + 7'd1;
                2'b01: occupancy <= occupancy - 7'd1;
                default: begin end
            endcase
            if (produce) begin
                output_fifo[write_pointer] <= {pending_identity[identity_read_pointer], aligned_results};
                write_pointer <= write_pointer + 6'd1;
            end
            if (retire) read_pointer <= read_pointer + 6'd1;
        end
    end
endmodule
`default_nettype wire
