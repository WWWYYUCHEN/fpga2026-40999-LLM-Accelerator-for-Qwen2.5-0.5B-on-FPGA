`timescale 1ns/1ps
module tb_array_pipeline;
    localparam COL = 32;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;
    reg load_valid = 0, load_bank = 0, compute_bank = 0;
    reg [2:0] load_row = 0;
    reg [2047:0] load_data = 0;
    reg valid = 0, advance = 0, output_ready = 1;
    reg [127:0] activation = 0;
    reg [159:0] meta = 0;
    wire ref_load_ready, new_load_ready, ref_ready, new_ready;
    wire ref_valid, new_valid, ref_busy, new_busy;
    wire [COL*32-1:0] ref_data, new_data;
    wire [167:0] ref_meta, new_meta;
    reg [COL*32+168-1:0] expected [0:511];
    integer written = 0, checked = 0, failures = 0;
    integer bank, row, item, lane, seed = 1459, watchdog = 0, cycles = 0;

    qwen_gemm_array_reference #(.COL(COL)) ref_dut (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .s_weight_load_valid(load_valid), .s_weight_load_ready(ref_load_ready),
        .s_weight_load_data(load_data), .s_weight_load_row(load_row), .s_weight_load_bank(load_bank),
        .s_compute_bank(compute_bank), .s_compute_bank_rows({8{compute_bank}}), .s_compute_release(1'b0),
        .s_valid(valid), .s_ready(ref_ready), .s_activation(activation), .s_weight_advance(advance), .s_meta(meta),
        .m_valid(ref_valid), .m_ready(output_ready), .m_data(ref_data), .m_meta(ref_meta),
        .busy(ref_busy), .protocol_error()
    );
    qwen_gemm_array_i #(.COL(COL)) new_dut (
        .clk(clk), .rst_n(rst_n), .clear(1'b0),
        .s_weight_load_valid(load_valid), .s_weight_load_ready(new_load_ready),
        .s_weight_load_data(load_data), .s_weight_load_row(load_row), .s_weight_load_bank(load_bank),
        .s_compute_bank(compute_bank), .s_compute_bank_rows({8{compute_bank}}), .s_compute_release(1'b0),
        .s_valid(valid), .s_ready(new_ready), .s_activation(activation), .s_weight_advance(advance), .s_meta(meta),
        .m_valid(new_valid), .m_ready(output_ready), .m_data(new_data), .m_meta(new_meta),
        .busy(new_busy), .protocol_error()
    );

    always @(posedge clk) if (rst_n) begin
        if (ref_valid && output_ready) begin
            expected[written] = {ref_meta, ref_data};
            written = written + 1;
        end
        if (new_valid && output_ready) begin
            if (checked >= written || (^{new_meta, new_data}) === 1'bx ||
                {new_meta, new_data} !== expected[checked]) begin
                $display("FAIL output %0d expected_count=%0d", checked, written);
                failures = failures + 1;
            end
            checked = checked + 1;
        end
        if (valid && ref_ready !== new_ready) begin
            $display("FAIL differing input readiness");
            failures = failures + 1;
        end
    end

    always @(negedge clk) if (rst_n) begin
        cycles = cycles + 1;
        output_ready = (cycles % 7 < 4);
    end

    initial begin
        repeat (4) @(negedge clk);
        rst_n = 1;
        repeat (3) @(negedge clk);
        for (bank = 0; bank < 2; bank = bank + 1) begin
            load_bank = bank;
            for (row = 0; row < 8; row = row + 1) begin
                for (item = 0; item < 128; item = item + 1)
                    load_data[item*16 +: 16] = $random(seed);
                load_row = row;
                load_valid = 1;
                while (!(ref_load_ready && new_load_ready)) @(negedge clk);
                @(negedge clk);
            end
            load_valid = 0;
            @(negedge clk);
        end
        for (bank = 0; bank < 2; bank = bank + 1) begin
            compute_bank = bank;
            for (item = 0; item < 32; item = item + 1) begin
                for (lane = 0; lane < 8; lane = lane + 1)
                    activation[lane*16 +: 16] = $random(seed);
                meta = bank*32 + item;
                advance = (item % 8 == 7);
                valid = 1;
                while (!(ref_ready && new_ready)) @(negedge clk);
                @(negedge clk);
            end
            valid = 0;
            advance = 0;
            repeat (2) @(negedge clk);
        end
        while ((ref_busy || new_busy) && watchdog < 1000) begin
            watchdog = watchdog + 1;
            @(negedge clk);
        end
        if (watchdog == 1000 || written != 64 || checked != 64) failures = failures + 1;
        if (failures == 0) $display("ARRAY_PIPELINE PASS checked=%0d", checked);
        else $fatal(1, "ARRAY_PIPELINE FAIL failures=%0d written=%0d checked=%0d", failures, written, checked);
        $finish;
    end
    initial begin
        #100000;
        $fatal(1, "ARRAY_PIPELINE timeout");
    end
endmodule
