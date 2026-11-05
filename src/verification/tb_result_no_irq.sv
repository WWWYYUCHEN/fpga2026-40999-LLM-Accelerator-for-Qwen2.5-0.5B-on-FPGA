`timescale 1ns/1ps
module tb_result_no_irq;
    reg clk = 0, rst_n = 0, write_enable = 0, commit = 0;
    reg [15:0] write_addr = 0, read_addr = 0;
    reg [63:0] write_data = 0;
    reg [7:0] write_strb = 8'hff;
    reg [255:0] status = 0;
    reg [17:0] token = 18'd321;
    wire mapped_w, mapped_r, read_allowed, start_busy, done, start, clear_result;
    wire [63:0] read_data;
    wire [17:0] best;
    always #5 clk = ~clk;
    qwen25_result_regs dut (
        .clk(clk), .rst_n(rst_n), .write_enable(write_enable), .write_addr(write_addr),
        .write_data(write_data), .write_strb(write_strb), .read_addr(read_addr),
        .write_mapped(mapped_w), .read_mapped(mapped_r), .read_data(read_data),
        .read_allowed(read_allowed), .start_while_busy(start_busy), .status(status),
        .hw_result_commit(commit), .hw_best_token_id(token), .best_token_id(best),
        .result_done(done), .task_start(start), .result_clear(clear_result),
        .abort_req(), .soft_reset_req(), .error_clear_req()
    );
    initial begin
        repeat (2) @(negedge clk);
        rst_n = 1;
        read_addr = 16'h01f8; #1;
        if (read_allowed !== 0) $fatal(1, "result accessible before commit");
        commit = 1; @(negedge clk); commit = 0; #1;
        if (!done || best != 321 || !read_allowed) $fatal(1, "commit failed");
        write_enable = 1; write_addr = 16'h0010; write_data = 1; #1;
        if (!start) $fatal(1, "start failed");
        @(negedge clk); write_enable = 0; #1;
        if (done || !clear_result) $fatal(1, "result clear failed");
        read_addr = 16'h0020; #1;
        if (!mapped_r || read_data !== 0) $fatal(1, "IRQ status is not reserved zero");
        read_addr = 16'h0028; #1;
        if (!mapped_r || read_data !== 0) $fatal(1, "IRQ mask is not reserved zero");
        $display("RESULT_NO_IRQ PASS");
        $finish;
    end
endmodule
