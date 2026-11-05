`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_dma_wr_latchup -- does a *bad* write command permanently kill the DMA
// write path?
//
// Motivation (read-only, from another agent's E2E probe runB_diag\dbg20.out):
//   the INT8 post FSM sits in PS_KV8_VSEND forever with wr_busy=1 and
//   weight_load_done=0.  qwen25_mmu_dma.v:54 refuses EVERY new write command
//   while flag_write_active is set, and flag_write_active is only cleared by
//   wr_done, which qwen25_dma_wr.v:199 only pulses when
//       (aw_out==1) && (w_left==0) && !cur_valid.
//   qwen25_dma_wr.v:91 additionally drops any command that has
//       cmd_bytes==0  OR  addr[3:0]!=0
//   (q_push = cmd_fire && (cmd_bytes != 0) && (s_cmd[3:0] == 0)), raising
//   sticky error_code 0x21 -- but qwen25_mmu_dma has ALREADY set
//   flag_write_active on that same handshake.  If that is true, one such
//   command latches the whole write path off for the rest of the run, which is
//   exactly the observed wr_busy=1-forever.
//
// This TB proves or disproves that at module level, with a minimal AXI slave.
// =============================================================================
module tb_kv8_dma_wr_latchup;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // ------------------------------------------------------------ command port
    reg  [127:0] cmd;
    reg          cmd_valid;
    wire         cmd_ready;

    // ------------------------------------------------------------ write stream
    reg  [127:0] s_data;
    reg  [15:0]  s_keep = 16'hffff;
    reg          s_last, s_valid;
    wire         s_ready;

    // ------------------------------------------------------------ AXI (4 ports)
    wire [255:0] awaddr;
    wire [31:0]  awlen;
    wire [11:0]  awsize;
    wire [7:0]   awburst;
    wire [3:0]   awvalid;
    reg  [3:0]   awready;
    wire [511:0] wdata;
    wire [63:0]  wstrb;
    wire [3:0]   wlast, wvalid;
    reg  [3:0]   wready;
    reg  [3:0]   bvalid;
    reg  [7:0]   bresp  = 8'd0;
    reg  [15:0]  bid    = 16'd0;
    wire [3:0]   bready;
    wire [255:0] araddr;
    wire [31:0]  arlen;
    wire [11:0]  arsize;
    wire [7:0]   arburst;
    wire [3:0]   arvalid, rready;

    wire         wr_busy, wr_done, wr_error, wr_aborted;
    wire [7:0]   wr_error_code;
    wire         rd_busy, rd_done, rd_error, rd_aborted;
    wire [7:0]   rd_error_code;

    qwen25_mmu_dma #(.RS_LV(0), .DIRECT_BUFFER_READ(0)) dut (
        .clk(clk), .rst_n(rst_n),
        .s_cmd(cmd), .s_cmd_meta(128'd0), .s_cmd_valid(cmd_valid), .s_cmd_ready(cmd_ready),
        .abort_req(1'b0), .soft_reset_req(1'b0),
        .m_axi_araddr(araddr), .m_axi_arlen(arlen), .m_axi_arsize(arsize), .m_axi_arburst(arburst),
        .m_axi_arvalid(arvalid), .m_axi_arready(4'd0),
        .m_axi_rid(16'd0), .m_axi_rdata(512'd0), .m_axi_rresp(8'd0), .m_axi_rlast(4'd0),
        .m_axi_rvalid(4'd0), .m_axi_rready(rready),
        .m_stream_data(), .m_stream_keep(), .m_stream_last(), .m_stream_meta(),
        .m_stream_tag(), .m_stream_resource(), .m_stream_error(),
        .m_stream_valid(), .m_stream_ready(1'b0),
        .m_buffer_valid(), .m_buffer_ready(4'd0), .m_buffer_data(), .m_buffer_word(),
        .m_buffer_keep(), .m_buffer_error(), .m_buffer_meta(),
        .rd_busy(rd_busy), .rd_done(rd_done), .rd_error(rd_error), .rd_aborted(rd_aborted),
        .rd_error_code(rd_error_code), .rd_completion_tag(), .rd_completion_meta(),
        .s_stream_data(s_data), .s_stream_keep(s_keep), .s_stream_last(s_last),
        .s_stream_meta(128'd0), .s_stream_valid(s_valid), .s_stream_ready(s_ready),
        .m_axi_awaddr(awaddr), .m_axi_awlen(awlen), .m_axi_awsize(awsize), .m_axi_awburst(awburst),
        .m_axi_awvalid(awvalid), .m_axi_awready(awready),
        .m_axi_wdata(wdata), .m_axi_wstrb(wstrb), .m_axi_wlast(wlast),
        .m_axi_wvalid(wvalid), .m_axi_wready(wready),
        .m_axi_bresp(bresp), .m_axi_bid(bid), .m_axi_bvalid(bvalid), .m_axi_bready(bready),
        .wr_busy(wr_busy), .wr_done(wr_done), .wr_error(wr_error), .wr_aborted(wr_aborted),
        .wr_error_code(wr_error_code), .wr_completion_tag(), .wr_completion_meta()
    );

    // ------------------------------------------------- minimal AXI write slave
    // One transaction at a time per port; B arrives 2 cycles after WLAST.
    reg        active  [0:3];
    reg        bpend   [0:3];
    reg [3:0]  bdelay  [0:3];
    integer pc, ps;
    always @(*) begin
        awready = 4'd0;
        wready  = 4'd0;
        for (pc = 0; pc < 4; pc = pc + 1) begin
            awready[pc] = rst_n && !active[pc] && !bpend[pc];
            wready[pc]  = rst_n && active[pc];
            bvalid[pc]  = bpend[pc] && (bdelay[pc] == 4'd0);
        end
    end
    always @(posedge clk) begin
        for (ps = 0; ps < 4; ps = ps + 1) begin
            if (!rst_n) begin
                active[ps] <= 1'b0; bpend[ps] <= 1'b0; bdelay[ps] <= 4'd0;
            end else begin
                if (awvalid[ps] && awready[ps]) begin
                    active[ps] <= 1'b1;
                end
                if (wvalid[ps] && wready[ps] && wlast[ps]) begin
                    active[ps] <= 1'b0;
                    bpend[ps]  <= 1'b1;
                    bdelay[ps] <= 4'd2;
                end
                if (bvalid[ps] && bready[ps]) bpend[ps] <= 1'b0;
                else if (bpend[ps] && bdelay[ps] != 4'd0) bdelay[ps] <= bdelay[ps] - 4'd1;
            end
        end
    end

    // ------------------------------------------------------------- scoreboard
    integer done_count = 0;
    integer cmd_fires  = 0;
    always @(posedge clk) begin
        if (wr_done) done_count <= done_count + 1;
        if (cmd_valid && cmd_ready) cmd_fires <= cmd_fires + 1;
    end

    // ------------------------------------------------------------- helpers
    task automatic do_reset;
        begin
            rst_n = 1'b0; cmd_valid = 1'b0; s_valid = 1'b0; s_last = 1'b0;
            cmd = 128'd0; s_data = 128'd0;
            repeat (6) @(negedge clk);
            rst_n = 1'b1;
            repeat (4) @(negedge clk);
        end
    endtask

    // one write command: addr, byte count, max-burst field (255 -> 16 beats/AW)
    task automatic send_cmd(input [63:0] addr, input [23:0] bytes, input [7:0] maxburst);
        begin
            cmd = 128'd0;
            cmd[63:0]    = addr;
            cmd[87:64]   = bytes;
            cmd[95:88]   = maxburst;
            cmd[115:112] = 4'd2;      // same op code the controller puts there
            cmd[118]     = 1'b1;      // write
            $display("  [dbg] send_cmd addr=%h bytes=%0d maxb=%0d", addr, bytes, maxburst);
            @(negedge clk);
            cmd_valid = 1'b1;
            @(posedge clk);
            while (!cmd_ready) @(posedge clk);
            $display("  [dbg]   accepted, wr_busy=%b", wr_busy);
            @(negedge clk);
            cmd_valid = 1'b0;
        end
    endtask

    task automatic send_beats(input integer n);
        integer i;
        begin
            for (i = 0; i < n; i = i + 1) begin
                @(negedge clk);
                s_data  = {i[15:0], 112'd0};
                s_last  = (i == n - 1);
                s_valid = 1'b1;
                @(posedge clk);
                while (!s_ready) @(posedge clk);
                @(negedge clk);
                s_valid = 1'b0;
                s_last  = 1'b0;
            end
        end
    endtask

    // wait for done (or time out); reports and returns the number of dones seen
    task automatic wait_done(input integer limit, output integer got, output integer waited);
        integer t;
        integer d0;
        begin
            d0 = done_count; t = 0;
            while ((done_count == d0) && (t < limit)) begin
                @(negedge clk);
                t = t + 1;
            end
            got = done_count - d0;
            waited = t;
        end
    endtask

    integer got, waited, d0;
    initial begin
        $display("=== tb_kv8_dma_wr_latchup : bad write command = permanent lockup? ===");
        wait_done(1, got, waited);   // let done_count initialise

        // ---- CASE 1: control. 128 B, 16 B aligned, 8 beats. Must complete. ----
        do_reset; done_count = 0; cmd_fires = 0;
        send_cmd(64'h0000_0000_1000_0000, 24'd128, 8'd255);
        send_beats(8);
        wait_done(200, got, waited);
        $display("CASE1 aligned_128B   done=%0d waited=%0d wr_busy=%b wr_error=%b code=%02h cmd_ready=%b",
                 got, waited, wr_busy, wr_error, wr_error_code, cmd_ready);
        if (got != 1) $display("  CASE1 FAIL: a well-formed 128 B write did not complete");

        // ---- CASE 2: bytes == 0 -------------------------------------------
        do_reset; done_count = 0;
        send_cmd(64'h0000_0000_1000_0000, 24'd0, 8'd255);
        wait_done(100, got, waited);
        $display("CASE2 zero_bytes     done=%0d waited=%0d wr_busy=%b wr_error=%b code=%02h cmd_ready=%b",
                 got, waited, wr_busy, wr_error, wr_error_code, cmd_ready);
        // recovery probe: is the write path still usable afterwards?  A good
        // command cannot even be presented (cmd_ready is stuck low), so we only
        // observe the port instead of driving it (a driver would wait forever).
        begin : recovery
            integer t;
            t = 0;
            while (!cmd_ready && t < 50) begin @(negedge clk); t = t + 1; end
            $display("CASE2b after_zero    cmd_ready_after_50cyc=%b wr_busy=%b done_since=%0d  (ready=0 -> path latched off)",
                     cmd_ready, wr_busy, done_count);
        end

        // ---- CASE 3: address not 16 B aligned ------------------------------
        do_reset; done_count = 0;
        send_cmd(64'h0000_0000_1000_0008, 24'd128, 8'd255);
        wait_done(100, got, waited);
        $display("CASE3 misaligned_addr done=%0d waited=%0d wr_busy=%b wr_error=%b code=%02h cmd_ready=%b",
                 got, waited, wr_busy, wr_error, wr_error_code, cmd_ready);

        // ---- CASE 4: control again after a reset, to prove the reset works --
        do_reset; done_count = 0;
        send_cmd(64'h0000_0000_1000_0000, 24'd256, 8'd255);
        send_beats(16);
        wait_done(200, got, waited);
        $display("CASE4 aligned_256B   done=%0d waited=%0d wr_busy=%b wr_error=%b",
                 got, waited, wr_busy, wr_error);
        if (got != 1) $display("  CASE4 FAIL: 256 B write did not complete");

        $display("LATCHUP_PROBE_DONE");
        $finish;
    end

    initial begin
        #(200*1000*1000);
        $display("FAIL: global timeout");
        $finish;
    end
endmodule
`default_nettype wire

