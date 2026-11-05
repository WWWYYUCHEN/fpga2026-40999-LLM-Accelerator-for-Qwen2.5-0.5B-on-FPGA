`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_i_scale_fifo -- stage-3 WIRING block testbench (self-checking)
//
// kv8_i_scale_fifo is the 16 x (8 x BF16) store between kv8_scale_ram and the
// score ingress.  This TB drives its fill side with a byte-accurate model of the
// RAM read port (3-cycle latency, in-order returns) and checks:
//
//   PHASE1  the fill port issues base+0..15 ascending, one per cycle, and
//           o_ready is LOW until the 16th word has returned.
//   PHASE2  the read side is indexed by {slice[2:0], sub} = the token word inside
//           the tile, and every one of the 8 query rows of a slice reads the
//           SAME word -- this is why it is a 16-entry indexed store and not a
//           128-entry pop-FIFO.
//   PHASE3  the mapping is not vacuous: all 16 words are distinct and each index
//           returns its own word (a one-word rotation would fail 15 of 16).
//   PHASE4  a refill for the next tile (different base) drops o_ready, replaces
//           the contents, and the new words read back correctly.
// =============================================================================
module tb_kv8_i_scale_fifo;
    localparam integer WORDS = 16;

    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg         i_fill_start = 0;
    reg  [11:0] i_fill_base  = 12'd0;
    wire        o_fill_en;
    wire [11:0] o_fill_word;
    reg         i_fill_valid = 0;
    reg  [127:0] i_fill_data = 0;
    reg  [7:0]   i_fill_eb   = 0;      // [M16-SCALE] block exponent of the word
    wire        o_ready;
    reg  [3:0]  i_rd_index = 4'd0;
    wire [135:0] o_scale;              // [M16-SCALE] {eb, 8 x m16}

    kv8_i_scale_fifo #(.WORDS(WORDS)) dut (
        .clk(clk), .rst_n(rst_n),
        .i_fill_start(i_fill_start), .i_fill_base(i_fill_base),
        .o_fill_en(o_fill_en), .o_fill_word(o_fill_word),
        .i_fill_valid(i_fill_valid), .i_fill_data(i_fill_data), .i_fill_eb(i_fill_eb),
        .o_ready(o_ready),
        .i_rd_index(i_rd_index), .o_scale(o_scale)
    );

    // ---- 128-bit payload of word w: 8 x 16-bit, lane b = 16'h0100*w + b*0x23,
    //      so every word and every lane is distinct and a permutation cannot pass ----
    function [127:0] payload(input [11:0] w);
        integer b;
        begin
            for (b=0;b<8;b=b+1) payload[b*16 +: 16] = (w*16'h0101) + b*16'h0023;
        end
    endfunction
    // the block exponent travels with the word
    function [7:0] payload_eb(input [11:0] w);
        begin payload_eb = 8'h80 + w[7:0]; end
    endfunction

    // ---- model of the kv8_scale_ram read port: 3 cycles, in-order ----
    reg        en_p0 = 0, en_p1 = 0, en_p2 = 0;
    reg [11:0] wd_p0 = 0, wd_p1 = 0, wd_p2 = 0;
    always @(posedge clk) begin
        en_p0 <= o_fill_en;   wd_p0 <= o_fill_word;
        en_p1 <= en_p0;       wd_p1 <= wd_p0;
        en_p2 <= en_p1;       wd_p2 <= wd_p1;
        i_fill_valid <= en_p2;
        i_fill_data  <= payload(wd_p2);
        i_fill_eb    <= payload_eb(wd_p2);
    end

    // ---- address sequence audit ----
    integer n_issue = 0;
    integer fail_seq = 0;
    reg [11:0] base_q = 0;
    always @(posedge clk) begin
        if (o_fill_en) begin
            if (o_fill_word !== base_q + n_issue[11:0]) begin
                fail_seq = fail_seq + 1;
                $display("FAIL fill address %h != %h", o_fill_word, base_q + n_issue[11:0]);
            end
            n_issue = n_issue + 1;
        end
    end

    integer errors = 0;
    integer checks = 0;
    integer i, sl, sb, rw;
    integer distinct_bad;
    reg [135:0] got;

    task automatic start_tile(input [11:0] base);
        begin
            @(negedge clk);
            i_fill_base  <= base;
            base_q        = base;
            n_issue       = 0;
            i_fill_start <= 1'b1;
            @(negedge clk);
            i_fill_start <= 1'b0;
            // 16 reads + 3 cycles of latency + margin
            for (rw=0; rw<80 && !o_ready; rw=rw+1) @(posedge clk);
        end
    endtask

    initial begin
        errors = 0; checks = 0;
        repeat (5) @(posedge clk);
        rst_n = 1;
        repeat (3) @(posedge clk);

        // ------------------------------------------------------------- PHASE1
        $display("--- PHASE1: fill port issues base+0..15, o_ready only after 16 ---");
        checks = checks + 1;
        if (o_ready) begin errors = errors + 1; $display("FAIL o_ready high before any fill"); end
        start_tile(12'd40);
        checks = checks + 2;
        if (!o_ready) begin errors = errors + 1; $display("FAIL o_ready did not rise"); end
        if (n_issue !== 16) begin errors = errors + 1; $display("FAIL issued %0d words", n_issue); end
        if (fail_seq !== 0) errors = errors + fail_seq;
        $display("  fill: issued=%0d address_errors=%0d ready=%b", n_issue, fail_seq, o_ready);

        // ------------------------------------------------------------- PHASE2
        $display("--- PHASE2: {slice,sub} indexed read, 8 rows share one word ---");
        for (sl=0; sl<8; sl=sl+1) begin
            for (sb=0; sb<2; sb=sb+1) begin
                i_rd_index = {sl[2:0], sb[0]};
                #1;
                for (rw=0; rw<8; rw=rw+1) begin     // 8 query rows, same token word
                    checks = checks + 1;
                    got = o_scale;
                    if (got[127:0] !== payload(12'd40 + sl*2 + sb) ||
                        got[135:128] !== payload_eb(12'd40 + sl*2 + sb)) begin
                        errors = errors + 1;
                        if (errors < 12)
                            $display("FAIL read idx=%0d (slice %0d sub %0d) got %h/%02h want %h/%02h",
                                     sl*2+sb, sl, sb, got[127:0], got[135:128],
                                     payload(12'd40 + sl*2 + sb), payload_eb(12'd40 + sl*2 + sb));
                    end
                end
            end
        end
        $display("  read: 128 checks (8 slices x 2 subs x 8 rows), errors=%0d", errors);

        // ------------------------------------------------------------- PHASE3
        $display("--- PHASE3: mapping is not vacuous (16 distinct words) ---");
        distinct_bad = 0;
        for (i=0; i<16; i=i+1) begin
            for (rw=i+1; rw<16; rw=rw+1)
                if (payload(12'd40 + i) === payload(12'd40 + rw)) distinct_bad = distinct_bad + 1;
        end
        checks = checks + 1;
        if (distinct_bad != 0) begin
            errors = errors + 1;
            $display("FAIL payload model is not injective");
        end
        $display("  distinct-payload collisions=%0d", distinct_bad);

        // ------------------------------------------------------------- PHASE4
        $display("--- PHASE4: refill for the next tile replaces the contents ---");
        start_tile(12'd200);
        checks = checks + 1;
        if (!o_ready) begin errors = errors + 1; $display("FAIL o_ready after refill"); end
        for (i=0; i<16; i=i+1) begin
            i_rd_index = i[3:0];
            #1;
            checks = checks + 1;
            if (o_scale[127:0] !== payload(12'd200 + i) ||
                o_scale[135:128] !== payload_eb(12'd200 + i)) begin
                errors = errors + 1;
                if (errors < 20)
                    $display("FAIL refill idx=%0d got %h/%02h want %h/%02h", i,
                             o_scale[127:0], o_scale[135:128],
                             payload(12'd200 + i), payload_eb(12'd200 + i));
            end
        end
        $display("  refill: 16 words, errors so far=%0d", errors);

        repeat (5) @(posedge clk);
        if (errors == 0)
            $display("STAGE3-WIRING PASS: checks=%0d", checks);
        else
            $display("STAGE3-WIRING FAIL: errors=%0d", errors);
        $finish;
    end

    // watchdog
    initial begin
        #200000;
        $display("STAGE3-WIRING TIMEOUT ready=%b issue=%0d", o_ready, n_issue);
        $finish;
    end
endmodule
`default_nettype wire
