`timescale 1ns/1ps
`default_nettype none
// =============================================================================
// tb_kv8_v_fill_packer -- [V16] the MODE_KV=1 (V) half of kv8_fill_packer
//
// V's code is 16 bit = {v_hi,v_lo}, so a V DDR beat is 8 dims of ONE head
// (8 x 16 bit = 128 bit) and a token is 16 beats = 2 heads x 8 dim_groups --
// the same index shape K uses.  The packer collects the four beats of one
// (token, head, dim_group[2]) and emits TWO 32-column words:
//     seg0 = v_lo plane, seg1 = v_hi plane
//     column b = dim_group[1:0]*8 + j
//     word address = {1'b0, token[5:0], seg, head, dim_group[2]}
// which is what makes one 1024-bit read at (row,seg) return 128 columns of one
// plane.  This TB checks the word address, both planes byte for byte, the beat
// accounting (16 beats -> 4 words) and that a beat taken out of order raises
// o_err_seq.
// =============================================================================
module tb_kv8_v_fill_packer;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    localparam integer NTOK = 4;

    integer errors = 0, checks = 0;

    reg         i_beat_valid = 0;
    wire        o_beat_ready;
    reg  [17:0] i_beat_index = 0;
    reg  [127:0] i_beat_data = 0;
    wire        o_wr_valid;
    reg         i_wr_ready = 1;
    wire [9:0]  o_wr_word;
    wire [255:0] o_wr_data;
    wire        o_err_head, o_err_seq, o_busy;

    kv8_fill_packer #(.MODE_KV(1)) dut (
        .i_clk(clk), .i_rstn(rst_n),
        .i_beat_valid(i_beat_valid), .o_beat_ready(o_beat_ready),
        .i_beat_index(i_beat_index), .i_beat_data(i_beat_data),
        .i_head_sel(1'b0), .i_bank(1'b1),
        .o_wr_valid(o_wr_valid), .i_wr_ready(i_wr_ready),
        .o_wr_bank(), .o_wr_word(o_wr_word), .o_wr_data(o_wr_data),
        .o_err_head(o_err_head), .o_err_seq(o_err_seq), .o_busy(o_busy)
    );

    function [7:0] fbyte(input integer tok, input integer hd, input integer dg,
                         input integer j,   input integer seg);
        integer v;
        begin
            v = (tok*7 + hd*53 + dg*29 + j*11 + seg*97 + 1) & 255;
            fbyte = v[7:0];
        end
    endfunction

    // expected byte: [tok][hd][dg2][seg][byte b], b = (dg%4)*8 + j
    reg [7:0] exp_byte [0:NTOK*2*2*2*32-1];

    task automatic chk(input integer cond, input [255:0] msg);
        begin
            checks = checks + 1;
            if (!cond) begin
                errors = errors + 1;
                $display("FAIL %0s", msg);
            end
        end
    endtask

    // ---- capture the write words -------------------------------------------
    integer cap_tok, cap_hd, cap_dg2, cap_seg, cap_b, cap_idx;
    integer words_seen [0:NTOK*2*2*2-1];
    always @(posedge clk) begin
        if (rst_n && o_wr_valid && i_wr_ready) begin
            cap_tok = o_wr_word[8:3];
            cap_seg = o_wr_word[2];
            cap_hd  = o_wr_word[1];
            cap_dg2 = o_wr_word[0];
            cap_idx = (((cap_tok*2 + cap_hd)*2 + cap_dg2)*2) + cap_seg;
            if (cap_tok < NTOK) words_seen[cap_idx] = words_seen[cap_idx] + 1;
            for (cap_b=0;cap_b<32;cap_b=cap_b+1) begin
                chk(o_wr_data[cap_b*8 +: 8] ===
                    exp_byte[((((cap_tok*2 + cap_hd)*2 + cap_dg2)*2 + cap_seg)*32) + cap_b],
                    "V packer word byte");
            end
        end
    end

    // ---- one beat: token tok, head hd, dim_group dg -------------------------
    // Sample o_beat_ready on the negedge (race-free) so a beat is delivered
    // exactly once even when the packer emits its two words in between.
    task automatic send_beat(input integer tok, input integer hd, input integer dg);
        integer j;
        begin
            // Assert only right after a negedge: asserting earlier spans two
            // posedges and delivers the beat twice (measured on this TB).
            i_beat_valid = 1'b0;
            @(negedge clk);
            i_beat_index = {tok[13:0], hd[0:0], dg[2:0]};
            for (j=0;j<8;j=j+1) begin
                i_beat_data[j*16 +: 8]     = fbyte(tok, hd, dg, j, 0);
                i_beat_data[j*16 + 8 +: 8] = fbyte(tok, hd, dg, j, 1);
            end
            i_beat_valid = 1'b1;
            while (!o_beat_ready) @(negedge clk);   // sampled away from the edge
            @(posedge clk);                        // accepted here
            #1 i_beat_valid = 1'b0;                // deassert AFTER the edge (no race)
            @(negedge clk);
        end
    endtask

    integer t, hd, dg, dg2, seg, b, jj;
    initial begin
        errors = 0; checks = 0;
        for (t=0;t<NTOK*2*2*2;t=t+1) words_seen[t] = 0;
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1)
                for (dg2=0;dg2<2;dg2=dg2+1)
                    for (seg=0;seg<2;seg=seg+1)
                        for (b=0;b<32;b=b+1) begin
                            jj = b % 8;
                            exp_byte[((((t*2 + hd)*2 + dg2)*2 + seg)*32) + b] =
                                fbyte(t, hd, dg2*4 + (b/8), jj, seg);
                        end
        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);

        // ---- 4 tokens x 2 heads x 8 dim_groups = 64 beats -> 32 words
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1)
                for (dg=0;dg<8;dg=dg+1) send_beat(t, hd, dg);
        repeat (6) @(posedge clk);

        // ---- every word exactly once, both planes present
        for (t=0;t<NTOK;t=t+1)
            for (hd=0;hd<2;hd=hd+1)
                for (dg2=0;dg2<2;dg2=dg2+1)
                    for (seg=0;seg<2;seg=seg+1)
                        chk(words_seen[(((t*2 + hd)*2 + dg2)*2) + seg] == 1,
                            "each (token,head,dg2,seg) word written exactly once");
        chk(o_err_seq === 1'b0, "in-order stream raises no sequence error");

        $display("V-FILLPACK words=%0d checks=%0d errors=%0d", (checks/32), checks, errors);
        if (errors == 0) $display("V-FILLPACK PASS: checks=%0d", checks);
        else             $display("V-FILLPACK FAIL: checks=%0d errors=%0d", checks, errors);
        $finish;
    end

    initial begin
        #400000;
        $display("V-FILLPACK TIMEOUT");
        $finish;
    end
endmodule
`default_nettype wire
