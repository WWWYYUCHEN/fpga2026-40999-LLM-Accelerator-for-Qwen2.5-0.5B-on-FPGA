// -----------------------------------------------------------------------------
// [V-MARGIN8] kv8_v_quant -- the V path's quantiser, now on the SAME mantissa rule
// as K (16-bit m, round(2^24/m), MARGIN configurable), with an 8-bit CODE.
//
// Why this changed (owner decision, 2026-11-02, after the K16 landing):
//   V's V rel_rms measured 1.1770 % while K's reconstruction had fallen to
//   0.0073 %, so V became the only error term left (corr 0.998010 vs the
//   0.9995-0.9997 target).  The cause is NOT V's 8-bit code (whose floor is
//   ~0.23 % rms) but the CEIL STEP OF THE SCALE: at MARGIN=2 the mantissa is
//   m = ceil(M8*2^2/127) in [5,9], so 1/m alone is 11..20 %.  At MARGIN=8 the
//   mantissa lands in [259,515] and the same ceil step is 0.19..0.39 %, i.e.
//   ~40x finer, while the published scale m*2^(eb-134) is UNCHANGED (m grows by
//   64 and eb drops by 6).
//
// What had to widen with it:
//   * m 8 -> 16 bits (259..515 does not fit in 8).  MTOP becomes 65535 so the
//     saturation target matches the field.
//   * the reciprocal R = round(2^RSHIFT/m) must move 16 -> 24 bits.  At m ~ 300,
//     round(2^16/m) is only 218 with a 0.23 % quantisation error of its own,
//     which would have re-introduced the very error we are removing.  This is
//     the same [M16-RECIP] argument the K quantiser carries.
//   * the code stays 8-bit and still saturates at 127, so |q| <= 127 by the
//     ceil'd-m construction and a later token that outgrows its lock saturates
//     the code (o_sat_lane) exactly as before.
//
// [RECIP-SEQ] 2026-11-03: the combinational (2^16 + m/2)/m and the 16-bit /127 are
//   replaced by ONE shared iterative engine (kv8_div_seq), exactly as in
//   kv8_k_quant.  Measured reason: this module's divider was the second worst path
//   in the accelerator (93 levels / 27.532 ns / WNS -22.535 ns at 5.000 ns, OOC)
//   and its /127 prefix another -9.768 ns.  The handshake semantics are unchanged
//   (o_done / o_lock_done / o_restore_done still mean "o_eb/o_m are valid now"),
//   only the latency moved from 1 cycle to 17..41 cycles.  Bit-identical
//   arithmetic: tb_kv8_quant_equiv.sv proves it over the full operand range.
//
// d-field headroom (the owner's point 2).  Two bounds, both enforced here:
//   * the block's first token: d = MARGIN = 8, so N = M8*2^8 <= 65 280 (24 bits
//     is ample).
//   * a later token of the same lock: d = E_max - eb_lock can reach 15 (K's own
//     measured bound).  The shifter now CLAMPS at d > 15 into 24'hFF_FFFF
//     instead of wrapping - the same guard kv8_k_quant has, and the same one
//     whose absence let a 9-bit quotient escape a clip test ([FIX-LOCKWIDTH]).
//     A clamped lane saturates the code, so o_sat_lane / o_clip_codes count it.
//   Above d = 15 the lock itself saturates (m_lock_sat at MTOP) and the chain
//   flags the append, so there is no silent path.
//
// Two streaming passes over the 8 beats of one head:
//   pass 0 (i_pass=0): magnitude accumulation, o_done pulses on i_last
//   pass 1 (i_pass=1): quantised bytes out, one beat per cycle
// -----------------------------------------------------------------------------
`timescale 1ns/1ps
`default_nettype none
module kv8_v_quant #(
    parameter integer MARGIN = 2
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         i_clear,       // new token
    input  wire         i_pass,        // 0 = magnitude, 1 = quantise
    input  wire         i_valid,
    input  wire         i_last,        // 8th beat of this head
    input  wire [127:0] i_k,           // 8 x BF16
    // ---- block-exponent lock (integration extension, default off) -----------
    input  wire         i_eb_lock_valid,
    input  wire [7:0]   i_eb_lock,
    output wire         o_lock_done,
    // Direct register restore: the two passes of a token are separate streaming
    // passes, and the second head's magnitude pass overwrites eb/m/recip.
    input  wire         i_restore_valid,
    input  wire [7:0]   i_restore_eb,
    input  wire [7:0]   i_restore_m,
    output wire         o_restore_done,
    output wire         o_done,        // pulse: eb/m valid for this head
    output wire [7:0]   o_eb,
    output wire [7:0]   o_m,
    output wire [7:0]   o_maxmag_exp,
    output wire         o_q_valid,
    output wire [63:0]  o_q,           // 8 x INT8 (single plane, V format unchanged)
    // [V-PERTOK-M] per-lane saturation flag of the quantise pass (display-only:
    // it feeds the chain's saturated-code counter, never the data).
    output wire [7:0]   o_sat_lane
);
    // ------------------------------------------------ magnitude pass
    wire [14:0] mag [0:7];
    reg  [14:0] l1 [0:3];
    reg  [14:0] l2 [0:1];
    reg  [14:0] max8;
    integer q;
    genvar i;
    generate
        for (i=0;i<8;i=i+1) begin : gen_mag
            assign mag[i] = i_k[i*16 +: 15];     // sign dropped: unsigned compares
        end
    endgenerate
    always @(*) begin
        for (q=0;q<4;q=q+1) l1[q] = (mag[2*q] >= mag[2*q+1]) ? mag[2*q] : mag[2*q+1];
        l2[0] = (l1[0] >= l1[1]) ? l1[0] : l1[1];
        l2[1] = (l1[2] >= l1[3]) ? l1[2] : l1[3];
        max8  = (l2[0] >= l2[1]) ? l2[0] : l2[1];
    end

    reg [14:0] run_max;
    reg first_magnitude;
    reg [7:0]  eb_r;
    reg [7:0]  m_r;
    reg [16:0] recip_r;
    reg        done_r;
    reg        lockdone_r;
    reg        restoredone_r;
    assign o_restore_done = restoredone_r;
    assign o_eb = (eb_r > 8'd10) ? (eb_r - 8'd10) : 8'd0;   // table convention (s_k/8)
    assign o_m  = m_r;
    assign o_maxmag_exp = run_max[14:7];
    assign o_done = done_r;
    assign o_lock_done = lockdone_r;

    // The magnitude of the last beat must join the maximum before eb/m are
    // computed, so the effective maximum is max(run_max, max8) on i_last.
    wire [14:0] eff_max = (i_valid && !i_pass && i_last && (first_magnitude || max8 > run_max)) ? max8 : run_max;
    // m = ceil(M8 * 2^MARGIN / 127), M8 = {1'b1, eff_max[6:0]}
    // [V-MARGIN8] 16-bit quotient: at MARGIN=8 the value is 259..515, which an
    // 8-bit field would have silently truncated to 3..3.
    wire [15:0] num = {1'b1,eff_max[6:0]} << MARGIN;
    wire [7:0]  eb_calc = (eff_max[14:7] > (MARGIN+1)) ? (eff_max[14:7]-MARGIN) : 8'd1;
    // The first accepted magnitude beat initializes each complete head.
    always @(posedge clk) begin
        if (!rst_n) begin
            run_max <= 15'd0;
            first_magnitude <= 1'b1;
        end else if (i_valid && !i_pass) begin
            first_magnitude <= i_last;
            if (first_magnitude || max8 > run_max) run_max <= max8;
        end
    end

    // =========================================================== [RECIP-SEQ]
    // One iterative divider serves all three request sources (restore > lock >
    // magnitude, the priority the shipped if/else had).  R = (2^16 + m/2)/m is
    // 17 bits at most (m = 1 gives exactly 2^16, which is why the engine is wider
    // than 16 bits), and the lock numerator lock_num is 23 bits, so NUMW = 24.
    localparam integer DIVW = 16;
    localparam integer NUMW = (16 + 1 >= 24) ? (16 + 1) : 24;   // V: 24
    localparam [NUMW-1:0] RONE = {{(NUMW-1){1'b0}},1'b1} << 16; // 2^16, 17 bits
    wire [NUMW-1:0] num_d127 = {{(NUMW-16){1'b0}}, num};

    // ---- block-exponent lock: same rule, but relative to i_eb_lock ----------
    // [FIX-LOCKMARGIN] eb_lock is ALREADY the margined exponent (eb = Eb - MARGIN),
    // so MARGIN belongs in eb_lock only.
    wire signed [9:0] lock_shift = $signed({2'b0,eff_max[14:7]}) - $signed({2'b0,i_eb_lock});
    // [FIX-LOCKWIDTH] 23-bit numerator with an explicit saturating shifter; m
    // SATURATES at the field top (MTOP) instead of wrapping.
    wire signed [9:0] lock_dn = -lock_shift;
    reg [22:0] lock_num;
    always @(*) begin
        if      (lock_shift >  10'sd15) lock_num = 23'h7F_FFFF;   // saturate -> m clamps
        else if (lock_shift < -10'sd15) lock_num = 23'd0;          // underflow -> m = 0, flagged
        else if (lock_shift < 0)        lock_num = {15'd0,1'b1,eff_max[6:0]} >> lock_dn[3:0];
        else if (lock_shift > 0)        lock_num = {15'd0,1'b1,eff_max[6:0]} << lock_shift[3:0];
        else                            lock_num = {15'd0,1'b1,eff_max[6:0]};
    end
    wire [7:0]  eb_lock_c = (i_eb_lock == 8'd0) ? 8'd1 : i_eb_lock;

    // ---- request sequencer + the single shared divider ----------------------
    localparam [1:0] K_REST = 2'd0, K_LOCK = 2'd1, K_MAG = 2'd2;
    localparam [1:0] S_IDLE = 2'd0, S_D127 = 2'd1, S_DREC = 2'd2;

    reg  [1:0]      st;
    reg  [1:0]      kind;
    reg  [7:0]      eb_hold;
    reg  [7:0]      m_hold;
    reg             div_start;
    reg  [NUMW-1:0] div_num;
    reg  [DIVW-1:0] div_den;
    wire            div_done;
    wire [NUMW-1:0] div_quo;
    wire [DIVW-1:0] div_rem;

    kv8_div_seq #(.NUMW(NUMW), .DIVW(DIVW)) div_Inst (
        .clk(clk), .rst_n(rst_n),
        .i_start(div_start), .i_num(div_num), .i_den(div_den),
        .o_busy(), .o_done(div_done), .o_quo(div_quo), .o_rem(div_rem)
    );

    // m from the ÷127 quotient/remainder, exactly as the combinational version did
    wire [DIVW-1:0] rem_nz = {DIVW{1'b0}};
    // [V-MARGIN8] the 8-bit field keeps the shipped truncation semantics: the
    // quotient's low byte plus the ceil bit, carry out dropped.
    wire [7:0]  m_from_num  = div_quo[7:0] + ((div_rem != rem_nz) ? 8'd1 : 8'd0);
    // [V-MARGIN8] the saturation target follows the mantissa width
    wire        m_lock_sat  = (div_quo > 24'd255) ||
                              ((div_quo == 24'd255) && (div_rem != rem_nz));
    wire [7:0]  m_from_lock = m_lock_sat ? 8'd255
                                         : (div_quo[7:0] + ((div_rem != rem_nz) ? 8'd1 : 8'd0));
    wire [7:0]  m_next      = (kind == K_LOCK) ? m_from_lock : m_from_num;

    always @(posedge clk) begin
        if (!rst_n) begin
            eb_r<=8'd0; m_r<=8'd0; recip_r<=17'd0; done_r<=1'b0; lockdone_r<=1'b0;
            restoredone_r<=1'b0; st<=S_IDLE; kind<=K_MAG; eb_hold<=8'd0; m_hold<=8'd0;
            div_start<=1'b0; div_num<={NUMW{1'b0}}; div_den<=16'd0;
        end else begin
            done_r<=1'b0;
            lockdone_r<=1'b0;
            restoredone_r<=1'b0;
            div_start<=1'b0;
            case (st)
            // ---- request arbitration: same priority the shipped if/else had -----
            S_IDLE: begin
                if (i_restore_valid) begin
                    kind    <= K_REST;
                    eb_hold <= i_restore_eb;
                    m_hold  <= i_restore_m;
                    if (i_restore_m == 8'd0) begin
                        // RONE guard, inherited verbatim: a zero m never divides
                        eb_r <= i_restore_eb;
                        m_r  <= 8'd0;
                        recip_r <= 17'd65536;
                        restoredone_r <= 1'b1;
                    end else begin
                        div_start <= 1'b1;
                        div_num   <= RONE + {1'b0, i_restore_m[7:1]};
                        div_den   <= {8'd0, i_restore_m};
                        st        <= S_DREC;
                    end
                end else if (i_eb_lock_valid) begin
                    kind      <= K_LOCK;
                    eb_hold   <= eb_lock_c;
                    div_start <= 1'b1;
                    div_num   <= {{(NUMW-23){1'b0}}, lock_num};
                    div_den   <= 16'd127;
                    st        <= S_D127;
                end else if (i_valid && !i_pass && i_last) begin
                    kind      <= K_MAG;
                    eb_hold   <= eb_calc;
                    div_start <= 1'b1;
                    div_num   <= num_d127;
                    div_den   <= 16'd127;
                    st        <= S_D127;
                end
            end
            // ---- phase 1: m = ceil(numerator/127) ------------------------------
            S_D127: if (div_done) begin
                if (m_next == 8'd0) begin
                    // RONE guard, inherited verbatim: m == 0 publishes R = 2^16
                    eb_r    <= eb_hold;
                    m_r     <= 8'd0;
                    recip_r <= 17'd65536;
                    if (kind == K_LOCK) lockdone_r <= 1'b1;
                    else                done_r     <= 1'b1;
                    st <= S_IDLE;
                end else begin
                    m_hold    <= m_next;
                    div_start <= 1'b1;
                    div_num   <= RONE + {1'b0, m_next[7:1]};
                    div_den   <= {8'd0, m_next};
                    st        <= S_DREC;
                end
            end
            // ---- phase 2: R = (2^16 + m/2)/m, then publish ---------------------
            S_DREC: if (div_done) begin
                eb_r    <= eb_hold;
                m_r     <= m_hold;
                recip_r <= div_quo[16:0];   // <= 2^16, 17 bits (m = 1 corner)
                if      (kind == K_REST) restoredone_r <= 1'b1;
                else if (kind == K_LOCK) lockdone_r    <= 1'b1;
                else                     done_r        <= 1'b1;
                st <= S_IDLE;
            end
            endcase
        end
    end

    // ------------------------------------------------ quantise pass
    wire [7:0] q8_lane [0:7];
    generate
        for (i=0;i<8;i=i+1) begin : gen_q
            wire [15:0] kk = i_k[i*16 +: 16];
            wire [7:0]  mm = {1'b1,kk[6:0]};
            wire signed [9:0] d = $signed({2'b0,kk[14:7]}) - $signed({2'b0,eb_r});
            wire [23:0] nn = (kk[14:7]==8'd0) ? 24'd0 :
                             (!d[9]) ? ({{16{1'b0}},mm} << d[7:0]) :
                                       ({{16{1'b0}},mm} >> (-d[7:0]));
            wire [40:0] pr = nn * {24'd0,recip_r} + 41'd32768;
            wire [24:0] qv = pr >> 16;
            assign q8_lane[i] = (qv > 25'd127) ? 8'd127 : qv[7:0];
            // [V-PERTOK-M] display-only saturation flag, aligned with the push
            assign o_sat_lane[i] = i_valid && i_pass && (qv > 25'd127);
        end
    endgenerate

    reg [63:0] q_r;
    reg        qv_r;
    assign o_q = q_r;
    assign o_q_valid = qv_r;
    always @(posedge clk) begin
        if (!rst_n) begin
            q_r<=64'd0; qv_r<=1'b0;
        end else begin
            qv_r <= i_valid && i_pass;
            if (i_valid && i_pass)
                q_r <= { i_k[127] ? (~q8_lane[7]+8'd1) : q8_lane[7],
                         i_k[111] ? (~q8_lane[6]+8'd1) : q8_lane[6],
                         i_k[ 95] ? (~q8_lane[5]+8'd1) : q8_lane[5],
                         i_k[ 79] ? (~q8_lane[4]+8'd1) : q8_lane[4],
                         i_k[ 63] ? (~q8_lane[3]+8'd1) : q8_lane[3],
                         i_k[ 47] ? (~q8_lane[2]+8'd1) : q8_lane[2],
                         i_k[ 31] ? (~q8_lane[1]+8'd1) : q8_lane[1],
                         i_k[ 15] ? (~q8_lane[0]+8'd1) : q8_lane[0] };
        end
    end
endmodule
`default_nettype wire
