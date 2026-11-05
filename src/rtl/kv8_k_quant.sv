`timescale 1ns/1ps
`default_nettype none
// -----------------------------------------------------------------------------
// kv8_k_quant -- per-token K quantiser (the write side of the INT8 KV cache).
//
// Answers "who computes max|K| over the 64 dims of each head": this block runs
// inside the layer's K-production step, between the K projection result (proj
// buffer, 64 dims per KV head, 8 x BF16 per beat) and the DDR / scale-RAM write.
//
// Quantisation rule (symmetric, shared exponent per (head, 256-token block))
//   1) maxmag = max over the 64 dims of |k|                    (magnitude pass)
//   2) Eb     = exponent(maxmag), M8 = 1.significand(maxmag)
//   3) eb     = Eb - MARGIN     (MARGIN=8 in the tree: growth headroom)
//   4) m      = ceil(M8 * 2^MARGIN / 127)  in 259..515 for MARGIN=8, top 65535
//   5) R      = round(2^24 / m)             (no combinational divider any more)
//   6) per dim: d = Ed - eb, N = M8d * 2^d, q = (N*R + 2^23)>>24, sat 127, sign
//   s' = m * 2^(eb-127) >= max|k|/127 by construction, so |q| <= 127 and no dim
//   clips; a later token of the same block that exceeds 65535*2^eb must raise
//   clip (i_app_clip of kv8_scale_ram).
//
// [M16-RECIP] Why 2^24 and not 2^16 (this is a precision requirement, not taste):
//   the shipped 8-bit format had m in [5,255], so R = round(2^16/m) had 257..13107
//   integer steps and its quantisation of 1/m was <= 0.06%.  With a 16-bit m the
//   reachable m at MARGIN=8 is [259, 16449], and round(2^16/m) collapses to 3..253
//   -- up to 0.8% error on the CODE itself, which would swamp the whole m16 gain.
//   round(2^24/m) keeps the relative error at 0.003% for the reachable m (worst
//   case 0.05% at the saturation corner) and stays EXACTLY consistent with the
//   *published* ceil'd m, so the reconstruction error is the code rounding alone.
//   The product stays 24x25 bits and the shift stays a constant 24 (no variable
//   shifter in the quantise path).
//
// [RECIP-SEQ] 2026-11-03: steps 4 and 5 are now executed by ONE small iterative
//   engine (kv8_div_seq) instead of two combinational dividers.  Measured reason:
//   the combinational 25/16 divider plus the 32-bit /127 were the worst paths in
//   the whole accelerator (142 levels / 34.874 ns / WNS -29.877 ns at 5.000 ns,
//   OOC, xck26).  The handshake semantics are unchanged -- o_done / o_lock_done /
//   o_restore_done still say "o_eb/o_m are valid now" -- only the latency moved
//   from 1 cycle to 24..40 cycles, which is invisible next to the ~13 M cycles a
//   token costs.  The arithmetic is bit-identical (tb_kv8_quant_equiv.sv proves
//   it over the full operand range).
//
// Two streaming passes over the 8 beats of one head:
//   pass 0 (i_pass=0): magnitude accumulation, o_done pulses on i_last
//   pass 1 (i_pass=1): quantised bytes out, one beat per cycle
// -----------------------------------------------------------------------------
module kv8_k_quant #(
    parameter integer MARGIN = 8,       // tree value; tb_kv8_dot overrides with 2
    // [M16-CLIP] top of the mantissa field.  The DEFAULT is 255, i.e. the original
    // 8-bit locked-mantissa rule, so any instance that does not opt in (the V chain)
    // behaves bit-for-bit as shipped.  The K chain passes 65535 for the m16 format.
    parameter integer MTOP   = 255,
    // [M16-RECIP] reciprocal base 2^RSHIFT.  K needs 24 (see the header); the DEFAULT
    // is 16, which is the shipped arithmetic and is kept for the V path: V's m never
    // exceeds 255, where round(2^16/m) is already exact enough (worst 0.19% at the
    // saturation corner, 0.007% at the reachable 5..9).
    parameter integer RSHIFT = 16,
    // [K16-CODE] extra low bits of the published code.
    //   0 (default) = the shipped 8-bit code, q = round(N*R/2^RSHIFT), sat 127.
    //   8           = the frozen K16 code: the SAME (m, eb) rule and the SAME R,
    //                 but the product keeps 8 more fractional bits:
    //                     q16 = round(N*R / 2^(RSHIFT-8)) = round(256 * N/m)
    //                 so |q16| <= 256*127 = 32512 by the ceil'd-m construction and
    //                 the code carries 15 significant bits instead of 7.
    //                 The dequantisation scale is therefore 256x smaller
    //                 (s16 = s8/256), which the K chain folds into the table
    //                 exponent (eb_table - 8) so that C_K stays 268 as frozen.
    //                 q16 is published as two signed 8-bit planes:
    //                     hi = round(q16/256),  lo = q16 - 256*hi
    //                 so that  q16 = 256*hi + lo  EXACTLY, |hi|,|lo| <= 127, and
    //                     k = s16*q16 = (s8/256)*(256*hi+lo) = s8*(hi + lo/256),
    //                 i.e. exactly the frozen `k = s_k*(k_hi + k_lo/256)`.
    parameter integer CODE_EXTRA = 0
) (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         i_clear,       // new token
    input  wire         i_pass,        // 0 = magnitude, 1 = quantise
    input  wire         i_valid,
    input  wire         i_last,        // 8th beat of this head
    input  wire [127:0] i_k,           // 8 x BF16
    // ---- block-exponent lock (integration extension, default off) -----------
    // kv8_scale_ram stores ONE exponent per (head, 256-token block), fixed by the
    // first token of the block.  A later token of the same block must therefore be
    // quantised with that stored exponent, not with its own.  Asserting
    // i_eb_lock_valid for one cycle re-derives m/recip for i_eb_lock and keeps the
    // pass-0 magnitude result (run_max) as the clip witness.  With the port tied
    // low the module is bit-identical to the verified original (see the report's
    // regression note: tb_kv8_dot re-run with lock off).
    input  wire         i_eb_lock_valid,
    input  wire [7:0]   i_eb_lock,
    output wire         o_lock_done,
    // Direct register restore: the two passes of a token are separate streaming
    // passes, and the second head's magnitude pass overwrites eb/m/recip.  Before
    // the first head's quantise pass the caller therefore restores that head's own
    // pair (this is the same pair pass 0 produced, or the locked one).
    input  wire         i_restore_valid,
    input  wire [7:0]   i_restore_eb,
    input  wire [15:0]  i_restore_m,
    output wire         o_restore_done,
    output wire         o_done,        // pulse: eb/m valid for this head
    output wire [7:0]   o_eb,
    output wire [15:0]  o_m,           // 16-bit since the m16 format (was 8)
    output wire [7:0]   o_maxmag_exp,
    output wire         o_q_valid,
    output wire [127:0] o_q            // 8 x {hi,lo} = one 128-bit KV8 beat (K16)
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
    reg [15:0] m_r;
    reg [24:0] recip_r;
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
    // m = ceil(M8 * 2^MARGIN / 127), M8 = {1'b1, eff_max[6:0]}  (MARGIN=8 -> 259..515)
    wire [31:0] num = {24'd0,1'b1,eff_max[6:0]} << MARGIN;
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
    // Round(2^RSHIFT/m) and ceil(num/127) are no longer combinational: they run on
    // ONE iterative divider (kv8_div_seq) shared by the three request sources.
    // Why one engine and not three:
    //   * the three request sources are mutually exclusive at the register level --
    //     recip_r is loaded from exactly one of them per cycle (restore > lock >
    //     magnitude, the priority the shipped code had) -- so one datapath serves
    //     all three;
    //   * the shipped code carried TWO combinational dividers (recip_calc and
    //     recip_lock were separate combinational functions of different inputs;
    //     the clocked i_restore_m copy could share one of them), so one engine is
    //     cheaper than what it replaces, not just later.
    // The numeric result is unchanged: R = (RONE + m/2)/m and lquo/lrem are the
    // same integer division, only spread over NUMW cycles.  Proven exhaustively by
    // tb_kv8_quant_equiv.sv (every m, every num, every lock shift, every restore m).
    localparam integer DIVW = 16;                                       // divisor width
    localparam integer NUMW = (RSHIFT + 1 >= 24) ? (RSHIFT + 1) : 24;   // K: 25
    localparam [NUMW-1:0] RONE = {{(NUMW-1){1'b0}},1'b1} << RSHIFT;
    // num <= 255<<8 = 65 280, so it always fits NUMW
    wire [NUMW-1:0] num_d127 = num[NUMW-1:0];

    // ---- block-exponent lock: same rule, but relative to i_eb_lock ----------
    // m = ceil(M8 * 2^(E_max - eb_lock + MARGIN) / 127); if a token's own
    // exponent is above the lock this saturates and the caller must flag a clip.
    // [FIX-LOCKMARGIN] eb_lock is ALREADY the margined exponent (eb = Eb - MARGIN),
    // so adding MARGIN again gave a 2^(2*MARGIN) scale for every token after the
    // block's first.  MARGIN belongs in eb_lock only.
    wire signed [9:0] lock_shift = $signed({2'b0,eff_max[14:7]}) - $signed({2'b0,i_eb_lock});
    // [FIX-LOCKWIDTH] 16-bit lock_num wrapped for shift >= 9 and the 9-bit
    // quotient wrapped m in [512,1023] to m-512, escaping the clip test.
    // Measured: 26.3% of appends published a wrong scale with no flag.  Widened
    // to 23 bits (M8*2^15 <= 2^23) with an explicit saturating shifter; m now
    // SATURATES at the field top (65535 for K, was 255) instead of wrapping,
    // which is exactly what the locked path's q_m == 65535 clip test consumes.
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
    reg  [15:0]     m_hold;
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

    // the two ceilings, computed from the ÷127 quotient/remainder exactly as the
    // combinational version did (m_calc = quo[15:0] + (rem != 0), carry dropped)
    wire [DIVW-1:0] rem_nz = {DIVW{1'b0}};
    wire [15:0] m_from_num  = div_quo[15:0] + ((div_rem != rem_nz) ? 16'd1 : 16'd0);
    // [M16-CLIP] the saturation target follows the mantissa width: >65535, not >255.
    wire        m_lock_sat  = (div_quo > MTOP) ||
                              ((div_quo == MTOP) && (div_rem != rem_nz));
    wire [15:0] m_from_lock = m_lock_sat ? MTOP[15:0]
                                         : (div_quo[15:0] + ((div_rem != rem_nz) ? 16'd1 : 16'd0));
    wire [15:0] m_next      = (kind == K_LOCK) ? m_from_lock : m_from_num;

    always @(posedge clk) begin
        if (!rst_n) begin
            eb_r<=8'd0; m_r<=16'd0; recip_r<=25'd0; done_r<=1'b0; lockdone_r<=1'b0;
            restoredone_r<=1'b0; st<=S_IDLE; kind<=K_MAG; eb_hold<=8'd0; m_hold<=16'd0;
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
                    if (i_restore_m == 16'd0) begin
                        // RONE guard, inherited verbatim: a zero m never divides
                        eb_r <= i_restore_eb;
                        m_r  <= 16'd0;
                        recip_r <= RONE;
                        restoredone_r <= 1'b1;
                    end else begin
                        div_start <= 1'b1;
                        div_num   <= RONE + {1'b0, i_restore_m[15:1]};
                        div_den   <= i_restore_m;
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
                if (m_next == 16'd0) begin
                    // RONE guard, inherited verbatim: m == 0 publishes R = RONE
                    eb_r    <= eb_hold;
                    m_r     <= 16'd0;
                    recip_r <= RONE;
                    if (kind == K_LOCK) lockdone_r <= 1'b1;
                    else                done_r     <= 1'b1;
                    st <= S_IDLE;
                end else begin
                    m_hold    <= m_next;
                    div_start <= 1'b1;
                    div_num   <= RONE + {1'b0, m_next[15:1]};
                    div_den   <= m_next;
                    st        <= S_DREC;
                end
            end
            // ---- phase 2: R = (RONE + m/2)/m, then publish ---------------------
            S_DREC: if (div_done) begin
                eb_r    <= eb_hold;
                m_r     <= m_hold;
                recip_r <= div_quo;
                if      (kind == K_REST) restoredone_r <= 1'b1;
                else if (kind == K_LOCK) lockdone_r    <= 1'b1;
                else                     done_r        <= 1'b1;
                st <= S_IDLE;
            end
            endcase
        end
    end

    // ------------------------------------------------ quantise pass
    // [K16-CODE] QSHIFT = RSHIFT - CODE_EXTRA is the only thing that changes with
    // the code width: R, m, eb and the whole reciprocal machinery are untouched.
    localparam integer QSHIFT = RSHIFT - CODE_EXTRA;
    // Signed 8+8 split: |q16| <= 127*256 + 127 must hold, so the code saturates
    // one bit below the 16-bit field top.  Unreachable by construction (the ceil'd
    // m makes q = 256*N/m <= 256*127), it exists so a locked token that runs away
    // saturates the code instead of wrapping the hi plane.
    localparam integer LIMIT  = (CODE_EXTRA != 0) ? 32639 : 127;
    wire [7:0] q_hi_lane [0:7];
    wire [7:0] q_lo_lane [0:7];
    reg [24:0] quant_recip_stage;
    reg quant_valid_stage, product_valid_stage;
    always @(posedge clk) begin
        if (!rst_n) begin
            quant_valid_stage <= 1'b0;
            product_valid_stage <= 1'b0;
        end else begin
            quant_valid_stage <= i_valid && i_pass;
            product_valid_stage <= quant_valid_stage;
            if (i_valid && i_pass) quant_recip_stage <= recip_r;
        end
    end
    generate
        for (i=0;i<8;i=i+1) begin : gen_q
            wire [15:0] kk = i_k[i*16 +: 16];
            wire [7:0]  mm = {1'b1,kk[6:0]};
            wire signed [9:0] d = $signed({2'b0,kk[14:7]}) - $signed({2'b0,eb_r});
            // d <= 15 keeps N in 24 bits.  d >= 16 can only happen when the token
            // rose more than the m16 field can follow, i.e. m_lock_sat already
            // flagged the clip; saturate the code instead of wrapping it.
            wire [23:0] nn = (kk[14:7]==8'd0) ? 24'd0 :
                             (!d[9]) ? ((d > 10'sd15) ? 24'hFF_FFFF
                                                      : ({{16{1'b0}},mm} << d[7:0])) :
                                       ({{16{1'b0}},mm} >> (-d[7:0]));
            // q = round(N/m) with R = round(2^RSHIFT/m): 24b x 25b -> 49b.
            // [K16-CODE] the rounding constant and the shift move together, so the
            // 8-bit case is bit-for-bit the shipped arithmetic.
            reg [23:0] nn_stage;
            reg sign_stage, sign_product;
            reg [49:0] pr;
            always @(posedge clk) begin
                if (i_valid && i_pass) begin
                    nn_stage <= nn;
                    sign_stage <= kk[15];
                end
                if (quant_valid_stage) begin
                    pr <= nn_stage * {26'd0,quant_recip_stage} + (50'd1 << (QSHIFT-1));
                    sign_product <= sign_stage;
                end
            end
            wire [33:0] qv = pr >> QSHIFT;
            // 16-bit compare against the frozen clip target, then the signed split.
            wire [15:0] qmag = (qv[33:16] != 18'd0 || qv[15:0] > LIMIT[15:0])
                             ? LIMIT[15:0] : qv[15:0];
            // hi = round(qmag/256), lo = qmag - 256*hi  (|lo| <= 128, two's complement)
            wire [16:0] qround = {1'b0,qmag} + 17'd128;
            wire [7:0]  hi_u = (CODE_EXTRA != 0) ? qround[16:8] : 8'd0;
            wire [7:0]  lo_u = qmag[7:0];
            assign q_hi_lane[i] = sign_product ? (~hi_u + 8'd1) : hi_u;
            assign q_lo_lane[i] = sign_product ? (~lo_u + 8'd1) : lo_u;
        end
    endgenerate

    reg [127:0] q_r;
    reg         qv_r;
    assign o_q = q_r;
    assign o_q_valid = qv_r;
    always @(posedge clk) begin
        if (!rst_n) begin
            q_r<=128'd0; qv_r<=1'b0;
        end else begin
            qv_r <= product_valid_stage;
            if (product_valid_stage)
                q_r <= { q_hi_lane[7], q_lo_lane[7], q_hi_lane[6], q_lo_lane[6],
                         q_hi_lane[5], q_lo_lane[5], q_hi_lane[4], q_lo_lane[4],
                         q_hi_lane[3], q_lo_lane[3], q_hi_lane[2], q_lo_lane[2],
                         q_hi_lane[1], q_lo_lane[1], q_hi_lane[0], q_lo_lane[0] };
        end
    end
endmodule
`default_nettype wire
