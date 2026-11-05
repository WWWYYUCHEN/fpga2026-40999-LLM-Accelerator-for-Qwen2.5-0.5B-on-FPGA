`timescale 1ns / 1ps
`default_nettype none

// ============================================================================
// qwen_gemm_array_i  ——  INT8 GEMM 阵列的集成版
//
// 端口/握手/输出宽度与 rtl/qwen_gemm_array.sv 保持兼容，差异只有：
//   * s_weight_load_data 的每个 16 bit 槽的低 8 bit 是 INT8 权重码
//     （高 8 bit 忽略；由 qwen25_weight_pack8 在 DDR 入口写入，
//       与今天 bf16(int8) 精确转换等价，见 REPORT §3）
//   * m_data 的低 384 bit = 16 列 × INT24 段和（高 128 bit 填 0）
//   * m_meta 是 168 bit = {block_exp[7:0], 原 160 bit 身份}
//
// 相对原型 newdma/work_int8/rtl/w8_gemm_array.sv 的四处修正（每处都有 TB 证据）：
//
//  I1 每 lane 错位（skew）。DSP48E2 的 PCIN->P 是**每级 1 拍**，所以第 j 级的乘积
//     在链尾晚 j 拍到达：P_out(t) = Σ_j M_j(t-8+j)。若不把 lane j 的激活和权重选择
//     都延迟 j 拍，链尾加起来的 8 个乘积来自 8 个**不同周期**的块 —— 原型只有在
//     操作数保持 8 拍不变时才正确（tb_w8_array2 正是每块等 60 拍，所以没抓到）。
//     错位后保持 1 块/拍吞吐；DSP 输入寄存器增加一拍延迟。
//
//  I2 块指数随身份流水。原型把组合的 emax 直接接到 w8_int24_to_fp32_v2，而 INT24 段和
//     要 9 拍后才到 —— 背靠背时用的是**当前周期**的指数（错 2 的幂）。集成版在 accept
//     拍把 emax 写进身份 FIFO，pending_identity 由 160 bit 变 168 bit，第二级从
//     m_meta[167:160] 取指数。
//
//  I3 权重 2 bank × 8 slot（与今天一致），INT8 存储（FF 由 32768 降到 16384）。
//
//  [ARRAY-256] COL 16 -> 32：256 PE = 32 N x 8 K。**算术一个比特都没动**：
//      每个 (row, slot) 拍的 8 条 lane、wmem 里的 INT8 码、PCIN 链、align_lane
//      的对齐组（8 lane）、INT24 裁剪、块指数、fp32 累加顺序与次数全部与
//      COL=16 时逐位相同。变的只有"一拍覆盖多少列"（16 -> 32），因此一个
//      bundle 的权重 slot 数从 8 变 4，计算拍数减半。
//      这不是 lane 加宽：把 8 维的两个平面塞进 16 lane 会让 emax = emax_lo+8，
//      低平面每条 lane 多右移 8 位（12 bit 对齐字只剩 4 位有效），实测 score
//      相对误差 3.19e-4 -> 3.55e-3、128 个 BF16 score 里 40 个变（见
//      arr_lane_probe.py）。所以 256 PE 走列方向。
//
//  [LANE-PARAM] 端口不再写死：s_activation = LANE*16 bit、s_weight_load_row =
//      $clog2(LANE) bit、LAT = LANE+1、装载末拍比较 = LANE-1。LANE=8 时逐位等价
//      （128 bit / 3 bit / 9 / 7），LANE=16 现在能过综合（历史上这里是
//      [Synth 8-524] part-select [142:135] out of range 的来源）。LANE=16 不是
//      本轮的落地形态，只是给面积对照表提供一个真能综合的读数。

//
//  I4 A 端口 27 bit 有符号量完整符号扩展（原型 {3'b000, xa} 只靠
//     DSP48E2 乘法器忽略 A[29:27] 才对；显式扩展不花钱且不依赖该细节）。
// ============================================================================
module qwen_gemm_array_i #(
    parameter integer DISTRIBUTED_COMPUTE_BANK = 0,
    parameter integer COL  = 16,
    parameter integer LANE = 8,
    parameter integer SLOT = 8,
    // DSP 输入寄存器加上 DSP 链和 lane 错位：LANE=8 时 LAT=10。
    parameter integer LAT  = LANE + 2
) (
    input  wire          clk,
    input  wire          rst_n,
    input  wire          clear,
    // 权重装载：一拍一个 K lane，128 个 (col,slot) 槽；每槽低 8 bit = INT8
    input  wire          s_weight_load_valid,
    output wire          s_weight_load_ready,
    input  wire [2047:0] s_weight_load_data,
    input  wire [((LANE<=2)?1:$clog2(LANE))-1:0] s_weight_load_row,
    input  wire          s_weight_load_bank,
    input  wire          s_compute_bank,
    input  wire [7:0]    s_compute_bank_rows,
    input  wire          s_compute_release,
    // 计算：一拍一个 K 块（token）
    input  wire          s_valid,
    output wire          s_ready,
    input  wire [LANE*16-1:0] s_activation,
    input  wire          s_weight_advance,
    input  wire [159:0]  s_meta,
    // 输出：COL x INT24 负载（高位补 0 到 COL x 32）+ 168 bit 身份（含块指数）
    output wire          m_valid,
    input  wire          m_ready,
    output wire [COL*32-1:0] m_data,
    output wire [167:0]  m_meta,
    output wire          busy,
    output wire          protocol_error
);
    localparam integer FIFO_DEPTH = 64;
    // 256 PE arrangement: the 128-slot INT8 weight store is read as
    //   (128/COL) weight slots x COL columns.
    // COL=16 -> 8 x 16 (the historical array), COL=32 -> 4 x 32 (256 PEs).
    // The slot field is truncated to the number of slots the GEMM issues, so a
    // stray slot index can never walk off the 128-entry store.
    localparam integer SLOT_N = 128 / COL;
    localparam integer SLOT_B = (SLOT_N <= 1) ? 1 : $clog2(SLOT_N);

    // ---------------------------------------------------------------- 权重存储
    // wmem[bank][slot*16+col][lane] = INT8 码
    logic [7:0] wmem [0:1][0:127][0:LANE-1];
    wire load_fire = s_weight_load_valid && s_weight_load_ready;
    integer n;
    always_ff @(posedge clk) begin
        if (load_fire) begin
            for (n = 0; n < 128; n = n + 1)
                wmem[s_weight_load_bank][n][s_weight_load_row] <= s_weight_load_data[n*16 +: 8];
        end
    end

    // -------------------------------------------------- 共享块指数与对齐移位
    wire [7:0] lane_exp [0:LANE-1];
    genvar gi;
    generate
        for (gi = 0; gi < LANE; gi = gi + 1) begin : gen_exp
            assign lane_exp[gi] = (s_activation[gi*16 + 7 +: 8] == 8'd0) ? 8'd0
                                                                        : s_activation[gi*16 + 7 +: 8];
        end
    endgenerate

    // 8 输入 8 bit max 树（全阵列共享，只跟激活有关）
    reg [7:0] emax_r;
    integer mi;
    always_comb begin
        emax_r = 8'd0;
        for (mi = 0; mi < LANE; mi = mi + 1)
            if (lane_exp[mi] > emax_r) emax_r = lane_exp[mi];
    end
    wire [7:0] emax = emax_r;

    function automatic [11:0] align_lane(input [15:0] a, input [7:0] emax_v);
        reg [7:0]  gap;
        reg [7:0]  e;
        reg [11:0] base;
        begin
            e    = a[14:7];
            gap  = (e == 8'd0) ? 8'd15 : ((emax_v >= e) ? (emax_v - e) : 8'd0);
            if (gap > 8'd15) gap = 8'd15;
            base = (e == 8'd0) ? 12'd0 : {2'b00, 1'b1, a[6:0], 4'b0000};
            align_lane = base >> gap[3:0];
        end
    endfunction

    // 对齐 + 带符号（13 bit 有符号）
    wire signed [12:0] xa [0:LANE-1];
    generate
        for (gi = 0; gi < LANE; gi = gi + 1) begin : gen_align
            wire [11:0] al = align_lane(s_activation[gi*16 +: 16], emax);
            assign xa[gi] = s_activation[gi*16 + 15] ? (-$signed({1'b0, al})) : $signed({1'b0, al});
        end
    endgenerate

    // ------------------------------------------------------- I1: lane 错位（skew）
    // xs[j] = xa[j] 延迟 j 拍；bs[j] = {bank,slot} 延迟 j 拍
    logic signed [12:0] xs [0:LANE-1];
    logic        [3:0]  bs [0:LANE-1];
    wire compute_bank_row = (DISTRIBUTED_COMPUTE_BANK != 0) ? s_compute_bank_rows[0] : s_compute_bank;
    logic [2:0] slot_idx [0:1];
    wire  accept = s_valid && s_ready;
    always_ff @(posedge clk) begin
        if (!rst_n ) begin
            slot_idx[0] <= 3'd0;
            slot_idx[1] <= 3'd0;
        end else begin
            // 与 qwen_gemm_pe 相同的槽推进语义：装载复位优先，token 组末尾（last_issue_row）前进一格。
            if (accept && s_weight_advance && !compute_bank_row) slot_idx[0] <= slot_idx[0] + 3'd1;
            if (accept && s_weight_advance &&  compute_bank_row) slot_idx[1] <= slot_idx[1] + 3'd1;
            if (load_fire && !s_weight_load_bank) slot_idx[0] <= 3'd0;
            if (load_fire &&  s_weight_load_bank) slot_idx[1] <= 3'd0;
        end
    end
    wire [3:0] bs_now = {compute_bank_row, slot_idx[compute_bank_row]};

    generate
        for (gi = 0; gi < LANE; gi = gi + 1) begin : gen_skew
            if (gi == 0) begin : gen_s0
                assign xs[0] = xa[0];
                assign bs[0] = bs_now;
            end else begin : gen_sn
                integer t;
                logic signed [12:0] xpipe [0:gi-1];
                logic        [3:0]  bpipe [0:gi-1];
                always_ff @(posedge clk) begin
                    xpipe[0] <= xa[gi];
                    bpipe[0] <= bs_now;
                    for (t = 1; t < gi; t = t + 1) begin
                        xpipe[t] <= xpipe[t-1];
                        bpipe[t] <= bpipe[t-1];
                    end
                end
                assign xs[gi] = xpipe[gi-1];
                assign bs[gi] = bpipe[gi-1];
            end
        end
    endgenerate

    // -------------------------------------------------------------- 身份 FIFO
    // 168 bit = {块指数, 160 bit 原身份}
    (* ram_style = "distributed" *) logic [167:0] pending_identity [0:63];
    logic [5:0] identity_write_pointer, identity_read_pointer;
    logic [6:0] pending_count;
    (* ram_style = "distributed" *) logic [COL*24+167:0] output_fifo [0:63];
    logic [5:0] read_pointer, write_pointer;
    logic [6:0] occupancy, credits;
    logic [1:0] reg_weights_loaded;
    logic reg_available;
    wire [COL*32-1:0] int24_pack;
    wire         out_valid;

    // ------------------------------------------------------------ 每列 DSP 链
    localparam [8:0] OPMODE_MAC = 9'b000010101;  // P = M + PCIN
    localparam [8:0] OPMODE_MUL = 9'b000000101;  // P = M

    wire signed [47:0] p [0:COL-1][0:LANE-1];
    wire [7:0] int24_flat [0:COL-1];
    genvar gc, gl;
    generate
        for (gc = 0; gc < COL; gc = gc + 1) begin : gen_col
            for (gl = 0; gl < LANE; gl = gl + 1) begin : gen_lane
                wire [7:0] wsel = wmem[bs[gl][3]][(bs[gl][SLOT_B-1:0] * COL) + gc][gl];
                logic signed [12:0] dsp_activation;
                logic [7:0] dsp_weight;
                always_ff @(posedge clk) begin
                    dsp_activation <= xs[gl];
                    dsp_weight <= wsel;
                end
                (* DONT_TOUCH = "yes" *) DSP48E2 #(
                    .ACASCREG(0), .ADREG(0), .ALUMODEREG(0), .AMULTSEL("A"), .AREG(0),
                    .BCASCREG(0), .BMULTSEL("B"), .BREG(0), .CARRYINREG(0),
                    .CARRYINSELREG(0), .CREG(0), .DREG(0), .INMODEREG(0),
                    .MREG(1), .OPMODEREG(0), .PREADDINSEL("A"), .PREG(1),
                    .USE_MULT("MULTIPLY"), .USE_SIMD("ONE48")
                ) mac_Inst (
                    .CLK(clk),
                    .A({{17{dsp_activation[12]}}, dsp_activation}),
                    .D(27'd0),
                    .B({{10{dsp_weight[7]}}, dsp_weight}),
                    .INMODE(5'd0),
                    .OPMODE((gl == 0) ? OPMODE_MUL : OPMODE_MAC),
                    .ALUMODE(4'd0),
                    .CARRYINSEL(3'd0),
                    .C(48'd0),
                    .PCIN((gl == 0) ? 48'd0 : p[gc][gl-1]),
                    .ACIN(30'd0), .BCIN(18'd0),
                    .CARRYIN(1'b0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
                    .CEA1(1'b0), .CEA2(1'b1), .CEAD(1'b1), .CEALUMODE(1'b1),
                    .CEB1(1'b0), .CEB2(1'b1), .CEC(1'b1), .CECARRYIN(1'b1),
                    .CECTRL(1'b1), .CED(1'b1), .CEINMODE(1'b1), .CEM(1'b1), .CEP(1'b1),
                    .RSTA(1'b0), .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTB(1'b0),
                    .RSTC(1'b0), .RSTCTRL(1'b0), .RSTD(1'b0), .RSTINMODE(1'b0),
                    .RSTM(1'b0), .RSTP(1'b0),
                    .P(p[gc][gl]),
                    .ACOUT(), .BCOUT(), .CARRYCASCOUT(), .CARRYOUT(), .MULTSIGNOUT(),
                    .OVERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(),
                    .PCOUT(), .UNDERFLOW(), .XOROUT()
                );
            end

            wire signed [47:0] ptot = p[gc][LANE-1];
            wire signed [23:0] sum_c =
                (ptot >  48'sd8388607) ?  24'sh7fffff :
                (ptot < -48'sd8388608) ?  24'sh800000 : ptot[23:0];
            assign int24_flat[gc] = sum_c;
            assign int24_pack[gc*24 +: 24] = sum_c;
        end
    endgenerate
    assign int24_pack[COL*32-1:COL*24] = {(COL*8){1'b0}};

    wire retire  = m_valid && m_ready;
    wire produce = out_valid && (pending_count != 0);

    assign s_weight_load_ready = reg_available &&
        (!s_valid || s_weight_load_bank != compute_bank_row);
    assign s_ready = reg_available && reg_weights_loaded[compute_bank_row] && (credits < FIFO_DEPTH) &&
        (!s_weight_load_valid || s_weight_load_bank != compute_bank_row);
    assign m_valid = (occupancy != 0);
    assign m_data = {{(COL*8){1'b0}}, output_fifo[read_pointer][COL*24-1:0]};
    assign m_meta = output_fifo[read_pointer][COL*24+167:COL*24];
    assign busy = (credits != 0);
    assign protocol_error = 1'b0;

    // 延迟对齐的 valid
    logic [LAT-1:0] vld;
    always_ff @(posedge clk) begin
        if (!rst_n) vld <= {LAT{1'b0}};
        else        vld <= {vld[LAT-2:0], accept};
    end
    assign out_valid = vld[LAT-1];

    always_ff @(posedge clk) begin
        if (!rst_n ) begin
            identity_write_pointer <= 0; identity_read_pointer <= 0; pending_count <= 0;
            read_pointer <= 0; write_pointer <= 0;
            occupancy <= 0; credits <= 0;
            reg_weights_loaded <= 0;
            reg_available <= 0;
        end else begin
            reg_available <= 1;
            if (accept) begin
                // I2：accept 拍的块指数随身份一起进 FIFO
                pending_identity[identity_write_pointer] <= {emax, s_meta};
                identity_write_pointer <= identity_write_pointer + 6'd1;
            end
            if (produce) identity_read_pointer <= identity_read_pointer + 6'd1;
            case ({accept, produce})
                2'b10: pending_count <= pending_count + 7'd1;
                2'b01: pending_count <= pending_count - 7'd1;
                default: begin end
            endcase
            if (load_fire && s_weight_load_row == 0) reg_weights_loaded[s_weight_load_bank] <= 1'b0;
            if (load_fire && s_weight_load_row == (LANE-1)) reg_weights_loaded[s_weight_load_bank] <= 1'b1;
            if (s_compute_release) reg_weights_loaded[compute_bank_row] <= 1'b0;
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
                // 552 bit = {168 bit 身份(含块指数), 384 bit INT24}
                output_fifo[write_pointer] <= {pending_identity[identity_read_pointer], int24_pack[COL*24-1:0]};
                write_pointer <= write_pointer + 6'd1;
            end
            if (retire) read_pointer <= read_pointer + 6'd1;
        end
    end
endmodule
`default_nettype wire
