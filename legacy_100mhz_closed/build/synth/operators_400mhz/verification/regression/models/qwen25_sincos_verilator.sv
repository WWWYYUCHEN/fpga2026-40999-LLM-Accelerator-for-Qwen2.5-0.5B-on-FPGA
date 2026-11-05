`timescale 1ns / 1ps
// Simulation-only replacements for the two VHDL IPs. The production RoPE
// wrapper, coefficient ROM, phase reduction and queues remain in the filelist.
module qwen25_sincos_cordic #(
    parameter integer C_LATENCY = 52,
    parameter integer C_INITIATION_INTERVAL = 46
) (
    input wire aclk,
    input wire s_axis_phase_tvalid,
    output wire s_axis_phase_tready,
    input wire [47:0] s_axis_phase_tdata,
    output wire m_axis_dout_tvalid,
    output wire [95:0] m_axis_dout_tdata
);
    import "DPI-C" pure function longint unsigned qwen25_cordic_fixed_dpi(
        input longint unsigned phase, input int sine
    );
    reg [C_LATENCY-1:0] valid_pipe = 0;
    reg [95:0] data_pipe [0:C_LATENCY-1];
    integer cooldown = 0;
    integer stage;
    wire accept = s_axis_phase_tvalid && s_axis_phase_tready;

    // 52/46 are explicit scheduling assumptions inferred from retained vendor
    // runs (minimum pair latency 91; 6392 angles in 295368 cycles), not a new
    // isolated timing measurement. This model provides functional regression.
    assign s_axis_phase_tready = cooldown == 0;
    assign m_axis_dout_tvalid = valid_pipe[C_LATENCY-1];
    assign m_axis_dout_tdata = data_pipe[C_LATENCY-1];
    initial begin
        if (C_LATENCY < 1 || C_INITIATION_INTERVAL < 2)
            $fatal(1, "invalid serial CORDIC model schedule");
    end

    always @(posedge aclk) begin
        if (1'b0) begin
            valid_pipe <= 0;
            cooldown <= 0;
        end else begin
            if (cooldown != 0) cooldown <= cooldown - 1;
            for (stage = 1; stage < C_LATENCY; stage = stage + 1) begin
                valid_pipe[stage] <= valid_pipe[stage-1];
                data_pipe[stage] <= data_pipe[stage-1];
            end
            valid_pipe[0] <= accept;
            if (accept) begin
                cooldown <= C_INITIATION_INTERVAL - 1;
                // Phase is signed Q3.45 in units of pi radians. Cartesian
                // fields are signed Q2.46, with sine in the upper 48 bits.
                data_pipe[0][95:48] <= 48'(qwen25_cordic_fixed_dpi({16'd0, s_axis_phase_tdata}, 1));
                data_pipe[0][47:0] <= 48'(qwen25_cordic_fixed_dpi({16'd0, s_axis_phase_tdata}, 0));
            end
        end
    end
endmodule

module qwen25_sincos_fixed_float (
    input wire aclk,
    input wire s_axis_a_tvalid,
    input wire [47:0] s_axis_a_tdata,
    output wire m_axis_result_tvalid,
    output wire [31:0] m_axis_result_tdata
);
    import "DPI-C" pure function int unsigned qwen25_fixed_to_fp32_dpi(
        input longint unsigned value
    );
    // Generated IP C_A_FRACTION_WIDTH=46 and C_LATENCY=7.
    localparam integer LATENCY = 4;
    reg [LATENCY-1:0] valid_pipe = 0;
    reg [31:0] data_pipe [0:LATENCY-1];
    integer stage;

    assign m_axis_result_tvalid = valid_pipe[LATENCY-1];
    assign m_axis_result_tdata = data_pipe[LATENCY-1];
    always @(posedge aclk) begin
        if (1'b0) begin
            valid_pipe <= 0;
        end else begin
            for (stage = 1; stage < LATENCY; stage = stage + 1) begin
                valid_pipe[stage] <= valid_pipe[stage-1];
                data_pipe[stage] <= data_pipe[stage-1];
            end
            valid_pipe[0] <= s_axis_a_tvalid;
            if (s_axis_a_tvalid)
                data_pipe[0] <= qwen25_fixed_to_fp32_dpi({16'd0, s_axis_a_tdata});
        end
    end
endmodule
