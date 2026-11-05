# qwen_gemm_array_i candidate

The candidate retains the baseline ports, signed INT8 weights, BF16 activation
alignment, INT24 column results, metadata, bank ownership, and ready/valid
interface. Clock is `clk`; `rst_n` is active low and sampled on the rising edge.

The selected weight and aligned activation of each DSP lane are registered
together before the DSP. This inserts one clock in the data path, and the valid
pipeline changes from `LANE + 1` to `LANE + 2`. Credits count accepted but not
retired transactions as before. The lane skew and relative DSP cascade timing
remain unchanged. No arithmetic rounding or saturation is changed.

For each accepted input, the output data and metadata must match the baseline
in order, including back-to-back inputs, bank transitions, and output stalls.
Reset, clear, bank reload, clock-domain assumptions, and parameter overrides
require additional coverage before this candidate can be promoted.
