# kv8_k_quant candidate

The interface and numerical format match the baseline module. `clk` is the
single clock and `rst_n` is sampled low on rising edges. Magnitude, lock and
restore requests retain the existing divider handshake and arithmetic.

Pass-one quantisation has three registered stages: exponent alignment and
reciprocal capture, rounded multiplication, then saturation and signed code
packing. `o_q_valid` follows each accepted `i_valid && i_pass` by three sampled
clock edges. The pipeline accepts one beat each cycle. Sign and reciprocal are
captured with their associated input; reset clears in-flight valid bits.

Verification covers the quantiser numerical model and the token-major K write
chain under backpressure. Full top-level timing and end-to-end verification are
still required before promotion.
