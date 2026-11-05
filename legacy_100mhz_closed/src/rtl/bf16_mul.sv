`timescale 1ns/1ps
`default_nettype none

// Exact BF16 multiply. Result is {sign, biased exponent, explicit 16-bit significand}.
// Three register stages; DAZ/FTZ; no rounding and no backpressure.
module bf16_mul (
    input  wire        clk,
    input  wire        input_valid,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output reg [24:0] result,
    output wire         output_valid
);
    // Configuration INIT clears valid without adding a reset port or datapath reset.
    reg [2:0] valid_pipe = 3'd0;
    assign output_valid = valid_pipe[2];

    reg [7:0] decode_mantissa_a, decode_mantissa_b;
    reg [7:0] decode_exponent_a, decode_exponent_b;
    reg decode_sign, decode_zero, decode_infinity, decode_nan;

    // Use the DSP48E2 multiplier while retaining all sixteen product bits.
    (* use_dsp = "yes" *) reg [15:0] product_mantissa;
    reg [8:0] product_exponent_sum;
    reg product_sign, product_zero, product_infinity, product_nan;

    wire [15:0] normal_mantissa;
    wire [8:0] normal_exponent_sum;
    wire [7:0] normal_exponent;

    // Normalization and range selection share the final register stage.
    assign normal_mantissa = product_mantissa[15] ? product_mantissa :
                             {product_mantissa[14:0], 1'b0};
    assign normal_exponent_sum = product_exponent_sum + {8'b0, product_mantissa[15]};
    assign normal_exponent = normal_exponent_sum[7:0] - 8'd127;

    always @(posedge clk) begin
        valid_pipe <= {valid_pipe[1:0], input_valid};

        // Decode includes all special cases. Exponent-zero inputs are signed zero.
        decode_mantissa_a <= {1'b1, a[6:0]};
        decode_mantissa_b <= {1'b1, b[6:0]};
        decode_exponent_a <= a[14:7];
        decode_exponent_b <= b[14:7];
        decode_sign <= a[15] ^ b[15];
        decode_zero <= (a[14:7] == 0) || (b[14:7] == 0);
        decode_infinity <= (a[14:7] == 8'hff) || (b[14:7] == 8'hff);
        decode_nan <= ((a[14:7] == 8'hff) && (a[6:0] != 0)) ||
                      ((b[14:7] == 8'hff) && (b[6:0] != 0)) ||
                      ((a[14:7] == 8'hff) && (b[14:7] == 0)) ||
                      ((b[14:7] == 8'hff) && (a[14:7] == 0));

        // Unsigned exponent sum is 0..510; normalization adds at most one.
        product_mantissa <= decode_mantissa_a * decode_mantissa_b;
        product_exponent_sum <= {1'b0,decode_exponent_a}+{1'b0,decode_exponent_b};
        product_sign <= decode_sign;
        product_zero <= decode_zero;
        product_infinity <= decode_infinity;
        product_nan <= decode_nan;

        // Canonical NaN takes precedence over infinity and zero.
        if (product_nan)
            result <= {1'b0, 8'hff, 16'hc000};
        else if (product_infinity)
            result <= {product_sign, 8'hff, 16'h8000};
        else if (product_zero || (normal_exponent_sum <= 9'd127))
            result <= {product_sign, 8'h00, 16'h0000};
        else if (normal_exponent_sum >= 9'd382)
            result <= {product_sign, 8'hff, 16'h8000};
        else
            result <= {product_sign, normal_exponent[7:0], normal_mantissa};
    end
endmodule

`default_nettype wire


