// Simulation-only scalar arithmetic. No HDL shortreal conversions are involved.
// Build without fast-math or floating point contraction. C++11 is sufficient.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr std::uint32_t sign_mask = 0x80000000u;
constexpr std::uint32_t exponent_mask = 0x7f800000u;
constexpr std::uint32_t fraction_mask = 0x007fffffu;
constexpr std::uint32_t quiet_nan = 0x7fc00000u;
static_assert(sizeof(unsigned int) == 4, "DPI int unsigned must be 32 bits");
static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559,
              "DPI math requires IEEE binary32");
static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
              "DPI math requires IEEE binary64");

std::uint32_t daz(std::uint32_t bits) {
    return (bits & exponent_mask) == 0 ? bits & sign_mask : bits;
}

bool is_nan(std::uint32_t bits) {
    return (bits & exponent_mask) == exponent_mask && (bits & fraction_mask) != 0;
}

bool is_inf(std::uint32_t bits) {
    return (bits & ~sign_mask) == exponent_mask;
}

int highest_bit(std::uint64_t value) {
    int bit = -1;
    while (value != 0) { value >>= 1; ++bit; }
    return bit;
}

// Integer right-shift with round-to-nearest, ties-to-even; no host rounding mode.
std::uint64_t round_shift(std::uint64_t value, int shift) {
    if (shift <= 0) return value << -shift;
    if (shift > 64) return 0;
    if (shift == 64) return value > (std::uint64_t(1) << 63) ? 1 : 0;
    const std::uint64_t retained = value >> shift;
    const std::uint64_t discarded = value & ((std::uint64_t(1) << shift) - 1);
    const std::uint64_t halfway = std::uint64_t(1) << (shift - 1);
    return retained + (discarded > halfway || (discarded == halfway && (retained & 1)));
}

// Pack the exact dyadic number (-1)^sign * magnitude * 2^power, then FTZ.
std::uint32_t pack_dyadic(std::uint32_t sign, std::uint64_t magnitude, int power, bool flush = true) {
    if (magnitude == 0) return sign;
    const int leading = highest_bit(magnitude);
    int exponent = leading + power;
    if (exponent < -126) {
        const std::uint64_t subnormal = round_shift(magnitude, -149 - power);
        if (subnormal >= (std::uint64_t(1) << 23)) return sign | 0x00800000u;
        return sign | (flush ? 0u : std::uint32_t(subnormal));
    }
    std::uint64_t significand = round_shift(magnitude, leading - 23);
    if (significand == (std::uint64_t(1) << 24)) { significand >>= 1; ++exponent; }
    if (exponent > 127) return sign | exponent_mask;
    return sign | (std::uint32_t(exponent + 127) << 23) |
           (std::uint32_t(significand) & fraction_mask);
}

std::uint32_t add_bits(std::uint32_t first, std::uint32_t second) {
    if (is_nan(first) || is_nan(second)) return quiet_nan;
    if (is_inf(first) || is_inf(second)) {
        if (is_inf(first) && is_inf(second) && ((first ^ second) & sign_mask)) return quiet_nan;
        return is_inf(first) ? first : second;
    }
    const int first_exp = int((first >> 23) & 255);
    const int second_exp = int((second >> 23) & 255);
    if (first_exp == 0 && second_exp == 0) return (first & second) & sign_mask;
    if (first_exp == 0) return second;
    if (second_exp == 0) return first;
    // At gaps over 26 bits the small normal cannot change a binary32 RNE result,
    // including subtraction just below a power-of-two boundary.
    if (first_exp < second_exp) return add_bits(second, first);
    const int gap = first_exp - second_exp;
    if (gap > 26) return first;
    const std::uint64_t large = std::uint64_t((first & fraction_mask) | 0x00800000u) << gap;
    const std::uint64_t small = (second & fraction_mask) | 0x00800000u;
    std::uint32_t sign = first & sign_mask;
    std::uint64_t magnitude;
    if (((first ^ second) & sign_mask) == 0) magnitude = large + small;
    else if (large >= small) magnitude = large - small;
    else { magnitude = small - large; sign = second & sign_mask; }
    if (magnitude == 0) return 0;  // Exact cancellation under RNE is +0.
    return pack_dyadic(sign, magnitude, second_exp - 127 - 23);
}

std::uint32_t multiply_bits(std::uint32_t first, std::uint32_t second) {
    if (is_nan(first) || is_nan(second)) return quiet_nan;
    const std::uint32_t sign = (first ^ second) & sign_mask;
    const int first_exp = int((first >> 23) & 255);
    const int second_exp = int((second >> 23) & 255);
    if (is_inf(first) || is_inf(second)) {
        if (first_exp == 0 || second_exp == 0) return quiet_nan;
        return sign | exponent_mask;
    }
    if (first_exp == 0 || second_exp == 0) return sign;
    const std::uint64_t product = std::uint64_t((first & fraction_mask) | 0x00800000u) *
                                  std::uint64_t((second & fraction_mask) | 0x00800000u);
    return pack_dyadic(sign, product, first_exp + second_exp - 254 - 46);
}

double decode(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<double>(value);
}

// Explicit binary64 -> binary32 RNE encoding, canonical NaN and output FTZ.
std::uint32_t encode(double value, bool flush = true) {
    std::uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = std::uint32_t(bits >> 32) & sign_mask;
    const int exponent = int((bits >> 52) & 2047);
    const std::uint64_t fraction = bits & 0x000fffffffffffffull;
    if (exponent == 2047) return fraction != 0 ? quiet_nan : sign | exponent_mask;
    if (exponent == 0) return pack_dyadic(sign, fraction, -1074, flush);
    return pack_dyadic(sign, fraction | (std::uint64_t(1) << 52), exponent - 1023 - 52, flush);
}
}  // namespace

extern "C" unsigned int qwen25_fp32_dpi(int operation, unsigned int first, unsigned int second) {
    const std::uint32_t a = daz(first);
    const std::uint32_t b = daz(second);
    if (operation == 0) return add_bits(a, b);
    if (operation == 1) return multiply_bits(a, b);
    if (is_nan(a) || (operation == 3 && is_nan(b))) return quiet_nan;
    const double av = decode(a);
    switch (operation) {
        case 2: return encode(std::exp(av));
        case 3: return encode(av / decode(b));
        case 4: return encode(1.0 / std::sqrt(av));
        case 5: return encode(1.0 / av);
        default: return 0;
    }
}

// Testbench replacements for bitstoshortreal/shortrealtobits. These helpers
// preserve subnormals; DAZ and FTZ are specific to the scalar IP function above.
extern "C" double qwen_fp32_to_double(unsigned int bits) {
    return decode(bits);
}

extern "C" unsigned int qwen_double_to_fp32(double value) {
    return encode(value, false);
}

extern "C" unsigned long long qwen25_cordic_fixed_dpi(unsigned long long phase, int sine) {
    constexpr std::uint64_t fixed_mask = 0x0000ffffffffffffull;
    constexpr double pi = 3.141592653589793238462643383279502884;
    const std::uint64_t raw = phase & fixed_mask;
    const std::int64_t signed_phase = (raw & 0x0000800000000000ull) != 0
        ? std::int64_t(raw) - (std::int64_t(1) << 48) : std::int64_t(raw);
    const double angle = std::ldexp(static_cast<double>(signed_phase), -45) * pi;
    const double value = sine != 0 ? std::sin(angle) : std::cos(angle);
    const double scaled = std::ldexp(std::fabs(value), 46);
    const double floor_value = std::floor(scaled);
    std::int64_t rounded = static_cast<std::int64_t>(floor_value);
    const double discarded = scaled - floor_value;
    if (discarded > 0.5 || (discarded == 0.5 && (rounded & 1))) ++rounded;
    if (std::signbit(value)) rounded = -rounded;
    return static_cast<std::uint64_t>(rounded) & fixed_mask;
}

extern "C" unsigned int qwen25_fixed_to_fp32_dpi(unsigned long long input) {
    const std::uint64_t raw = input & 0x0000ffffffffffffull;
    const bool negative = (raw & 0x0000800000000000ull) != 0;
    const std::uint64_t magnitude = negative ? (std::uint64_t(1) << 48) - raw : raw;
    return pack_dyadic(negative ? sign_mask : 0u, magnitude, -46);
}
