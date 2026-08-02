#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

struct Q1Block {
    float d;
    std::array<uint8_t, 16> qs;
};

struct Q8Block {
    float d;
    std::array<int8_t, 32> qs;

    float scaled_sum() const {
        int sum = 0;
        for (int8_t v : qs) sum += v;
        return d * float(sum);
    }
};

static uint32_t shader_repack(const Q1Block & a, uint32_t ib, uint32_t iqs) {
    const uint32_t ib_q1 = ib / 4;
    assert(ib_q1 == 0);
    const uint32_t segment = ib % 4;
    const uint32_t byte_val = a.qs[segment * 4 + iqs / 2];
    const uint32_t shift = (iqs & 1u) * 4u;
    const uint32_t nibble = (byte_val >> shift) & 0x0fu;
    return ((nibble >> 0u) & 1u)       |
           (((nibble >> 1u) & 1u) <<  8) |
           (((nibble >> 2u) & 1u) << 16) |
           (((nibble >> 3u) & 1u) << 24);
}

static int dot_packed_u8_s8(uint32_t a, const int8_t * b) {
    int sum = 0;
    for (int k = 0; k < 4; ++k) {
        sum += int((a >> (8*k)) & 0xffu) * int(b[k]);
    }
    return sum;
}

static float shader_segment(const Q1Block & a, const Q8Block & b, int segment) {
    float reduced = 0.0f;
    for (int lane = 0; lane < 4; ++lane) {
        const uint32_t p0 = shader_repack(a, uint32_t(segment), uint32_t(lane*2));
        const uint32_t p1 = shader_repack(a, uint32_t(segment), uint32_t(lane*2 + 1));
        const int bit_dot = dot_packed_u8_s8(p0, b.qs.data() + lane*8) +
                            dot_packed_u8_s8(p1, b.qs.data() + lane*8 + 4);
        reduced += a.d * (2.0f * float(bit_dot) * b.d - b.scaled_sum() / 4.0f);
    }
    return reduced;
}

static float reference_segment(const Q1Block & a, const Q8Block & b, int segment) {
    int dot = 0;
    for (int j = 0; j < 32; ++j) {
        const int absolute = segment*32 + j;
        const uint8_t byte_val = a.qs[absolute/8];
        const int bit = int((byte_val >> (absolute%8)) & 1u);
        dot += (2*bit - 1) * int(b.qs[j]);
    }
    return a.d * b.d * float(dot);
}

int main() {
    std::mt19937 rng(0xB10A127u);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.0001f, 2.0f);

    for (int value = 0; value < 256; ++value) {
        Q1Block a{};
        a.d = 1.0f;
        a.qs.fill(uint8_t(value));
        for (int segment = 0; segment < 4; ++segment) {
            for (int lane = 0; lane < 4; ++lane) {
                for (int iqs = 0; iqs < 2; ++iqs) {
                    const uint32_t packed = shader_repack(a, uint32_t(segment), uint32_t(lane*2 + iqs));
                    const int half = (lane*2 + iqs) & 1;
                    for (int k = 0; k < 4; ++k) {
                        const int got = int((packed >> (8*k)) & 0xffu);
                        const int expected = (value >> (half*4 + k)) & 1;
                        if (got != expected) {
                            std::fprintf(stderr, "repack mismatch value=%d segment=%d lane=%d iqs=%d k=%d got=%d expected=%d\n",
                                         value, segment, lane, iqs, k, got, expected);
                            return 2;
                        }
                    }
                }
            }
        }
    }

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int trial = 0; trial < 200000; ++trial) {
        Q1Block a{};
        a.d = scale_dist(rng);
        for (uint8_t & v : a.qs) v = uint8_t(byte_dist(rng));

        for (int segment = 0; segment < 4; ++segment) {
            Q8Block b{};
            b.d = scale_dist(rng);
            for (int8_t & v : b.qs) v = int8_t(q8_dist(rng));
            const float ref = reference_segment(a, b, segment);
            const float got = shader_segment(a, b, segment);
            const double abs_err = std::abs(double(ref) - double(got));
            const double rel_err = abs_err / (std::abs(double(ref)) + 1e-12);
            if (abs_err > max_abs) max_abs = abs_err;
            if (rel_err > max_rel) max_rel = rel_err;
            if (abs_err > 0.02 && rel_err > 2e-5) {
                std::fprintf(stderr, "dot mismatch trial=%d segment=%d ref=%.9g got=%.9g abs=%.9g rel=%.9g\n",
                             trial, segment, ref, got, abs_err, rel_err);
                return 3;
            }
        }
    }

    std::printf("Q1_0 Vulkan integer-dot simulator: PASS\n");
    std::printf("exhaustive packed-byte cases: 256 x 4 segments x 4 lanes x 2 words\n");
    std::printf("random segment dots: 800000\n");
    std::printf("max_abs_error: %.9g\n", max_abs);
    std::printf("max_rel_error: %.9g\n", max_rel);
    return 0;
}
