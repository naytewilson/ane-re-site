#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

static int32_t repack_shader(uint8_t v) {
    return int32_t((uint32_t(v) & 0x03u)
                 | ((uint32_t(v) & 0x0cu) << 6)
                 | ((uint32_t(v) & 0x30u) << 12)
                 | ((uint32_t(v) & 0xc0u) << 18));
}

static int dot_packed_u8_s8(int32_t packed, const int8_t * q) {
    const uint32_t u = uint32_t(packed);
    int sum = 0;
    for (int i = 0; i < 4; ++i) {
        sum += int((u >> (i * 8)) & 0xffu) * int(q[i]);
    }
    return sum;
}

static float shader_block(
    const std::array<uint8_t, 32> & q2,
    const std::array<int8_t, 128> & q8,
    const std::array<float, 4> & d8,
    float d2) {

    float total = 0.0f;
    for (int chunk = 0; chunk < 4; ++chunk) {
        int q8_sum = 0;
        for (int i = 0; i < 32; ++i) q8_sum += int(q8[chunk * 32 + i]);

        float reduced_invocations = 0.0f;
        for (int invocation = 0; invocation < 4; ++invocation) {
            int q_sum = 0;
            for (int pair = 0; pair < 2; ++pair) {
                const int byte_index = chunk * 8 + invocation * 2 + pair;
                const int q8_index = chunk * 32 + invocation * 8 + pair * 4;
                q_sum += dot_packed_u8_s8(repack_shader(q2[byte_index]), &q8[q8_index]);
            }
            reduced_invocations += d2 * (float(q_sum) * d8[chunk] - (d8[chunk] * float(q8_sum)) / 4.0f);
        }
        total += reduced_invocations;
    }
    return total;
}

static float scalar_block(
    const std::array<uint8_t, 32> & q2,
    const std::array<int8_t, 128> & q8,
    const std::array<float, 4> & d8,
    float d2) {

    float total = 0.0f;
    for (int chunk = 0; chunk < 4; ++chunk) {
        int dot = 0;
        for (int i = 0; i < 32; ++i) {
            const int global = chunk * 32 + i;
            const uint8_t byte = q2[global / 4];
            const int code = int((byte >> ((global % 4) * 2)) & 3u) - 1;
            dot += code * int(q8[global]);
        }
        total += d2 * d8[chunk] * float(dot);
    }
    return total;
}

int main() {
    // Exhaustively validate every packed byte and every possible signed Q8 value
    // in each of the four byte lanes.
    for (int packed = 0; packed < 256; ++packed) {
        for (int lane = 0; lane < 4; ++lane) {
            for (int qv = -127; qv <= 127; ++qv) {
                int8_t q[4] = {0, 0, 0, 0};
                q[lane] = int8_t(qv);
                const int got = dot_packed_u8_s8(repack_shader(uint8_t(packed)), q);
                const int code = ((packed >> (lane * 2)) & 3) * qv;
                if (got != code) {
                    std::fprintf(stderr, "repack mismatch packed=%d lane=%d q=%d got=%d expected=%d\n", packed, lane, qv, got, code);
                    return 1;
                }
            }
        }
    }

    std::mt19937 rng(0xB05A1u);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> q8_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(0.0001f, 3.0f);

    float worst_abs = 0.0f;
    float worst_rel = 0.0f;
    constexpr int cases = 200000;
    for (int c = 0; c < cases; ++c) {
        std::array<uint8_t, 32> q2{};
        std::array<int8_t, 128> q8{};
        std::array<float, 4> d8{};
        for (auto & v : q2) v = uint8_t(byte_dist(rng));
        for (auto & v : q8) v = int8_t(q8_dist(rng));
        for (auto & v : d8) v = scale_dist(rng);
        const float d2 = scale_dist(rng);

        const float ref = scalar_block(q2, q8, d8, d2);
        const float got = shader_block(q2, q8, d8, d2);
        const float abs_err = std::fabs(ref - got);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(ref));
        worst_abs = std::max(worst_abs, abs_err);
        worst_rel = std::max(worst_rel, rel_err);
        if (rel_err > 2.0e-5f && abs_err > 2.0e-3f) {
            std::fprintf(stderr, "block mismatch case=%d ref=%.9g got=%.9g abs=%.9g rel=%.9g\n", c, ref, got, abs_err, rel_err);
            return 2;
        }
    }

    std::printf("repack_exhaustive: PASS (256 bytes x 4 lanes x 255 signed values)\n");
    std::printf("full_block_random: PASS (%d cases)\n", cases);
    std::printf("worst_abs_error: %.9g\n", worst_abs);
    std::printf("worst_rel_error: %.9g\n", worst_rel);
    return 0;
}
