#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static inline int hsum_i32_8(__m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

static inline __m256i unpack_mul(const uint8_t * src) {
    const __m128i idxlo = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m128i idxhi = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    const __m256i mul = _mm256_setr_epi16(64,16,4,1, 64,16,4,1, 64,16,4,1, 64,16,4,1);
    const __m256i three = _mm256_set1_epi16(3);
    const __m128i s = _mm_loadl_epi64((const __m128i *) src);
    const __m256i rep = _mm256_set_m128i(_mm_shuffle_epi8(s, idxhi), _mm_shuffle_epi8(s, idxlo));
    __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
    __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
    r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
    r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
}

static inline __m256i unpack_lut(const uint8_t * src) {
    // Each lookup byte packs two decoded 2-bit codes as low/high nibbles.
    const __m128i lut = _mm_setr_epi8(
        0x00,0x01,0x02,0x03, 0x10,0x11,0x12,0x13,
        0x20,0x21,0x22,0x23, 0x30,0x31,0x32,0x33);
    const __m128i s = _mm_loadl_epi64((const __m128i *) src);
    const __m128i mask0f = _mm_set1_epi8(0x0f);
    const __m128i lo = _mm_and_si128(s, mask0f);
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(s, 4), mask0f);
    const __m128i packed_lo = _mm_shuffle_epi8(lut, lo);
    const __m128i packed_hi = _mm_shuffle_epi8(lut, hi);
    const __m128i pairs = _mm_unpacklo_epi8(packed_lo, packed_hi);
    const __m256i w = _mm256_cvtepu8_epi16(pairs);
    const __m256i low = _mm256_and_si256(w, _mm256_set1_epi16(0x000f));
    const __m256i high = _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x00f0)), 4);
    return _mm256_or_si256(low, high);
}

static inline int original(const uint8_t * qs, const int8_t * qy) {
    const __m256i codes = unpack_mul(qs);
    const __m256i y = _mm256_loadu_si256((const __m256i *) qy);
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    const int dp = hsum_i32_8(_mm256_dpbusd_epi32(z, codes, y));
    const int sy = hsum_i32_8(_mm256_dpbusd_epi32(z, ones, y));
    return dp - sy;
}

static inline int reduce_once(const uint8_t * qs, const int8_t * qy) {
    const __m256i codes = unpack_mul(qs);
    const __m256i y = _mm256_loadu_si256((const __m256i *) qy);
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i dp = _mm256_dpbusd_epi32(z, codes, y);
    const __m256i sy = _mm256_dpbusd_epi32(z, ones, y);
    return hsum_i32_8(_mm256_sub_epi32(dp, sy));
}

static inline int lut_reduce_once(const uint8_t * qs, const int8_t * qy) {
    const __m256i codes = unpack_lut(qs);
    const __m256i y = _mm256_loadu_si256((const __m256i *) qy);
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i dp = _mm256_dpbusd_epi32(z, codes, y);
    const __m256i sy = _mm256_dpbusd_epi32(z, ones, y);
    return hsum_i32_8(_mm256_sub_epi32(dp, sy));
}

static int scalar(const uint8_t * qs, const int8_t * qy) {
    int sum = 0;
    for (int b = 0; b < 8; ++b) {
        for (int k = 0; k < 4; ++k) {
            sum += (((qs[b] >> (2*k)) & 3) - 1) * qy[b*4 + k];
        }
    }
    return sum;
}

static int validate_exhaustive(void) {
    uint8_t qs[8] = {0};
    int8_t qy[32];
    for (int i = 0; i < 32; ++i) qy[i] = (int8_t) ((i * 37) % 255 - 127);

    for (int slot = 0; slot < 8; ++slot) {
        for (int byte = 0; byte < 256; ++byte) {
            memset(qs, 0, sizeof(qs));
            qs[slot] = (uint8_t) byte;
            const int a = scalar(qs, qy);
            const int b = original(qs, qy);
            const int c = reduce_once(qs, qy);
            const int d = lut_reduce_once(qs, qy);
            if (a != b || a != c || a != d) {
                fprintf(stderr, "exhaustive mismatch slot=%d byte=%d: %d %d %d %d\n", slot, byte, a, b, c, d);
                return 0;
            }
        }
    }
    return 1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef int (*kernel_fn)(const uint8_t *, const int8_t *);

static double bench(kernel_fn fn, const uint8_t * qs, const int8_t * qy, int n, int repeats, volatile int * sink) {
    const double start = now_s();
    for (int r = 0; r < repeats; ++r) {
        for (int i = 0; i < n; ++i) *sink += fn(qs + i*8, qy + i*32);
    }
    return now_s() - start;
}

int main(void) {
    enum { N = 1 << 17, REPEATS = 160 };
    uint8_t * qs = aligned_alloc(64, (size_t) N * 8);
    int8_t * qy = aligned_alloc(64, (size_t) N * 32);
    if (!qs || !qy) return 2;

    if (!validate_exhaustive()) return 3;

    srand(12345);
    for (int i = 0; i < N*8; ++i) qs[i] = (uint8_t) rand();
    for (int i = 0; i < N*32; ++i) qy[i] = (int8_t) ((rand() % 255) - 127); // Q8_0 invariant

    for (int i = 0; i < N; ++i) {
        const int a = scalar(qs + i*8, qy + i*32);
        const int b = original(qs + i*8, qy + i*32);
        const int c = reduce_once(qs + i*8, qy + i*32);
        const int d = lut_reduce_once(qs + i*8, qy + i*32);
        if (a != b || a != c || a != d) {
            fprintf(stderr, "random mismatch at %d: %d %d %d %d\n", i, a, b, c, d);
            return 4;
        }
    }

    volatile int sink = 0;
    (void) bench(lut_reduce_once, qs, qy, N, 2, &sink);
    const double t_original = bench(original, qs, qy, N, REPEATS, &sink);
    const double t_reduce = bench(reduce_once, qs, qy, N, REPEATS, &sink);
    const double t_lut = bench(lut_reduce_once, qs, qy, N, REPEATS, &sink);
    const double ops = (double) N * REPEATS;

    printf("correctness: PASS (2048 exhaustive byte placements + %d randomized blocks)\n", N);
    printf("original_ns_per_32w: %.3f\n", t_original * 1e9 / ops);
    printf("reduce_once_ns_per_32w: %.3f\n", t_reduce * 1e9 / ops);
    printf("lut_reduce_once_ns_per_32w: %.3f\n", t_lut * 1e9 / ops);
    printf("reduce_once_speedup: %.4fx\n", t_original / t_reduce);
    printf("lut_speedup_vs_original: %.4fx\n", t_original / t_lut);
    printf("lut_speedup_vs_reduce_once: %.4fx\n", t_reduce / t_lut);
    printf("sink=%d\n", sink);

    free(qs);
    free(qy);
    return 0;
}
