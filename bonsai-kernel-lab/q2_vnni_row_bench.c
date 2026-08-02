#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline int hsum_i32_8(__m256i a) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    __m128i h = _mm_unpackhi_epi64(s, s);
    s = _mm_add_epi32(s, h);
    h = _mm_shuffle_epi32(s, _MM_SHUFFLE(2,3,0,1));
    return _mm_cvtsi128_si32(_mm_add_epi32(s, h));
}

static inline float hsum_f32_8(__m256 a) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline __m256i unpack32(const uint8_t * src) {
    const __m128i il = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m128i ih = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    const __m256i mul = _mm256_setr_epi16(64,16,4,1,64,16,4,1,64,16,4,1,64,16,4,1);
    const __m256i three = _mm256_set1_epi16(3);
    const __m128i s = _mm_loadl_epi64((const __m128i *) src);
    const __m256i rep = _mm256_set_m128i(_mm_shuffle_epi8(s, ih), _mm_shuffle_epi8(s, il));
    __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
    __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
    r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
    r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
}

static float baseline(const uint8_t * x, const int8_t * y, const float * dx, const float * dy, int nb) {
    const __m256i ones = _mm256_set1_epi8(1);
    float sum = 0.0f;
    for (int i = 0; i < nb; ++i) {
        float block = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const __m256i codes = unpack32(x + i*32 + k*8);
            const __m256i qy = _mm256_loadu_si256((const __m256i *)(y + i*128 + k*32));
            const __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), codes, qy);
            const __m256i sy = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, qy);
            block += dy[i*4+k] * (float) hsum_i32_8(_mm256_sub_epi32(dp, sy));
        }
        sum += dx[i] * block;
    }
    return sum;
}

static float candidate(const uint8_t * x, const int8_t * y, const float * dx, const float * dy, int nb) {
    const __m256i ones = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        __m256 block = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) {
            const __m256i codes = unpack32(x + i*32 + k*8);
            const __m256i qy = _mm256_loadu_si256((const __m256i *)(y + i*128 + k*32));
            const __m256i dp = _mm256_dpbusd_epi32(_mm256_setzero_si256(), codes, qy);
            const __m256i sy = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, qy);
            const __m256 partial = _mm256_cvtepi32_ps(_mm256_sub_epi32(dp, sy));
            block = _mm256_fmadd_ps(_mm256_set1_ps(dy[i*4+k]), partial, block);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(dx[i]), block, acc);
    }
    return hsum_f32_8(acc);
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(void) {
    enum { NB = 128, R = 20000 };
    uint8_t * x = aligned_alloc(64, NB*32);
    int8_t * y = aligned_alloc(64, NB*128);
    float * dx = aligned_alloc(64, NB*sizeof(float));
    float * dy = aligned_alloc(64, NB*4*sizeof(float));
    if (!x || !y || !dx || !dy) return 2;
    srand(11);
    for (int i = 0; i < NB*32; ++i) x[i] = (uint8_t) rand();
    for (int i = 0; i < NB*128; ++i) y[i] = (int8_t) ((rand()%255)-127);
    for (int i = 0; i < NB; ++i) {
        dx[i] = .001f + (rand()%1000)/1000.f;
        for (int k = 0; k < 4; ++k) dy[i*4+k] = .001f + (rand()%1000)/1000.f;
    }
    const float a = baseline(x, y, dx, dy, NB);
    const float b = candidate(x, y, dx, dy, NB);
    const float rel = fabsf(a-b)/(fabsf(a)+1e-9f);
    printf("relative_error %.9g\n", rel);
    if (rel > 2e-5f) return 3;
    volatile float sink = 0;
    double t0 = now_s();
    for (int r = 0; r < R; ++r) sink += baseline(x, y, dx, dy, NB);
    double t1 = now_s();
    for (int r = 0; r < R; ++r) sink += candidate(x, y, dx, dy, NB);
    double t2 = now_s();
    printf("correctness PASS\nbaseline_us %.3f\ncandidate_us %.3f\nspeedup %.3fx\nsink %g\n",
           (t1-t0)*1e6/R, (t2-t1)*1e6/R, (t1-t0)/(t2-t1), sink);
    free(x); free(y); free(dx); free(dy);
    return 0;
}
