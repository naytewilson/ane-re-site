#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline int hsum_i32_8(__m256i a) {
    __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(hi64, sum128);
    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

static inline __m256i unpack_codes_32(const uint8_t *src) {
    const __m128i idxlo = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m128i idxhi = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    const __m256i mul = _mm256_setr_epi16(64,16,4,1, 64,16,4,1, 64,16,4,1, 64,16,4,1);
    const __m256i three = _mm256_set1_epi16(3);
    const __m128i s = _mm_loadl_epi64((const __m128i *)src);
    const __m256i rep = _mm256_set_m128i(_mm_shuffle_epi8(s, idxhi), _mm_shuffle_epi8(s, idxlo));
    __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
    __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
    r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
    r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
    return _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
}

static inline int baseline(const uint8_t *qs, const int8_t *qy) {
    const __m256i codes = unpack_codes_32(qs);
    const __m256i y = _mm256_loadu_si256((const __m256i *)qy);
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    const int dp = hsum_i32_8(_mm256_dpbusd_epi32(z, codes, y));
    const int sy = hsum_i32_8(_mm256_dpbusd_epi32(z, ones, y));
    return dp - sy;
}

static inline int reduce_once(const uint8_t *qs, const int8_t *qy) {
    const __m256i codes = unpack_codes_32(qs);
    const __m256i y = _mm256_loadu_si256((const __m256i *)qy);
    const __m256i z = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi8(1);
    const __m256i dp = _mm256_dpbusd_epi32(z, codes, y);
    const __m256i sy = _mm256_dpbusd_epi32(z, ones, y);
    return hsum_i32_8(_mm256_sub_epi32(dp, sy));
}

static int scalar(const uint8_t *qs, const int8_t *qy) {
    int s = 0;
    for (int b = 0; b < 8; ++b) {
        uint8_t v = qs[b];
        for (int k = 0; k < 4; ++k) s += (((v >> (2*k)) & 3) - 1) * qy[b*4+k];
    }
    return s;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    enum { N = 1<<15, R = 400 };
    uint8_t *qs = aligned_alloc(64, N*8);
    int8_t *qy = aligned_alloc(64, N*32);
    if (!qs || !qy) return 2;
    srand(1);
    for (int i=0;i<N*8;i++) qs[i]=(uint8_t)rand();
    for (int i=0;i<N*32;i++) qy[i]=(int8_t)((rand()%255)-127);
    for (int i=0;i<N;i++) {
        int a=scalar(qs+i*8,qy+i*32), b=baseline(qs+i*8,qy+i*32), c=reduce_once(qs+i*8,qy+i*32);
        if (a!=b || a!=c) { fprintf(stderr,"mismatch %d %d %d at %d\n",a,b,c,i); return 3; }
    }
    volatile int sink=0;
    double t0=now_s();
    for(int r=0;r<R;r++) for(int i=0;i<N;i++) sink += baseline(qs+i*8,qy+i*32);
    double t1=now_s();
    for(int r=0;r<R;r++) for(int i=0;i<N;i++) sink += reduce_once(qs+i*8,qy+i*32);
    double t2=now_s();
    double ops=(double)N*R;
    printf("correctness: PASS (%d blocks)\n",N);
    printf("baseline_ns_per_32w: %.3f\n",(t1-t0)*1e9/ops);
    printf("reduce_once_ns_per_32w: %.3f\n",(t2-t1)*1e9/ops);
    printf("speedup: %.3fx\n",(t1-t0)/(t2-t1));
    printf("sink=%d\n",sink);
    free(qs); free(qy);
    return 0;
}
