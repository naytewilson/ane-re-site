#!/usr/bin/env python3
"""Apply the Bonsai Q2_0 g128 x86 LUT/VNNI candidate.

Candidate behavior:
- AVX-512 VNNI / AVX-VNNI: decode 8 packed Q2_0 bytes with a 16-entry
  nibble lookup table, keep eight int32 dot-product lanes live across the
  whole row, then perform one horizontal reduction.
- AVX2: add the proven direct packed Q2_0 x Q8_0 kernel instead of the
  scalar fallback.

The script fails closed if the pinned Prism source layout drifts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

FUNCTION = "void ggml_vec_dot_q2_0_q8_0("
NEXT_FUNCTION = "void ggml_vec_dot_q1_0_q8_0("
VNNI_MARKER = "#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)"

OPTIMIZED = r'''#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)
    // Decode each 2-bit code with two pshufb lookups per eight packed bytes.
    // A LUT byte packs the two codes from a nibble as low/high nibbles.
    const __m128i lut = _mm_setr_epi8(
            0x00, 0x01, 0x02, 0x03, 0x10, 0x11, 0x12, 0x13,
            0x20, 0x21, 0x22, 0x23, 0x30, 0x31, 0x32, 0x33);
    const __m128i mask_0f = _mm_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        __m256 acc_block = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb->qs);
            const __m128i src = _mm_loadl_epi64((const __m128i *) &x[i].qs[k * 8]);

            const __m128i lo_idx = _mm_and_si128(src, mask_0f);
            const __m128i hi_idx = _mm_and_si128(_mm_srli_epi16(src, 4), mask_0f);
            const __m128i lo_pair = _mm_shuffle_epi8(lut, lo_idx);
            const __m128i hi_pair = _mm_shuffle_epi8(lut, hi_idx);
            const __m128i pairs = _mm_unpacklo_epi8(lo_pair, hi_pair);
            const __m128i c02 = _mm_and_si128(pairs, mask_0f);
            const __m128i c13 = _mm_and_si128(_mm_srli_epi16(pairs, 4), mask_0f);
            const __m256i codes = _mm256_set_m128i(
                    _mm_unpackhi_epi8(c02, c13),
                    _mm_unpacklo_epi8(c02, c13));

            const __m256i dp = GGML_DPBUSD_256(_mm256_setzero_si256(), codes, qy);
            const __m256i sy = GGML_DPBUSD_256(_mm256_setzero_si256(), ones, qy);
            const __m256 partial = _mm256_cvtepi32_ps(_mm256_sub_epi32(dp, sy));
            acc_block = _mm256_fmadd_ps(
                    _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(yb->d)), partial, acc_block);
        }
        acc = _mm256_fmadd_ps(
                _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d)), acc_block, acc);
    }
    sumf = hsum_float_8(acc);
#elif defined(__AVX2__)
    // Native AVX2 path for hosts without VNNI. The previous implementation
    // dropped to scalar for Q2_0 on these CPUs.
    const __m256i ones_8  = _mm256_set1_epi8(1);
    const __m256i ones_16 = _mm256_set1_epi16(1);
    const __m128i idxlo   = _mm_setr_epi8(0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3);
    const __m128i idxhi   = _mm_setr_epi8(4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7);
    const __m256i mul     = _mm256_setr_epi16(64,16,4,1, 64,16,4,1, 64,16,4,1, 64,16,4,1);
    const __m256i three   = _mm256_set1_epi16(3);
    __m256 acc = _mm256_setzero_ps();

    for (int i = 0; i < nb; ++i) {
        __m256 acc_block = _mm256_setzero_ps();
        for (int k = 0; k < 4; ++k) {
            const block_q8_0 * GGML_RESTRICT yb = &y[i * 4 + k];
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb->qs);
            const __m128i src = _mm_loadl_epi64((const __m128i *) &x[i].qs[k * 8]);
            const __m256i rep = _mm256_set_m128i(_mm_shuffle_epi8(src, idxhi), _mm_shuffle_epi8(src, idxlo));
            __m256i r0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(rep));
            __m256i r1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(rep, 1));
            r0 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r0, mul), 6), three);
            r1 = _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(r1, mul), 6), three);
            const __m256i codes = _mm256_permute4x64_epi64(_mm256_packus_epi16(r0, r1), 0xD8);
            const __m256i prod16 = _mm256_maddubs_epi16(codes, qy);
            const __m256i sum16  = _mm256_maddubs_epi16(ones_8, qy);
            const __m256i partial32 = _mm256_madd_epi16(
                    _mm256_sub_epi16(prod16, sum16), ones_16);
            acc_block = _mm256_fmadd_ps(
                    _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(yb->d)),
                    _mm256_cvtepi32_ps(partial32), acc_block);
        }
        acc = _mm256_fmadd_ps(
                _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[i].d)), acc_block, acc);
    }
    sumf = hsum_float_8(acc);
#else'''


def patch(path: Path) -> None:
    text = path.read_text()
    f0 = text.find(FUNCTION)
    if f0 < 0:
        raise SystemExit("Q2_0 function not found")
    f1 = text.find(NEXT_FUNCTION, f0)
    if f1 < 0:
        raise SystemExit("Q1_0 function boundary not found")
    body = text[f0:f1]
    p0 = body.find(VNNI_MARKER)
    if p0 < 0:
        raise SystemExit("VNNI branch marker not found")
    p_else = body.find("\n#else\n", p0)
    if p_else < 0:
        raise SystemExit("scalar branch marker not found")
    p_end = body.find("\n#endif", p_else)
    if p_end < 0:
        raise SystemExit("Q2_0 preprocessor terminator not found")
    scalar = body[p_else + len("\n#else"):p_end]
    replacement = OPTIMIZED + scalar + "\n#endif"
    patched = body[:p0] + replacement + body[p_end + len("\n#endif"):]
    if patched.count("#elif defined(__AVX2__)") != 1:
        raise SystemExit("candidate insertion invariant failed")
    if patched.count("const __m128i lut") != 1:
        raise SystemExit("LUT insertion invariant failed")
    path.write_text(text[:f0] + patched + text[f1:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    patch(args.source)


if __name__ == "__main__":
    main()
