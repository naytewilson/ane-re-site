#!/usr/bin/env python3
"""Patch the pinned Prism Q2_0 4x8 CPU repack hot path.

The candidate is intentionally narrow and fails closed on source drift:
- use AVX-512 VBMI vpmultishiftqb for packed 2-bit expansion when available;
- preserve the existing expansion on AVX-512 VNNI CPUs without VBMI;
- seed vpdpbusd with -sum(q8) in Q2_0 GEMV and GEMM, eliminating a vector
  subtraction while remaining bit-exact.

Target source:
  ggml/src/ggml-cpu/arch/x86/repack.cpp
Pinned Prism commit:
  9ca265a57f85f2117942490f421f64a226dd9847
"""

from __future__ import annotations

import argparse
from pathlib import Path

GEMV_START = "void ggml_gemv_q2_0_4x8_q8_0("
GEMM_START = "void ggml_gemm_q2_0_4x8_q8_0("

OLD_HELPER = r'''static inline void __q2_0_expand_x4(const uint8_t * qs, __m512i * w01, __m512i * w23) {
    const __m512i m3 = _mm512_set1_epi32(0x03030303);
    const __m256i packed = _mm256_loadu_si256((const __m256i *) qs);
    // dword d = packed byte d; spread the four 2-bit fields to byte lanes
    const __m512i v01 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
    const __m512i v23 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
    const __m512i r01 = _mm512_or_si512(_mm512_or_si512(v01, _mm512_slli_epi32(v01, 6)),
                                        _mm512_or_si512(_mm512_slli_epi32(v01, 12), _mm512_slli_epi32(v01, 18)));
    const __m512i r23 = _mm512_or_si512(_mm512_or_si512(v23, _mm512_slli_epi32(v23, 6)),
                                        _mm512_or_si512(_mm512_slli_epi32(v23, 12), _mm512_slli_epi32(v23, 18)));
    *w01 = _mm512_and_si512(r01, m3);
    *w23 = _mm512_and_si512(r23, m3);
}'''

NEW_HELPER = r'''static inline void __q2_0_expand_x4(const uint8_t * qs, __m512i * w01, __m512i * w23) {
#if defined(__AVX512VBMI__)
    // vpmultishiftqb extracts eight 2-bit fields from each 64-bit source lane.
    // The interleaved repack layout stores one 64-bit lane per model column;
    // duplicate columns {0,1} and {2,3} four times to produce 32 code bytes
    // per column in the same layout expected by the existing VNNI kernels.
    alignas(64) static const uint8_t shifts[64] = {
         0, 2, 4, 6, 8,10,12,14, 16,18,20,22,24,26,28,30,
        32,34,36,38,40,42,44,46, 48,50,52,54,56,58,60,62,
         0, 2, 4, 6, 8,10,12,14, 16,18,20,22,24,26,28,30,
        32,34,36,38,40,42,44,46, 48,50,52,54,56,58,60,62,
    };
    const __m512i ctrl = _mm512_load_si512((const void *) shifts);
    const __m512i m3 = _mm512_set1_epi8(3);
    const __m256i packed = _mm256_loadu_si256((const __m256i *) qs);
    const __m512i all = _mm512_broadcast_i64x4(packed);
    const __m512i idx01 = _mm512_setr_epi64(0, 0, 0, 0, 1, 1, 1, 1);
    const __m512i idx23 = _mm512_setr_epi64(2, 2, 2, 2, 3, 3, 3, 3);
    const __m512i src01 = _mm512_permutexvar_epi64(idx01, all);
    const __m512i src23 = _mm512_permutexvar_epi64(idx23, all);
    *w01 = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, src01), m3);
    *w23 = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctrl, src23), m3);
#else
    // Preserve the existing AVX-512 VNNI expansion on CPUs without VBMI.
    const __m512i m3 = _mm512_set1_epi32(0x03030303);
    const __m256i packed = _mm256_loadu_si256((const __m256i *) qs);
    const __m512i v01 = _mm512_cvtepu8_epi32(_mm256_castsi256_si128(packed));
    const __m512i v23 = _mm512_cvtepu8_epi32(_mm256_extracti128_si256(packed, 1));
    const __m512i r01 = _mm512_or_si512(_mm512_or_si512(v01, _mm512_slli_epi32(v01, 6)),
                                        _mm512_or_si512(_mm512_slli_epi32(v01, 12), _mm512_slli_epi32(v01, 18)));
    const __m512i r23 = _mm512_or_si512(_mm512_or_si512(v23, _mm512_slli_epi32(v23, 6)),
                                        _mm512_or_si512(_mm512_slli_epi32(v23, 12), _mm512_slli_epi32(v23, 18)));
    *w01 = _mm512_and_si512(r01, m3);
    *w23 = _mm512_and_si512(r23, m3);
#endif
}'''

OLD_DOT = r'''                    const __m512i qa  = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) a_blk->qs));
                    const __m512i sq  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, qa);
                    const __m512i i01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, qa);
                    const __m512i i23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, qa);

                    // signed dot partials: dot(codes, qy) - sum(qy), codes-1 in {-1,0,1,2}
                    const __m512 f01 = _mm512_cvtepi32_ps(_mm512_sub_epi32(i01, sq));
                    const __m512 f23 = _mm512_cvtepi32_ps(_mm512_sub_epi32(i23, sq));'''

NEW_DOT = r'''                    const __m512i qa  = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *) a_blk->qs));
                    const __m512i sq  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, qa);
                    const __m512i negsq = _mm512_sub_epi32(_mm512_setzero_si512(), sq);
                    const __m512i i01 = _mm512_dpbusd_epi32(negsq, w01, qa);
                    const __m512i i23 = _mm512_dpbusd_epi32(negsq, w23, qa);

                    // Seed VNNI with -sum(qy): dot(codes, qy)-sum(qy).
                    const __m512 f01 = _mm512_cvtepi32_ps(i01);
                    const __m512 f23 = _mm512_cvtepi32_ps(i23);'''

OLD_DOT_GEMM = r'''                            const __m512i qa  = _mm512_permutex2var_epi64(qa_lo, rowidx[m], qa_hi);
                            const __m512i sq  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, qa);
                            const __m512i i01 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w01, qa);
                            const __m512i i23 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w23, qa);

                            const __m512 f01 = _mm512_cvtepi32_ps(_mm512_sub_epi32(i01, sq));
                            const __m512 f23 = _mm512_cvtepi32_ps(_mm512_sub_epi32(i23, sq));'''

NEW_DOT_GEMM = r'''                            const __m512i qa  = _mm512_permutex2var_epi64(qa_lo, rowidx[m], qa_hi);
                            const __m512i sq  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, qa);
                            const __m512i negsq = _mm512_sub_epi32(_mm512_setzero_si512(), sq);
                            const __m512i i01 = _mm512_dpbusd_epi32(negsq, w01, qa);
                            const __m512i i23 = _mm512_dpbusd_epi32(negsq, w23, qa);

                            const __m512 f01 = _mm512_cvtepi32_ps(i01);
                            const __m512 f23 = _mm512_cvtepi32_ps(i23);'''


def replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one exact match, found {count}")
    return text.replace(old, new, 1)


def replace_within(text: str, start: str, end: str | None, old: str, new: str, label: str) -> str:
    first = text.find(start)
    if first < 0:
        raise SystemExit(f"{label}: start boundary not found")
    last = len(text) if end is None else text.find(end, first + len(start))
    if last < 0:
        raise SystemExit(f"{label}: end boundary not found")
    body = replace_exact(text[first:last], old, new, label)
    return text[:first] + body + text[last:]


def patch(path: Path) -> None:
    text = path.read_text()
    if "_mm512_multishift_epi64_epi8" in text:
        raise SystemExit("candidate already applied")
    text = replace_exact(text, OLD_HELPER, NEW_HELPER, "Q2_0 expand helper")
    text = replace_within(text, GEMV_START, GEMM_START, OLD_DOT, NEW_DOT, "Q2_0 GEMV correction")
    text = replace_within(text, GEMM_START, None, OLD_DOT_GEMM, NEW_DOT_GEMM, "Q2_0 GEMM correction")
    if text.count("_mm512_multishift_epi64_epi8") != 2:
        raise SystemExit("VBMI insertion invariant failed")
    if text.count("const __m512i negsq") < 2:
        raise SystemExit("seeded VNNI insertion invariant failed")
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    patch(args.source)


if __name__ == "__main__":
    main()
