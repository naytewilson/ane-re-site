#!/usr/bin/env python3
"""Add a Q2_0 integer-dot Vulkan GEMV path to a pinned Prism llama.cpp tree.

The patch is intentionally narrow:
- mark Q2_0 as legacy-like for mul_mat_vecq.comp's K_PER_ITER=8 path;
- generate Q2_0 q8_1 integer-dot shader variants;
- decode four packed 2-bit codes into one packed int32;
- apply the Q2_0 zero-point correction using the q8_1 block sum.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{path}: expected one match, found {count}: {old!r}")
    path.write_text(text.replace(old, new, 1))


def patch(root: Path) -> None:
    types = root / "ggml/src/ggml-vulkan/vulkan-shaders/types.glsl"
    funcs = root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl"
    gen = root / "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"

    replace_once(
        types,
        """#if defined(DATA_A_Q2_0)\n#define QUANT_K QUANT_K_Q2_0\n#define QUANT_R QUANT_R_Q2_0\n#define QUANT_AUXF 1\n#define A_TYPE block_q2_0\n#endif\n""",
        """#if defined(DATA_A_Q2_0)\n#define QUANT_K QUANT_K_Q2_0\n#define QUANT_R QUANT_R_Q2_0\n#define QUANT_AUXF 1\n#define A_TYPE block_q2_0\n#define DATA_A_QUANT_LEGACY\n#endif\n""",
    )

    replace_once(
        funcs,
        """#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)\nFLOAT_TYPEV2 get_dm(uint ib) {\n""",
        """#if defined(DATA_A_Q2_0)\n// mul_mat_vecq indexes Q2_0 as four virtual 32-value blocks per physical\n// 128-value block so it aligns with one Q8_1 activation block.\nFLOAT_TYPE get_dm(uint ib) {\n    return FLOAT_TYPE(data_a[ib / 4].d);\n}\n#endif\n\n#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)\nFLOAT_TYPEV2 get_dm(uint ib) {\n""",
    )

    q2_block = r'''#if defined(DATA_A_Q2_0)
// Each physical byte contains four sequential 2-bit codes. Expand them into
// four unsigned bytes for dotPacked4x8EXT. The encoded values are {0,1,2,3};
// the model values are code-1, corrected below with the Q8_1 block sum.
int32_t repack(uint ib, uint iqs) {
    const uint block = ib / 4;
    const uint chunk = ib % 4;
    const uint v = uint(data_a[block].qs[chunk * 8 + iqs]);
    return int32_t((v & 0x03u)
                 | ((v & 0x0cu) << 6)
                 | ((v & 0x30u) << 12)
                 | ((v & 0xc0u) << 18));
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    // Four invocations cover one 32-value Q8_1 block. Each invocation owns
    // one quarter of the zero-point correction, so the reduced workgroup
    // subtracts exactly 1 * sum(q8) for code-1.
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - dsb.y / float(sum_divisor)));
}
#endif

'''
    replace_once(
        funcs,
        "// Each iqs value maps to a 32-bit integer\n#if defined(DATA_A_Q4_0)\n",
        "// Each iqs value maps to a 32-bit integer\n" + q2_block + "#if defined(DATA_A_Q4_0)\n",
    )

    replace_once(
        gen,
        'if (is_legacy_quant(tname) || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m") {',
        'if (is_legacy_quant(tname) || tname == "q2_0" || tname == "mxfp4" || is_k_quant(tname) || tname == "iq1_s" || tname == "iq1_m") {',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    patch(args.root)


if __name__ == "__main__":
    main()
