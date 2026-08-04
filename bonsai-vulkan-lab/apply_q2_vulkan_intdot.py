#!/usr/bin/env python3
"""Add a direct Q2_0(g128) x Q8_1 integer-dot Vulkan decode path.

The pinned Prism runtime already supports Q2_0 through generic dequantize +
float matvec shaders. This candidate adds a packed integer-dot MMVQ path only;
it deliberately does not classify Q2_0 as a generic legacy quant globally,
so the existing float, dequantize, copy, and matmul shader paths remain intact.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_shader_main(path: Path) -> None:
    text = path.read_text()
    old = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4)\n#define K_PER_ITER 8"
    new = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4) || defined(DATA_A_Q2_0)\n#define K_PER_ITER 8"
    path.write_text(replace_once(text, old, new, "Q2_0 K_PER_ITER selection"))


def patch_shader_funcs(path: Path) -> None:
    text = path.read_text()
    marker = "#if defined(DATA_A_Q4_0)\n// 2-byte loads for Q4_0 blocks (18 bytes)"
    if text.count(marker) != 1:
        raise SystemExit(f"Q4_0 insertion marker: expected one match, found {text.count(marker)}")

    q2 = r'''#if defined(DATA_A_Q2_0)
// Q2_0 g128: ib addresses a 32-weight Q8_1-sized segment. Four such
// segments share one 128-weight Q2_0 block. Each source byte stores four
// unsigned 2-bit codes; represented values are code - 1.
int32_t repack(uint ib, uint iqs) {
    const uint ib_q2 = ib / 4;
    const uint segment = ib % 4;
    const uint byte_val = uint(data_a[ib_q2].qs[segment * 8 + iqs]);
    return int32_t(( byte_val        & 3u)       |
                  (((byte_val >> 2u) & 3u) <<  8) |
                  (((byte_val >> 4u) & 3u) << 16) |
                  (((byte_val >> 6u) & 3u) << 24));
}

FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(data_a[ib / 4].d);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    // Four lanes cover one 32-value Q8_1 block. Across their reduction this
    // subtracts exactly dsb.y, implementing dot(code - 1, q8).
    return FLOAT_TYPE(da * (float(q_sum) * dsb.x - dsb.y / float(sum_divisor)));
}
#endif

'''
    text = text.replace(marker, q2 + marker, 1)

    old_dispatch = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4)\nFLOAT_TYPE mmvq_dot_product"
    new_dispatch = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4) || defined(DATA_A_Q2_0)\nFLOAT_TYPE mmvq_dot_product"
    text = replace_once(text, old_dispatch, new_dispatch, "Q2_0 MMVQ dispatch")
    path.write_text(text)


def patch_generator(path: Path) -> None:
    text = path.read_text()
    old = 'return type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";'
    new = 'return type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";'
    path.write_text(replace_once(text, old, new, "generator integer-dot quant list"))


def patch_backend(path: Path) -> None:
    text = path.read_text()
    normal_marker = 'ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_q8_1_f32[w][GGML_TYPE_Q4_0][i], "mul_mat_vec_q4_0_q8_1_f32"'
    id_marker = 'ggml_vk_create_pipeline(device, device->pipeline_dequant_mul_mat_vec_id_q8_1_f32[w][GGML_TYPE_Q4_0], "mul_mat_vec_id_q4_0_q8_1_f32"'

    lines = text.splitlines(keepends=True)
    normal = [i for i, line in enumerate(lines) if normal_marker in line]
    ident = [i for i, line in enumerate(lines) if id_marker in line]
    if len(normal) != 1 or len(ident) != 1:
        raise SystemExit(f"backend markers: normal={len(normal)} id={len(ident)}")

    normal_line = lines[normal[0]]
    q2_normal = normal_line.replace("GGML_TYPE_Q4_0", "GGML_TYPE_Q2_0").replace("q4_0", "q2_0")
    lines.insert(normal[0], q2_normal)

    ident = [i for i, line in enumerate(lines) if id_marker in line]
    id_line = lines[ident[0]]
    q2_id = id_line.replace("GGML_TYPE_Q4_0", "GGML_TYPE_Q2_0").replace("q4_0", "q2_0")
    lines.insert(ident[0], q2_id)
    text = "".join(lines)

    # The generic vector-pipeline selectors explicitly whitelist the types
    # allowed to use Q8_1 activations. Add Q2_0 to both the normal and ID
    # selectors; otherwise the new pipeline is compiled but never selected.
    selector = """if (b_type == GGML_TYPE_Q8_1) {
        switch (a_type) {
            case GGML_TYPE_Q4_0:"""
    selected = """if (b_type == GGML_TYPE_Q8_1) {
        switch (a_type) {
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_Q4_0:"""
    count = text.count(selector)
    if count != 2:
        raise SystemExit(f"Q8_1 vector selector markers: expected two, found {count}")
    text = text.replace(selector, selected)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Prism llama.cpp checkout")
    args = parser.parse_args()
    root = args.root
    patch_shader_main(root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp")
    patch_shader_funcs(root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl")
    patch_generator(root / "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp")
    patch_backend(root / "ggml/src/ggml-vulkan/ggml-vulkan.cpp")


if __name__ == "__main__":
    main()
