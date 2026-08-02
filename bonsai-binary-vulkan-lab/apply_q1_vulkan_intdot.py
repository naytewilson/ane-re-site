#!/usr/bin/env python3
"""Add a direct Q1_0 x Q8_1 integer-dot Vulkan decode path.

Q1_0 stores one sign bit per weight. Packed bits are expanded to byte lanes
containing {0,1}, then the shader evaluates 2*dot(bits,q8)-sum(q8), which is
exactly dot(sign,q8) for sign in {-1,+1}.
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
    new = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4) || defined(DATA_A_Q1_0)\n#define K_PER_ITER 8"
    path.write_text(replace_once(text, old, new, "Q1_0 K_PER_ITER selection"))


def patch_shader_funcs(path: Path) -> None:
    text = path.read_text()
    marker = "#if defined(DATA_A_Q4_0)\n// 2-byte loads for Q4_0 blocks (18 bytes)"
    if text.count(marker) != 1:
        raise SystemExit(f"Q4_0 insertion marker: expected one match, found {text.count(marker)}")

    q1 = r'''#if defined(DATA_A_Q1_0)
// Q1_0: ib addresses a 32-weight Q8_1-sized segment. Four segments share
// one 128-weight Q1_0 block. Each repack returns four {0,1} byte lanes.
int32_t repack(uint ib, uint iqs) {
    const uint ib_q1 = ib / 4;
    const uint segment = ib % 4;
    const uint byte_val = uint(data_a[ib_q1].qs[segment * 4 + iqs / 2]);
    const uint shift = (iqs & 1u) * 4u;
    const uint nibble = (byte_val >> shift) & 0x0fu;
    return int32_t(((nibble >> 0u) & 1u)       |
                  (((nibble >> 1u) & 1u) <<  8) |
                  (((nibble >> 2u) & 1u) << 16) |
                  (((nibble >> 3u) & 1u) << 24));
}

FLOAT_TYPE get_dm(uint ib) {
    return FLOAT_TYPE(data_a[ib / 4].d);
}

FLOAT_TYPE mul_q8_1(const int32_t q_sum, const float da, const vec2 dsb, const int32_t sum_divisor) {
    // Across four reduced lanes this subtracts the complete Q8_1 sum.
    return FLOAT_TYPE(da * (2.0f * float(q_sum) * dsb.x - dsb.y / float(sum_divisor)));
}
#endif

'''
    text = text.replace(marker, q1 + marker, 1)

    old_dispatch = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4)\nFLOAT_TYPE mmvq_dot_product"
    new_dispatch = "#if defined(DATA_A_QUANT_LEGACY) || defined(DATA_A_MXFP4) || defined(DATA_A_Q1_0)\nFLOAT_TYPE mmvq_dot_product"
    text = replace_once(text, old_dispatch, new_dispatch, "Q1_0 MMVQ dispatch")
    path.write_text(text)


def patch_generator(path: Path) -> None:
    text = path.read_text()
    old = 'return type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";'
    new = 'return type_name == "q1_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";'
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
    q1_normal = normal_line.replace("GGML_TYPE_Q4_0", "GGML_TYPE_Q1_0").replace("q4_0", "q1_0")
    lines.insert(normal[0], q1_normal)

    ident = [i for i, line in enumerate(lines) if id_marker in line]
    id_line = lines[ident[0]]
    q1_id = id_line.replace("GGML_TYPE_Q4_0", "GGML_TYPE_Q1_0").replace("q4_0", "q1_0")
    lines.insert(ident[0], q1_id)
    text = "".join(lines)

    selector = """if (b_type == GGML_TYPE_Q8_1) {
        switch (a_type) {
            case GGML_TYPE_Q4_0:"""
    selected = """if (b_type == GGML_TYPE_Q8_1) {
        switch (a_type) {
            case GGML_TYPE_Q1_0:
            case GGML_TYPE_Q4_0:"""
    count = text.count(selector)
    if count != 2:
        raise SystemExit(f"Q8_1 vector selector markers: expected two, found {count}")
    text = text.replace(selector, selected)
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    patch_shader_main(root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq.comp")
    patch_shader_funcs(root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl")
    patch_generator(root / "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp")
    patch_backend(root / "ggml/src/ggml-vulkan/ggml-vulkan.cpp")


if __name__ == "__main__":
    main()
