#!/usr/bin/env python3
"""Add a direct Q2_0(g128) x Q8_1 integer-dot Vulkan decode path.

The pinned Prism runtime already supports Q2_0 through the generic dequantize +
float matvec shaders. This candidate wires Q2_0 into the packed integer-dot
MMVQ path used by other quant types. The Q2_0 code values are unsigned
{0,1,2,3}; the represented value is code-1, so the shader subtracts one quarter
of the Q8_1 block sum in each of the four participating lanes.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch_types(path: Path) -> None:
    text = path.read_text()
    old = """#if defined(DATA_A_Q2_0)
#define QUANT_K QUANT_K_Q2_0
#define QUANT_R QUANT_R_Q2_0
#define QUANT_AUXF 1
#define A_TYPE block_q2_0
#endif"""
    new = """#if defined(DATA_A_Q2_0)
#define QUANT_K QUANT_K_Q2_0
#define QUANT_R QUANT_R_Q2_0
#define QUANT_AUXF 1
#define A_TYPE block_q2_0
#define DATA_A_QUANT_LEGACY
#endif"""
    path.write_text(replace_once(text, old, new, "types Q2_0 legacy marker"))


def patch_shader_funcs(path: Path) -> None:
    text = path.read_text()
    old_cond = "#if defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0)"
    new_cond = "#if defined(DATA_A_Q2_0) || defined(DATA_A_Q4_0) || defined(DATA_A_Q5_0) || defined(DATA_A_Q8_0)"
    text = replace_once(text, old_cond, new_cond, "Q2_0 get_dm condition")

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
    path.write_text(text)


def patch_generator(path: Path) -> None:
    text = path.read_text()
    old = 'return type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";'
    new = 'return type_name == "q2_0" || type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";'
    path.write_text(replace_once(text, old, new, "generator legacy quant list"))


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
    path.write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Prism llama.cpp checkout")
    args = parser.parse_args()
    root = args.root
    patch_types(root / "ggml/src/ggml-vulkan/vulkan-shaders/types.glsl")
    patch_shader_funcs(root / "ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_funcs.glsl")
    patch_generator(root / "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp")
    patch_backend(root / "ggml/src/ggml-vulkan/ggml-vulkan.cpp")


if __name__ == "__main__":
    main()
