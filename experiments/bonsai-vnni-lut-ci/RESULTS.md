# Bonsai Q2_0 VNNI LUT candidate

Pinned runtime: `PrismML-Eng/llama.cpp@9ca265a57f85f2117942490f421f64a226dd9847`.

The candidate replaces the existing multiply/shift 2-bit unpack sequence in the AVX-512 VNNI path with a 16-entry nibble lookup, retains eight integer dot-product lanes through the row, and performs one final horizontal reduction. The existing direct AVX2 path is retained for non-VNNI hosts.

## Remote sandbox microbenchmark

Host: AMD EPYC 9V74 VM, 5 exposed CPUs, AVX-512F/BW/VL/VNNI.

Command:

```bash
cc -O3 -march=icelake-server -mavx512vnni -mavx512vl -mavx512bw -mavx512f -mfma \
  q2_lut_row_bench.c -lm -o q2_lut_row_bench
```

Five independent median-of-nine rotated-order executions all passed scalar, multiply-unpack, and LUT-unpack correctness. LUT speedup over the already improved global-lane multiply path:

```text
1.277x
1.258x
1.269x
1.275x
1.269x
```

Median: `1.269x` over global-lane multiply unpack. Median speedup over the original per-chunk-horizontal-reduction implementation was approximately `1.98x` in this isolated row kernel.

These are microkernel results, not an end-to-end token-generation claim. Full-model validation is performed separately with the exact 7,165,121,600-byte GGUF on an AVX-512 VNNI runtime.
