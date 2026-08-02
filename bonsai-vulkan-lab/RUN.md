# Bonsai Q2_0 Vulkan integer-dot experiment

Pinned runtime: `PrismML-Eng/llama.cpp@9ca265a57f85f2117942490f421f64a226dd9847`.

This lane adds and validates a direct packed Q2_0(g128) x Q8_1 integer-dot decode shader, compiles the Vulkan runtime, runs shader and backend correctness tests on Lavapipe, and attempts a one-token full-model smoke test. It is experimental and must not be merged into the public site.

Validation trigger: 2026-08-02 q2-g128-intdot-1.
