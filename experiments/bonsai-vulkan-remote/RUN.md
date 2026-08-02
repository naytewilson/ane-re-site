# Bonsai Vulkan Q2_0 remote validation

Pinned Prism runtime: `9ca265a57f85f2117942490f421f64a226dd9847`.

This branch is an isolated compute lane. It validates the Q2_0 integer-dot decode candidate by compiling all generated Vulkan shader variants, running exhaustive CPU-side shader simulation, exercising the Vulkan backend through Lavapipe, and loading the complete verified 27B GGUF. It must not be merged into the site branch.
