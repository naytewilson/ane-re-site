# Bonsai remote kernel CI

This branch is an isolated compute lane. It pins PrismML-Eng/llama.cpp at `9ca265a57f85f2117942490f421f64a226dd9847`, validates the combined Q2_0 AVX2 and VNNI candidate, builds native and AVX-512 compile-only runtimes, runs correctness tests, and executes a matched pristine-versus-candidate ternary benchmark. Nothing in this branch is intended for the public site or for merge into `main`.

Validation trigger: 2026-08-02 combined-x86-candidate-1.
