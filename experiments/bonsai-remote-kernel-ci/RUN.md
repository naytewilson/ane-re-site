# Bonsai remote kernel CI

This branch is an isolated compute lane. It pins PrismML-Eng/llama.cpp at `9ca265a57f85f2117942490f421f64a226dd9847`, validates the Q2_0 reduce-once candidate, builds the runtime, runs tests, and executes a matched pristine-versus-candidate ternary benchmark. Nothing in this branch is intended for the public site or for merge into `main`.
