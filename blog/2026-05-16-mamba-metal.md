---
title: "#3 mamba-metal: running Mamba on Apple Silicon"
slug: 3
date: 2026-05-16
authors: [createcentury]
tags: [ml, ssm, cuda, metal]
---

Mamba's ([state-spaces/mamba](https://github.com/state-spaces/mamba)) selective scan is written as a CUDA kernel and doesn't run on Apple Silicon out of the box. I rewrote it in Metal Shading Language (MSL), wired it up to load HuggingFace weights directly, and got end-to-end inference working — that's [mamba-metal](https://github.com/createcentury/mamba-metal). This post is the design + results notebook.

{/* truncate */}

## Motivation

The heart of Mamba's official implementation is the CUDA kernel at `csrc/selective_scan/selective_scan_fwd_kernel.cuh`. It's what turns Mamba from "paper architecture" into "fast running model". Being NVIDIA-only, it doesn't move to Apple Silicon.

The pure-PyTorch reference (`selective_scan_ref`) does exist, but it's a naïve for-loop evaluation of the recurrence — fine for small toy cases, way too slow for real sequences. The whole point of Mamba (parallel prefix scan) is missing from it.

So: write the equivalent kernel in MSL. JIT-compile and dispatch through MLX's `mx.fast.metal_kernel`, keeping the `.metal` files as first-class artefacts.

## The essence of selective scan

Mamba's hidden state evolves as

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, u_t,\qquad y_t = C_t^\top h_t
$$

an input-dependent recurrence. $\bar{A}_t = \exp(\Delta_t A)$, and $\Delta_t, B_t, C_t$ are computed from the input $x_t$ — that's the "selective" part (the gate opens and closes based on the input).

A recurrence $h_t = a_t h_{t-1} + b_t$ is naturally sequential, but the pair composition

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\ a_2 b_1 + b_2)
$$

is **associative**, so a prefix scan runs it in $O(\log T)$ parallel steps (Blelloch 1990 / Martin & Cundy 2017). That is exactly what Mamba's kernel does.

## The core in MSL

### SIMD-group primitives

Metal has a **SIMD-group** of 32 threads (Apple's equivalent of a CUDA warp), with built-ins like `simd_prefix_inclusive_sum` and `simd_shuffle_up`. These are float-scalar-only, so the pair composition has to be written by hand:

```metal
for (uint d = 1u; d < 32u; d <<= 1) {
    float a_prev = simd_shuffle_up(a, d);
    float b_prev = simd_shuffle_up(b, d);
    if (lane >= d) {
        b = a * b_prev + b;   // order matters: update b first (uses old a)
        a = a * a_prev;
    }
}
```

That's a full inclusive scan within one 32-lane SIMD-group.

### Block-level scan (two-tier)

To scan over an entire threadgroup of 1024 threads (= 32 SIMD-groups × 32 lanes), each SIMD-group writes its total into threadgroup memory; the first SIMD-group then scans the 32 group totals; finally each thread adds the carry. This is the MSL analogue of CUB's `BlockScan + WARP_SCANS`.

### Inter-chunk running prefix

When `seqlen > 1024` we chunk. Each SSM state index $s$ keeps a `(carry_a[s], carry_b[s])` pair in threadgroup memory across chunks. At the start of a new chunk the previous chunk's accumulated pair is composed in from the left:

$$
(a, b)_\text{new} = (a_\text{local}, b_\text{local}) \circ (\text{carry}_a, \text{carry}_b)
$$

This lets a single kernel launch handle sequences of any length.

### Observation: tg memory isn't a win for simple reuse

The classic "load data into threadgroup memory and read it K times" pattern barely helped: Apple Silicon's System Level Cache (shared between CPU and GPU) absorbs the reuse for you. Where threadgroup memory is genuinely needed is **inter-thread communication** — scan intermediates, running prefix — not as a data cache.

## From kernel to inference

Once the kernel works, stack the Python model on top:

```
selective_scan (Metal kernel)
  ↓
MambaBlock           = in_proj → conv1d → SiLU → x_proj/dt_proj → SSM → out_proj
  ↓
MambaResidualBlock   = pre-norm RMSNorm + MambaBlock + residual
  ↓
MambaModel           = embeddings → N × ResidualBlock → norm_f → tied LM head
  ↓
generate / generate_fast
```

Loading HF `state-spaces/mamba-*-hf` weights into MLX takes only two transforms:

1. Strip the `backbone.` prefix from every key
2. Transpose `conv1d.weight` from PyTorch `(out, in/g, k)` to MLX `(out, k, in/g)`

Everything else (Linear, embeddings, A_log, D, norm) goes straight into `mx.array`. One catch: prefer the HF transformers-standard field names (`hidden_size`, `intermediate_size`, `num_hidden_layers`); the legacy `d_model` field is broken in some checkpoints (e.g. 790m).

## O(L) incremental decoding

Mamba's headline property is "constant per-token cost at long context". To realise it in practice you need to carry the SSM hidden state and the conv1d sliding window across decode calls:

```python
conv_states, ssm_states = model.init_state(batch_size=1)
for token in prompt:
    logits, conv_states, ssm_states = model.step(token, conv_states, ssm_states)
# from here, every new token is O(1)
```

Each step is just elementwise math — no scan, because the state is already maintained:

$$
h_\text{new}^{(s)} = \exp(\Delta_t A_s) \cdot h^{(s)} + \Delta_t \cdot x_t \cdot B_{s,t},\qquad
y_t = \sum_s h_\text{new}^{(s)} \cdot C_{s,t} + D \cdot x_t
$$

followed by the z gate and out_proj.

Measured (M4 Max, mamba-130m, greedy decode):

| new tokens | O(L²) re-forward | **O(L) `generate_fast`** | speedup |
|---:|---:|---:|---:|
| 10  | 0.24 s | **0.06 s** | 4.3× |
| 100 | 3.24 s | **0.51 s** | 6.3× |
| 1000 | ~32 s (extrapolated) | **6.84 s** | ~5× |
| 2000 | ~80 s (extrapolated) | **14.08 s** | ~6× |

`generate_fast` holds at **~7 ms/token from n=50 to n=2000**. That's what "linear-time decode" actually looks like.

## Across model sizes

All five `state-spaces/mamba-*-hf` checkpoints load and generate:

| model | params | load (s) | tok/s | ms/tok | continuation of "The capital of Japan is" |
|---|---:|---:|---:|---:|---|
| 130m | 129 M | 1.3 | 175 | 5.7 | Tokyo, Japan. The city is located in the northern part of the country… |
| 370m | 372 M | 3.4 | 82 | 12.2 | Tokyo. (repeats) |
| 790m | 702 M | 4.8 | 42 | 23.7 | Tokyo, and the capital of the country is Osaka. (factually mixed) |
| 1.4b | 1372 M | 11.6 | 30 | 33.2 | Tokyo. … Washington, D.C. … London. |
| **2.8b** | **2.7 B** | **19.6** | **12** | **80.6** | **"Tokyo, which is also the largest city in the country"** (clean) |

130m loops at this scale, but 2.8b already volunteers the "largest city" fact — all greedy, just from scaling.

Kernel-level peaks for the selective scan alone: **~187 GFLOPS** at `seqlen=32k`, with vec4 loads measuring **~290 GB/s** on the Unified Memory bus (≈ 70% of the M4 Max's 410 GB/s theoretical peak).

## What's left

- **Faster prefill.** Right now prompts are walked through one token at a time via `step`, which costs seconds at long context. If the selective_scan kernel exposes the final SSM state, prefill can run the parallel scan once and hand the state to the O(1)-per-token decode
- **iPhone Transformer vs Mamba benchmark.** Compare like-for-like on a phone, plot the long-context advantage
- **Backward kernel.** Still missing — training would need it

## Reflection

Mamba's equations are short on paper. The "linear-time at practical speed" claim, though, lives in the kernel. Porting it to another piece of hardware is where the details actually become legible:

- What belongs in SRAM and what belongs in HBM (and how that shifts when the platform has caches that absorb naïve reuse)
- Why $A$ has to be diagonal (per-state independence is what lets the state loop sit on the outside)
- Why `exp2f + LOG2E` is preferred over `expf` (a small win that multiplies across every step)
- The actual demonstration that state caching makes decode $O(L)$ rather than just claiming it

Reading the paper alone you miss the resolution. Writing the kernel gives it to you.

---

## References

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.
- [state-spaces/mamba](https://github.com/state-spaces/mamba) — official implementation
- [createcentury/mamba-metal](https://github.com/createcentury/mamba-metal) — the project from this post

---

*Created: 2026-05-16 / Updated: 2026-05-18*
