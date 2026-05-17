---
title: "#1 mamba"
slug: 1
date: 2026-05-11
authors: [createcentury]
tags: [ml, ssm, cuda]
---

Mamba ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) is a sequence model called a **Selective State Space Model**. It's a candidate replacement for the Transformer's attention, with linear-time compute in context length and a per-token decode cost that is **independent of context length**. This post walks through the architecture, what "selective" actually means, why the recurrence can still be parallelised, and finally the official CUDA kernel [`csrc/selective_scan/selective_scan_fwd_kernel.cuh`](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh) line by line.

{/* truncate */}

## Why Mamba

The Transformer's attention scales as $O(L^2)$ in compute and (during decode) $O(L)$ in memory — the KV cache. At long context this becomes the dominant cost.

RNNs are $O(L)$ in compute, but their state is **sequentially dependent**, so they can't be parallelised across time and training is slow.

State Space Models (SSMs) sit between the two: S4, S5 and their relatives. **Training is parallel** (an LTI SSM has a convolutional form), and **inference is recurrent** (one step at a time). The catch is that classical SSMs are **linear time-invariant (LTI)**: they can't change their coefficients based on the current input, so they can't reproduce attention's content-aware behaviour.

Mamba's contribution is to make the SSM parameters **input-dependent** while keeping training parallel. That is what "**selective**" refers to.

## SSM primer

Continuous-time state equation:

$$
\dot{h}(t) = A\, h(t) + B\, x(t),\qquad y(t) = C\, h(t)
$$

Discretise with timestep $\Delta$. The standard discretisation is Zero-order hold ([ZOH](https://en.wikipedia.org/wiki/Zero-order_hold)):

$$
\bar{A} = \exp(\Delta A),\qquad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B
$$

When $A$ is **diagonal**, each state component becomes an independent scalar exponential — that's the structural benefit. Mamba's official implementation uses the approximation

$$
\bar{A}_t = \exp(\Delta_t A),\qquad \bar{B}_t \approx \Delta_t B
$$

so the discrete recurrence is

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t,\qquad y_t = C_t^\top h_t + D\, x_t
$$

— this much is shared with classical SSMs.

## What "selective" means

In an LTI model like S4, $A, B, C, \Delta$ are **learned parameters** that don't depend on time. Mamba instead defines

$$
\Delta_t = \mathrm{softplus}(W_\Delta\, x_t + b_\Delta),\quad
B_t = W_B\, x_t,\quad
C_t = W_C\, x_t
$$

so $B, C, \Delta$ are **functions of the input $x_t$** — a time-varying system. Concretely:

- Large $\Delta_t$: the state $h$ takes a **big step** ("absorb this token strongly")
- Small $\Delta_t$: the state is **almost frozen** ("ignore this token")

The essence of selectivity is **input-controlled step size**. You gain gate-like behaviour, but you lose the LTI property — the system no longer has a convolutional form and can't be parallelised via FFT. The trick instead is the **parallel scan** described below.

## The Mamba block

From the paper's Section 3.4. One Mamba layer is:

```
x ∈ ℝ^{B×L×D}                              # input
  ├── in_proj (Linear D → 2 D')
  ├── split → (x_main, z)                 # each ℝ^{B×L×D'}
  ├── x_main = SiLU(conv1d(x_main))       # causal, kernel=4
  ├── x_dbl  = x_proj(x_main)              # Linear D' → dt_rank + 2N
  ├── (dt_pre, B_ssm, C_ssm) = split(x_dbl)
  ├── dt = softplus(dt_proj(dt_pre))      # ℝ^{B×L×D'}
  ├── A  = -exp(A_log)                    # ℝ^{D'×N}, always negative
  ├── y_ssm = selective_scan(x_main, dt, A, B_ssm, C_ssm, D, z)
  └── y = out_proj(y_ssm)                 # Linear D' → D
```

Points:

- **`in_proj` doubles the channel count**; half goes down the z branch and re-enters at the end as `y *= SiLU(z)` (a SwiGLU-style gate)
- **`conv1d` is a short causal filter** (kernel=4) that mixes the immediate neighbourhood before handing off to the SSM
- **A is diagonal**, parameterised through $A_\text{log}$ so that $A = -\exp(A_\text{log})$ is always negative (the system doesn't blow up)
- **B, C, Δ are input-dependent** (selectivity). `x_proj` produces all three at once and `dt_proj` lifts Δ to the full inner dimension

A real Mamba model (e.g. `state-spaces/mamba-130m`) stacks **24–64** such layers with **pre-norm RMSNorm + residual** wrapping each one. The LM head shares weight with the token embedding.

## Mamba vs Transformer at a glance

| Quantity | Transformer | Mamba |
|---|---|---|
| Training compute | $O(L^2 D)$ (attention) | $O(L D N)$ (scan; $N$ = state dim) |
| Decode compute / token | $O(L D)$ (with KV cache) | $O(D N)$ (constant in $L$) |
| Decode memory | $O(L D)$ (KV cache, growing) | $O(D N)$ (state, constant) |
| Where input controls mixing | attention's softmax | input-dependent $B_t, C_t, \Delta_t$ |

Mamba wins as context grows. **$N \ll L$** (typically $N=16$), so the real question is how well the state compresses the past.

## The recurrence to implement

From here it's implementation-oriented. The state $h_t \in \mathbb{R}^{N}$ obeys

$$
h_t = \exp(\Delta_t A)\, h_{t-1} + (\Delta_t B_t)\, x_t,\qquad y_t = C_t^\top h_t + D\, x_t
$$

with diagonal $A$. Each state component is therefore an independent scalar recurrence $h_t = a_t h_{t-1} + b_t$. That's the starting point for the kernel.

## Why "scan"

The recurrence $h_t = a_t h_{t-1} + b_t$ is a **left fold**, but for pairs $(a, b)$ the composition

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\ a_2 b_1 + b_2)
$$

is **associative**. So you can prefix-scan it (Blelloch 1990) in $O(\log T)$ parallel steps. Martin & Cundy (2017) and S5 (Smith et al. 2022) brought this idea to linear RNNs and SSMs. Mamba's kernel realises it on the GPU with `cub::BlockScan`.

`thread_data[i]` in the source is exactly this pair:

```cpp
// L221-222
thread_data[i] = make_float2(
    exp2f(delta_vals[r][i] * A_val[r]),                 // a_i = exp(Δ A)
    !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]  // b_i = ΔB · u
);
```

`exp2f` is used because $A$ is multiplied by `LOG2E` once when loaded (L174–179) — `exp2f` is faster than `expf`.

## Kernel layout

The file has three layers:

| Role | Symbol | Lines |
|---|---|---|
| Type / template constants | `Selective_Scan_fwd_kernel_traits` | L24–70 |
| GPU kernel | `selective_scan_fwd_kernel` | L72–308 |
| Host launch | `selective_scan_fwd_launch` / `..._cuda` | L310–376 |

### Thread / block layout

```cpp
// L322
dim3 grid(params.batch, params.dim / kNRows);
```

One CUDA block = `(batch_id, dim_id)`. Each block reads

- input $u, \Delta \in \mathbb{R}^{T}$ (one channel)
- weights $A \in \mathbb{R}^{N}$, input-dependent $B, C \in \mathbb{R}^{N \times T}$

and produces $y \in \mathbb{R}^{T}$. `kNThreads` is picked based on `seqlen`:

```cpp
// L353-364
if (params.seqlen <= 128)  launch<32,  4>();
else if (seqlen <= 256)    launch<32,  8>();
else if (seqlen <= 512)    launch<32, 16>();
else if (seqlen <= 1024)   launch<64, 16>();
else                       launch<128, 16>();
```

For short sequences too many threads is just overhead, so the dispatch is tuned.

### Chunking

The number of tokens one block iteration covers is

$$
\text{kChunkSize} = \text{kNThreads} \times \text{kNItems}
$$

— at most 2048 (128×16). For longer sequences the block loops:

```cpp
// L137
for (int chunk = 0; chunk < params.n_chunks; ++chunk) { ... }
```

State is carried across chunks via `smem_running_prefix` (L100, L244–247, L257–258): the last scan prefix is saved in shared memory and read at the start of the next chunk. This avoids ever spilling the state back through HBM — the **hardware-aware** part of the paper's title.

## One chunk in pseudo-code

```text
1. load_input: read u, delta coalesced
2. apply delta_softplus → delta_vals
3. delta_u_vals = delta * u, out_vals = D * u (skip connection)
4. for state_idx in [0, dstate):
     a. read A_val (already multiplied by LOG2E)
     b. read B_val, C_val (BlockLoad if selective, direct if constant)
     c. thread_data = (exp2f(Δ A), ΔB u)  ← scan tuple
     d. cub::BlockScan InclusiveScan(SSMScanOp)
        → carry running_prefix
     e. out_vals += scan_output.y * C
5. store_output: write y
6. (optional) if kHasZ: out *= z * sigmoid(z)  ← SwiGLU-style gate
```

The state dimension `dstate` ($N$) is the **outer** loop. The parallel scan runs along the **time axis**; state is sequential. This works because $A$ is diagonal — each state component is independent (the upside of a diagonal SSM).

## Shared-memory budget

`Selective_Scan_fwd_kernel_traits::kSmemSize` (L63–69) sums:

- BlockLoad / Store TempStorage (reused as a union)
- BlockScan TempStorage

The kernel then appends `MAX_DSTATE * sizeof(scan_t) * kNRows` for the running-prefix area:

```cpp
// L321
kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(scan_t);
```

If this exceeds 48 KB, `cudaFuncSetAttribute` raises the dynamic shared-memory limit (L331–340).

## Optimisations worth pointing out

- **`exp2f` + LOG2E preprocessing**: `exp2f` is faster than `expf`; multiply `A` by `LOG2E` once and the per-step cost drops
- **WARP_TRANSPOSE BlockLoad**: warp-level transpose to coalesce strided accesses
- **WARP_SCANS BlockScan**: warp-level parallel scan (faster than RAKING; comment L60–61 leaves the alternatives in)
- **`kIsEvenLen` branch**: when seqlen divides the chunk size, switch to `BLOCK_LOAD_DIRECT` (L47–59)
- **Custom `cexp2f`**: PyTorch's `thrust::complex_exp` is slow, so a hand-rolled version is used (L229)
- **`kIsVariableB/C` compile-time branches**: in the LTI case the extra BlockLoads are removed entirely (L186–212)
- **`__launch_bounds__`**: `kMinBlocks=3 or 5` pins occupancy (L33, L73)

## Observations and open questions

- Only `kNRows == 1` has been validated in practice (L312–314). Processing multiple dims per block to reuse loads is left on the table
- The `delta_softplus` cutoff at `<= 20.f` (L160) is just an overflow guard
- `MAX_DSTATE` is defined in `selective_scan.h` (would need to read that to know the cap on state dim)

The Mamba **architecture** itself is just SSM + selectivity. But the paper's claim of "linear-time at practical speed" only holds once you combine the kernel's **chunking × associative scan × in-SRAM state** — and the file analysed above is where those three meet.

---

## References

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" Technical Report CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.
- Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman. "[Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)" arXiv:2208.04933, 2022.
- Wikipedia. "[Leaky integrator](https://en.wikipedia.org/wiki/Leaky_integrator)"
- Wikipedia. "[Zero-order hold](https://en.wikipedia.org/wiki/Zero-order_hold)"

---

*Created: 2026-05-11 / Updated: 2026-05-18*
