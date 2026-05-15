---
title: "#1 mamba"
slug: 1
date: 2026-05-11
authors: [createcentury]
tags: [ml, ssm, cuda]
---

Mamba の中核は **selective scan** という線形時間の系列演算。論文だけ読むと「ハードウェア意識した実装」で済まされてしまう部分を、公式実装の forward カーネル [`csrc/selective_scan/selective_scan_fwd_kernel.cuh`](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh) を読んで分解する。

{/* truncate */}

## 動かす漸化式

Mamba の隠れ状態 $h_t \in \mathbb{R}^{N}$ は、入力依存の係数で動く線形時不変ではない SSM：

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t,\qquad y_t = C_t^\top h_t + D\, x_t
$$

$\Delta_t$ をステップサイズとして、連続系から離散化する典型は ZOH だが、Mamba 公式実装は近似的に

$$
\bar{A}_t = \exp(\Delta_t A),\qquad \bar{B}_t \approx \Delta_t B
$$

を使う（diagonal $A$ の各成分について scalar exp）。$\Delta_t, B, C$ は入力 $x_t$ から線形写像 + softplus などで都度作る — これが「**selective**」 (入力に応じてゲートが開閉する)。

## なぜ「scan」か

漸化式 $h_t = a_t h_{t-1} + b_t$ は **左畳み込み**だが、ペア $(a, b)$ に対する次の演算

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\ a_2 b_1 + b_2)
$$

は**結合的** (associative)。よって prefix scan (Blelloch 1990) で $O(\log T)$ 段の並列ステップに落とせる。Martin & Cundy (2017) と S5 (Smith et al. 2022) はこの観察を線形 RNN・SSM に持ち込んだ。Mamba のカーネルもこれを GPU の `cub::BlockScan` で具体化している。

実装上の `thread_data[i]` の中身がまさにこのペア：

```cpp
// L221-222
thread_data[i] = make_float2(
    exp2f(delta_vals[r][i] * A_val[r]),                 // a_i = exp(Δ A)
    !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]  // b_i = ΔB · u
);
```

`exp2f` が使われているのは、$A$ を読み込む際に `LOG2E` を一度かけておく前処理 (L174-179) があるため。`expf` より高速。

## カーネルの全体構造

ファイルは大きく3レイヤ：

| 役割 | シンボル | 行 |
|---|---|---|
| 型・テンプレ定数 | `Selective_Scan_fwd_kernel_traits` | L24-70 |
| GPU カーネル本体 | `selective_scan_fwd_kernel` | L72-308 |
| Host ローンチ | `selective_scan_fwd_launch` / `..._cuda` | L310-376 |

### スレッド/ブロックの並び

```cpp
// L322
dim3 grid(params.batch, params.dim / kNRows);
```

1つの CUDA ブロック = `(batch_id, dim_id)`。各ブロックは

- 入力 $u, \Delta \in \mathbb{R}^{T}$ (1チャネル分)
- 重み $A \in \mathbb{R}^{N}$、入力依存 $B, C \in \mathbb{R}^{N \times T}$

を読んで、$y \in \mathbb{R}^{T}$ を返す。`kNThreads` が `seqlen` に応じて 32〜128 で切り替わる：

```cpp
// L353-364
if (params.seqlen <= 128)  launch<32,  4>();
else if (seqlen <= 256)    launch<32,  8>();
else if (seqlen <= 512)    launch<32, 16>();
else if (seqlen <= 1024)   launch<64, 16>();
else                       launch<128, 16>();
```

短い系列で多くのスレッドを使うとオーバーヘッドが勝つのでチューニングされている。

### チャンク化

ブロック内 1 イテレーションで処理する系列長は

$$
\text{kChunkSize} = \text{kNThreads} \times \text{kNItems}
$$

つまり最大でも 2048 トークン (128×16)。`seqlen` がこれを超える場合は

```cpp
// L137
for (int chunk = 0; chunk < params.n_chunks; ++chunk) { ... }
```

でチャンクをループする。チャンク境界で状態を引き継ぐのが `smem_running_prefix` (L100, L244-247, L257-258)：scan の最後の prefix を共有メモリに保存し、次チャンクの初期 prefix として読む。これにより HBM への状態 readback を避ける（**ハードウェア意識** の本体）。

## 1チャンク内の処理フロー

```text
1. load_input で u, delta を coalesced 読み込み
2. delta_softplus 適用 → delta_vals
3. delta_u_vals = delta * u, out_vals = D * u (skip connection)
4. for state_idx in [0, dstate):
     a. A_val を読み (LOG2E 倍済み)
     b. B_val, C_val を読み (selective なら BlockLoad、定数なら直接)
     c. thread_data = (exp2f(Δ A), ΔB u)  ← scan の入力タプル
     d. cub::BlockScan で InclusiveScan(SSMScanOp)
        → running_prefix を carry
     e. out_vals += scan_output.y * C
5. store_output で y を書き出し
6. (オプション) kHasZ: out *= z * sigmoid(z)  ← SwiGLU 風ゲート
```

state 次元 `dstate` (N) は外側の `for` ループになっていることに注意。並列 scan は**時間方向**で取り、状態次元は逐次。これは $A$ が対角行列だから各 state 成分が独立しているのを利用している（diagonal SSM の旨味）。

## 共有メモリ設計

`Selective_Scan_fwd_kernel_traits::kSmemSize` (L63-69) は

- BlockLoad/Store の TempStorage (union 的に再利用)
- BlockScan の TempStorage

を合算したサイズ。さらにカーネル本体で `kSmemSize` の後ろに `MAX_DSTATE * sizeof(scan_t) * kNRows` を継ぎ足し、running prefix 用領域を確保する：

```cpp
// L321
kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(scan_t);
```

48KB を超える場合は `cudaFuncSetAttribute` でダイナミック共有メモリの上限を引き上げる (L331-340)。

## 主要な最適化テクニック

- **`exp2f` + LOG2E 前処理**: 浮動小数指数を `expf` でなく `exp2f` で。`A` 側に LOG2E を 1 回かけるだけで全 step に効く
- **WARP_TRANSPOSE BlockLoad**: ストライドアクセスをワープ単位で転置して coalesce
- **WARP_SCANS BlockScan**: warp-level 並列スキャンを採用 (RAKING より高速、コメント L60-61 に他の選択肢が残されている)
- **kIsEvenLen 分岐**: 系列長がチャンクで割り切れる場合は `BLOCK_LOAD_DIRECT` に切替 (L47-59)
- **complex 数の自前 `cexp2f`**: PyTorch の `thrust::complex_exp` が遅いので独自実装 (L229)
- **`kIsVariableB/C` の compile-time 分岐**: selective 性が無いケース (LTI) の不要な BlockLoad を消去 (L186-212)
- **`__launch_bounds__`**: `kMinBlocks=3 or 5` で occupancy を明示 (L33, L73)

## 観察と疑問

- `kNRows == 1` しか実機で検証されていない (L312-314)。複数の dim を 1 ブロックで処理して reuse する余地が残っているが現状未開拓
- `delta_softplus` の境界 `<= 20.f` (L160): 浮動オーバーフロー対策のショートカット
- `MAX_DSTATE` の値は `selective_scan.h` 側にあるはず（読まないと不明）— state 次元の上限を決めている

Mamba のアーキテクチャ自体は SSM + selectivity のみだが、論文の主張する「線形時間で実用速度」は、このカーネルの **チャンク化 × 結合的 scan × 状態は SRAM** の 3 点で初めて成立している。

---

## 参考文献

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" Technical Report CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.
- Jimmy T.H. Smith, Andrew Warrington, Scott W. Linderman. "[Simplified State Space Layers for Sequence Modeling](https://arxiv.org/abs/2208.04933)" arXiv:2208.04933, 2022.
- Wikipedia. "[Leaky integrator](https://en.wikipedia.org/wiki/Leaky_integrator)"
- Wikipedia. "[Zero-order hold](https://en.wikipedia.org/wiki/Zero-order_hold)"

---

*作成日: 2026-05-11 / 最終更新日: 2026-05-14*

