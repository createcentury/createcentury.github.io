---
title: "#1 mamba"
slug: 1
date: 2026-05-11
authors: [createcentury]
tags: [ml, ssm, cuda]
---

Mamba ([Gu & Dao 2023](https://arxiv.org/abs/2312.00752)) は **Selective State Space Model** という系列モデル。Transformer の attention に代わる候補で、文脈長に対し**線形時間**で計算でき、推論時の per-token コストが**文脈長によらず一定**という性質を持つ。本記事ではアーキテクチャの中身、selective という言葉が何を意味するか、なぜ並列に解けるか、そして公式 CUDA カーネルの実装 [`csrc/selective_scan/selective_scan_fwd_kernel.cuh`](https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh) を分解して読む。

{/* truncate */}

## なぜ Mamba か

Transformer の attention は文脈長 $L$ に対し計算 $O(L^2)$、メモリも decode で $O(L)$（KV cache）かかる。長文脈で支配的なコストになる。

一方 RNN は計算 $O(L)$ だが、状態に**逐次依存**するため並列化できず、訓練が遅い。

その中間として State Space Model (SSM) 系がある（S4, S5, …）。**訓練時は並列**に解け（線形時不変なら畳み込み形に変形可）、**推論時は再帰的**に 1 ステップずつ進められる。ただし古典 SSM は**線形時不変 (LTI)**：つまり「現在の入力を見て係数を変える」ことができず、attention の content-aware な振る舞いが再現できなかった。

Mamba の貢献は **SSM のパラメータを入力依存**にし、その上でも並列訓練を可能にしたこと。これが "**selective**" の意味。

## SSM の基礎

連続時間系の状態方程式：

$$
\dot{h}(t) = A\, h(t) + B\, x(t),\qquad y(t) = C\, h(t)
$$

これをタイムステップ $\Delta$ で離散化する。Zero-order hold（[ZOH](https://en.wikipedia.org/wiki/Zero-order_hold)）の標準離散化は

$$
\bar{A} = \exp(\Delta A),\qquad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\cdot \Delta B
$$

で、$A$ が**対角行列**のとき各状態成分が独立にスカラー指数になるのが嬉しい点。Mamba 公式実装は近似的に

$$
\bar{A}_t = \exp(\Delta_t A),\qquad \bar{B}_t \approx \Delta_t B
$$

を採用する。離散漸化式は

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, x_t,\qquad y_t = C_t^\top h_t + D\, x_t
$$

— ここまでは古典 SSM 含めて共通。

## Selectivity — 何が新しいのか

LTI な S4 では $A, B, C, \Delta$ はすべて**学習済みパラメータ**（時刻に依らず一定）。一方 Mamba は

$$
\Delta_t = \mathrm{softplus}(W_\Delta\, x_t + b_\Delta),\quad
B_t = W_B\, x_t,\quad
C_t = W_C\, x_t
$$

と、**$B, C, \Delta$ を入力 $x_t$ から都度作る**。つまり時間に応じて遷移が変わる時変系。これにより：

- $\Delta_t$ が大きい時刻：状態 $h$ が**大きくアップデート**される（「今の入力を強く取り込む」）
- $\Delta_t$ が小さい時刻：状態が**ほぼ凍結**される（「無視する」）

selective の本質は**ステップサイズ自体を入力で制御する**ことにある。これでゲート的振る舞いが手に入る代わりに、係数が時刻 $t$ で変わるので**畳み込み形に展開できない**（FFT で並列化できない）。代わりに後述の**並列 scan**を使う。

## Mamba ブロック

論文の Section 3.4。1 つの Mamba 層は次のデータフロー：

```
x ∈ ℝ^{B×L×D}                              # 入力
  ├── in_proj (Linear D → 2 D')
  ├── split → (x_main, z)                 # 各 ℝ^{B×L×D'}
  ├── x_main = SiLU(conv1d(x_main))       # causal, kernel=4
  ├── x_dbl  = x_proj(x_main)              # Linear D' → dt_rank + 2N
  ├── (dt_pre, B_ssm, C_ssm) = split(x_dbl)
  ├── dt = softplus(dt_proj(dt_pre))      # ℝ^{B×L×D'}
  ├── A  = -exp(A_log)                    # ℝ^{D'×N}, 必ず負
  ├── y_ssm = selective_scan(x_main, dt, A, B_ssm, C_ssm, D, z)
  └── y = out_proj(y_ssm)                 # Linear D' → D
```

要点：

- **`in_proj` で 2 倍に膨らませて半分は z 経路に**回す。z は最終的に `y *= SiLU(z)` でゲートとして使う（SwiGLU 風）
- **conv1d は causal な短距離フィルタ**（kernel=4）。直前数 token の文脈を SSM に渡す前に圧縮しておく役割
- **A は対角**で、$A_\text{log}$ をパラメータにして $A = -\exp(A_\text{log})$ とすることで**常に負**（系が発散しない）
- **B, C, Δ は入力依存**（selective）。`x_proj` が一発で 3 つまとめて生成し、`dt_proj` が Δ の次元を拡張

実際の Mamba モデル（`state-spaces/mamba-130m` 等）はこのブロックを **24-64 層** 積み、各層を **pre-norm RMSNorm + residual** で包む。LM head は token embedding と weight-tied。

## Transformer との対比

| 量 | Transformer | Mamba |
|---|---|---|
| 訓練の計算量 | $O(L^2 D)$ (attention) | $O(L D N)$ (scan、$N$ は状態次元) |
| 推論 decode の計算量 | $O(L D)$ (with KV cache) | $O(D N)$ (constant in $L$) |
| 推論 decode のメモリ | $O(L D)$ (KV cache, growing) | $O(D N)$ (state, constant) |
| 入力依存の混合 | attention の softmax | $B_t, C_t, \Delta_t$ の入力依存性 |

長文脈で効くのは Mamba。**$N \ll L$**（典型的に $N=16$）なので、状態が文脈をどう圧縮するかが本質的勝負どころ。

## 動かす漸化式（再掲）

ここからは実装目線。Mamba の隠れ状態 $h_t \in \mathbb{R}^{N}$ の漸化式は

$$
h_t = \exp(\Delta_t A)\, h_{t-1} + (\Delta_t B_t)\, x_t,\qquad y_t = C_t^\top h_t + D\, x_t
$$

で $A$ は対角行列。各 state 成分は独立にスカラー漸化式 $h_t = a_t h_{t-1} + b_t$ になる。これがカーネル実装の出発点。

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

*作成日: 2026-05-11 / 最終更新日: 2026-05-17*

