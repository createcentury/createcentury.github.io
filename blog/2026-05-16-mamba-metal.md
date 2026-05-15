---
title: "#4 mamba-metal: Apple Silicon で Mamba を動かす"
slug: 4
date: 2026-05-16
authors: [createcentury]
tags: [ml, ssm, cuda, metal]
---

Mamba ([state-spaces/mamba](https://github.com/state-spaces/mamba)) の selective scan は CUDA カーネル前提で書かれており、Apple Silicon ではそのまま走らない。Metal Shading Language (MSL) で書き直し、HuggingFace の重みを直接ロードして推論まで通すプロジェクト [mamba-metal](https://github.com/createcentury/mamba-metal) を作った。本記事はその設計と検証結果の備忘録。

{/* truncate */}

## 動機

Mamba 公式実装の本体は `csrc/selective_scan/selective_scan_fwd_kernel.cuh` にある CUDA カーネル。これが速度の核であり、Mamba を「論文上の理論」から「実機で動くアーキテクチャ」へ変えている部分。NVIDIA GPU 専用なので Apple Silicon では本来動かない。

参照実装の純 PyTorch 版（`selective_scan_ref`）も存在するが、for ループの素朴な漸化式評価で、長系列では非実用。Mamba の本質的な並列化（プレフィックススキャン）が抜けている。

そこで MSL で同等のカーネルを書くことにした。MLX の `mx.fast.metal_kernel` を介して JIT コンパイル・ディスパッチさせ、`.metal` ファイルを第一級資産として残す。

## Selective scan の本質

Mamba の隠れ状態は

$$
h_t = \bar{A}_t\, h_{t-1} + \bar{B}_t\, u_t,\qquad y_t = C_t^\top h_t
$$

という入力依存の係数を持つ漸化式。$\bar{A}_t = \exp(\Delta_t A)$ で、$\Delta_t, B_t, C_t$ は入力 $x_t$ から計算される（selective: 入力に応じてゲートが開閉する）。

漸化式 $h_t = a_t h_{t-1} + b_t$ は素直には逐次にしか解けないが、ペア $(a, b)$ に対する次の演算

$$
(a_2, b_2) \circ (a_1, b_1) = (a_2 a_1,\ a_2 b_1 + b_2)
$$

は**結合的** (associative)。よって prefix scan で $O(\log T)$ 段の並列ステップに落とせる（Blelloch 1990 / Martin & Cundy 2017）。Mamba カーネルがやっているのも本質的にこれ。

## MSL での核心

### SIMD-group プリミティブ

Metal は CUDA の warp に相当する **SIMD-group**（32 スレッド）を持ち、`simd_prefix_inclusive_sum`、`simd_shuffle_up` などの組み込み関数がある。ただしこれらは float スカラー専用。$(a, b)$ ペアの結合演算は自分で書く必要がある：

```metal
for (uint d = 1u; d < 32u; d <<= 1) {
    float a_prev = simd_shuffle_up(a, d);
    float b_prev = simd_shuffle_up(b, d);
    if (lane >= d) {
        b = a * b_prev + b;   // 順序重要: 先に b を更新（古い a を使う）
        a = a * a_prev;
    }
}
```

これで 32 レーンの SIMD-group 内で inclusive scan が完了。

### Block-level scan（two-tier）

1024 スレッド（= 32 SIMD-group × 32 lane）の threadgroup 全体で scan するために、SIMD-group の合計値を threadgroup memory に書き出し、1 つ目の SIMD-group がそれをさらに scan し、各スレッドが carry を加える、という二段構成にする。これは CUB の `BlockScan + WARP_SCANS` 戦略の MSL 版。

### チャンク間 running prefix

`seqlen > 1024` の場合、`smem_running_prefix` 方式：各 SSM 状態 $s$ ごとに `(carry_a[s], carry_b[s])` をチャンク間で持ち越す。新しいチャンクの先頭で前チャンクの累積を「左から」結合してから scan を実行：

$$
(a, b)_\text{new} = (a_\text{local}, b_\text{local}) \circ (\text{carry}_a, \text{carry}_b)
$$

これにより任意長の系列が単一カーネル呼び出しで処理できる。

### 観察：tg memory は単純な再利用には効かない

愚直な「データを threadgroup memory に置いて K 回読み返す」パターンは、Apple Silicon の System Level Cache（CPU/GPU 共有）が黙って吸収してしまうため、global memory 直読みと差がほぼ出なかった。tg memory が真に必要なのは**スレッド間通信**（scan の中間値交換、running prefix の保管）であって、データキャッシュ代用ではない、というのが実測の結論。

## カーネルから推論まで

カーネルが組めたら、上に Python のモデル層を積む。

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

HF の `state-spaces/mamba-*-hf` 重みは：
1. `backbone.` 接頭辞を剥がす
2. `conv1d.weight` だけ PyTorch (out, in/g, k) → MLX (out, k, in/g) で transpose
3. それ以外（Linear, embeddings, A_log, D, norm）はそのまま `mx.array` に変換

の 2 ステップだけで MLX 側にロードできる。`hidden_size` / `intermediate_size` / `num_hidden_layers` という HF transformers 標準フィールドを優先するのがコツ（790m などで legacy `d_model` フィールドが壊れているため）。

## O(L) インクリメンタルデコード

Mamba の論文上の最大の魅力は「長文脈で一定速度」。これを実機で具現化するには、推論時に SSM の隠れ状態と conv1d の sliding window を呼び出し間で持ち越す必要がある。

```python
conv_states, ssm_states = model.init_state(batch_size=1)
for token in prompt:
    logits, conv_states, ssm_states = model.step(token, conv_states, ssm_states)
# 以降は 1 トークンあたり O(1)
```

毎ステップは elementwise 演算のみ（SSM scan は不要、なぜなら状態を既に持っているから）：

$$
h_\text{new}^{(s)} = \exp(\Delta_t A_s) \cdot h^{(s)} + \Delta_t \cdot x_t \cdot B_{s,t},\qquad
y_t = \sum_s h_\text{new}^{(s)} \cdot C_{s,t} + D \cdot x_t
$$

これを z ゲートと out_proj で締める。

実測（M4 Max, mamba-130m, greedy decode）：

| 生成トークン数 | O(L²) 再 forward | **O(L) `generate_fast`** | speedup |
|---:|---:|---:|---:|
| 10  | 0.24 s | **0.06 s** | 4.3× |
| 100 | 3.24 s | **0.51 s** | 6.3× |
| 1000 | 約 32 s（外挿） | **6.84 s** | ~5× |
| 2000 | 約 80 s（外挿） | **14.08 s** | ~6× |

`generate_fast` は **n=50 から n=2000 まで一貫して ~7 ms/token**。これが Mamba の "linear-time decode" の正体。

## モデルサイズ別の結果

`state-spaces/mamba-*-hf` の全 5 サイズが load & generate 可能：

| model | params | load (s) | tok/s | ms/tok | 出力例（"The capital of Japan is" の続き） |
|---|---:|---:|---:|---:|---|
| 130m | 129 M | 1.3 | 175 | 5.7 | Tokyo, Japan. The city is located in the northern part of the country... |
| 370m | 372 M | 3.4 | 82 | 12.2 | Tokyo.（繰り返し） |
| 790m | 702 M | 4.8 | 42 | 23.7 | Tokyo, and the capital of the country is Osaka.（誤り混在） |
| 1.4b | 1372 M | 11.6 | 30 | 33.2 | Tokyo. ... Washington, D.C. ... London. |
| **2.8b** | **2.7 B** | **19.6** | **12** | **80.6** | **"Tokyo, which is also the largest city in the country"**（正確かつ自然） |

130m はサイズの限界で繰り返しに陥りやすいが、2.8b では「東京は最大の都市でもある」と付加的な事実までまとめて出してくる。greedy だけでこの差。

selective scan カーネル単体のピーク性能は `seqlen=32k` で **~187 GFLOPS**、Unified Memory の実効帯域は vec4 ロードで **~290 GB/s**（M4 Max の理論ピーク 410 GB/s の約 70%）。

## 残り課題

- **Prefill の高速化**: 現状プロンプトを 1 トークンずつ step で流すため、長文脈プロンプトでは秒オーダー。selective_scan カーネルから最終 SSM 状態を抽出できれば、parallel scan で prefill して decode に O(1)/token で接続できる
- **iPhone 上での Transformer vs Mamba ベンチマーク**: 同じ規模で速度・精度を比較し、長文脈での Mamba 優位を可視化する
- **後方カーネル**: 学習用の backward pass はまだ未実装

## 振り返り

Mamba は論文の数式自体は短いが、「実機で線形時間」を実現する部分はカーネルにある。それを別ハードウェア向けに書き直してみると、初めて論文の主張の細部が手触りとして理解できる：

- 何を SRAM に閉じるべきで何を HBM に出すべきか（Apple Silicon ではキャッシュが吸収するので少し違う）
- なぜ A は対角でなければならないか（per-state の独立性で外側ループに置けるから）
- なぜ exp(ΔA) を `exp2f + LOG2E` で書くか（少しでも速い）
- 状態キャッシュがあれば本当に O(L) になるという主張の確認

論文を読むだけでは抜けていた解像度が、書いてみると一気に上がる。

---

## 参考文献

- Albert Gu, Tri Dao. "[Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)" arXiv:2312.00752, 2023.
- Guy E. Blelloch. "[Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)" CMU-CS-90-190, 1993.
- Eric Martin, Chris Cundy. "[Parallelizing Linear Recurrent Neural Nets Over Sequence Length](https://arxiv.org/abs/1709.04057)" arXiv:1709.04057, 2017.
- [state-spaces/mamba](https://github.com/state-spaces/mamba) — 公式実装
- [createcentury/mamba-metal](https://github.com/createcentury/mamba-metal) — 本記事のプロジェクト

---

*作成日: 2026-05-16 / 最終更新日: 2026-05-16*
