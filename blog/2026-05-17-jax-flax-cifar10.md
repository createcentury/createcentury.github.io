---
title: "#5 JAX + Flax + Optax で CIFAR-10 を CNN で学習する"
slug: 5
date: 2026-05-17T08:00:00+09:00
authors: [createcentury]
tags: [ml]
---

JAX / Flax / Optax の 3 点セットで CIFAR-10 を CNN で訓練するハンズオン。pure-functional な書き方、Optax の合成、`jit` した訓練ループ、訓練曲線まで。

{/* truncate */}

:::note 作成中
本記事は執筆中。コード・数値・プロットは順次差し替え。
:::

## なぜ JAX / Flax / Optax

- **JAX**：NumPy + 自動微分 + `jit` / `vmap` / `pmap` の合成可能な変換系。pure-functional で副作用を明示
- **Flax**：JAX 上の neural network library。Module を**関数 + パラメータ**として扱う設計
- **Optax**：optimizer ライブラリ。`chain` / `apply_updates` で組合せが綺麗

PyTorch の `nn.Module` ＋ `Optimizer.step()` パターンと対比すると、JAX 系は「**訓練ループを自分で書きやすい**」のが強み。

## 環境

- Python 3.12 (uv 管理)
- `jax`, `flax`, `optax`, `tensorflow-datasets`, `matplotlib`
- ハードウェア：M4 Max (M4 GPU を Metal バックエンド で使う場合は `jax-metal` も)

```bash
uv venv && source .venv/bin/activate
pip install jax flax optax tensorflow-datasets matplotlib
```

## データセット

CIFAR-10：32×32×3 のカラー画像、10 クラス（airplane / automobile / bird / …）、訓練 50,000、テスト 10,000。

TODO: 数枚サンプル可視化

## モデル

3〜4 段の Conv + BatchNorm + ReLU + GlobalAveragePool + Linear の小さな CNN。

```python
# TODO: Flax で書いた Module をここに
```

## Optax で optimizer を組む

```python
# TODO: optax.chain(clip + adam(lr)) や cosine schedule の例
```

## 訓練ループ

```python
# TODO: jit した train_step、loss / acc 集計
```

## 結果

TODO: 訓練曲線、test accuracy

## 振り返り

TODO

---

## 参考文献

- [JAX 公式ドキュメント](https://docs.jax.dev/)
- [Flax 公式ドキュメント](https://flax.readthedocs.io/)
- [Optax 公式ドキュメント](https://optax.readthedocs.io/)

---

*作成日: 2026-05-17 / 最終更新日: 2026-05-17*
