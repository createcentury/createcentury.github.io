---
title: "#4 Training a CNN on CIFAR-10 with JAX + Flax + Optax"
slug: 4
date: 2026-05-17T08:00:00+09:00
authors: [createcentury]
tags: [ml]
---

A hands-on with JAX / Flax / Optax — training a small CNN on CIFAR-10. Covers the pure-functional style, Optax composition, a `jit`-ed training loop, and the training curves.

{/* truncate */}

:::note Work in progress
This post is being written. Code, numbers and plots will be filled in as they're produced.
:::

## Why JAX / Flax / Optax

- **JAX**: NumPy + autodiff + composable transforms (`jit` / `vmap` / `pmap`). Pure-functional — side effects are explicit.
- **Flax**: neural-network library on top of JAX. Modules are **function + parameters**, kept separate.
- **Optax**: optimiser library. `chain` / `apply_updates` compose cleanly.

Compared to PyTorch's `nn.Module` + `Optimizer.step()`, the JAX stack makes the **training loop trivial to write by hand** — there's nothing hidden.

## Environment

- Python 3.12 (managed by `uv`)
- `jax`, `flax`, `optax`, `tensorflow-datasets`, `matplotlib`
- Hardware: M4 Max (and `jax-metal` if running the GPU backend on Apple Silicon)

```bash
uv venv && source .venv/bin/activate
pip install jax flax optax tensorflow-datasets matplotlib
```

## Dataset

CIFAR-10: 32×32×3 colour images, 10 classes (airplane / automobile / bird / …), 50,000 train / 10,000 test.

TODO: a few sample images.

## Model

A small CNN — 3–4 stages of Conv + BatchNorm + ReLU, then GlobalAveragePool + Linear.

```python
# TODO: the Flax Module
```

## Optimiser via Optax

```python
# TODO: optax.chain(clip + adam(lr)) with a cosine schedule
```

## Training loop

```python
# TODO: jit-compiled train_step, loss / accuracy accumulation
```

## Results

TODO: training curves, test accuracy.

## Reflection

TODO.

---

## References

- [JAX docs](https://docs.jax.dev/)
- [Flax docs](https://flax.readthedocs.io/)
- [Optax docs](https://optax.readthedocs.io/)

---

*Created: 2026-05-17 / Updated: 2026-05-18*
