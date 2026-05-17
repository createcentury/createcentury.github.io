---
title: "#4 Training a CNN on CIFAR-10 with JAX + Flax + Optax"
slug: 4
date: 2026-05-17T08:00:00+09:00
authors: [createcentury]
tags: [ml]
---

A hands-on with JAX / Flax / Optax — training a small CNN on CIFAR-10. The full runnable notebook lives in [createcentury/jax-flax-optax-lab](https://github.com/createcentury/jax-flax-optax-lab/blob/main/notebooks/01-cifar10-cnn.ipynb); this post is the design notes + numbers.

{/* truncate */}

## Why JAX / Flax / Optax

- **JAX**: NumPy + autodiff + composable transforms (`jit` / `vmap` / `pmap`). Pure-functional — side effects are explicit.
- **Flax**: neural-network library on top of JAX. Modules are **function + parameters**, kept separate.
- **Optax**: optimiser library. `chain` / `apply_updates` compose cleanly.

Compared to PyTorch's `nn.Module` + `Optimizer.step()`, the JAX stack makes the **training loop trivial to write by hand** — there's nothing hidden.

## Setup

- Small VGG-style CNN: three `Conv → BN → ReLU` ×2 → `MaxPool` stages (32 → 64 → 128 channels), GAP, then a Linear(10). ~0.3M params.
- Dataset: CIFAR-10 via `tensorflow-datasets`. Standard augmentation — 4-px pad + random crop + horizontal flip.
- Optax: `clip_by_global_norm(1.0) + add_decayed_weights(5e-4) + sgd(cosine schedule, momentum=0.9, nesterov=True)`.
- 20 epochs, batch size 128, 1-epoch warmup → cosine decay to 0.
- Runtime: a free Colab T4.

## The training step

```python
@jax.jit
def train_step(state, batch, dropout_key):
    x, y = batch
    def loss_fn(params):
        logits, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=True,
            rngs={'dropout': dropout_key},
            mutable=['batch_stats'],
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss, (logits, new_model_state)
    (loss, (logits, new_model_state)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    acc = (jnp.argmax(logits, axis=-1) == y).mean()
    return state, {'loss': loss, 'acc': acc}
```

A few things to read off:

- `state.apply_fn(...)` is `model.apply(...)` — Flax `linen` modules are pure functions; params are passed in explicitly.
- `mutable=['batch_stats']` tells Flax that BatchNorm running stats are mutable state, distinct from learnable params.
- `jax.value_and_grad(..., has_aux=True)` returns both the loss and the auxiliary outputs (logits + new BN stats), with gradients only w.r.t. `params`.
- The whole thing is wrapped in `@jax.jit` — first call traces and compiles to XLA; later calls bypass Python entirely.

## Results

| epoch | train acc | test acc | wall time |
|---:|---:|---:|---:|
| 1  | 46.1% | 47.7% | 29.7s (JIT compile) |
| 5  | 78.6% | 76.3% | 9.2s |
| 10 | 84.4% | 82.0% | 9.6s |
| 15 | 89.1% | 86.1% | 9.6s |
| 20 | 92.5% | **88.5%** | 9.5s |

Total: **226.6s** for 20 epochs on a Colab T4. After the first step pays the JIT-compile cost (~20s), per-epoch settles at ~10s.

Final test accuracy **88.53%** — solid for a 0.3M-param model with a vanilla recipe.

The gap between train (92.5%) and test (88.5%) at the end is 4 points, which is in the healthy range. Heavier augmentation (mixup, cutout) or a larger model would close it further at the cost of wall time.

## What the stack feels like

End-to-end, a few things stand out:

- **No hidden state.** `state.params`, `state.batch_stats`, and the optax state are all explicit. After training, `state` is just a PyTree you can `jax.tree_util.tree_map` over, save, ship to another device.
- **JIT changes the failure mode.** First call: cryptic compilation errors with tracer-vs-array warnings. After it runs once: rock-solid and fast. The cost is concentrated at the start.
- **Functional update is the API.** `state = state.apply_gradients(grads=...)` returns a new state; the old one is unchanged. Closer to Clojure than to PyTorch.
- **Optax composition is genuinely nice.** `chain(clip_by_global_norm, add_decayed_weights, sgd)` reads top-to-bottom. Each transform is a `(state, params, grads) -> (state, grads)` step that minds its own business.

## What's next

- Move beyond `tensorflow-datasets`: a Hugging Face datasets loader, or a custom one.
- A bigger experiment: a small ViT at matched parameter count vs this CNN.
- Compare against Flax NNX (the newer API that hides the param/state plumbing) — same recipe, different ergonomics.
- `pmap` on multi-device is a TPU-only story on free tiers; M4 has a single logical GPU, so there's nothing to parallelise over locally.

---

## References

- Notebook: [createcentury/jax-flax-optax-lab — 01-cifar10-cnn](https://github.com/createcentury/jax-flax-optax-lab/blob/main/notebooks/01-cifar10-cnn.ipynb)
- [JAX docs](https://docs.jax.dev/)
- [Flax docs (linen)](https://flax.readthedocs.io/)
- [Optax docs](https://optax.readthedocs.io/)

---

*Created: 2026-05-17 / Updated: 2026-05-18*
