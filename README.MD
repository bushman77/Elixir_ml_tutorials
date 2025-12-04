# Elixir ML Tutorials 🧠✨

A practical, script-first course for learning machine learning **in Elixir**, from first principles to real models.

This repo is organized as a growing set of runnable lessons (`.exs` files) plus supporting notes and exercises. The goal is to make ML feel *native* in Elixir: clear data shapes, understandable math, repeatable experiments, and code you can actually read.

---

## What you’ll learn (the roadmap)

This course is designed to build confidence in layers. Every section adds one “new superpower”, while reusing what you already learned.

### Phase 1 — Foundations (tensors + gradients)
You’ll learn:
- What a **tensor** is (shape, dtype, broadcasting)
- How Elixir does numerical computing (Nx)
- What **autodiff** is and how gradients are computed
- How gradient descent actually updates parameters

Planned lessons:
- **01 — Linear regression from scratch (Nx + autodiff)**
- 02 — Tensors: shapes, types, slicing, broadcasting, reductions
- 03 — Loss functions: MSE, cross-entropy (and why they work)
- 04 — Optimization: SGD, momentum, Adam (implement + compare)
- 05 — Debugging numerics: NaNs, exploding gradients, normalization

### Phase 2 — Data and “classic ML”
You’ll learn:
- Data wrangling workflows (dataframes/series)
- Feature engineering and evaluation
- Classic algorithms (regression, clustering, metrics)

Planned lessons:
- 06 — Dataframes + dataset hygiene (train/val/test, leakage)
- 07 — Logistic regression for classification
- 08 — Metrics: accuracy, precision/recall, F1, ROC-AUC
- 09 — Clustering: k-means and distance metrics
- 10 — Regularization: L1/L2, bias/variance intuition

### Phase 3 — Neural nets “the Elixir way”
You’ll learn:
- How to build networks as composable functions
- Training loops and batching
- Using higher-level APIs when you’re ready

Planned lessons:
- 11 — A tiny neural net from scratch (dense layer + backprop)
- 12 — Mini-batching + shuffling + dataloaders
- 13 — Feedforward classifier on a real dataset
- 14 — Convolutions (CNN basics) + image-ish toy data
- 15 — Recurrent ideas (sequence batching + masking)

### Phase 4 — Performance, serving, and real-world usage
You’ll learn:
- Backends/JIT concepts and why they matter
- How to run inference cleanly in production-ish setups
- Reproducibility and experiment discipline

Planned lessons:
- 16 — Performance basics: vectorization vs loops
- 17 — JIT/backends (when/why you’d switch)
- 18 — Model saving/loading + versioning
- 19 — Inference pipelines & simple “serving” patterns
- 20 — Capstone: build a complete mini project end-to-end

---

## Current lessons

- `ml_hello_linear_annotated.exs`
  - Linear regression trained with gradient descent
  - Shows `Mix.install`, tensors, `Nx.Defn`, and `value_and_grad`
  - This is **Lesson 01** (foundations)

As the course grows, lessons may be moved into a `lessons/` folder with numbering.
For now, everything is kept simple and runnable from the repo root.

---

## How to run a lesson

Requirements:
- Elixir installed on your system
- Internet access the first time you run a script (to fetch deps via `Mix.install`)

Run:
```bash
elixir ml_hello_linear_annotated.exs
```
