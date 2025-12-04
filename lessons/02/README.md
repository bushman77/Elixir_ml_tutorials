# Lesson 02 — Tensors in Elixir (Nx): shape, dtype, broadcasting, slicing

This lesson turns Nx tensors from “mysterious blobs” into something you can *reason about* at a glance. You’ll build a small tensor playground and learn the core rules that make every ML model work.

---

## What you’ll be able to do after this lesson

By the end, you should be able to:

- Look at a tensor and confidently state its **shape** and **dtype**
- Create tensors (scalars, vectors, matrices) intentionally
- Use **broadcasting** on purpose (and spot it when it happens implicitly)
- Slice/select along axes and understand which dimension you’re operating on
- Use reductions like `sum/mean` with the right `axis:` and `keep_axes:` behavior
- Fix the #1 ML bug: **shape mismatch** (without guessing)

---

## Prerequisites

From Lesson 01, you should recognize:
- `Mix.install/1` scripts
- `%Nx.Tensor{}` values
- The idea that values can “live” in tensor-land (and may need conversion for printing)

If Lesson 01 ran successfully on your machine, you’re good.

---

## Files in this lesson (recommended)

- `lesson.exs`  
  A runnable *tensor playground* script. It prints shapes/types and demonstrates each concept.
- `ERRORS.md`  
  A running log of real errors and what they mean (we’ll keep adding to it as we go).

If you haven’t created them yet, that’s fine—this README is the guide for what we’ll add next.

---

## How to run

From repo root:
```bash
elixir lessons/02/lesson.exs
```

From within the folder:
```bash
cd lessons/02
elixir lesson.exs
```

---

## Core concepts (the Nx mental model)

### 1) Tensor
A tensor is an N-dimensional array:
- scalar: shape `{}` (rank 0)
- vector: shape `{d}` (rank 1)
- matrix: shape `{n, d}` (rank 2)
- higher ranks: `{b, t, d}` etc.

### 2) Shape
Shape answers: “how many items along each axis?”
Examples:
- `{200, 3}` = 200 rows, 3 columns
- `{3}` = 3 elements
- `{}` = a single scalar

### 3) Dtype (type)
Dtype answers: “what kind of number is stored?”
Common:
- `{:f, 32}` float32
- `{:f, 64}` float64
- `{:s, 64}` int64 (signed)
- `{:u, 8}`  uint8

Rule of thumb for ML: **use floats** (`{:f, 32}` is a great default).

### 4) Axis
Axis is *which dimension* you’re talking about.
For a shape `{n, d}`:
- `axis: 0` → across rows (down the column)
- `axis: 1` → across columns (across a row)

### 5) Broadcasting (the superpower)
Broadcasting is how Nx makes shapes compatible for elementwise ops.

Example idea:
- tensor `{200, 3}` plus vector `{3}` “stretches” the `{3}` across the 200 rows.

Broadcasting is *why* you can do:
- per-column normalization
- adding a bias vector
- scaling features

### 6) Reductions (sum/mean/etc.)
Reductions shrink dimensions:
- `Nx.mean(x)` → scalar (shape `{}`)
- `Nx.mean(x, axes: [0])` → per-column means (shape `{d}`)
- `Nx.mean(x, axes: [0], keep_axes: true)` → shape `{1, d}` (broadcast-friendly)

---

## Mini cheat sheet (you’ll use these constantly)

Create tensors:
```elixir
Nx.tensor([1, 2, 3], type: {:f, 32})
Nx.iota({2, 3}, type: {:f, 32})
Nx.broadcast(0.0, {5}) |> Nx.as_type({:f, 32})
```

Inspect:
```elixir
Nx.shape(t)
Nx.type(t)
Nx.rank(t)
```

Reshape / transpose:
```elixir
Nx.reshape(t, {rows, cols})
Nx.transpose(t) # (rank 2) or Nx.transpose(t, axes: [...]) for general
```

Slice/select:
```elixir
Nx.slice(t, [start0, start1], [len0, len1])
Nx.take(t, Nx.tensor([0, 2, 4]), axis: 0)
```

Concatenate / stack:
```elixir
Nx.concatenate([a, b], axis: 1)
Nx.stack([a, b], axis: 0)
```

Reductions:
```elixir
Nx.sum(t)
Nx.mean(t, axes: [0])
Nx.mean(t, axes: [0], keep_axes: true)
```

---

## Exercises (do at least 2)

1) **Column means and std dev**
- Make `x` with shape `{200, 3}`
- Compute per-column means: `Nx.mean(x, axes: [0])`
- Compute per-column std dev (hint: variance = mean((x - mean)^2))

2) **Standardize features**
Create:
```elixir
x_std = (x - mean) / (std + 1.0e-6)
```
Use `keep_axes: true` so broadcasting is clean.

3) **Add a bias column**
Turn `{n, d}` into `{n, d+1}` by concatenating a column of ones.

4) **Broadcasting intuition**
- Add a `{3}` vector to a `{200, 3}` tensor
- Then add a `{200}` vector and observe why it fails (shape mismatch)

---

## Common gotchas (so you don’t get stuck)

- “Why did this crash with shape mismatch?”  
  Because elementwise ops require shapes that are equal *or broadcastable*.

- “Why did my dtype become integer?”  
  If you create tensors from ints without specifying `type: {:f, 32}`,
  you sometimes end up with integer tensors. Cast with `Nx.as_type/2`.

- “Why did operators behave differently inside vs outside `defn`?”  
  Inside `defn`, operators become tensor ops. Outside, they are normal Elixir ops.
  When in doubt outside `defn`, use explicit `Nx.add`, `Nx.subtract`, etc.

---

## What’s next (Lesson 03)

Lesson 03 will focus on **loss functions**:
- MSE (regression)
- cross-entropy (classification)
- how loss shape + reduction choices affect gradients

That’s where tensors start turning into “training.”
