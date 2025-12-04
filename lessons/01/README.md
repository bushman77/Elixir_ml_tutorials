# Lesson 01 — Linear Regression + Autodiff (Nx)

In this lesson we train a tiny linear regression model in **Elixir** using **Nx** tensors and **automatic differentiation**.  
You’ll see how “ML training” is really just: **make a prediction → compute loss → compute gradient → update weights**.

---

## What’s in this lesson

### Files
- `lesson.exs`
  - A fully runnable script that:
    - creates a simple synthetic dataset
    - defines a linear model `ŷ = X · w`
    - defines mean-squared-error loss (MSE)
    - uses autodiff to compute gradients of the loss w.r.t. `w`
    - performs gradient descent updates in a training loop
    - prints loss during training and compares learned vs true weights

---

## Concepts you’ll learn

### 1) Tensors (Nx)
- A **tensor** is an N-dimensional array (scalar, vector, matrix, …).
- Key tensor properties:
  - **shape**: sizes of each dimension (ex: `{200, 3}`)
  - **dtype**: number type (ex: `{:f, 32}` for float32)

### 2) Linear model
We’re learning weights `w` so that:
- `X` has shape `{n, d}` (n samples, d features)
- `w` has shape `{d}`
- predictions `y_hat = X · w` has shape `{n}`

### 3) Loss (MSE)
We measure how wrong the predictions are:
- `loss = mean((y_hat - y)^2)`
- A good training run drives this number down.

### 4) Autodiff (automatic differentiation)
Instead of manually deriving gradients, Nx can compute:
- the loss value
- the gradient **d(loss)/d(w)**

That gradient tells us how to change `w` to reduce the loss.

### 5) Gradient descent update
Each training step updates weights:
- `w_new = w - lr * grad`
where `lr` is the learning rate.

---

## How to run

From the repo root:
```bash
elixir lessons/01/lesson.exs

