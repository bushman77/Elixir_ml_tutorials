# ml_hello_linear_annotated.exs
#
# Goal:
#   Train a tiny linear regression model in Elixir using Nx autodiff.
#
# Key libraries/modules used (and where they live):
#
#   - Mix.install/1
#       * Module: Mix (ships with Elixir)
#       * Purpose: Install dependencies for THIS script at runtime, without creating a Mix project.
#
#   - Nx
#       * Hex package: :nx
#       * Purpose: Tensor math (think: N-dimensional arrays) + numerical ops.
#       * Primary type: Nx.Tensor (a struct holding shape/type and pointing to a backend that owns data)
#
#   - Nx.Defn
#       * Part of Nx
#       * Purpose: "Numerical definitions" (defn) that are compiled to tensor graphs.
#                 It also includes automatic differentiation utilities like value_and_grad.
#
# Important mental model:
#   - Regular Elixir functions (def) run on normal values right away.
#   - Nx numerical functions (defn) build/compile a computation over tensors.
#     Inside a defn, your values might be "expressions" until executed by a backend.

Mix.install([
  # This pulls the Nx library from Hex. You can change the version constraint if needed.
  {:nx, "~> 0.10"}
])

defmodule HelloML.LinearRegression do
  # import Nx.Defn gives:
  #   - defn macro: define "numerical functions" over tensors
  #   - value_and_grad: compute function value + gradient (autodiff)
  #   - lots of operator overloads/macros that become elementwise tensor ops
  import Nx.Defn

  # predict/2:
  #   Implements the linear model y_hat = X · w
  #
  # Where these are found:
  #   - Nx.dot/2 is in the Nx module (tensor dot product / matrix-vector multiply).
  #
  # Shapes:
  #   x: {n, d}  (n samples, d features)
  #   w: {d}
  #   result: {n}
  defn predict(x, w) do
    Nx.dot(x, w)
  end

  # loss/3:
  #   Mean squared error (MSE) = mean((y_hat - y)^2)
  #
  # Where these are found:
  #   - Nx.mean/1 -> reduction that averages tensor elements
  #   - Nx.pow/2  -> elementwise power; Nx also supports "**" inside defn
  #
  # Shapes:
  #   y_hat, y: {n}
  #   loss: scalar tensor (shape {})
  defn loss(x, y, w) do
    y_hat = predict(x, w)
    Nx.mean(Nx.pow(y_hat - y, 2))
  end

  # step/5:
  #   One gradient descent step.
  #
  # New concept: autodiff gradient
  #   value_and_grad(vars, fun) computes:
  #     - the scalar output of fun(vars)
  #     - the gradient d(fun)/d(vars)
  #
  # Where this is found:
  #   - Nx.Defn.value_and_grad/3 (in Nx.Defn)
  #
  # Learning rate:
  #   lr is a scalar tensor or number (we'll pass a float).
  defn step(x, y, w, lr) do
    {l, g} =
      value_and_grad(w, fn w ->
        loss(x, y, w)
      end)

    # Gradient descent update:
    #   w_new = w - lr * gradient
    {w - lr * g, l}
  end
end

# -----------------------------
# DATASET (synthetic / deterministic)
# -----------------------------
#
# We're generating a simple dataset so you can see the training loop work.
#
# Where these are found:
#   - Nx.iota/2: creates a tensor filled with increasing numbers
#   - Nx.divide/2: elementwise divide (also supports "/")
#   - Nx.tensor/2: create a tensor from an Elixir list
#   - Nx.dot/2: produces the target y from X and the "true" weights
#
# Note:
#   Nx operations done HERE (outside defn) run eagerly and return Nx.Tensor data immediately,
#   using the default backend (usually Nx.BinaryBackend unless you configure something else).

n = 200
d = 3

x =
  Nx.iota({n, d}, type: {:f, 32})
  |> Nx.divide(50.0)

true_w = Nx.tensor([0.8, -0.2, 1.5], type: {:f, 32})

# Simple linear target with a tiny constant offset (not random noise to keep it deterministic):

y = Nx.add(Nx.dot(x, true_w), 0.1)

# Starting weights: all zeros
w0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {d})

# Learning rate (a normal float is fine; Nx.Defn will lift it as needed)
lr = 0.005

# -----------------------------
# TRAINING LOOP (plain Elixir)
# -----------------------------
#
# We loop in normal Elixir with Enum.reduce/3.
# Each iteration calls HelloML.LinearRegression.step/4 (a defn),
# which returns:
#   - updated weights (tensor)
#   - current loss (scalar tensor)
#
# Converting Nx tensors to normal Elixir values:
#   - Nx.to_number/1 extracts a scalar tensor into an Elixir number
#   - Nx.to_flat_list/1 converts a tensor to a plain list (outside defn)

{w, last_loss} =
  Enum.reduce(1..200, {w0, nil}, fn i, {w, _loss} ->
    {w2, l2} = HelloML.LinearRegression.step(x, y, w, lr)

    if rem(i, 50) == 0 do
      loss_number = Nx.to_number(l2)
      IO.puts("step=#{i} loss=#{:erlang.float_to_binary(loss_number, decimals: 6)}")
    end

    {w2, l2}
  end)

IO.puts("\nLearned w:  #{inspect(Nx.to_flat_list(w))}")
IO.puts("True w:     #{inspect(Nx.to_flat_list(true_w))}")
IO.puts("Final loss: #{Nx.to_number(last_loss)}")

# -----------------------------
# Optional: JIT compilation (later)
# -----------------------------
#
# Nx.Defn supports just-in-time compilation via compiler backends.
# In practice you’d typically add EXLA as a dep and enable it, then you can jit defn calls.
#
# This script doesn't enable a JIT backend to keep it minimal.
# When you're ready, we can do a second version that adds EXLA and shows
# the smallest possible JIT toggle.

