# lessons/02/lesson.exs
#
# Lesson 02 — Tensors in Elixir (Nx): shape, dtype, broadcasting, slicing
#
# Run:
#   elixir lessons/02/lesson.exs
#
# This is a "tensor playground" script. It's intentionally verbose and print-heavy.
# The goal is to build intuition for:
#   - shapes and dtypes
#   - axes (dimensions)
#   - broadcasting rules
#   - slicing/selecting
#   - reductions like mean/sum
#   - standardizing features + adding a bias column
#
# NOTE (from Lesson 01):
#   Outside Nx.Defn `defn`, avoid using plain `+ - * /` with tensors.
#   Use Nx.add/Nx.subtract/Nx.multiply/Nx.divide to prevent Elixir arithmetic errors.

Mix.install([
  {:nx, "~> 0.10"}
])

defmodule L2 do
  def hr(), do: IO.puts(String.duplicate("-", 72))

  def header(title) do
    hr()
    IO.puts(title)
    hr()
  end

  def info(label, t) do
    IO.puts("#{label}")
    IO.puts("  shape: #{inspect(Nx.shape(t))}   rank: #{Nx.rank(t)}   type: #{inspect(Nx.type(t))}")
    t
  end

  def sample(label, t, rows \\ 5, cols \\ 5) do
    shape = Nx.shape(t)

    preview =
      case tuple_size(shape) do
        0 ->
          t

        1 ->
          n = elem(shape, 0)
          k = min(rows, n)
          Nx.slice(t, [0], [k])

        2 ->
          n = elem(shape, 0)
          m = elem(shape, 1)
          r = min(rows, n)
          c = min(cols, m)
          Nx.slice(t, [0, 0], [r, c])

        _ ->
          # For higher-rank tensors, just show the first element as a scalar-ish sample.
          Nx.take(t, Nx.tensor([0]), axis: 0)
      end

    IO.puts("#{label} (preview)")
    IO.inspect(preview)
    t
  end

  def try_op(label, fun) do
    IO.puts(label)

    try do
      result = fun.()
      IO.puts("  ✅ ok")
      result
    rescue
      e ->
        IO.puts("  ❌ error: #{Exception.message(e)}")
        :error
    end
  end
end

# ------------------------------------------------------------------------------
# 1) Creating tensors: scalar, vector, matrix
# ------------------------------------------------------------------------------

L2.header("1) Creating tensors (scalar / vector / matrix)")

scalar = Nx.tensor(3.14, type: {:f, 32})
vector = Nx.tensor([1, 2, 3, 4], type: {:f, 32})
matrix = Nx.iota({2, 3}, type: {:f, 32})

L2.info("scalar", scalar) 
 L2.sample("scalar", scalar)
L2.info("vector", vector) 
 L2.sample("vector", vector)
L2.info("matrix = iota({2,3})", matrix) 
L2.sample("matrix", matrix)

# ------------------------------------------------------------------------------
# 2) Shape + dtype basics (and why dtype matters)
# ------------------------------------------------------------------------------

L2.header("2) Shape + dtype basics (why dtype matters)")

ints = Nx.tensor([1, 2, 3])                 # dtype may default to integer
floats = Nx.tensor([1, 2, 3], type: {:f, 32})

L2.info("ints (no type specified)", ints) 
L2.sample("ints", ints)
L2.info("floats (explicit float32)", floats) 
L2.sample("floats", floats)

IO.puts("""
Rule of thumb for ML:
  Prefer float32 tensors (type {:f, 32}) to avoid integer math surprises.
""")

# ------------------------------------------------------------------------------
# 3) Axes: what axis 0 vs axis 1 means
# ------------------------------------------------------------------------------

L2.header("3) Axes (dimensions): axis 0 vs axis 1")

x = Nx.iota({4, 3}, type: {:f, 32})
L2.info("x (shape {4,3})", x) 
L2.sample("x", x, 4, 3)

col_means = Nx.mean(x, axes: [0]) # per-column mean => shape {3}
row_means = Nx.mean(x, axes: [1]) # per-row mean => shape {4}

L2.info("mean over axes: [0] (per-column)", col_means) 
L2.sample("col_means", col_means)
L2.info("mean over axes: [1] (per-row)", row_means) 
L2.sample("row_means", row_means)

col_means_keep = Nx.mean(x, axes: [0], keep_axes: true) # shape {1,3}
L2.info("mean axes:[0], keep_axes:true (broadcast-friendly)", col_means_keep) 
L2.sample("col_means_keep", col_means_keep, 1, 3)

# ------------------------------------------------------------------------------
# 4) Broadcasting: making shapes compatible for elementwise ops
# ------------------------------------------------------------------------------

L2.header("4) Broadcasting rules (and how to spot them)")

# Make a bigger matrix like we use in ML: {n,d}
n = 6
d = 3

xb =
  Nx.iota({n, d}, type: {:f, 32})
  |> Nx.divide(10.0)

shift = Nx.tensor([1.0, 10.0, 100.0], type: {:f, 32})  # shape {3}

L2.info("xb (shape {#{n},#{d}})", xb)
 L2.sample("xb", xb, 6, 3)
L2.info("shift (shape {3})", shift) 
L2.sample("shift", shift)

# This will broadcast shift {3} across rows of xb {n,3}
xb_shifted = Nx.add(xb, shift)
L2.info("xb_shifted = Nx.add(xb, shift)  (broadcast shift across rows)", xb_shifted) 
L2.sample("xb_shifted", xb_shifted, 6, 3)

# Show an example that *fails* (shape mismatch):
bad = Nx.iota({n}, type: {:f, 32}) # shape {n}
L2.info("bad (shape {#{n}})", bad) 
L2.sample("bad", bad, 6)

L2.try_op("Try: Nx.add(xb {#{n},#{d}}, bad {#{n}})  (should fail: not broadcastable)", fn ->
  Nx.add(xb, bad)
end)

IO.puts("""
Broadcasting intuition:
  {n, d} + {d}   ✅ works (vector stretches across rows)
  {n, d} + {n}   ❌ fails (Nx doesn't guess whether {n} is rows or columns)
""")

# ------------------------------------------------------------------------------
# 5) Slicing / selecting: grabbing parts of tensors
# ------------------------------------------------------------------------------

L2.header("5) Slicing / selecting (Nx.slice, Nx.take)")

big =
  Nx.iota({10, 5}, type: {:f, 32})
  |> Nx.divide(10.0)

L2.info("big (shape {10,5})", big) 
L2.sample("big", big, 6, 5)

first3_rows = Nx.slice(big, [0, 0], [3, 5])
L2.info("first3_rows = slice(big, [0,0], [3,5])", first3_rows) 
L2.sample("first3_rows", first3_rows, 3, 5)

col2 = Nx.slice(big, [0, 2], [10, 1])    # shape {10, 1}
col2_vec = Nx.squeeze(col2, axes: [1])   # shape {10}

L2.info("col2 = slice(big, [0,2], [10,1])", col2) 
L2.sample("col2", col2, 6, 1)
L2.info("col2_vec = squeeze(col2, axes:[1])", col2_vec) 
L2.sample("col2_vec", col2_vec, 10)

picked_rows = Nx.take(big, Nx.tensor([0, 3, 9]), axis: 0)
L2.info("picked_rows = take(big, [0,3,9], axis:0)", picked_rows)
 L2.sample("picked_rows", picked_rows, 3, 5)

# ------------------------------------------------------------------------------
# 6) Reshape / transpose: reorganizing dimensions
# ------------------------------------------------------------------------------

L2.header("6) Reshape / transpose")

a = Nx.iota({2, 6}, type: {:f, 32})
L2.info("a (shape {2,6})", a) 
L2.sample("a", a, 2, 6)

a_reshaped = Nx.reshape(a, {3, 4})
L2.info("a_reshaped = reshape(a, {3,4})", a_reshaped) 
L2.sample("a_reshaped", a_reshaped, 3, 4)

a_t = Nx.transpose(a_reshaped) # {4,3}
L2.info("a_t = transpose(a_reshaped)", a_t) 
L2.sample("a_t", a_t, 4, 3)

# ------------------------------------------------------------------------------
# 7) Reductions: sum/mean over axes (+ keep_axes)
# ------------------------------------------------------------------------------

L2.header("7) Reductions (sum/mean over axes)")

r = Nx.iota({4, 3}, type: {:f, 32})
L2.info("r", r) 
L2.sample("r", r, 4, 3)

sum_all = Nx.sum(r)
sum_cols = Nx.sum(r, axes: [0])
sum_rows = Nx.sum(r, axes: [1])

L2.info("sum_all = sum(r)  (shape {})", sum_all) 
L2.sample("sum_all", sum_all)
L2.info("sum_cols = sum(r, axes:[0])  (shape {3})", sum_cols) 
L2.sample("sum_cols", sum_cols)
L2.info("sum_rows = sum(r, axes:[1])  (shape {4})", sum_rows) 
L2.sample("sum_rows", sum_rows)

# ------------------------------------------------------------------------------
# 8) Standardization (feature scaling) + bias column (ML-prep patterns)
# ------------------------------------------------------------------------------

L2.header("8) Standardization + bias column (ML-prep patterns)")

n = 200
d = 3

x =
  Nx.iota({n, d}, type: {:f, 32})
  |> Nx.divide(50.0)

L2.info("x (dataset) shape {200,3}", x) 
L2.sample("x", x, 5, 3)

# Per-column mean/std
mean = Nx.mean(x, axes: [0], keep_axes: true) # {1,3}
centered = Nx.subtract(x, mean)

var =
  centered
  |> Nx.pow(2)
  |> Nx.mean(axes: [0], keep_axes: true)

# Add tiny epsilon before sqrt to avoid divide-by-zero if a column is constant
eps = 1.0e-6
std = Nx.sqrt(Nx.add(var, eps))

x_std = Nx.divide(centered, std)

L2.info("mean (keep_axes) shape {1,3}", mean) 
L2.sample("mean", mean, 1, 3)
L2.info("std  (keep_axes) shape {1,3}", std) 
L2.sample("std", std, 1, 3)
L2.info("x_std = (x-mean)/std", x_std) 
L2.sample("x_std", x_std, 5, 3)

# Add a bias column of ones: {n, d} -> {n, d+1}
ones = Nx.broadcast(Nx.tensor(1.0, type: {:f, 32}), {n, 1})
x_bias = Nx.concatenate([x, ones], axis: 1)

L2.info("ones column shape {200,1}", ones) 
L2.sample("ones", ones, 5, 1)
L2.info("x_bias = concat([x, ones], axis:1) shape {200,4}", x_bias) 
L2.sample("x_bias", x_bias, 5, 4)

IO.puts("""
✅ Lesson 02 complete if you can explain:
  - what shape and dtype mean
  - what "axis" means in reductions
  - why keep_axes:true helps broadcasting
  - how standardization uses broadcasting on purpose
  - why we add a bias column in linear models
""")
