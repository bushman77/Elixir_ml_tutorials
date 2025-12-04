# Errors encountered (growing log)

This file tracks real errors we hit while building/running Lesson 01, what they mean, and the usual causes/fixes.

---
1)

## **(ArithmeticError) bad argument in arithmetic expression**
**Error:**
```
** (ArithmeticError) bad argument in arithmetic expression
    lesson.exs:121: (file)
    (elixir 1.18.4) lib/code.ex:1525: Code.require_file/2
```

### What it means
Elixir tried to do normal arithmetic (`+`, `-`, `*`, `/`) but at least one operand was **not a number** (for plain Elixir arithmetic).

### Why it showed up here
In this lesson we build **Nx tensors** (e.g. `%Nx.Tensor{...}`) outside `defn`. Outside of `defn`, the `+` operator is still **normal Elixir arithmetic**, so this crashes:

```elixir
y = Nx.dot(x, true_w) + 0.1
```

`Nx.dot/2` returns an `Nx.Tensor`, and Elixir can’t do `tensor + 0.1` using the normal operator.

### Fix
Use explicit Nx ops outside `defn`:

```elixir
y = Nx.add(Nx.dot(x, true_w), 0.1)
# or:
y = x |> Nx.dot(true_w) |> Nx.add(0.1)
```

### Notes
The stacktrace mentions `Code.require_file/2` because `.exs` scripts are loaded via `Code.require_file/2`. The real problem is the arithmetic on that line.

---
2)

## **(ArgumentError) unknown key :type in [type: {:f, 32}], expected one of [:axes, :names]**
**Error:**
```
** (ArgumentError) unknown key :type in [type: {:f, 32}], expected one of [:axes, :names]
    (nx 0.10.0) lib/nx/defn/kernel.ex:905: Nx.Defn.Kernel.keyword!/2
    (nx 0.10.0) lib/nx.ex:3712: Nx.broadcast/3
    lesson.exs:123: (file)
```

### What it means
You’re passing an option (`type: {:f, 32}`) to `Nx.broadcast/3` that it **doesn’t accept**.

In Nx `0.10.0`, `Nx.broadcast/3` only accepts a small set of keyword options (here it’s telling you: only `:axes` and `:names` are valid). So `type:` is treated as an unknown key and raises.

### Why it might show up in this lesson
It usually happens when you try to do something like:

```elixir
w0 = Nx.broadcast(0.0, {d}, type: {:f, 32})
```

The intention is good (you want float32 weights), but `broadcast` doesn’t set dtype that way.

### Fix options (pick one)

**Option A (recommended): create a typed tensor first, then broadcast**
```elixir
w0 = Nx.broadcast(Nx.tensor(0.0, type: {:f, 32}), {d})
```

**Option B: broadcast, then cast**
```elixir
w0 =
  Nx.broadcast(0.0, {d})
  |> Nx.as_type({:f, 32})
```

### Quick sanity check (debug)
Print the dtype and shape so you know you got what you intended:
```elixir
IO.inspect({:w0, shape: Nx.shape(w0), type: Nx.type(w0)})
```

---
3)

## **(ArgumentError) 1st argument: not a float — :erlang.float_to_binary(:nan, [decimals: 6])**
**Error:**
```
** (ArgumentError) errors were found at the given arguments:
    * 1st argument: not a float
    :erlang.float_to_binary(:nan, [decimals: 6])
    lesson.exs:148: anonymous fn/4 in :elixir_compiler_0.__FILE__/1
    (elixir 1.18.4) lib/enum.ex:4507: Enum.reduce_range/5
    lesson.exs:143: (file)
```

### What it means
You tried to format a number with:
```elixir
:erlang.float_to_binary(value, decimals: 6)
```
…but the value is **not a float**. In this crash, the value is literally the atom `:nan`.

### Why it showed up here
This usually comes from the training loop print line, e.g.:
```elixir
loss_number = Nx.to_number(l2)
IO.puts("loss=#{:erlang.float_to_binary(loss_number, decimals: 6)}")
```

When training goes unstable, the loss can become **NaN** (not-a-number). In your case, it’s showing up as `:nan`, and `float_to_binary/2` refuses it because it only accepts real float values.

### Root causes of NaN in this lesson (common)
NaN typically means the math blew up somewhere:
- Learning rate too high → weights explode → loss becomes NaN
- Inputs too large (or not scaled) → gradients get huge
- Some intermediate produced invalid math (e.g., overflow)

### Fix (two parts)

#### A) Make the print line robust (don’t crash on NaN)
Replace your formatting with a safe formatter:

```elixir
loss_number = Nx.to_number(l2)

loss_str =
  case loss_number do
    n when is_float(n) -> :erlang.float_to_binary(n, decimals: 6)
    n when is_integer(n) -> Integer.to_string(n)
    other -> inspect(other)
  end

IO.puts("step=#{i} loss=#{loss_str}")
```

This won’t crash even if the value is `:nan`.

#### B) Prevent NaN (stabilize training)
Try one or more:
- Reduce `lr` (e.g. `0.25 -> 0.05` or `0.01`)
- Ensure inputs are scaled (your `x |> Nx.divide(50.0)` is good—keep it)
- Keep everything float32 (don’t accidentally create int tensors)

### Quick sanity checks (debug)
Add these when NaNs appear:
```elixir
IO.inspect({:loss, Nx.to_number(l2)})
IO.inspect({:w_type, Nx.type(w2), :w_sample, Nx.to_flat_list(w2) |> Enum.take(5)})
```
If weights become huge or the loss becomes non-finite, dial back `lr`.
