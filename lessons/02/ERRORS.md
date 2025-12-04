# Errors encountered (growing log)

This file tracks real errors we hit while building/running Lesson 02, what they mean, and the usual causes/fixes.

---
1)

## **(ArgumentError) expected a shape ... Got: "scalar"**
**Error:**
```
** (ArgumentError) expected a shape. A shape is a n-element tuple with the size of each dimension.
Alternatively, you can pass a tensor (or a number) and the shape will be retrieved from the tensor. Got: "scalar"
    (nx 0.10.0) lib/nx.ex:4162: Nx.shape/1
    lesson.exs:41: L2.sample/4
    lesson.exs:95: (file)
```

### What it means
`Nx.shape/1` was called with a **string** (`"scalar"`) instead of a tensor (or a number).

### Why it showed up here
This is caused by piping into `L2.sample/4` with the wrong argument order:

```elixir
L2.info("scalar", scalar) |> L2.sample("scalar", scalar)
```

Because `|>` inserts the left value as the *first argument*, that becomes:

```elixir
L2.sample(L2.info("scalar", scalar), "scalar", scalar)
```

So inside `sample(label, t, ...)`, `t` is `"scalar"`, and we end up doing:

```elixir
Nx.shape("scalar")
```

### Fix
Don’t pipe here—call them separately:

```elixir
L2.info("scalar", scalar)
L2.sample("scalar", scalar)
```

Or keep a pipeline safely:

```elixir
scalar
|> tap(&L2.info("scalar", &1))
|> tap(&L2.sample("scalar", &1))
```

---
2)

## **(ArgumentError) expected a shape ... Got: "ints"**
**Error:**
```
** (ArgumentError) expected a shape. A shape is a n-element tuple with the size of each dimension. Alternatively, you can pass a tensor (or a number) and the shape will be retrieved from the tensor. Got: "ints"
    (nx 0.10.0) lib/nx.ex:4162: Nx.shape/1
    lesson.exs:41: L2.sample/4
    lesson.exs:111: (file)
```

### What it means
Same exact issue as (1), but this time the string passed to `Nx.shape/1` was `"ints"`.

### Why it showed up here
You likely repeated the same pipe pattern in the “ints vs floats” section:

```elixir
L2.info("ints", ints) |> L2.sample("ints", ints)
```

Which becomes:

```elixir
L2.sample(L2.info("ints", ints), "ints", ints)
```

So `t` becomes `"ints"` and we hit:

```elixir
Nx.shape("ints")
```

### Fix
Call separately:

```elixir
L2.info("ints", ints)
L2.sample("ints", ints)
```

Or use `tap/2`:

```elixir
ints
|> tap(&L2.info("ints", &1))
|> tap(&L2.sample("ints", &1))
```

---
3)

## **(UndefinedFunctionError) function L2.sample/5 is undefined or private**
**Error:**
```
** (UndefinedFunctionError) function L2.sample/5 is undefined or private. Did you mean:

      * sample/2
      * sample/3
      * sample/4

    L2.sample(#Nx.Tensor<...>, "x", #Nx.Tensor<...>, 4, 3)
    lesson.exs:128: (file)
```

### What it means
Elixir tried to call `L2.sample/5` (**sample with 5 arguments**), but your module only defines:

- `sample/2`
- `sample/3`
- `sample/4`

In Elixir, functions are identified by **name + arity** (arity = number of args).  
So `sample/4` and `sample/5` are completely different functions.

### Why it showed up here
This is the *next level* of the same piping pitfall.

You probably wrote something like:

```elixir
L2.info("x", x) |> L2.sample("x", x, 4, 3)
```

But the pipe inserts the left value as the first argument, so you end up with **five** args:

```elixir
L2.sample(L2.info("x", x), "x", x, 4, 3)
#        ^ inserted here
```

Which matches the stacktrace shape:

```elixir
L2.sample(<tensor>, "x", <tensor>, 4, 3)
```

### Fix (recommended)
Call separately:

```elixir
L2.info("x", x)
L2.sample("x", x, 4, 3)
```

### Safe pipeline alternatives
Use `tap/2` to keep a pipeline feel without changing arity:

```elixir
x
|> tap(&L2.info("x", &1))
|> tap(&L2.sample("x", &1, 4, 3))
```

Or use `then/2`:

```elixir
L2.info("x", x)
|> then(fn t -> L2.sample("x", t, 4, 3) end)
```

### Quick mental rule
If the *next* function call already includes the tensor argument, **don’t pipe into it**—you’ll duplicate the tensor and accidentally increase arity.

---
4)

## **(ArgumentError) expected a shape ... Got: "col_means"**
**Error:**
```
** (ArgumentError) expected a shape. A shape is a n-element tuple with the size of each dimension. Alternatively, you can pass a tensor (or a number) and the shape will be retrieved from the tensor. Got: "col_means"
    (nx 0.10.0) lib/nx.ex:4162: Nx.shape/1
    lesson.exs:41: L2.sample/4
    lesson.exs:134: (file)
```

### What it means
Same error family as (1) and (2): `Nx.shape/1` was called with a **string** (`"col_means"`) instead of a tensor.

### Why it showed up here
It happens when the piping mistake (argument swap) is repeated for `col_means`, for example:

```elixir
L2.info("mean over axes: [0] (per-column)", col_means) |> L2.sample("col_means", col_means)
```

The pipe turns that into:

```elixir
L2.sample(L2.info(...), "col_means", col_means)
```

So inside `sample/4`, `t` becomes `"col_means"` and `Nx.shape("col_means")` crashes.

### Fix
Same fix pattern—don’t pipe into `sample/4` like that:

```elixir
L2.info("mean over axes: [0] (per-column)", col_means)
L2.sample("col_means", col_means)
```

Or use `tap/2`:

```elixir
col_means
|> tap(&L2.info("mean over axes: [0] (per-column)", &1))
|> tap(&L2.sample("col_means", &1))
```
