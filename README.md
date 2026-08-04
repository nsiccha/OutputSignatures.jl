# OutputSignatures.jl

Compute a function's **output size and element type from immediately-available
info — without running a pilot element**. The size/allocate twin of
[MutatingFunctions.jl](https://github.com/nsiccha/MutatingFunctions.jl) (which
*fills* a preallocated buffer): this one tells you how to *make* it.

```julia
output_signature(f, x, args...)   # -> (; size, eltype), no pilot run
alloc_output(f, x, args...)       # -> similar(x, eltype, size)
```

## Why

`broadcast` can preallocate its output because the **shape is fixed by the input
axes** and only the **eltype** needs inference (`promote_op`). A reduction /
reshaping kernel is harder: its output **shape** is runtime data (e.g.
`quantile(v, p)` -> `length(p)` values, independent of `length(v)`), and **shape
is not recoverable from types**. That is exactly why `Base.mapslices` runs `f` on
the first slice (a *pilot*) to size + type its output — and is
[type-unstable](https://github.com/JuliaLang/julia/issues/26232) as a result.

`OutputSignatures` avoids the pilot by **declaring** the output signature:

- **size** — runtime data => declared per function (default: same shape).
- **eltype** — inferable => declared or delegated to `promote_op` (default: same eltype).

```julia
using OutputSignatures, Statistics
output_size(quantile, size(v), p)   # (length(p),)  — no pilot, no length(v) dependence
alloc_output(quantile, v, p)        # Vector{Float64}(undef, length(p))
```

Opt in for a function by adding `output_size` / `output_eltype` methods. Common
cases ship declared — Base reductions (`sum`/`prod`/`maximum`/…) + shape transforms
(`diff`/`vec`/`permutedims`), the `A[idx]` **gather** (`getindex` with an array of
indices → shape of `idx`), plus `Statistics` (`quantile`/`mean`/…) and
`LinearAlgebra` (`*`/`dot`/`norm`/`diag`) extensions.

`broadcast(f, args...)` is **function-first** (the elementwise `f` comes before
the data), so it can't use the `x`-first `output_size` contract. It gets a
dedicated, parallel API instead:

```julia
broadcast_size(args...)          # combined broadcast axes of the data args, as lengths
broadcast_eltype(f, args...)     # via Base's own combine_eltypes
broadcast_signature(f, args...)  # -> (; size, eltype)
alloc_broadcast(f, args...)      # similar(Base.broadcasted(f, args...), …)
```

Standalone; `v0.1.0`.
