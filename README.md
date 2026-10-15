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

- **size** — runtime data, declared per supported call shape.
- **eltype** — declared per supported call shape, or computed from the operand
  types by a rule that explicitly chooses inference.

There is **no input-preserving default**. An unsupported `output_size`,
`output_eltype`, `output_signature`, or `alloc_output` call raises an
`ArgumentError` instead of reporting invented facts or allocating from them.
For example, `output_signature(x -> length(x), v)` is unsupported even though
that function can be run: no rule has established its scalar result. A caller
analyzing arbitrary Julia calls must retain unknown information or ordinary
runtime behavior rather than treating an unsupported call as shape preserving.

```julia
using OutputSignatures, Statistics
output_size(quantile, size(v), p)   # (length(p),)  — no pilot, no length(v) dependence
alloc_output(quantile, v, p)        # Vector{Float64}(undef, length(p))
```

Opt in for a function by adding **both** `output_size` and `output_eltype`
methods for its supported argument forms. Common cases ship declared — Base
reductions (`sum`/`prod`/`maximum`/…) and transforms
(`diff`/`vec`/`permutedims`), explicitly preserving operations
(`identity`/`copy`/`reverse`/`sort`), scans (`cumsum`/`cumprod`, including
integer widening), and `getindex` (scalar and array indices, except Boolean
masks). Cartesian `A[i, group]` / `A[:, cols]` with 2+ indices follow Base's
result shape. The `Statistics` (`quantile`/`mean`/…) and `LinearAlgebra`
(`*`/`dot`/`norm`/`diag`) extensions provide additional rules.

A Boolean mask has a value-dependent output length. The current allocation
query refuses it instead of scanning the mask during signature analysis.

Every declared method follows one contract — **the first arg is the input SIZE,
the rest are the function's ACTUAL args**: `output_size(getindex, size(A), i, idx)`,
`output_size(*, size(L), z)` (the whole `z` matrix, not `size(z)`). Pass a size where
an arg belongs and the call is unsupported or fails; no fallback repairs it.
`output_signature`/`alloc_output` take the arrays directly and thread the sizing
for supported operations. This first-input-size convention is specific to the
current allocation API; it cannot describe unavailable intermediate operands
uniformly and is not the compiler metadata contract planned for RK/RKPPL.

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

## Compiler migration plan

The current API preallocates from an actual first input. It cannot propagate
signatures through a graph whose intermediate arrays do not exist yet.
ReactiveKernels (RK) and ReactiveKernelsPPL (RKPPL) need a separate, uniform
metadata query. The following is the target contract and migration scope; it
is **not** an API that `v0.1.0` already implements.

**Rules for the new query:**

1. Describe every positional argument and keyword value with the same
   nonparametric value descriptor. A callable passed as an argument is a value
   too. No rule receives one operand's size and another operand's live array.
2. Keep Julia value type, array element type, rank and individual axes,
   tuple/named-tuple components, tagged known constants, and simple symbolic
   dimension relationships as separate facts. Unknown rank is not scalar;
   unknown length is not a guessed input length. Facts must hold for every
   successful call consistent with the inputs.
3. Missing rules return unknown or partial facts. An optimization or
   allocation requiring an unknown fact retains the runtime operation or
   refuses that optimization. A query never executes the numerical call,
   samples parameters, scans model data, or assumes purity or absence of
   effects. Known constants are explicit; an exemplar's contents do not
   silently become constants.
4. OutputSignatures owns numerical transfer rules and the shared meaning of
   unknown facts. RK owns graph traversal, dependencies, effects, aliasing,
   guards and code generation. RKPPL owns distributions and observation-axis
   meaning; equal numeric lengths alone do not imply the same observation
   axis.
5. Keep descriptors, keywords and dimension expressions as ordinary cold
   data. Array lengths, constants, argument counts and whole expression trees
   must not enter their Julia type parameters. Specialize the executable at
   the final RK compilation boundary, and measure preparation cost rather
   than assuming the representation made it cheap.

For the first chain, the shared numerical facts are concrete: `alpha` and
`beta` are scalars; `x`, `b` and `group` are vectors whose lengths may be
symbolic; `b[group]` is a vector indexed by `group`; the broadcast result is
a vector only when its axes are compatible; `sum` reduces that vector to a
scalar with the rule's justified result type. The current duplicate to
replace is the **rank/length** part of RKPPL's `_PointwiseValue` propagation
in `packages/ReactiveKernelsPPL/src/generator.jl`. Its sampled-value
dependence has a different purpose and remains with RKPPL. Neither a matching
numeric length nor a scalar rank establishes observation alignment.

**Incremental delivery:**

0. **Guidance and safety now:** reject undeclared signatures in the current
   allocation API, declare supported preserving rules explicitly, compare
   them with Julia results, and inventory the numerical facts consumed by the
   first compiler slice. The rejection is interim protection, not the final
   behavior for arbitrary valid Julia calls.
1. **Minimal OS foundation:** add one metadata-only calling convention for all
   positional and keyword arguments. Start with a nonparametric descriptor
   containing a Julia type bound, partial rank/axes and an explicit known-value
   tag. Unknown is a first-class fact. Implement only the rules needed by the
   first consumer path: ordinary numerical scalars and one-based dense
   vectors, scalar arithmetic, scalar/vector integer indexing, pointwise
   broadcast arithmetic, and an ordinary sum reduction. All operands,
   including unavailable intermediates, use descriptors. A missing rule
   stays unknown rather than rejecting an otherwise valid program.
2. **One RK/RKPPL path:** propagate the shared facts over one RK graph such
   as `sum(alpha .+ beta .* x .+ b[group])`, then use the same numerical facts
   in one RKPPL predictor. Replace the rank/length portion of RKPPL's
   `_PointwiseValue` analysis on this path; retain sampled-value dependence
   and observation alignment as separate PPL facts. Check values and ordinary
   Enzyme gradients, then measure graph construction, preparation, first
   primal and first gradient across expression counts and data lengths.
   Compare baseline and candidate in the same environment, including
   generated-code size and specialization evidence. A longer observation
   vector must not unroll into a larger emitted program.
3. **Expand operation by operation:** add matrix products, reductions with
   `dims`, lazy wrappers, structured returns, joins and value-dependent
   operations as justified by consumers. Each batch removes a named
   duplicate rule or fixes a demonstrated gap; unknown lengths stay unknown
   until a runtime computation or explicit constant proves them. Retire the
   first-input-size allocation convention once consumers have migrated;
   preserving it is not a constraint on the replacement.

Phases 1 and 2 form the first implementation scope. The first slice is
intentionally bounded: no universal Julia tracer, full RKPPL rewrite, RK
planner replacement, general buffer-reuse engine, or execution of a helper
merely to discover its shape. A helper with an opaque body can have a declared
boundary signature or remain partial; inspecting its internals for fusion is
separate work. New compiler code should not add a PPL-only numerical rule for
an operation that belongs in the shared layer, and a migrated rule should
replace its old numerical copy in the same slice.

## License

MIT — see [LICENSE](LICENSE).
