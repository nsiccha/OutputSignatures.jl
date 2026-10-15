# OutputSignatures.jl

Describe a function's **output from numerical metadata without executing it**.
The metadata API works even when intermediate values do not exist yet. The
allocation API is the size/allocate twin of
[MutatingFunctions.jl](https://github.com/nsiccha/MutatingFunctions.jl) (which
*fills* a preallocated buffer): this one tells you how to *make* it.

```julia
infer_call(f, operands::Vector{ValueInfo}) # -> CallInfo, metadata only
output_signature(f, x, args...)   # -> (; size, eltype), no pilot run
alloc_output(f, x, args...)       # -> similar(x, eltype, size)
```

## Metadata queries (v0.1.2)

Every positional operand and keyword value uses a nonparametric `ValueInfo`.
Dimensions, constants, and operand counts are ordinary data, not Julia type
parameters. For example, analyze `sum(alpha .+ beta .* x .+ b[group])` without
constructing its intermediate arrays:

```julia
using OutputSignatures

alpha, beta = scalar_info(Float64), scalar_info(Float64)
x, b, group = vector_info(Float64, 10), vector_info(Float64, 3), vector_info(Int, 10)
gather = infer_call(getindex, ValueInfo[b, group]).value
scaled = infer_call(broadcast, ValueInfo[known_info(*), beta, x]).value
shifted = infer_call(broadcast, ValueInfo[known_info(+), alpha, scaled, gather]).value
result = infer_call(sum, ValueInfo[shifted])
type_bound(result.value)  # Float64
value_axes(result.value)  # ()
```

`CallInfo` contains `.value`, `.status`, and `.reason`. A supported rule returns
`:supported`, possibly with unknown extents. Missing rules or information return
`:unknown` with an explanation and any sound partial facts. A proved broadcast
shape conflict returns `:incompatible`. These facts describe **successful
returns**, not purity or freedom from exceptions. Keep the ordinary operation
and its bounds/shape checks; unknown facts disable only optimizations that need
those facts. For example, broadcasting unknown length with length 3 has length
3 on success, but the unknown operand must still be checked at runtime.

| Constructor/accessor | Meaning |
| --- | --- |
| `ValueInfo()` | Unknown type, rank, axes, and constant |
| `ValueInfo(; type_bound=Any, element_type=Any, axes=nothing)` | Explicit partial facts, without assuming operation semantics |
| `scalar_info(T)` | Numerical scalar with type bound `T` |
| `array_info(T, dims...; container_type=Array{T,N})` | Ordinary one-based Base array; each dim is a length, `SymbolicAxis`, or `nothing` |
| `vector_info(T, n=nothing; ...)`, `matrix_info(T, m=nothing, n=nothing; ...)` | `array_info` for rank 1 and 2 |
| `SymbolicAxis(name)` | A named axis of unavailable length, e.g. `SymbolicAxis(:(levels(g)))` |
| `value_info(x)` | Extract type/axes, without reading elements or asserting constant contents |
| `known_info(x)` | Explicit constant, including a callable, `:` or `nothing` |
| `type_bound(v)`, `element_type(v)` | Sound Julia type and element-type bounds |
| `value_axes(v)`, `value_rank(v)`, `value_length(v)` | Axes, rank, and length; unavailable facts are `nothing` |
| `axis_length(ax)` | A known axis's length, else `nothing` |
| `known_value(v)` | `KnownValue` with a `.value`, or `UnknownValue()` |

`axes=nothing`, `axes=()`, and `axes=(nothing,)` distinguish unknown rank,
rank zero, and a vector of unknown length. Type bounds distinguish a numerical
scalar from a zero-dimensional array. Known empty/singleton axes remain exact.
Array type bounds are covariant bounds where needed: unknown element type does
not imply an exact `Vector{Any}` container.

**Symbolic axes.** A `SymbolicAxis` names a dimension whose length is not
available yet, such as an unbound declaration's `levels(g)` or `axes(M, 2)`.
The name is ordinary data. Equal names denote the same axis and length;
different names may still have equal lengths. Rules carry names through
broadcasts, products, reductions, indexing and concatenation:

```julia
N, L = SymbolicAxis(:N), SymbolicAxis(:(levels(g)))
X, b = matrix_info(Float64, N, L), vector_info(Float64, L)
value_axes(infer_call(*, ValueInfo[X, b]).value)     # (N,)
value_axes(infer_call(adjoint, ValueInfo[b]).value)  # (Base.OneTo(1), L)
value_axes(infer_call(getindex, ValueInfo[X, known_info(:), known_info(2)]).value)  # (N,)
```

Operations whose success requires equal axes (a product's inner dimension,
undotted `+`/`-`, `hcat` rows) let a known length decide the result, then a
name: success implies the other lengths equal it. Known unequal lengths are
`:incompatible`. Broadcasting is different because a length-1 axis
stretches: a symbolic axis may itself have length 1, so only a single
distinct name among singletons stays named. `n .+ n` is `n`, `n .+ 3` is 3,
but `n .+ p` and `n .+ unknown` have unknown length.

`array_info(T, dims...; container_type=AbstractArray{T,N})` supports
unavailable ordinary views, such as a view of packed parameters. It explicitly
asserts **standard Base indexing and broadcast semantics** separately from its
Julia type bound. Do not use it to bless arbitrary custom arrays. `value_info`
recognizes dense and bit arrays, ordinary `SubArray` views of Base arrays with
scalar/range/slice indices, and (with LinearAlgebra loaded) their adjoints and
transposes. Custom arrays retain type/axes facts but receive no
standard-family claim. The adjoint or transpose of a vector is tracked as a
row vector, because Julia multiplies, adds and concatenates it differently
from a 1×n `Matrix` (`v' * w` is a scalar).

Rules cover Base machine integers (including `Bool`) and `Float16/32/64`:

- scalar `+`, `-`, `*`, `/`, `^`, `exp`, `log`, `sqrt`, and their broadcasts
  over standard arrays of any rank;
- Julia's `Base.literal_pow` for a known `^` and integer `Val` exponent;
- indexing: one linear index or one index per axis, each a machine integer,
  `:` (`known_info(:)`), or a standard integer array (Boolean masks keep
  rank with an unknown length);
- `length`, `adjoint`, `transpose`;
- `*` for scalar × array, matrix–vector, matrix–matrix, row-vector products
  (dot, row × matrix, outer products) and chains of these; undotted
  array `+`/`-` and array `/` scalar;
- `sum`, and with the Statistics extension `mean`/`std`, fully or over
  known `dims`;
- `vcat` of scalars/vectors and `hcat` of scalars, vectors and matrices.

Arithmetic rules account for Boolean results, integer-power typing, `sum`
widening and LinearAlgebra's product element type. The closed primitive family
is the boundary where these Base rules apply; custom numerical types require
their own justified rules, not guessed promotion. Where Julia's own result
type depends on the Julia version or on runtime sizes, the fact stays unknown:
the dot product of two `Bool` vectors, and the element type of a chained
product with `Bool` operands. `broadcast(f, v', 1)` returns a row vector
whereas `v' .+ 1` returns a `Matrix`, so a broadcast over row vectors and
scalars only keeps partial facts.
Literal powers retain Julia's special semantics: `Int8[2] .^ -1` produces
`Vector{Float64}`, whereas `broadcast(^, Int8[2], -1)` throws. Describe the
former with `infer_call(broadcast, ValueInfo[known_info(Base.literal_pow),
known_info(^), vector_info(Int8, 1), known_info(Val(-1))])`. A scalar literal
power uses `infer_call(Base.literal_pow, ValueInfo[known_info(^),
scalar_info(Int8), known_info(Val(-1))])`. The exponent is an explicitly supplied
Julia value, never an exponent parameter on an OS descriptor or traversal.
Boolean masks retain vector rank but unknown length without inspecting values.
Custom helpers are never run, even inside broadcast; broadcast can retain axes
while its element result remains unknown. No Julia compiler inference is used.

Keywords have the same descriptor convention,
`kwargs=Pair{Symbol,ValueInfo}[:dims => known_info(1)]`. The reductions above
accept a known `dims`; other keywords, structured returns, arithmetic between
dimension names (`n + p`), and general control flow return unknown.
Rules assume the documented Base methods have not been redefined. Known mutable
constants and boundary shapes must remain valid for the analysis using them.

## Allocation API

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
allocation API; use `infer_call` for unavailable intermediate operands.

`broadcast(f, args...)` is **function-first** (the elementwise `f` comes before
the data), so it can't use the `x`-first `output_size` contract. It gets a
dedicated, parallel API instead:

```julia
broadcast_size(args...)          # combined broadcast axes of the data args, as lengths
broadcast_eltype(f, args...)     # via Base's own combine_eltypes
broadcast_signature(f, args...)  # -> (; size, eltype)
alloc_broadcast(f, args...)      # similar(Base.broadcasted(f, args...), …)
```

The allocation functions remain source compatible. They keep their explicit
unsupported-call errors and opt-in inference behavior; `infer_call` never
delegates to them. This bridge belongs to OutputSignatures and remains for its
existing allocation consumers until those consumers migrate. It does not
restrict the metadata representation or make unknown facts allocatable.

## Compiler migration plan

The allocation API preallocates from an actual first input. It cannot propagate
signatures through a graph whose intermediate arrays do not exist yet.
ReactiveKernels (RK) and ReactiveKernelsPPL (RKPPL) need a separate, uniform
metadata query. The bounded foundation above is implemented in `v0.1.1`;
`v0.1.2` adds symbolic axes and the matrix, product, concatenation and
reduction rules RKPPL's declaration shape inference needs. The following
remains the broader target and incremental migration scope; tuple components
and arithmetic between dimension names are not implemented yet.

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
1. **Minimal OS foundation (implemented):** one metadata-only calling convention for all
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
