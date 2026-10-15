module OutputSignatures

export output_size, output_eltype, output_signature, alloc_output
export broadcast_size, broadcast_eltype, broadcast_signature, alloc_broadcast

"""
    output_size(f, insize::Tuple, args...) -> Tuple

Size of `f`'s output given the input size `insize` and `args`, **without running
`f`**. Output size is runtime data — *not* recoverable from types (this is why
`Base.mapslices` runs `f` on a pilot slice and is type-unstable) — so it is
DECLARED per function. An operation without a rule raises an `ArgumentError`;
an input-preserving result is never assumed.
"""
output_size(f, insize::Tuple, args...) =
    throw(ArgumentError("no output size rule for $(typeof(f)) with argument types $(typeof(args))"))

"""
    output_eltype(f, ineltype::Type, args...) -> Type

Element type of `f`'s output. Unlike size, eltype is often inferable; declare it
for robustness, or delegate to [`inferred_eltype`](@ref) / `Base.promote_op`.
An operation without a rule raises an `ArgumentError`.
"""
output_eltype(f, ineltype::Type, args...) =
    throw(ArgumentError("no output element-type rule for $(typeof(f)) with argument types $(typeof(args))"))

"""
    output_signature(f, x, args...) -> (; size, eltype)

The `(size, eltype)` of `f(x, args...)`'s output, from immediately-available info
— no pilot run.
"""
output_signature(f, x, args...) =
    (; size = output_size(f, size(x), args...), eltype = output_eltype(f, eltype(x), args...))

"""
    alloc_output(f, x, args...)

Preallocate `f`'s output container with `similar`, from its declared/inferred
signature — no pilot run. Pairs with MutatingFunctions.jl's in-place fill step:
this makes the buffer, that fills it.
"""
function alloc_output(f, x, args...)
    sig = output_signature(f, x, args...)
    similar(x, sig.eltype, sig.size)
end

"""
    inferred_eltype(f, ineltype::Type) -> Type

Opt-in `promote_op`-based eltype for the *elementwise* case. Not the default:
inference can fall back to `Any`, which would silently over-widen an allocation.
"""
inferred_eltype(f, ineltype::Type) = Base.promote_op(f, ineltype)

# These single-argument operations preserve both facts. They are declarations,
# not a general default for arbitrary functions.
for f in (:identity, :copy, :reverse, :sort)
    @eval begin
        output_size(::typeof($f), insize::Tuple) = insize
        output_eltype(::typeof($f), T::Type) = T
    end
end

for (f, op) in ((:cumsum, :(Base.add_sum)), (:cumprod, :(Base.mul_prod)))
    @eval begin
        output_size(::typeof($f), insize::Tuple) = insize
        output_eltype(::typeof($f), T::Type) = Base.promote_op($op, T, T)
    end
end

# --- Base: reductions to a scalar (size ()) -----------------------------------

output_size(::typeof(maximum), ::Tuple) = ()
output_eltype(::typeof(maximum), T::Type) = T
output_size(::typeof(minimum), ::Tuple) = ()
output_eltype(::typeof(minimum), T::Type) = T

output_size(::typeof(count), ::Tuple) = ()
output_eltype(::typeof(count), ::Type) = Int

output_size(::typeof(any), ::Tuple) = ()
output_eltype(::typeof(any), ::Type) = Bool
output_size(::typeof(all), ::Tuple) = ()
output_eltype(::typeof(all), ::Type) = Bool

# sum/prod widen sub-word integers (`Bool`/`Int8`/`Int16`/`Int32`-on-64-bit) to
# `Int` internally — `Base.add_sum`/`Base.mul_prod` are the exact (non-public)
# ops Base reduces with, so `promote_op` against them matches `sum`/`prod`'s
# real return type precisely (user decision: exactness over public-API-only).
output_size(::typeof(sum), ::Tuple) = ()
output_eltype(::typeof(sum), T::Type) = Base.promote_op(Base.add_sum, T, T)
output_size(::typeof(prod), ::Tuple) = ()
output_eltype(::typeof(prod), T::Type) = Base.promote_op(Base.mul_prod, T, T)

# `extrema` is NOT declared: it returns a `Tuple`, which `alloc_output`/`similar`
# cannot produce — doesn't fit this package's array-container contract.

# --- Base: shape transforms with known size -----------------------------------

output_size(::typeof(diff), insize::Tuple{Int}) = (max(insize[1] - 1, 0),)
output_eltype(::typeof(diff), T::Type) = Base.promote_op(-, T, T)
output_size(::typeof(vec), insize::Tuple) = (prod(insize),)
output_eltype(::typeof(vec), T::Type) = T
output_size(::typeof(permutedims), insize::Tuple{Int,Int}) = reverse(insize)
output_size(::typeof(permutedims), insize::Tuple, perm) = map(i -> insize[i], Tuple(perm))
output_eltype(::typeof(permutedims), T::Type) = T
output_eltype(::typeof(permutedims), T::Type, perm) = T

# `transpose`/`adjoint` are NOT declared: both return lazy wrapper types
# (`Transpose`/`Adjoint`) with zero allocation in the general case — there is
# nothing for `alloc_output` to preallocate.

# gather: `A[idx]` with an ARRAY of indices returns an array shaped like `idx`
# (NOT like `A`) — the `b[group]` gather that is ubiquitous in hierarchical
# models. Fits the `x`-first contract cleanly: `getindex(A, idx)`, `A` first.
# Eltype is preserved for array and scalar indexing, and is declared below.
output_size(::typeof(getindex), _insize::Tuple, idx::AbstractArray) = size(idx)
output_size(::typeof(getindex), _insize::Tuple, ::AbstractArray{Bool}) =
    throw(ArgumentError("Boolean-mask gather size depends on mask values"))
output_eltype(::typeof(getindex), T::Type, ::AbstractArray) = T
output_size(::typeof(getindex), _insize::Tuple, ::Integer) = ()
output_eltype(::typeof(getindex), T::Type, ::Integer) = T
output_size(::typeof(getindex), _insize::Tuple, ::Bool) =
    throw(ArgumentError("Boolean indexing needs a separate result-shape rule"))
output_size(::typeof(getindex), _insize::Tuple, ::CartesianIndex) = ()
output_eltype(::typeof(getindex), T::Type, ::CartesianIndex) = T
output_size(::typeof(getindex), insize::Tuple, ::Colon) = (prod(insize),)
output_eltype(::typeof(getindex), T::Type, ::Colon) = T

# multi-dim (Cartesian) gather: `A[i, group]`, `A[:, cols]`, `A[rows, cols]`, … — a
# `getindex` with 2+ index args, where the result shape is the concatenation of each
# index's own axes: a scalar index drops its dim, an `AbstractVector` contributes its
# length, `:` the source dim's extent, a range/`AbstractArray`/`CartesianIndex` its
# own. The single-`AbstractArray`-index method above is the LINEAR-indexing case
# (`A[idx]` → `size(idx)`, whatever `ndims(A)`); THIS is the Cartesian case, and the
# `i1, i2, …` arity (≥2 index args) keeps the two from overlapping — a lone index
# still hits the linear/scalar methods, not this one. Sized with Base's own
# `to_indices`/`index_shape` over `CartesianIndices(insize)` (lazy — no allocation),
# so mixed scalar/vector/colon/range indices size EXACTLY as the real `getindex`
# would — the same match-Base stance as the reductions above. Eltype is preserved
# by an explicit rule. This is the
# `b[i, group]` gather in a grouped-varying model body (BRM's julianic 0-alloc pool).
function output_size(::typeof(getindex), insize::Tuple, i1, i2, irest...)
    idxs = (i1, i2, irest...)
    any(i -> i isa AbstractArray{Bool}, idxs) &&
        throw(ArgumentError("Boolean-mask gather size depends on mask values"))
    map(length, Base.index_shape(Base.to_indices(CartesianIndices(insize), idxs)...))
end
output_eltype(::typeof(getindex), T::Type, i1, i2, irest...) = T

# `count(f, itr)`/`any(f, itr)`/`all(f, itr)`/`map(f, itrs...)`/`filter(f, itr)`
# (predicate/function-first forms) are NOT declared: their real argument order is
# predicate first, data second — the opposite of this package's documented
# `output_size(f, insize, args...)` contract ("size of `f(x, args...)`", `x`
# first). Only the predicate-free single-arg forms above are covered. `broadcast`
# is function-first for the same reason, but it IS the load-bearing PPL case, so
# it ships via its own `broadcast_*`/`alloc_broadcast` API below rather than a
# contorted `output_size(::typeof(broadcast), …)` (see the note there).

# --- broadcast: function-first, so it gets a dedicated API --------------------
#
# `broadcast(f, args...)` (equivalently `f.(args...)`) is function-first — the
# elementwise `f` comes BEFORE the data — so it does not fit the `x`-first
# `output_size(f, insize, args...)` contract, exactly like the `map`/`count`/
# `filter` forms above. A `output_size(::typeof(broadcast), _insize, f, args...)`
# method would also silently BREAK when routed through `alloc_output`/
# `output_signature`: those consume the first DATA arg as `x` and pass only its
# SIZE, dropping the array itself from the axis combination. So broadcast gets its
# own entry points instead. The output SHAPE is the combined broadcast axes of the
# DATA args (independent of `f`); the ELTYPE needs the same inference Base itself
# uses to size a broadcast (`combine_eltypes`), so — like `inferred_eltype` — it
# can widen to `Any` when inference fails. This is the linear predictor / `exp.()`
# / `tau .* (L*z)` case that dominates a PPL log-density body.

"""
    broadcast_size(args...) -> Tuple

Size of `broadcast(f, args...)` (equivalently `f.(args...)`): the combined
broadcast axes of the data `args`, as lengths. Independent of `f`, so `f` is
not an argument. No pilot run.
"""
broadcast_size(args...) = map(length, Base.Broadcast.combine_axes(args...))

"""
    broadcast_eltype(f, args...) -> Type

Element type of `broadcast(f, args...)`, via `Base.Broadcast.combine_eltypes` —
the exact inference `Base` uses to allocate a broadcast's output. May widen to
`Any` when inference cannot pin it down (same caveat as [`inferred_eltype`](@ref)).
"""
broadcast_eltype(f, args...) = Base.Broadcast.combine_eltypes(f, args)

"""
    broadcast_signature(f, args...) -> (; size, eltype)

The `(size, eltype)` of `broadcast(f, args...)`, from immediately-available info
— no pilot run. The function-first companion of [`output_signature`](@ref).
"""
broadcast_signature(f, args...) =
    (; size = broadcast_size(args...), eltype = broadcast_eltype(f, args...))

"""
    alloc_broadcast(f, args...)

Preallocate the output of `broadcast(f, args...)` (`f.(args...)`) — no pilot run.
Uses `similar(Base.broadcasted(f, args...), …)`, so the container type and shape
match what `Base` itself would allocate. Pairs with MutatingFunctions.jl's
`apply!!(cache, broadcast, f, args...)` fill step: this makes the buffer, that
fills it.
"""
alloc_broadcast(f, args...) =
    similar(Base.broadcasted(f, args...), broadcast_eltype(f, args...))

end # module
