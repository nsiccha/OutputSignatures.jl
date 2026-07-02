module OutputSignatures

export output_size, output_eltype, output_signature, alloc_output

"""
    output_size(f, insize::Tuple, args...) -> Tuple

Size of `f`'s output given the input size `insize` and `args`, **without running
`f`**. Output size is runtime data — *not* recoverable from types (this is why
`Base.mapslices` runs `f` on a pilot slice and is type-unstable) — so it is
DECLARED per function. Default: same shape as the input (the broadcast/`map` case).
"""
output_size(f, insize::Tuple, args...) = insize

"""
    output_eltype(f, ineltype::Type, args...) -> Type

Element type of `f`'s output. Unlike size, eltype is often inferable; declare it
for robustness, or delegate to [`inferred_eltype`](@ref) / `Base.promote_op`.
Default: same as the input eltype.
"""
output_eltype(f, ineltype::Type, args...) = ineltype

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
alloc_output(f, x, args...) =
    similar(x, output_eltype(f, eltype(x), args...), output_size(f, size(x), args...))

"""
    inferred_eltype(f, ineltype::Type) -> Type

Opt-in `promote_op`-based eltype for the *elementwise* case. Not the default:
inference can fall back to `Any`, which would silently over-widen an allocation.
"""
inferred_eltype(f, ineltype::Type) = Base.promote_op(f, ineltype)

# --- Base: reductions to a scalar (size ()) -----------------------------------

output_size(::typeof(maximum), ::Tuple) = ()
output_size(::typeof(minimum), ::Tuple) = ()

output_size(::typeof(count), ::Tuple) = ()
output_eltype(::typeof(count), ::Type) = Int

output_size(::typeof(any), ::Tuple) = ()
output_eltype(::typeof(any), ::Type) = Bool
output_size(::typeof(all), ::Tuple) = ()
output_eltype(::typeof(all), ::Type) = Bool

# `sum`/`prod` are deliberately NOT declared here: Base widens sub-word integers
# (`Bool`/`Int8`/`Int16`/`Int32`-on-64-bit) to `Int` via non-public
# `Base.add_sum`/`Base.mul_prod`, so a public-API-only eltype declaration would be
# WRONG (not just missing) for those inputs — a silent misallocation, exactly what
# this package exists to prevent. Falls back to `output_signature`'s default
# (input eltype), which is at least honestly wrong-shaped (whole-array `size`) for
# a caller who forgot to declare, rather than plausible-looking and wrong.

# `extrema` is NOT declared: it returns a `Tuple`, which `alloc_output`/`similar`
# cannot produce — doesn't fit this package's array-container contract.

# --- Base: shape transforms with known size -----------------------------------

output_size(::typeof(diff), insize::Tuple{Int}) = (insize[1] - 1,)
output_size(::typeof(vec), insize::Tuple) = (prod(insize),)
output_size(::typeof(permutedims), insize::Tuple{Int,Int}) = reverse(insize)
output_size(::typeof(permutedims), insize::Tuple, perm) = map(i -> insize[i], Tuple(perm))

# `transpose`/`adjoint` are NOT declared: both return lazy wrapper types
# (`Transpose`/`Adjoint`) with zero allocation in the general case — there is
# nothing for `alloc_output` to preallocate.

# `count(f, itr)`/`any(f, itr)`/`all(f, itr)`/`map(f, itrs...)`/`filter(f, itr)`
# (predicate/function-first forms) are NOT declared: their real argument order is
# predicate first, data second — the opposite of this package's documented
# `output_size(f, insize, args...)` contract ("size of `f(x, args...)`", `x`
# first). Only the predicate-free single-arg forms above are covered.

end # module
