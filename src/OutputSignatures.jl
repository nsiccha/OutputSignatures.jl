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
signature — no pilot run. Pairs with `MutatingFunctions.forward`: this makes the
buffer, `forward` fills it.
"""
alloc_output(f, x, args...) =
    similar(x, output_eltype(f, eltype(x), args...), output_size(f, size(x), args...))

"""
    inferred_eltype(f, ineltype::Type) -> Type

Opt-in `promote_op`-based eltype for the *elementwise* case. Not the default:
inference can fall back to `Any`, which would silently over-widen an allocation.
"""
inferred_eltype(f, ineltype::Type) = Base.promote_op(f, ineltype)

end # module
