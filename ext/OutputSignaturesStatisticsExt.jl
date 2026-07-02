module OutputSignaturesStatisticsExt

import OutputSignatures: output_size, output_eltype
import Statistics

# quantile(v, p::AbstractVector) → length(p) values — the textbook case where the
# output SIZE is a runtime value only `p` knows, independent of length(v).
output_size(::typeof(Statistics.quantile), _insize::Tuple, p::AbstractVector) = (length(p),)
output_eltype(::typeof(Statistics.quantile), T::Type, p::AbstractVector) = float(T)

# quantile(v, p::Number) → a single scalar. Without this method, the default
# (same shape as `v`) silently misallocates — `p` isn't a vector, so the
# `AbstractVector` method above doesn't apply either.
output_size(::typeof(Statistics.quantile), _insize::Tuple, ::Number) = ()
output_eltype(::typeof(Statistics.quantile), T::Type, ::Number) = float(T)

# mean/std/var/median: full reduction to a scalar, always floating-point.
output_size(::typeof(Statistics.mean), ::Tuple) = ()
output_eltype(::typeof(Statistics.mean), T::Type) = float(T)
output_size(::typeof(Statistics.std), ::Tuple) = ()
output_eltype(::typeof(Statistics.std), T::Type) = float(T)
output_size(::typeof(Statistics.var), ::Tuple) = ()
output_eltype(::typeof(Statistics.var), T::Type) = float(T)
output_size(::typeof(Statistics.median), ::Tuple) = ()
output_eltype(::typeof(Statistics.median), T::Type) = float(T)

end # module
