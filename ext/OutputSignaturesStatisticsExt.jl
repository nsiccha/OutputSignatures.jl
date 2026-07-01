module OutputSignaturesStatisticsExt

import OutputSignatures: output_size, output_eltype
import Statistics

# quantile(v, p::AbstractVector) → length(p) values — the textbook case where the
# output SIZE is a runtime value only `p` knows, independent of length(v).
output_size(::typeof(Statistics.quantile), _insize::Tuple, p::AbstractVector) = (length(p),)
output_eltype(::typeof(Statistics.quantile), T::Type, p::AbstractVector) = float(T)

end # module
