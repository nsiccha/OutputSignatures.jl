module OutputSignaturesLinearAlgebraExt

import OutputSignatures: output_size, output_eltype
import LinearAlgebra

# A*B / A*v: the textbook shape rule, (m,k)*(k,n)->(m,n) or (m,k)*(k,)->(m,).
output_size(::typeof(*), sizeA::Tuple{Int,Int}, B::AbstractMatrix) = (sizeA[1], size(B, 2))
output_eltype(::typeof(*), TA::Type, B::AbstractMatrix) = promote_type(TA, eltype(B))
output_size(::typeof(*), sizeA::Tuple{Int,Int}, ::AbstractVector) = (sizeA[1],)
output_eltype(::typeof(*), TA::Type, v::AbstractVector) = promote_type(TA, eltype(v))

# dot(u, v): scalar, promoted across both vectors' eltypes.
output_size(::typeof(LinearAlgebra.dot), ::Tuple, ::AbstractVector) = ()
output_eltype(::typeof(LinearAlgebra.dot), Tu::Type, v::AbstractVector) = promote_type(Tu, eltype(v))

# norm(v): scalar, always real (norm of a complex vector is a real number).
output_size(::typeof(LinearAlgebra.norm), ::Tuple) = ()
output_eltype(::typeof(LinearAlgebra.norm), T::Type) = real(float(T))

# diag(A): the main diagonal, length min(m,n); same eltype as the input.
output_size(::typeof(LinearAlgebra.diag), sizeA::Tuple{Int,Int}) = (min(sizeA...),)

end # module
