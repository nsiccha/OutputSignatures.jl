module OutputSignaturesLinearAlgebraExt

import OutputSignatures: output_size, output_eltype
import OutputSignatures: _standard_storage, _array_family
import LinearAlgebra
using LinearAlgebra: Adjoint, Transpose

# A*B / A*v: the textbook shape rule, (m,k)*(k,n)->(m,n) or (m,k)*(k,)->(m,).
output_size(::typeof(*), sizeA::Tuple{Int,Int}, B::AbstractMatrix) = (sizeA[1], size(B, 2))
_product_eltype(::Type{Bool}, ::Type{Bool}) = Int
_product_eltype(TA::Type, TB::Type) = promote_type(TA, TB)
output_eltype(::typeof(*), TA::Type, B::AbstractMatrix) = _product_eltype(TA, eltype(B))
output_size(::typeof(*), sizeA::Tuple{Int,Int}, ::AbstractVector) = (sizeA[1],)
output_eltype(::typeof(*), TA::Type, v::AbstractVector) = _product_eltype(TA, eltype(v))

# dot(u, v): scalar, promoted across both vectors' eltypes.
output_size(::typeof(LinearAlgebra.dot), ::Tuple, ::AbstractVector) = ()
output_eltype(::typeof(LinearAlgebra.dot), Tu::Type, v::AbstractVector) = _product_eltype(Tu, eltype(v))

# norm(v): scalar, always real (norm of a complex vector is a real number).
output_size(::typeof(LinearAlgebra.norm), ::Tuple) = ()
output_eltype(::typeof(LinearAlgebra.norm), T::Type) = real(float(T))

# diag(A): the main diagonal, length min(m,n); same eltype as the input.
output_size(::typeof(LinearAlgebra.diag), sizeA::Tuple{Int,Int}) = (min(sizeA...),)
output_eltype(::typeof(LinearAlgebra.diag), T::Type) = T

# Metadata input boundary: an adjoint/transpose of standard storage indexes and
# broadcasts like it; a flipped vector is additionally a row vector.
_standard_storage(x::Union{Adjoint,Transpose}) = _standard_storage(parent(x))
_array_family(x::Adjoint{<:Any,<:AbstractVector}) = _row_family(x, :adjoint_vector)
_array_family(x::Transpose{<:Any,<:AbstractVector}) = _row_family(x, :transpose_vector)
_row_family(x, kind::Symbol) =
    _standard_storage(parent(x)) && axes(parent(x), 1) isa Base.OneTo{Int} ? kind : :unknown

end # module
