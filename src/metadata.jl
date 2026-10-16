export ValueInfo, CallInfo, KnownValue, UnknownValue, SymbolicAxis
export value_info, known_info, scalar_info, vector_info, matrix_info, array_info, infer_call
export type_bound, element_type, value_axes, value_rank, value_length, known_value, axis_length

"""An explicitly unavailable constant (distinct from the known value `nothing`)."""
struct UnknownValue end

"""An explicitly supplied constant. Its value is ordinary data, not a type parameter."""
struct KnownValue
    value::Any
end

"""
    SymbolicAxis(name)

An axis whose length is not available yet but which has an identity, such as
an unbound declaration's `levels(g)` or `axes(M, 2)`. The name is ordinary data,
never a type parameter. Equal names denote the same axis, hence the same
length; different names may still have equal lengths. A symbolic axis can have
length 0 or 1, so broadcasting never assumes it is non-singleton.
"""
struct SymbolicAxis
    name::Any
end
Base.:(==)(a::SymbolicAxis, b::SymbolicAxis) = isequal(a.name, b.name)
Base.hash(a::SymbolicAxis, h::UInt) = hash(a.name, hash(SymbolicAxis, h))
Base.show(io::IO, a::SymbolicAxis) = print(io, "SymbolicAxis(", repr(a.name), ")")

"""
    axis_length(axis)

The length of a known axis, or `nothing` for a [`SymbolicAxis`](@ref) or an
unknown (`nothing`) axis.
"""
axis_length(ax::AbstractUnitRange) = length(ax)
axis_length(::Union{Nothing,SymbolicAxis}) = nothing

"""
    ValueInfo()
    ValueInfo(; type_bound=Any, element_type=Any, axes=nothing)

Cold numerical facts about a value, including one that does not exist yet.
`type_bound` and `element_type` are upper bounds. `axes=nothing` means unknown
rank; `()` means rank zero; `(nothing,)` means a vector of unknown length.
Each axis is an immutable integer unit range, a [`SymbolicAxis`](@ref), or
`nothing`. This general constructor makes no claim about a container's
indexing or broadcast methods. Prefer [`scalar_info`](@ref),
[`array_info`](@ref), or [`value_info`](@ref) when their stronger contracts
apply. All supplied facts must describe the same value.
"""
struct ValueInfo
    julia_type::Type
    element_type::Type
    axes::Union{Nothing,Tuple}
    constant::Union{UnknownValue,KnownValue}
    family::Symbol
end

_is_axis(a) = a === nothing || a isa AbstractUnitRange{Int} || a isa SymbolicAxis
function ValueInfo(; type_bound::Type=Any, element_type::Type=Any, axes=nothing)
    axes === nothing || all(_is_axis, axes) ||
        throw(ArgumentError("axes must contain integer unit ranges, SymbolicAxis entries, or nothing"))
    ValueInfo(type_bound, element_type, axes, UnknownValue(), :unknown)
end

"""Describe a numerical scalar of type `T` (or a subtype of `T`), without a value."""
function scalar_info(T::Type=Number)
    T <: Number || throw(ArgumentError("scalar_info requires a Number type bound"))
    ValueInfo(T, T, (), UnknownValue(), :scalar)
end

_axis(n::Integer) = n >= 0 ? Base.OneTo(Int(n)) :
    throw(ArgumentError("an axis length must be nonnegative"))
_axis(a::Union{Nothing,SymbolicAxis}) = a
_axis(@nospecialize(a)) =
    throw(ArgumentError("an axis is a nonnegative length, a SymbolicAxis, or nothing; got $(repr(a))"))

"""
    array_info(T, dims...; container_type=Array{T,length(dims)})

Describe an ordinary one-based Base array with element type `T` and rank
`length(dims) ≥ 1`. Each dimension is a nonnegative length, a
[`SymbolicAxis`](@ref), or `nothing` (unknown length).
`container_type=AbstractArray{T,N}` permits an unavailable dense array or
ordinary view without claiming a concrete container. This constructor
*asserts* standard Base indexing and broadcast semantics independently of that
type bound. It must not describe an arbitrary custom AbstractArray.
`value_info(x)` checks this family for actual inputs.
"""
function array_info(T::Type, dims...; container_type::Type=Array{T,length(dims)})
    isempty(dims) && throw(ArgumentError("array_info requires rank ≥ 1; use scalar_info for scalars"))
    container_type <: AbstractArray{T,length(dims)} ||
        throw(ArgumentError("container_type must be a subtype of AbstractArray{T,$(length(dims))}"))
    ValueInfo(container_type, T, map(_axis, dims), UnknownValue(), :standard_array)
end

"""
    vector_info(T, n=nothing; container_type=Vector{T})

[`array_info`](@ref) for a vector. `n` is a nonnegative length, a
[`SymbolicAxis`](@ref), or `nothing`. For example,
`container_type=AbstractVector{T}` describes an unavailable vector view.
"""
vector_info(T::Type, n=nothing; container_type::Type=Vector{T}) =
    array_info(T, n; container_type)

"""
    matrix_info(T, m=nothing, n=nothing; container_type=Matrix{T})

[`array_info`](@ref) for a matrix with `m` rows and `n` columns.
"""
matrix_info(T::Type, m=nothing, n=nothing; container_type::Type=Matrix{T}) =
    array_info(T, m, n; container_type)

type_bound(v::ValueInfo) = v.julia_type
element_type(v::ValueInfo) = v.element_type
value_axes(v::ValueInfo) = v.axes
value_rank(v::ValueInfo) = v.axes === nothing ? nothing : length(v.axes)
known_value(v::ValueInfo) = v.constant
function value_length(v::ValueInfo)
    v.axes === nothing && return nothing
    n = 1
    for ax in v.axes
        m = axis_length(ax)
        m === nothing && return nothing
        n = Base.checked_mul(n, m)
    end
    n
end

# This is a semantic boundary, not a guess from AbstractVector's type name.
# User-defined arrays/numbers may change broadcast/indexing/arithmetic methods.
const _MachineInteger = Union{Bool,Base.BitInteger}
const _MachineReal = Union{_MachineInteger,Float16,Float32,Float64}
_machine(T::Type) = isconcretetype(T) && T <: _MachineReal
_standard_storage(::Any) = false
_standard_storage(::Array) = true
_standard_storage(::BitArray) = true
_standard_index(::Any) = false
_standard_index(::_MachineInteger) = true
_standard_index(::Union{Base.OneTo{Int},UnitRange{Int},StepRange{Int,Int}}) = true
_standard_index(::Base.Slice{Base.OneTo{Int}}) = true
_standard_storage(x::SubArray) =
    _standard_storage(getfield(x, :parent)) && all(_standard_index, getfield(x, :indices))
# The LinearAlgebra extension adds the adjoint/transpose row-vector families.
_array_family(x::AbstractArray) =
    ndims(x) >= 1 && _standard_storage(x) && all(a -> a isa Base.OneTo{Int}, axes(x)) ?
        :standard_array : :unknown

"""
    value_info(x)

Extract type and axes at an input boundary, without reading array elements or
declaring their values constant. Custom arrays retain their facts but do not
acquire the standard-array family. No numerical operation is evaluated.
"""
value_info(@nospecialize(x)) = ValueInfo(type_bound=typeof(x))
value_info(x::Number) = scalar_info(typeof(x))
value_info(x::AbstractArray) = ValueInfo(typeof(x), eltype(x), axes(x), UnknownValue(), _array_family(x))

"""As `value_info(x)`, but explicitly declare `x` constant. Never mutates `x`."""
function known_info(@nospecialize(x))
    v = value_info(x)
    ValueInfo(v.julia_type, v.element_type, v.axes, KnownValue(x), v.family)
end

"""
Result of `infer_call`: `.value::ValueInfo`, `.status`, and `.reason::String`.
`:supported` means a rule applies, even if some extents remain unknown;
`:unknown` means missing coverage/information (possibly with partial facts);
`:incompatible` means a proved shape conflict. Facts describe successful returns
only. They do not prove purity, termination, or absence of bounds/shape errors.
"""
struct CallInfo
    value::ValueInfo
    status::Symbol
    reason::String
end
_supported(v::ValueInfo) = CallInfo(v, :supported, "")
_unknown(reason::String, v::ValueInfo=ValueInfo()) = CallInfo(v, :unknown, reason)
_incompatible(reason::String) = CallInfo(ValueInfo(), :incompatible, reason)

"""
    infer_call(f, args::Vector{ValueInfo}; kwargs=Pair{Symbol,ValueInfo}[]) -> CallInfo

Analyze a Julia call using only metadata. Every operand, including broadcast's
callable and keyword values, has the same representation. No numerical helper,
compiler inference, or legacy allocation signature is invoked. The covered
operations are listed in the README; unsupported cases return explicit unknown
facts.

The original runtime operation must retain its bounds and shape checks. In
particular, broadcast facts may be exact *conditional on compatible axes*.
Rules describe the Base and LinearAlgebra methods of these functions (and
Statistics' or LogExpFunctions', when that extension is loaded); redefining
them invalidates the rules.
"""
function infer_call(@nospecialize(f), args::Vector{ValueInfo};
                    kwargs::Vector{Pair{Symbol,ValueInfo}}=Pair{Symbol,ValueInfo}[])
    _infer_call(f, args, kwargs)
end
_infer_call(@nospecialize(f), args::Vector{ValueInfo}, kwargs::Vector{Pair{Symbol,ValueInfo}}) =
    isempty(kwargs) ? _infer_call(f, args) :
        _unknown("keyword arguments have no metadata rule for this call")
_infer_call(@nospecialize(f), ::Vector{ValueInfo}) = _unknown("no metadata rule for this callable")

# ── families and result descriptors ─────────────────────────────────────

# :standard_array promises Base's one-based indexing and broadcasting. An
# adjoint/transpose of a vector indexes and broadcasts the same way, but Julia
# multiplies, adds and concatenates it as a row vector (`v' * w` is a scalar).
_is_scalar(v::ValueInfo) = v.family === :scalar
_machine_scalar(v::ValueInfo) = _is_scalar(v) && _machine(type_bound(v))
_row(v::ValueInfo) = v.family === :adjoint_vector || v.family === :transpose_vector
_array(v::ValueInfo) = (v.family === :standard_array || _row(v)) && something(value_rank(v), 0) >= 1
_vector(v::ValueInfo) = v.family === :standard_array && value_rank(v) == 1
_matrix(v::ValueInfo) = v.family === :standard_array && value_rank(v) == 2

_element_info(T::Type) = T <: Number ? scalar_info(T) : ValueInfo(type_bound=T)
_array_result(T::Type, axs::Tuple; container_type::Type) =
    ValueInfo(container_type, T, axs, UnknownValue(), :standard_array)
_row_result(kind::Symbol, T::Type, ax) =
    ValueInfo(AbstractMatrix{T}, T, (Base.OneTo(1), ax), UnknownValue(), kind)
# Base materializes elementwise results as Arrays, packing Bool into BitArrays.
# Arithmetic on a wrapper such as a flipped matrix may keep the wrapper, so
# only operands of plain storage promise the materialized container.
_materialized(T::Type, N::Int) = T === Bool ? BitArray{N} : Array{T,N}
_plain(v::ValueInfo) = type_bound(v) <: Union{Array,BitArray,SubArray}
_elementwise_container(T::Type, N::Int, operands) =
    all(_plain, operands) ? _materialized(T, N) : AbstractArray{T,N}
_rebuilt(v::ValueInfo, T::Type) =
    _row(v) ? ValueInfo(AbstractMatrix{T}, T, value_axes(v), UnknownValue(), v.family) :
        _array_result(T, value_axes(v); container_type=_elementwise_container(T, value_rank(v), (v,)))

# ── axis algebra ────────────────────────────────────────────────────────

struct _AxisConflict end
const _CONFLICT = _AxisConflict()

# Axes that must be equal for the call to succeed (contractions, `+`,
# concatenated rows). A known length decides the result; otherwise a name
# does, since success implies the unknown or other named lengths equal it.
function _equal_axis(axs)
    known = nothing
    named = nothing
    for a in axs
        if a isa AbstractUnitRange
            known === nothing ? (known = a) : length(known) == length(a) || return _CONFLICT
        elseif a isa SymbolicAxis && named === nothing
            named = a
        end
    end
    known === nothing ? named : known
end

# One broadcast dimension; operands of lower rank behave as singletons. A known
# non-singleton length decides the successful result (0 with 1 is 0). Only one
# distinct name with singletons stays named: an unknown or second name may be
# the stretched singleton.
function _broadcast_axis(axs)
    known = nothing
    named = nothing
    ambiguous = false
    for a in axs
        if a isa AbstractUnitRange
            length(a) == 1 && continue
            known === nothing ? (known = a) : length(known) == length(a) || return _CONFLICT
        elseif a isa SymbolicAxis
            named === nothing ? (named = a) : named == a || (ambiguous = true)
        else
            ambiguous = true
        end
    end
    known !== nothing && return known
    ambiguous && return nothing
    named === nothing ? Base.OneTo(1) : named
end

# The axis of a concatenation along one dimension: a single part keeps its
# axis (named or not); several parts are known only when every length is.
function _concatenated_axis(axs)
    length(axs) == 1 && return only(axs)
    n = 0
    for a in axs
        m = axis_length(a)
        m === nothing && return nothing
        n = Base.checked_add(n, m)
    end
    Base.OneTo(n)
end

# The single axis of `A[:]` (linear order). Known lengths multiply; a known
# zero empties it; one named axis among singletons keeps its name.
function _linear_axis(axs)
    n = 1
    rest = Any[]
    for a in axs
        m = axis_length(a)
        m == 0 && return Base.OneTo(0)
        m === nothing ? push!(rest, a) : (n = Base.checked_mul(n, m))
    end
    isempty(rest) && return Base.OneTo(n)
    length(rest) == 1 && n == 1 && only(rest) isa SymbolicAxis ? only(rest) : nothing
end

# ── scalar type algebra ─────────────────────────────────────────────────

# Type algebra is only applied to Base's closed primitive numeric families.
# No promote_op/return_type, exemplar arithmetic, or user promotion method runs.
_scalar_type(@nospecialize(f), ::Vector{Type}) = nothing
_scalar_result_type(@nospecialize(f), args::Vector{ValueInfo}) =
    _scalar_type(f, Type[element_type(v) for v in args])

# Julia lowers literal powers through literal_pow, whose negative-integer
# behavior differs from an ordinary ^ call. Inspect the explicit Val constant
# as data; do not specialize a metadata walker on its exponent parameter.
function _literal_exponent(v::ValueInfo)
    c = known_value(v)
    c isa KnownValue && c.value isa Val || return nothing
    p = typeof(c.value).parameters[1]
    p isa _MachineInteger ? p : nothing
end
function _scalar_result_type(::typeof(Base.literal_pow), args::Vector{ValueInfo})
    length(args) == 3 || return nothing
    f = known_value(args[1])
    f isa KnownValue && f.value === (^) || return nothing
    p = _literal_exponent(args[3])
    p === nothing && return nothing
    T = element_type(args[2])
    _machine(T) || return nothing
    T <: Integer && p < 0 ? Float64 : T
end

# Real scalar functions of one argument. The floating ones return Float64 for
# machine integers (Bool included) and keep Float16/32/64; abs keeps its
# argument's type (abs(true) === true).
const _FloatingUnary = Union{typeof(exp),typeof(expm1),typeof(log),typeof(log10),typeof(log1p),
                             typeof(sqrt),typeof(tanh)}
const _Unary = Union{_FloatingUnary,typeof(abs)}
const _Arithmetic = Union{typeof(+),typeof(-),typeof(*),typeof(/),typeof(^),_Unary}
_floating(T::Type) = T <: Integer ? Float64 : T
_unary_type(::Union{typeof(+),typeof(-)}, T::Type) = T === Bool ? Int : T
_unary_type(::Union{typeof(*),typeof(abs)}, T::Type) = T
_unary_type(::_FloatingUnary, T::Type) = _floating(T)
_binary_type(::Union{typeof(+),typeof(-)}, A::Type, B::Type) =
    A === Bool && B === Bool ? Int : promote_type(A, B)
_binary_type(::typeof(*), A::Type, B::Type) = promote_type(A, B)
_binary_type(::typeof(/), A::Type, B::Type) =
    A <: Integer && B <: Integer ? Float64 : promote_type(A, B)
_binary_type(::typeof(^), A::Type, B::Type) =
    B <: Integer ? A : promote_type(A, B)
# LinearAlgebra's product element type, promote_op(matprod, A, B): x*y + x*y.
_matprod_type(A::Type, B::Type) = (P = _binary_type(*, A, B); _binary_type(+, P, P))

function _scalar_type(f::_Arithmetic, ts::Vector{Type})
    all(_machine, ts) || return nothing
    _arithmetic_type(f, ts)
end
function _arithmetic_type(f::Union{typeof(+),typeof(*)}, ts::Vector{Type})
    isempty(ts) && return nothing
    length(ts) == 1 && return _unary_type(f, ts[1])
    T = ts[1]
    for i in 2:length(ts)
        T = _binary_type(f, T, ts[i])
    end
    T
end
function _arithmetic_type(f::typeof(-), ts::Vector{Type})
    length(ts) == 1 && return _unary_type(f, ts[1])
    length(ts) == 2 ? _binary_type(f, ts[1], ts[2]) : nothing
end
_arithmetic_type(f::Union{typeof(/),typeof(^)}, ts::Vector{Type}) =
    length(ts) == 2 ? _binary_type(f, ts[1], ts[2]) : nothing
_arithmetic_type(f::_Unary, ts::Vector{Type}) =
    length(ts) == 1 ? _unary_type(f, ts[1]) : nothing

# ── arithmetic: scalars, then the array methods of the same functions ───

# A scalar function's scalar call follows its type algebra; array operands
# use that function's array methods, if it has any. Extensions reuse this for
# their own scalar functions.
function _scalar_function_call(@nospecialize(f), args::Vector{ValueInfo})
    all(_is_scalar, args) || return _array_call(f, args)
    T = _scalar_result_type(f, args)
    T === nothing ? _unknown("no scalar rule for these operand types or arity") : _supported(scalar_info(T))
end
_infer_call(f::_Arithmetic, args::Vector{ValueInfo}) = _scalar_function_call(f, args)
_array_call(@nospecialize(f), ::Vector{ValueInfo}) =
    _unknown("this operation has no rule for array operands")

function _infer_call(f::typeof(Base.literal_pow), args::Vector{ValueInfo})
    length(args) == 3 && _is_scalar(args[2]) ||
        return _unknown("literal_pow requires a scalar base and known callable/exponent")
    T = _scalar_result_type(f, args)
    T === nothing ? _unknown("literal_pow requires Base.^, a machine-real base and a known integer Val") :
        _supported(scalar_info(T))
end

_machine_operands(args::Vector{ValueInfo}) =
    all(v -> (_array(v) || _is_scalar(v)) && _machine(element_type(v)), args)

# Undotted `+`/`-` require equal axes (no singleton expansion) and equal rank.
# Row vectors of one kind stay row vectors; any other mix materializes.
function _array_call(f::Union{typeof(+),typeof(-)}, args::Vector{ValueInfo})
    all(_array, args) && _machine_operands(args) ||
        return _unknown("undotted +/- apply to standard machine-real arrays of equal shape")
    T = _arithmetic_type(f, Type[element_type(v) for v in args])
    T === nothing && return _unknown("no rule for this arity")
    length(args) == 1 && return _unary_array(f, args[1], T)
    N = value_rank(args[1])
    all(v -> value_rank(v) == N, args) || return _unknown("operands of different rank")
    axs = map(d -> _equal_axis(Any[value_axes(v)[d] for v in args]), Tuple(1:N))
    any(a -> a === _CONFLICT, axs) && return _incompatible("undotted +/- operands have different axes")
    kind = args[1].family
    _row(args[1]) && all(v -> v.family === kind, args) && return _supported(_row_result(kind, T, axs[2]))
    _supported(_array_result(T, axs; container_type=_elementwise_container(T, N, args)))
end

# Unary - materializes the negation (rows stay rows). Unary + broadcasts an
# Array (Bool becomes Int) but returns any other numeric array itself.
_unary_array(::typeof(-), a::ValueInfo, T::Type) = _supported(_rebuilt(a, T))
function _unary_array(::typeof(+), a::ValueInfo, T::Type)
    B, E = type_bound(a), element_type(a)
    B <: Array && return _supported(_array_result(T, value_axes(a); container_type=Array{T,value_rank(a)}))
    _row(a) || typeintersect(B, Array) === Union{} || E === T ||
        return _unknown("unary + of a Bool array depends on whether it is an Array")
    # Either result is an input-compatible container with the same elements.
    _supported(ValueInfo(B, E, value_axes(a), UnknownValue(), a.family))
end

# Scalar scaling (`s * A`, `A * s`, `A / s`) is elementwise and keeps A's kind.
function _scaled(a::ValueInfo, s::ValueInfo, T::Type)
    _machine_operands(ValueInfo[a, s]) || return _unknown("scaling requires machine-real operands")
    _supported(_rebuilt(a, T))
end
function _array_call(f::typeof(/), args::Vector{ValueInfo})
    length(args) == 2 && _array(args[1]) && _is_scalar(args[2]) ||
        return _unknown("undotted / covers an array divided by a scalar")
    a, s = args
    _scaled(a, s, _binary_type(f, element_type(a), element_type(s)))
end

# Each binary product follows LinearAlgebra's method for its operand kinds.
function _product(a::ValueInfo, b::ValueInfo)
    _machine_operands(ValueInfo[a, b]) || return _unknown("products require standard machine-real operands")
    A, B = element_type(a), element_type(b)
    _is_scalar(a) && _is_scalar(b) && return _supported(scalar_info(_binary_type(*, A, B)))
    _is_scalar(a) && return _scaled(b, a, _binary_type(*, A, B))
    _is_scalar(b) && return _scaled(a, b, _binary_type(*, A, B))
    P = _matprod_type(A, B)
    ra, rb = value_axes(a), value_axes(b)
    if _row(a)
        if _vector(b)
            # The dot product's type for two Bool vectors differs across Julia
            # versions and lengths (an empty one is Bool on Julia 1.10).
            A === Bool && B === Bool && return _unknown("Bool row-vector dot product type is version dependent",
                                                        ValueInfo(type_bound=Integer, element_type=Integer, axes=()))
            return _contracted(ra[2], rb[1], () -> scalar_info(P))
        end
        _matrix(b) && return _contracted(ra[2], rb[1], () -> _row_result(a.family, P, rb[2]))
    elseif _vector(a)
        # An outer product with a row vector is elementwise; with a one-row
        # matrix it is a matrix product.
        _row(b) && return _supported(_array_result(_binary_type(*, A, B), (ra[1], rb[2]);
                                                   container_type=_materialized(_binary_type(*, A, B), 2)))
        _matrix(b) && return _contracted(Base.OneTo(1), rb[1],
            () -> _array_result(P, (ra[1], rb[2]); container_type=Matrix{P}))
    elseif _matrix(a)
        _vector(b) && return _contracted(ra[2], rb[1],
            () -> _array_result(P, (ra[1],); container_type=Vector{P}))
        (_matrix(b) || _row(b)) && return _contracted(ra[2], rb[1],
            () -> _array_result(P, (ra[1], rb[2]); container_type=Matrix{P}))
    end
    _unknown("no product rule for these operand ranks")
end
_contracted(x, y, result) = _equal_axis(Any[x, y]) === _CONFLICT ?
    _incompatible("product inner dimensions differ") : _supported(result())

# Julia's three- and four-operand methods pick a multiplication order from the
# runtime sizes. Shapes do not depend on the order; element types do only when
# Bool operands mix additions into some orders but not others.
function _array_call(::typeof(*), args::Vector{ValueInfo})
    length(args) >= 2 || return _unknown("a product needs at least two operands")
    result = _product(args[1], args[2])
    for b in args[3:end]
        result.status === :supported || return result
        result = _product(result.value, b)
    end
    length(args) > 2 && result.status === :supported && any(v -> element_type(v) === Bool, args) &&
        return _unknown("the element type of a chained product with Bool operands depends on its order",
                        ValueInfo(type_bound=Any, element_type=Number, axes=value_axes(result.value)))
    result
end

# ── shape functions ─────────────────────────────────────────────────────

function _infer_call(::typeof(length), args::Vector{ValueInfo})
    length(args) == 1 && (_array(args[1]) || _machine_scalar(args[1])) ||
        return _unknown("length covers standard arrays and machine-real scalars")
    _supported(scalar_info(Int))
end

_row_kind(::typeof(adjoint)) = :adjoint_vector
_row_kind(::typeof(transpose)) = :transpose_vector
# For real elements both are value-preserving flips: a vector becomes a row,
# a row returns its parent vector, and a matrix swaps its axes.
function _infer_call(f::Union{typeof(adjoint),typeof(transpose)}, args::Vector{ValueInfo})
    length(args) == 1 || return _unknown("adjoint/transpose take one operand")
    a = args[1]
    T = element_type(a)
    _machine(T) && (_array(a) || _is_scalar(a)) ||
        return _unknown("adjoint/transpose rules cover standard machine-real values")
    _is_scalar(a) && return _supported(a)
    axs = value_axes(a)
    _row(a) && return _supported(_array_result(T, (axs[2],); container_type=AbstractVector{T}))
    _vector(a) && return _supported(_row_result(_row_kind(f), T, axs[1]))
    _matrix(a) && return _supported(_array_result(T, reverse(axs); container_type=AbstractMatrix{T}))
    _unknown("adjoint/transpose of an array with rank above 2")
end

# ── indexing ────────────────────────────────────────────────────────────

_position(i::ValueInfo) =
    _is_scalar(i) && _machine(type_bound(i)) && type_bound(i) <: Integer && type_bound(i) !== Bool
_integer_array(i::ValueInfo) = _array(i) && _machine(element_type(i)) && element_type(i) <: Integer

# Julia's result has, per index, nothing for a position, the indexed axis for
# `:`, and the index array's own axes for an integer gather. A single index is
# linear over all axes. A Boolean mask selects a value-dependent count.
function _infer_call(::typeof(getindex), args::Vector{ValueInfo})
    length(args) >= 2 || return _unknown("getindex requires an array and indices")
    a = args[1]
    idx = args[2:end]
    _array(a) || return _unknown("indexing requires a standard Base array")
    T = element_type(a)
    # LinearAlgebra re-wraps `row[:, :]` and `row[:, I::AbstractArray{Int}]`
    # as a row of the gathered parent; other index types use Base's method.
    if _row(a) && length(idx) == 2 && type_bound(idx[1]) === Colon
        j = idx[2]
        type_bound(j) === Colon && return _supported(_row_result(a.family, T, value_axes(a)[2]))
        if _integer_array(j) && element_type(j) === Int
            value_rank(j) == 1 || return _unknown("a row gathered by an Int matrix becomes an adjoint matrix")
            return _supported(_row_result(a.family, T, value_axes(j)[1]))
        end
    end
    if length(idx) == 1
        source = (_linear_axis(value_axes(a)),)
    elseif length(idx) == value_rank(a)
        source = value_axes(a)
    else
        return _unknown("indexing requires one linear index or one index per axis")
    end
    out = Any[]
    masked = false
    for (i, ax) in zip(idx, source)
        if _position(i)
            continue
        elseif type_bound(i) === Colon
            push!(out, ax)
        elseif _integer_array(i) && element_type(i) === Bool
            value_rank(i) == 1 || return _unknown("only Boolean vector masks are covered")
            masked = true
            push!(out, nothing)
        elseif _integer_array(i)
            append!(out, value_axes(i))
        else
            return _unknown("each index must be a machine integer, `:`, or a standard integer array")
        end
    end
    isempty(out) && return _supported(_element_info(T))
    # BitArray and views can select a different container; keep a sound bound.
    result = _array_result(T, Tuple(out); container_type=AbstractArray{<:T,length(out)})
    masked ? _unknown("Boolean-mask length is value dependent", result) : _supported(result)
end

# ── broadcast ───────────────────────────────────────────────────────────

function _broadcast_scalar(v::ValueInfo)
    _machine_scalar(v) && return true
    # Base makes these literal_pow control arguments scalar-like via Ref.
    c = known_value(v)
    c isa KnownValue && (c.value === (^) || _literal_exponent(v) !== nothing)
end

function _infer_call(::typeof(broadcast), args::Vector{ValueInfo})
    length(args) >= 2 || return _unknown("broadcast requires a callable and data operands")
    f = known_value(args[1])
    f isa KnownValue || return _unknown("broadcast callable is not known")
    operands = args[2:end]
    all(v -> _array(v) || _broadcast_scalar(v), operands) ||
        return _unknown("broadcast container/scalar semantics are not established")
    T = _scalar_result_type(f.value, operands)
    arrays = filter(_array, operands)
    isempty(arrays) && return T === nothing ?
        _unknown("no scalar broadcast rule") : _supported(scalar_info(T))
    N = maximum(value_rank, arrays)
    axs = map(d -> _broadcast_axis(Any[value_axes(v)[d] for v in arrays if value_rank(v) >= d]), Tuple(1:N))
    any(a -> a === _CONFLICT, axs) && return _incompatible("broadcast axes are incompatible")
    E = T === nothing ? Any : T
    # broadcast(f, ...) over row vectors of one kind (and scalars) returns that
    # row kind, whereas dot syntax materializes a Matrix; neither is promised.
    kind = arrays[1].family
    if _row(arrays[1]) && all(v -> v.family === kind, arrays)
        return _unknown("broadcast and dot syntax materialize row vectors differently",
                        ValueInfo(type_bound=AbstractMatrix{<:E}, element_type=E, axes=axs))
    end
    T === nothing && return _unknown("broadcast element result type is unknown",
                                     _array_result(Any, axs; container_type=AbstractArray{<:Any,N}))
    _supported(_array_result(T, axs; container_type=_materialized(T, N)))
end

# ── reductions ──────────────────────────────────────────────────────────

_sum_type(T::Type{Bool}) = Int
_sum_type(T::Type{<:Signed}) = sizeof(T) < sizeof(Int) ? Int : T
_sum_type(T::Type{<:Unsigned}) = sizeof(T) < sizeof(UInt) ? UInt : T
_sum_type(T::Type{<:AbstractFloat}) = T
const _Extremum = Union{typeof(minimum),typeof(maximum)}
_reduction_type(::typeof(sum), T::Type) = _sum_type(T)
_reduction_type(::_Extremum, T::Type) = T
_reduces_scalar(@nospecialize(f)) = false
_reduces_scalar(::typeof(sum)) = true

function _known_dims(v::ValueInfo)
    c = known_value(v)
    c isa KnownValue || return nothing
    d = c.value
    dim(x) = x isa Int && x >= 1
    dim(d) && return (d,)
    d isa Tuple && all(dim, d) ? d : nothing
end

# Base allocates a `dims` reduction as similar(A, S, dims) and a scan as
# similar(A, S): for a standard array an Array, except that a Bool result over
# BitArray storage (or a view of it) is a BitArray. An abstract bound cannot
# tell those apart.
function _similar_container(a::ValueInfo, S::Type, N::Int)
    S === Bool || return Array{S,N}
    B = type_bound(a)
    B <: Array ? Array{Bool,N} : B <: BitArray ? BitArray{N} : AbstractArray{Bool,N}
end

# A full reduction is a scalar. With `dims`, each reduced axis that exists has
# length 1 and the rank is unchanged, as in Base and Statistics.
function _reduction_call(f, args::Vector{ValueInfo}, kwargs::Vector{Pair{Symbol,ValueInfo}})
    length(args) == 1 || return _unknown("only the single-argument reduction is covered")
    a = args[1]
    (_array(a) || _reduces_scalar(f) && _is_scalar(a)) && _machine(element_type(a)) ||
        return _unknown("this reduction requires standard machine-real operands")
    S = _reduction_type(f, element_type(a))
    isempty(kwargs) && return _supported(scalar_info(S))
    _array(a) && length(kwargs) == 1 && first(kwargs[1]) === :dims ||
        return _unknown("only the dims keyword on an array is covered")
    dims = _known_dims(last(kwargs[1]))
    dims === nothing && return _unknown("dims must be a known positive integer or tuple of them")
    axs = Any[value_axes(a)...]
    for d in dims
        d <= length(axs) && (axs[d] = Base.OneTo(1))
    end
    _supported(_array_result(S, Tuple(axs); container_type=_similar_container(a, S, length(axs))))
end
_infer_call(f::Union{typeof(sum),_Extremum}, args::Vector{ValueInfo}, kwargs::Vector{Pair{Symbol,ValueInfo}}) =
    _reduction_call(f, args, kwargs)

# ── concatenation ───────────────────────────────────────────────────────

function _infer_call(::typeof(vcat), args::Vector{ValueInfo})
    isempty(args) && return _unknown("empty argument list has no vcat rule")
    all(v -> (_vector(v) || _is_scalar(v)) && _machine(element_type(v)), args) ||
        return _unknown("vcat requires standard machine-real scalars/vectors")
    T = mapreduce(element_type, promote_type, args)
    n = _concatenated_axis(Any[_is_scalar(v) ? Base.OneTo(1) : value_axes(v)[1] for v in args])
    _supported(_array_result(T, (n,); container_type=AbstractVector{T}))
end

# hcat treats a scalar as 1×1 and a vector as one column; all parts need equal
# row counts. Rows of one kind with scalars stay that row kind (LinearAlgebra
# concatenates their parents); all scalars or any other mix give a matrix.
function _infer_call(::typeof(hcat), args::Vector{ValueInfo})
    isempty(args) && return _unknown("empty argument list has no hcat rule")
    all(v -> (_array(v) && value_rank(v) <= 2 || _is_scalar(v)) && _machine(element_type(v)), args) ||
        return _unknown("hcat requires standard machine-real scalars, vectors and matrices")
    T = mapreduce(element_type, promote_type, args)
    rows = _equal_axis(Any[_is_scalar(v) ? Base.OneTo(1) : value_axes(v)[1] for v in args])
    rows === _CONFLICT && return _incompatible("hcat operands have different row counts")
    cols = _concatenated_axis(Any[value_rank(v) == 2 ? value_axes(v)[2] : Base.OneTo(1) for v in args])
    parts = filter(!_is_scalar, args)
    isempty(parts) && return _supported(_array_result(T, (rows, cols); container_type=Matrix{T}))
    kind = parts[1].family
    _row(parts[1]) && all(v -> v.family === kind, parts) && return _supported(_row_result(kind, T, cols))
    _supported(_array_result(T, (rows, cols); container_type=AbstractMatrix{T}))
end

# ── scans ───────────────────────────────────────────────────────────────

# cumsum/cumprod accumulate with Base.add_sum/mul_prod, which widen small
# integers as sum does; Bool sums are Int but Bool products stay Bool.
_prod_type(::Type{Bool}) = Bool
_prod_type(T::Type) = _sum_type(T)
_scan_type(::typeof(cumsum), T::Type) = _sum_type(T)
_scan_type(::typeof(cumprod), T::Type) = _prod_type(T)

# A scan's `dims` is one positive integer; a tuple is a TypeError in Base.
function _known_dim(v::ValueInfo)
    dims = _known_dims(v)
    dims !== nothing && known_value(v).value isa Int ? only(dims) : nothing
end

# A scan keeps its operand's axes along every `dims`, including one beyond the
# rank (a copy). Only a vector may omit `dims`; Base requires it otherwise. A
# row vector stays a row of the same kind, since LinearAlgebra re-wraps
# similar(parent). Base iterates a scalar into a zero-dimensional Array.
function _infer_call(f::Union{typeof(cumsum),typeof(cumprod)}, args::Vector{ValueInfo},
                     kwargs::Vector{Pair{Symbol,ValueInfo}})
    length(args) == 1 || return _unknown("only the single-argument cumsum/cumprod is covered")
    a = args[1]
    (_array(a) || _is_scalar(a)) && _machine(element_type(a)) ||
        return _unknown("cumsum/cumprod require standard machine-real operands")
    S = _scan_type(f, element_type(a))
    if isempty(kwargs)
        _is_scalar(a) && return _supported(ValueInfo(Array{S,0}, S, (), UnknownValue(), :unknown))
        _vector(a) || return _unknown("cumsum/cumprod of a row vector or an array of rank ≥ 2 require dims")
    else
        _array(a) && length(kwargs) == 1 && first(kwargs[1]) === :dims ||
            return _unknown("only the dims keyword on an array is covered")
        _known_dim(last(kwargs[1])) === nothing &&
            return _unknown("dims must be a known positive integer")
    end
    _row(a) && return _supported(_row_result(a.family, S, value_axes(a)[2]))
    _supported(_array_result(S, value_axes(a); container_type=_similar_container(a, S, value_rank(a))))
end
