export ValueInfo, CallInfo, KnownValue, UnknownValue
export value_info, known_info, scalar_info, vector_info, infer_call
export type_bound, element_type, value_axes, value_rank, value_length, known_value

"""An explicitly unavailable constant (distinct from the known value `nothing`)."""
struct UnknownValue end

"""An explicitly supplied constant. Its value is ordinary data, not a type parameter."""
struct KnownValue
    value::Any
end

"""
    ValueInfo()
    ValueInfo(; type_bound=Any, element_type=Any, axes=nothing)

Cold numerical facts about a value, including one that does not exist yet.
`type_bound` and `element_type` are upper bounds. `axes=nothing` means unknown
rank; `()` means rank zero; `(nothing,)` means a vector of unknown length.
Known axes are immutable integer unit ranges. This general constructor makes
no claim about a container's indexing or broadcast methods. Prefer
[`scalar_info`](@ref), [`vector_info`](@ref), or [`value_info`](@ref) when their
stronger contracts apply. All supplied facts must describe the same value.
"""
struct ValueInfo
    julia_type::Type
    element_type::Type
    axes::Union{Nothing,Tuple}
    constant::Union{UnknownValue,KnownValue}
    family::Symbol
end

function ValueInfo(; type_bound::Type=Any, element_type::Type=Any, axes=nothing)
    axes === nothing || all(a -> a === nothing || a isa AbstractUnitRange{Int}, axes) ||
        throw(ArgumentError("axes must contain integer unit ranges or nothing"))
    ValueInfo(type_bound, element_type, axes, UnknownValue(), :unknown)
end

"""Describe a numerical scalar of type `T` (or a subtype of `T`), without a value."""
function scalar_info(T::Type=Number)
    T <: Number || throw(ArgumentError("scalar_info requires a Number type bound"))
    ValueInfo(T, T, (), UnknownValue(), :scalar)
end

"""
    vector_info(T, n=nothing; container_type=Vector{T})

Describe an ordinary one-based Base vector with element type `T`. `n=nothing`
keeps its length unknown. `container_type=AbstractVector{T}` permits an
unavailable dense vector or ordinary vector view without claiming a concrete
container. This constructor *asserts* standard Base indexing and broadcast
semantics independently of that type bound. It must not describe an arbitrary
custom AbstractVector. `value_info(x)` checks this family for actual inputs.
"""
function vector_info(T::Type, n::Union{Integer,Nothing}=nothing;
                     container_type::Type=Vector{T})
    container_type <: AbstractVector{T} ||
        throw(ArgumentError("container_type must be a subtype of AbstractVector{T}"))
    n === nothing || n >= 0 || throw(ArgumentError("vector length must be nonnegative"))
    ax = n === nothing ? nothing : Base.OneTo(Int(n))
    ValueInfo(container_type, T, (ax,), UnknownValue(), :standard_vector)
end

type_bound(v::ValueInfo) = v.julia_type
element_type(v::ValueInfo) = v.element_type
value_axes(v::ValueInfo) = v.axes
value_rank(v::ValueInfo) = v.axes === nothing ? nothing : length(v.axes)
known_value(v::ValueInfo) = v.constant
function value_length(v::ValueInfo)
    v.axes === nothing && return nothing
    n = 1
    for ax in v.axes
        ax === nothing && return nothing
        n = Base.checked_mul(n, length(ax))
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

"""
    value_info(x)

Extract type and axes at an input boundary, without reading array elements or
declaring their values constant. Custom arrays retain their facts but do not
acquire the standard-vector family. No numerical operation is evaluated.
"""
value_info(@nospecialize(x)) = ValueInfo(type_bound=typeof(x))
value_info(x::Number) = scalar_info(typeof(x))
function value_info(x::AbstractArray)
    ax = axes(x)
    family = ndims(x) == 1 && _standard_storage(x) && ax[1] isa Base.OneTo{Int} ?
             :standard_vector : :unknown
    ValueInfo(typeof(x), eltype(x), ax, UnknownValue(), family)
end

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
compiler inference, or legacy allocation signature is invoked. This bounded
implementation covers machine-real scalar arithmetic, standard-vector integer
indexing, broadcast, `sum`, and `vcat`; unsupported cases return explicit unknown
facts. Keywords are represented but currently have no transfer rules.

The original runtime operation must retain its bounds and shape checks. In
particular, broadcast facts may be exact *conditional on compatible axes*.
Rules describe Base semantics; overriding Base methods invalidates those rules.
"""
function infer_call(@nospecialize(f), args::Vector{ValueInfo};
                    kwargs::Vector{Pair{Symbol,ValueInfo}}=Pair{Symbol,ValueInfo}[])
    isempty(kwargs) || return _unknown("keyword arguments have no metadata rule for this call")
    _infer_call(f, args)
end
_infer_call(@nospecialize(f), ::Vector{ValueInfo}) = _unknown("no metadata rule for this callable")

# Type algebra is only applied to Base's closed primitive numeric families.
# No promote_op/return_type, exemplar arithmetic, or user promotion method runs.
_scalar_type(@nospecialize(f), ::Vector{Type}) = nothing
_unary_type(::Union{typeof(+),typeof(-)}, T::Type) = T === Bool ? Int : T
_unary_type(::typeof(*), T::Type) = T
_unary_type(::typeof(exp), T::Type) = T <: Integer ? Float64 : T
_binary_type(::Union{typeof(+),typeof(-)}, A::Type, B::Type) =
    A === Bool && B === Bool ? Int : promote_type(A, B)
_binary_type(::typeof(*), A::Type, B::Type) = promote_type(A, B)
_binary_type(::typeof(/), A::Type, B::Type) =
    A <: Integer && B <: Integer ? Float64 : promote_type(A, B)
_binary_type(::typeof(^), A::Type, B::Type) =
    B <: Integer ? A : promote_type(A, B)

function _scalar_type(f::Union{typeof(+),typeof(-),typeof(*),typeof(/),typeof(^),typeof(exp)},
                      ts::Vector{Type})
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
_arithmetic_type(f::typeof(exp), ts::Vector{Type}) =
    length(ts) == 1 ? _unary_type(f, ts[1]) : nothing

function _infer_call(f::Union{typeof(+),typeof(-),typeof(*),typeof(/),typeof(^),typeof(exp)},
                     args::Vector{ValueInfo})
    all(v -> v.family === :scalar, args) || return _unknown("scalar arithmetic requires scalar operands")
    T = _scalar_type(f, Type[type_bound(v) for v in args])
    T === nothing ? _unknown("no scalar rule for these operand types or arity") : _supported(scalar_info(T))
end

_vector(v::ValueInfo) = v.family === :standard_vector && value_rank(v) == 1
_element_info(T::Type) = T <: Number ? scalar_info(T) : ValueInfo(type_bound=T)
function _vector_result(T::Type, n; container_type::Type=AbstractVector)
    ax = n === nothing ? nothing : Base.OneTo(n)
    ValueInfo(container_type, T, (ax,), UnknownValue(), :standard_vector)
end

function _infer_call(::typeof(getindex), args::Vector{ValueInfo})
    length(args) == 2 || return _unknown("only single-index vector indexing is covered")
    a, i = args
    _vector(a) || return _unknown("indexing requires a standard Base vector")
    T = element_type(a)
    if i.family === :scalar && _machine(type_bound(i)) && type_bound(i) <: Integer && type_bound(i) !== Bool
        return _supported(_element_info(T))
    end
    _vector(i) && _machine(element_type(i)) && element_type(i) <: Integer ||
        return _unknown("index must be a machine integer or standard integer vector")
    # A Bool mask is not an integer gather: its output length depends on values.
    element_type(i) === Bool && return _unknown("Boolean-mask length is value dependent", _vector_result(T, nothing))
    # BitArray and views can select a different container; preserve a sound bound.
    container = AbstractVector{<:T}
    _supported(_vector_result(T, value_length(i); container_type=container))
end

function _infer_call(::typeof(broadcast), args::Vector{ValueInfo})
    length(args) >= 2 || return _unknown("broadcast requires a callable and data operands")
    f = known_value(args[1])
    f isa KnownValue || return _unknown("broadcast callable is not known")
    operands = args[2:end]
    all(v -> _vector(v) || (v.family === :scalar && _machine(type_bound(v))), operands) ||
        return _unknown("broadcast container/scalar semantics are not established")
    T = _scalar_type(f.value, Type[element_type(v) for v in operands])
    any(_vector, operands) || return T === nothing ?
        _unknown("no scalar broadcast rule") : _supported(scalar_info(T))
    n = 1
    unknown_length = false
    for v in operands
        _vector(v) || continue
        m = value_length(v)
        if m === nothing
            unknown_length = true
        elseif n == 1
            n = m
        elseif m != 1 && m != n
            return _incompatible("broadcast vector axes are incompatible")
        end
    end
    # A known non-singleton axis determines the successful result even when
    # another length is unknown. In particular, broadcasting length 0 with 1
    # produces length 0, not max(0, 1).
    n = n == 1 && unknown_length ? nothing : n
    T === nothing && return _unknown("broadcast element result type is unknown", _vector_result(Any, n))
    container = T === Bool ? BitVector : Vector{T}
    _supported(_vector_result(T, n; container_type=container))
end

_sum_type(T::Type{Bool}) = Int
_sum_type(T::Type{<:Signed}) = sizeof(T) < sizeof(Int) ? Int : T
_sum_type(T::Type{<:Unsigned}) = sizeof(T) < sizeof(UInt) ? UInt : T
_sum_type(T::Type{<:AbstractFloat}) = T
function _infer_call(::typeof(sum), args::Vector{ValueInfo})
    length(args) == 1 || return _unknown("only ordinary single-argument sum is covered")
    a = args[1]
    (_vector(a) || a.family === :scalar) && _machine(element_type(a)) ||
        return _unknown("sum requires standard machine-real operands")
    _supported(scalar_info(_sum_type(element_type(a))))
end

function _infer_call(::typeof(vcat), args::Vector{ValueInfo})
    isempty(args) && return _unknown("empty argument list has no vcat rule")
    all(v -> (_vector(v) || v.family === :scalar) && _machine(element_type(v)), args) ||
        return _unknown("vcat requires standard machine-real scalars/vectors")
    T = element_type(args[1])
    n = 0
    for v in args
        T = promote_type(T, element_type(v))
        m = value_length(v)
        n = n === nothing || m === nothing ? nothing : Base.checked_add(n, m)
    end
    _supported(_vector_result(T, n; container_type=AbstractVector{T}))
end
