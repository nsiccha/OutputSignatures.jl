using Test, OutputSignatures

# Use actual Julia results as the oracle, including their container types.
function check_metadata(f, xs...)
    result = infer_call(f, ValueInfo[value_info(x) for x in xs])
    actual = f(xs...)
    @test result.status == :supported
    @test actual isa type_bound(result.value)
    @test eltype(actual) <: element_type(result.value)
    @test value_axes(result.value) == axes(actual)
    @test value_rank(result.value) == ndims(actual)
    @test value_length(result.value) == length(actual)
    result.value
end

function check_broadcast_metadata(f, xs...)
    result = infer_call(broadcast, ValueInfo[known_info(f); value_info.(collect(xs))])
    actual = broadcast(f, xs...)
    @test result.status == :supported
    @test actual isa type_bound(result.value)
    @test eltype(actual) == element_type(result.value)
    @test value_axes(result.value) == axes(actual)
    result.value
end

struct MetadataCustomVector <: AbstractVector{Float64}
    values::Vector{Float64}
end
Base.size(x::MetadataCustomVector) = size(x.values)
Base.getindex(x::MetadataCustomVector, i::Int) = x.values[i]

struct MetadataCustomNumber <: Number end
const metadata_custom_calls = Ref(0)
Base.:+(::MetadataCustomNumber, ::MetadataCustomNumber) =
    (metadata_custom_calls[] += 1; [1.0, 2.0])

@testset "metadata descriptors and unknown facts" begin
    @test value_rank(ValueInfo()) === nothing
    @test value_length(ValueInfo()) === nothing
    @test type_bound(ValueInfo()) === Any
    @test value_rank(scalar_info(Float64)) == 0
    @test value_rank(vector_info(Float64)) == 1
    @test value_length(vector_info(Float64)) === nothing
    @test value_length(vector_info(Float64, 0)) == 0
    @test value_length(vector_info(Float64, 1)) == 1
    @test value_axes(vector_info(Float64)) == (nothing,)
    @test value_axes(vector_info(Float64, 3)) == (Base.OneTo(3),)
    @test known_value(value_info(3)) isa UnknownValue
    @test known_value(value_info([1, 2])) isa UnknownValue
    @test known_value(known_info(nothing)) isa KnownValue
    @test known_value(known_info(nothing)).value === nothing
    @test known_value(ValueInfo()) isa UnknownValue
    @test typeof(vector_info(Float64, 2)) === typeof(vector_info(Float64, 2000)) === ValueInfo
    @test isempty(ValueInfo.parameters) && isempty(CallInfo.parameters) && isempty(KnownValue.parameters)
    @test value_rank(value_info(fill(1.0))) == 0
    @test type_bound(value_info(fill(1.0))) == Array{Float64,0}
    @test type_bound(value_info(fill(1.0))) != type_bound(scalar_info(Float64))
    @test value_rank(ValueInfo(type_bound=AbstractArray)) === nothing
    @test value_rank(ValueInfo(type_bound=AbstractMatrix, axes=(nothing, Base.OneTo(2)))) == 2
    abstract_view = vector_info(Float64, 3; container_type=AbstractVector{Float64})
    @test type_bound(abstract_view) === AbstractVector{Float64}
    @test value_axes(abstract_view) == (Base.OneTo(3),)
end

@testset "scalar rules agree with Base type semantics" begin
    types = (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16,
             UInt32, UInt64, UInt128, Float16, Float32, Float64)
    for T in types
        for f in (+, -, *, exp, sum)
            check_metadata(f, one(T))
        end
        for S in types, f in (+, -, *, /, ^)
            check_metadata(f, one(T), one(S))
        end
        for f in (+, *)
            check_metadata(f, one(T), true, Float32(2))
        end
    end
    @test type_bound(infer_call(+, ValueInfo[scalar_info(Bool), scalar_info(Bool)]).value) == Int
    @test type_bound(infer_call(^, ValueInfo[scalar_info(Int8), scalar_info(Int64)]).value) == Int8
end

@testset "scalar and vector integer indexing" begin
    for x in ([2., 4., 6.], Int8[2,4,6], [true,false,true], trues(3),
              view([2.,4.,6.], :), view([2.,4.,6.,8.,10.], 1:2:5))
        check_metadata(getindex, x, 2)
        for idx in (Int[], [1], [3,1,3], Int8[2,1], view([3,2,1], :))
            check_metadata(getindex, x, idx)
        end
    end
    source = vector_info(Float64; container_type=AbstractVector{Float64})
    @test value_rank(infer_call(getindex, ValueInfo[source, scalar_info(Int)]).value) == 0
    gather = infer_call(getindex, ValueInfo[source, vector_info(Int)])
    @test gather.status == :supported
    @test value_rank(gather.value) == 1 && value_length(gather.value) === nothing
    @test element_type(gather.value) == Float64
    mask = infer_call(getindex, ValueInfo[source, vector_info(Bool, 10)])
    @test mask.status == :unknown && value_length(mask.value) === nothing
    @test value_rank(mask.value) == 1
    opaque_elements = infer_call(getindex, ValueInfo[vector_info(Any), scalar_info(Int)])
    @test value_rank(opaque_elements.value) === nothing
    @test type_bound(opaque_elements.value) === Any
end

@testset "literal powers preserve Julia's special semantics" begin
    types = (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16,
             UInt32, UInt64, UInt128, Float16, Float32, Float64)
    for T in types, p in (-3, -2, -1, 0, 1, 2, 3, 4)
        args = ValueInfo[known_info(^), scalar_info(T), known_info(Val(p))]
        result = infer_call(Base.literal_pow, args)
        actual = Base.literal_pow(^, one(T), Val(p))
        @test result.status == :supported
        @test type_bound(result.value) == typeof(actual)
        @test value_axes(result.value) == axes(actual)
        scalar_broadcast = infer_call(broadcast, ValueInfo[known_info(Base.literal_pow); args])
        @test scalar_broadcast.status == :supported
        @test type_bound(scalar_broadcast.value) == typeof(broadcast(Base.literal_pow, ^, one(T), Val(p)))
        for n in (0, 1, 3)
            x = fill(one(T), n)
            for input in (x, view(x, :))
                operands = ValueInfo[known_info(Base.literal_pow), known_info(^), value_info(input), known_info(Val(p))]
                br = infer_call(broadcast, operands)
                y = broadcast(Base.literal_pow, ^, input, Val(p))
                @test br.status == :supported
                @test y isa type_bound(br.value)
                @test element_type(br.value) == eltype(y)
                @test value_axes(br.value) == axes(y)
            end
        end
    end
    literal = infer_call(broadcast, ValueInfo[known_info(Base.literal_pow), known_info(^),
                                             vector_info(Int8, 1), known_info(Val(-1))])
    @test type_bound(literal.value) == typeof(Int8[2] .^ -1) == Vector{Float64}
    @test (Int8[2] .^ -1) == [0.5]
    # Ordinary ^ intentionally retains its different Base semantics.
    ordinary = infer_call(broadcast, ValueInfo[known_info(^), vector_info(Int8, 1), known_info(-1)])
    @test type_bound(ordinary.value) == Vector{Int8}
    @test try broadcast(^, Int8[2], -1); false catch e; e isa DomainError end
    partial = infer_call(broadcast, ValueInfo[known_info(Base.literal_pow), known_info(^),
                                             vector_info(Int8), known_info(Val(-1))])
    @test partial.status == :supported
    @test value_rank(partial.value) == 1 && value_length(partial.value) === nothing
    for exponent in (value_info(Val(-1)), known_info(Val(:opaque)), ValueInfo())
        @test infer_call(Base.literal_pow, ValueInfo[known_info(^), scalar_info(Float64), exponent]).status == :unknown
        br = infer_call(broadcast, ValueInfo[known_info(Base.literal_pow), known_info(^), vector_info(Float64,3), exponent])
        @test br.status == :unknown
    end
    calls = Ref(0)
    other_power(x, p) = (calls[] += 1; x^p)
    @test infer_call(Base.literal_pow, ValueInfo[known_info(other_power), scalar_info(Int8), known_info(Val(-1))]).status == :unknown
    @test infer_call(broadcast, ValueInfo[known_info(Base.literal_pow), known_info(other_power), vector_info(Int8,3), known_info(Val(-1))]).status == :unknown
    @test calls[] == 0
end

@testset "broadcast singleton, empty, mixed types and views" begin
    for n in (0, 1, 3), T in (Bool, Int8, UInt8, Float16, Float32, Float64)
        x = fill(one(T), n)
        for f in (+, -, *, /, ^)
            for y in (one(T), [one(T)], copy(x), view(copy(x), :))
                check_broadcast_metadata(f, x, y)
            end
        end
        for f in (+, -, *, exp)
            check_broadcast_metadata(f, x)
        end
        check_broadcast_metadata(+, x, Float64(2))
        check_broadcast_metadata(*, view(x, :), Float32(2))
        check_metadata(sum, x)
        check_metadata(sum, view(x, :))
    end
    check_broadcast_metadata(+, true, true)
    check_broadcast_metadata(*, [true], [false])
    partial = vector_info(Float64)
    for n in (0, 1, 3)
        r = infer_call(broadcast, ValueInfo[known_info(+), partial, vector_info(Float64, n)])
        @test r.status == :supported
        @test value_length(r.value) === (n == 1 ? nothing : n)
        # All compatible concretizations obey the conditional output facts.
        for m in (0, 1, 3)
            (m == n || m == 1 || n == 1) || continue
            y = zeros(m) .+ zeros(n)
            @test value_length(r.value) === nothing || value_length(r.value) == length(y)
        end
    end
    r = infer_call(broadcast, ValueInfo[known_info(+), partial, partial])
    @test value_rank(r.value) == 1 && value_length(r.value) === nothing
    mismatch = infer_call(broadcast, ValueInfo[known_info(+), vector_info(Float64,0), vector_info(Float64,3)])
    @test mismatch.status == :incompatible
    @test try zeros(0) .+ zeros(3); false catch e; e isa DimensionMismatch end
    @test infer_call(broadcast, ValueInfo[known_info(+), ValueInfo(), scalar_info(Float64)]).status == :unknown
end

@testset "unavailable intermediate chain and vcat" begin
    for n in (0, 1, 5)
        alpha, beta, x, b, group = 0.5, 2.0, collect(1.0:n), [0.2,0.4], fill(2,n)
        # There are no numerical intermediates at query time.
        gather = infer_call(getindex, ValueInfo[vector_info(Float64,2), vector_info(Int,n)]).value
        scale = infer_call(broadcast, ValueInfo[known_info(*), scalar_info(Float64), vector_info(Float64,n)]).value
        shift = infer_call(broadcast, ValueInfo[known_info(+), scalar_info(Float64), scale, gather]).value
        total = infer_call(sum, ValueInfo[shift])
        actual = sum(alpha .+ beta .* x .+ b[group])
        @test total.status == :supported
        @test type_bound(total.value) == typeof(actual)
        @test value_axes(total.value) == axes(actual)
        @test value_length(shift) == n
        @test known_value(total.value) isa UnknownValue
        check_metadata(vcat, alpha, x, b)
        check_metadata(vcat, view(x,:), b)
        check_metadata(vcat, true, trues(n), Int8[1])
    end
    r = infer_call(vcat, ValueInfo[scalar_info(Float64), vector_info(Float64)])
    @test value_rank(r.value) == 1 && value_length(r.value) === nothing
end

@testset "unknown operations never run and never borrow allocation inference" begin
    calls = Ref(0)
    opaque(x) = (calls[] += 1; length(x))
    v = vector_info(Float64, 3)
    r = infer_call(opaque, ValueInfo[v])
    @test r.status == :unknown && value_rank(r.value) === nothing
    @test !isempty(r.reason)
    b = infer_call(broadcast, ValueInfo[known_info(opaque), v])
    @test b.status == :unknown && value_length(b.value) == 3
    @test type_bound(b.value) == AbstractVector
    # Partial element bounds must not become an exact Vector{Any} container.
    g = infer_call(getindex, ValueInfo[b.value, vector_info(Int,2)])
    @test [1.,2.] isa type_bound(g.value)
    @test trues(2) isa type_bound(g.value)
    @test calls[] == 0
    n = scalar_info(MetadataCustomNumber)
    @test infer_call(+, ValueInfo[n,n]).status == :unknown
    @test metadata_custom_calls[] == 0
    custom = value_info(MetadataCustomVector([1.,2.]))
    @test value_length(custom) == 2
    @test infer_call(getindex, ValueInfo[custom, scalar_info(Int)]).status == :unknown
    @test infer_call(broadcast, ValueInfo[known_info(+), custom, v]).status == :unknown
    @test infer_call(sum, ValueInfo[v]; kwargs=Pair{Symbol,ValueInfo}[:init=>known_info(0.0)]).status == :unknown
    @test infer_call(vcat, ValueInfo[v]; kwargs=Pair{Symbol,ValueInfo}[:dims=>known_info(1)]).status == :unknown
    @test infer_call(sum, ValueInfo[ValueInfo()]).status == :unknown
    @test infer_call(sum, ValueInfo[vector_info(Real, 3)]).status == :unknown
    @test infer_call(broadcast, ValueInfo[value_info(+), v, v]).status == :unknown
    # Both APIs remain callable; metadata never uses the legacy promote_op path.
    @test output_signature(sum, Int8[1,2]).eltype == Int
    @test broadcast_signature(+, [1.], [2.]).size == (1,)
end
