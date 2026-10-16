using Test, OutputSignatures, LinearAlgebra, Statistics

# Property checks for symbolic/unknown axes. A case is an expression over
# leaves whose axes are known lengths, symbolic names, or unknown lengths. For
# every assignment of lengths 0:3 to those names, the actual Julia call is the
# oracle: a supported result's facts must hold whenever the call succeeds, and
# an incompatible result must fail for every assignment.

struct Unknown
    name::Symbol
end
struct Leaf
    T::Type
    dims::Tuple
    container::Symbol
end
struct Known
    value::Any
end
struct Call
    f::Any
    args::Vector{Any}
    kwargs::Vector{Pair{Symbol,Any}}
end
call(f, args...; kwargs...) = Call(f, Any[args...], Pair{Symbol,Any}[k => v for (k, v) in kwargs])

scalar(T) = Leaf(T, (), :scalar)
arr(T, dims...; container=:array) = Leaf(T, dims, container)
flip(f, x) = call(f, x)

_descr_axis(d::Int) = d
_descr_axis(d::Symbol) = SymbolicAxis(d)
_descr_axis(::Unknown) = nothing
_length(d::Int, b) = d
_length(d::Symbol, b) = b[d]
_length(d::Unknown, b) = b[d.name]
names_of(d::Int) = Symbol[]
names_of(d::Symbol) = [d]
names_of(d::Unknown) = [d.name]
names_of(x::Leaf) = reduce(vcat, map(names_of, x.dims); init=Symbol[])
names_of(::Known) = Symbol[]
names_of(c::Call) = reduce(vcat, map(names_of, c.args); init=Symbol[])

function meta(x::Leaf)
    x.container === :scalar && return scalar_info(x.T)
    N = length(x.dims)
    container_type = x.container === :bit ? BitArray{N} :
                     x.container === :view ? AbstractArray{x.T,N} : Array{x.T,N}
    array_info(x.T, map(_descr_axis, x.dims)...; container_type)
end
meta(x::Known) = known_info(x.value)
meta(c::Call) = infer(c).value
infer(c::Call) = infer_call(c.f, ValueInfo[meta(a) for a in c.args];
                            kwargs=Pair{Symbol,ValueInfo}[k => meta(v) for (k, v) in c.kwargs])

function actual(x::Leaf, b)
    x.container === :scalar && return one(x.T)
    lens = map(d -> _length(d, b), x.dims)
    x.container === :bit && return trues(lens...)
    x.container === :view && return view(fill(one(x.T), (lens .+ 1)...), map(n -> 1:n, lens)...)
    fill(one(x.T), lens...)
end
actual(x::Known, b) = x.value
actual(c::Call, b) = c.f(map(a -> actual(a, b), c.args)...;
                         (k => actual(v, b) for (k, v) in c.kwargs)...)

axis_holds(ax::AbstractUnitRange, n, b) = length(ax) == n
axis_holds(ax::SymbolicAxis, n, b) = b[ax.name] == n
axis_holds(::Nothing, n, b) = true

# A supported descriptor is exact: element type, axes, rank, Julia type bound
# and the container family that later rules rely on (row vector or array).
function facts_hold(info::ValueInfo, x, b; exact::Bool)
    x isa type_bound(info) || return false
    E = x isa AbstractArray ? eltype(x) : typeof(x)
    (exact ? E == element_type(info) : E <: element_type(info)) || return false
    axs = value_axes(info)
    axs === nothing && return true
    length(axs) == ndims(x) || return false
    all(axis_holds(a, n, b) for (a, n) in zip(axs, size(x))) || return false
    !exact || value_info(x).family === info.family
end

bindings(names) = isempty(names) ? [Dict{Symbol,Int}()] :
    vec([Dict(zip(names, ls)) for ls in Iterators.product(ntuple(_ -> 0:3, length(names))...)])

show_case(x::Leaf) = "$(x.container){$(x.T)}$(x.dims)"
show_case(x::Known) = repr(x.value)
show_case(c::Call) = "$(c.f)(" * join(map(show_case, c.args), ", ") * join(("; $k=$(show_case(v))" for (k, v) in c.kwargs)) * ")"

# Operands built by calls (row vectors, chains) must themselves be supported.
supported_operands(c::Call) = all(a -> !(a isa Call) || infer(a).status === :supported && supported_operands(a), c.args)

function check_case(c::Call; status=:supported)
    @test supported_operands(c)
    result = infer(c)
    @test result.status === status
    status === :supported && @test isempty(result.reason)
    successes = 0
    for b in bindings(unique(names_of(c)))
        x = try
            actual(c, b)
        catch err
            # Facts only describe successful calls. Some lengths fail in Julia
            # itself (Julia 1.10 cannot reduce an empty adjoint along dims).
            err isa Union{DimensionMismatch,BoundsError,ArgumentError,MethodError} || rethrow()
            continue
        end
        successes += 1
        @test status !== :incompatible
        status === :incompatible && break
        ok = facts_hold(result.value, x, b; exact=status === :supported)
        ok || println("FACTS FAIL: ", show_case(c), " => ", typeof(x), " ", axes(x), " claimed ", type_bound(result.value), " ", value_axes(result.value), " ", result.value.family, " ", b)
        @test ok
    end
    # A rule that never meets a successful call would prove nothing.
    status === :incompatible ? @test(successes == 0) : @test(successes > 0)
    result
end

# Shape rules do not depend on the element type beyond Bool's special
# arithmetic; element types are checked exhaustively below.
const SHAPE_TYPES = (Bool, Int8, Float16, Float64)

@testset "symbolic axes are ordinary data with name identity" begin
    a, b = SymbolicAxis(:(levels(g))), SymbolicAxis(:(levels(g)))
    @test a == b && isequal(a, b) && hash(a) == hash(b)
    @test a != SymbolicAxis(:(levels(h))) && a != SymbolicAxis(:(axes(M, 1)))
    @test isempty(SymbolicAxis.parameters)
    @test typeof(vector_info(Float64, a)) === typeof(matrix_info(Int, 3, a)) === ValueInfo
    @test axis_length(a) === nothing && axis_length(Base.OneTo(4)) == 4 && axis_length(nothing) === nothing
    @test value_length(vector_info(Float64, a)) === nothing
    @test value_rank(matrix_info(Float64, a, nothing)) == 2
    @test value_axes(ValueInfo(type_bound=AbstractMatrix, axes=(a, nothing))) == (a, nothing)
    @test_throws ArgumentError array_info(Float64)
    @test_throws ArgumentError vector_info(Float64, -1)
    @test_throws ArgumentError vector_info(Float64, :n)
    @test_throws ArgumentError ValueInfo(axes=(:n,))
    @test_throws ArgumentError matrix_info(Float64, 2, 2; container_type=Vector{Float64})
end

@testset "matrix reads: positions, colons, gathers and linear indexing" begin
    for T in SHAPE_TYPES, container in (:array, :view, (T === Bool ? (:bit,) : ())...)
        M = arr(T, :m, :n; container)
        v = arr(T, :n; container)
        g, h = arr(Int, :k), arr(Int8, Unknown(:u))
        G = arr(Int, 2, :k)
        i, c = Known(1), Known(:)
        for source in (M, flip(adjoint, arr(T, :n, :m; container)))
            for args in ((i, i), (c, i), (i, c), (c, c), (g, c), (c, g), (g, h), (G, i), (i, G))
                check_case(call(getindex, source, args...))
            end
        end
        for idx in (i, c, g, G)
            check_case(call(getindex, M, idx))
            check_case(call(getindex, v, idx))
        end
        check_case(call(getindex, arr(T, :m, 1), c))      # linear keeps the one named axis
        check_case(call(getindex, arr(T, 1, Unknown(:u)), c))
        check_case(call(getindex, arr(T, 0, :n), c))
        for f in (adjoint, transpose)
            row = flip(f, v)
            for args in ((i, c), (c, i), (c, c), (g,), (c,), (i, g), (c, g), (c, h), (g, c), (G,))
                check_case(call(getindex, row, args...))
            end
            check_case(call(getindex, row, c, G); status=:unknown)
        end
        mask = arr(Bool, :m)
        check_case(call(getindex, M, mask, c); status=:unknown)
        check_case(call(getindex, arr(T, :m), mask); status=:unknown)
    end
    M = matrix_info(Float64, 3, 2)
    @test infer_call(getindex, ValueInfo[M, known_info(1), known_info(1), known_info(1)]).status === :unknown
    @test infer_call(getindex, ValueInfo[M, known_info(true), known_info(1)]).status === :unknown
    @test infer_call(getindex, ValueInfo[M, ValueInfo(), known_info(1)]).status === :unknown
    @test infer_call(getindex, ValueInfo[M, value_info(1:2), known_info(1)]).status === :unknown
end

@testset "length, adjoint and transpose" begin
    for T in SHAPE_TYPES
        for x in (arr(T, :n), arr(T, :m, :n), scalar(T), flip(adjoint, arr(T, :n)), arr(T, Unknown(:u), 2))
            check_case(call(length, x))
        end
        for f in (adjoint, transpose), g in (adjoint, transpose)
            for x in (arr(T, :n), arr(T, :m, :n), scalar(T), arr(T, Unknown(:u), 3; container=:view))
                check_case(call(f, x))
                check_case(call(g, flip(f, x)))
            end
        end
    end
    @test infer_call(adjoint, ValueInfo[array_info(Float64, 2, 2, 2)]).status === :unknown
    @test infer_call(adjoint, ValueInfo[vector_info(Complex{Float64}, 2)]).status === :unknown
    @test infer_call(length, ValueInfo[ValueInfo(type_bound=AbstractVector)]).status === :unknown
end

@testset "products carry outer axes and check inner axes" begin
    for T in SHAPE_TYPES, S in (Bool, Int8, Float64)
        M, N = arr(T, :m, :n), arr(S, :n, :p)
        v, w = arr(S, :n), arr(T, :m)
        row(x, f=adjoint) = flip(f, x)
        cases = Any[call(*, M, v), call(*, M, N), call(*, w, row(v)), call(*, w, row(v, transpose)),
                    call(*, arr(T, :m), arr(S, 1, :p)), call(*, row(w), arr(S, :m, :p)),
                    call(*, row(w, transpose), arr(S, :m, :p)), call(*, arr(T, :m, 1), row(v)),
                    call(*, scalar(T), N), call(*, M, scalar(S)), call(*, scalar(S), row(w)),
                    call(*, row(w), scalar(S)), call(/, M, scalar(S)), call(/, row(w), scalar(S)),
                    call(*, arr(T, :m, :n; container=:view), arr(S, :n; container=:view)),
                    call(*, row(arr(T, :n, :m)), v), call(*, scalar(S), row(arr(T, :n, :m))),
                    call(*, row(arr(T, :n, :m), transpose), scalar(S)), call(/, row(arr(T, :n, :m)), scalar(S)),
                    call(*, scalar(S), arr(T, :m, :n; container=:view)), call(/, arr(T, :n; container=:view), scalar(S)),
                    call(*, row(arr(T, :n, :m)), row(arr(S, :p, :n), transpose)), call(*, M, row(arr(S, :p, :n))),
                    call(*, row(w), row(arr(S, :p, :m)))]
        T === Bool && S === Bool || push!(cases, call(*, row(arr(T, :n)), v))
        foreach(check_case, cases)
        # Known inner lengths that differ can never succeed.
        check_case(call(*, arr(T, 2, 3), arr(S, 2)); status=:incompatible)
        check_case(call(*, arr(T, :m, 3), arr(S, 2, :p)); status=:incompatible)
        check_case(call(*, arr(T, :m), arr(S, 2, :p)); status=:incompatible)
    end
    for T in (Int8, Int64, Float16, Float64), S in (UInt8, Float32)
        M, v = arr(T, :m, :n), arr(S, :n)
        for c in (call(*, M, arr(S, :n, :p), v), call(*, flip(adjoint, arr(T, :m)), M, v),
                  call(*, scalar(S), M, v), call(*, arr(T, :m), flip(adjoint, v), arr(S, :n, :p)),
                  call(*, flip(transpose, arr(T, :m)), M, arr(S, :n, :n)), call(*, M, v, scalar(S)),
                  call(*, flip(adjoint, arr(T, :m)), arr(S, :m), arr(T, 1, :p)),
                  call(*, M, arr(S, :n, :p), arr(T, :p, :q), arr(S, :q)))
            check_case(c)
        end
    end
    # Bool dot products and chained Bool products stay explicit unknowns.
    check_case(call(*, flip(adjoint, arr(Bool, :n)), arr(Bool, :n)); status=:unknown)
    check_case(call(*, arr(Int8, :m, :n), arr(Bool, :n, :p), arr(Bool, :p)); status=:unknown)
    @test infer_call(*, ValueInfo[vector_info(Float64, 2), vector_info(Float64, 2)]).status === :unknown
    @test infer_call(*, ValueInfo[matrix_info(Float64, 2, 2)]).status === :unknown
    @test infer_call(/, ValueInfo[scalar_info(Float64), vector_info(Float64, 2)]).status === :unknown
    @test infer_call(/, ValueInfo[matrix_info(Float64, 2, 2), matrix_info(Float64, 2, 2)]).status === :unknown
end

@testset "undotted +/- need equal axes" begin
    for T in SHAPE_TYPES, S in (Bool, Int8, Float64)
        for (x, y) in ((arr(T, :n), arr(S, :n)), (arr(T, :m, :n), arr(S, :m, :n; container=:view)),
                       (arr(T, :n), arr(S, 3)), (arr(T, Unknown(:u)), arr(S, :n)),
                       (flip(adjoint, arr(T, :n)), flip(adjoint, arr(S, :n))),
                       (flip(transpose, arr(T, :n)), flip(adjoint, arr(S, :n))),
                       (flip(adjoint, arr(T, :n)), arr(S, 1, :n)),
                       (flip(transpose, arr(T, :n, :m)), flip(transpose, arr(S, :n, :m))),
                       (flip(adjoint, arr(T, :n, :m)), arr(S, :m, :n)))
            check_case(call(+, x, y))
            check_case(call(-, x, y))
        end
        check_case(call(+, arr(T, :n), arr(S, :n), arr(T, :n)))
        for x in (arr(T, :n), arr(T, :m, :n; container=:view), flip(adjoint, arr(T, :n)),
                  flip(transpose, arr(T, :m, :n)), (T === Bool ? (arr(T, :n; container=:bit),) : ())...)
            check_case(call(-, x))
            # Bool + is identity for wrappers but promotes Arrays; an abstract
            # bound cannot tell which, so it stays unknown.
            T === Bool && (x isa Leaf ? x.container === :view : length(x.args[1].dims) == 2) ?
                check_case(call(+, x); status=:unknown) : check_case(call(+, x))
        end
        check_case(call(+, arr(T, 2), arr(S, 3)); status=:incompatible)
        check_case(call(-, arr(T, :m, 2), arr(S, :m, 3)); status=:incompatible)
    end
    @test infer_call(+, ValueInfo[scalar_info(Float64), vector_info(Float64, 2)]).status === :unknown
    @test infer_call(+, ValueInfo[vector_info(Float64, 2), matrix_info(Float64, 2, 1)]).status === :unknown
    @test infer_call(-, ValueInfo[vector_info(Float64, 2), vector_info(Float64, 2), vector_info(Float64, 2)]).status === :unknown
end

@testset "hcat and vcat" begin
    for T in SHAPE_TYPES, S in (Bool, Int8, Float64)
        for parts in ((arr(T, :m), arr(S, :m)), (arr(T, :m, :n), arr(S, :m)), (arr(T, :m),),
                      (arr(T, :m, :n),), (scalar(T), scalar(S)), (arr(T, :m), arr(S, Unknown(:u))),
                      (flip(adjoint, arr(T, :n)), scalar(S)), (scalar(T), flip(transpose, arr(S, :n))),
                      (flip(adjoint, arr(T, :n)), flip(adjoint, arr(S, :p))),
                      (flip(adjoint, arr(T, :n)), flip(transpose, arr(S, :n))),
                      (flip(adjoint, arr(T, :n)), arr(S, 1, :p)), (flip(adjoint, arr(T, :n)),),
                      (arr(T, :m, :n), arr(S, 3, :p), arr(T, :m)))
            check_case(call(hcat, parts...))
        end
        check_case(call(hcat, arr(T, 2), arr(S, 3)); status=:incompatible)
        check_case(call(hcat, scalar(T), arr(S, 2)); status=:incompatible)
        for parts in ((arr(T, :n),), (arr(T, :n), arr(S, :n)), (scalar(T), arr(S, :n)),
                      (arr(T, 2), scalar(S), arr(S, 3)), (scalar(T),))
            check_case(call(vcat, parts...))
        end
    end
    @test infer_call(hcat, ValueInfo[]).status === :unknown
    @test infer_call(hcat, ValueInfo[array_info(Float64, 2, 2, 2)]).status === :unknown
end

@testset "reductions with and without dims" begin
    for T in SHAPE_TYPES, f in (sum, mean, std)
        for x in (arr(T, :m, :n), arr(T, :n), arr(T, Unknown(:u), :n; container=:view),
                  flip(adjoint, arr(T, :n)), flip(transpose, arr(T, :n, :m)),
                  (T === Bool ? (arr(T, :m, :n; container=:bit),) : ())...)
            check_case(call(f, x))
            for dims in (1, 2, 3, (1, 2), (2,), (), (2, 2))
                check_case(call(f, x; dims=Known(dims)))
            end
        end
        @test infer_call(f, ValueInfo[matrix_info(T, 2, 2)]; kwargs=[:dims => scalar_info(Int)]).status === :unknown
        @test infer_call(f, ValueInfo[matrix_info(T, 2, 2)]; kwargs=[:dims => known_info(0)]).status === :unknown
        @test infer_call(f, ValueInfo[matrix_info(T, 2, 2)]; kwargs=[:dims => known_info(true)]).status === :unknown
        @test infer_call(f, ValueInfo[matrix_info(T, 2, 2)]; kwargs=[:init => known_info(zero(T))]).status === :unknown
    end
    check_case(call(sum, scalar(Int8)))
    @test infer_call(mean, ValueInfo[scalar_info(Float64)]).status === :unknown
    @test infer_call(sum, ValueInfo[scalar_info(Float64)]; kwargs=[:dims => known_info(1)]).status === :unknown
end

@testset "cumsum and cumprod keep axes, with and without dims" begin
    for T in SHAPE_TYPES, f in (cumsum, cumprod)
        bit = T === Bool ? (:bit,) : ()
        for container in (:array, :view, bit...)
            for x in (arr(T, :n; container), arr(T, Unknown(:u); container), arr(T, 0; container))
                check_case(call(f, x))
            end
        end
        for x in (arr(T, :n), arr(T, :m, :n), arr(T, Unknown(:u), :n; container=:view),
                  arr(T, :m, 2, :p), flip(adjoint, arr(T, :n)), flip(transpose, arr(T, :n)),
                  flip(adjoint, arr(T, :n, :m)), flip(transpose, arr(T, :n, :m; container=:view)),
                  (T === Bool ? (arr(T, :m, :n; container=:bit), flip(transpose, arr(T, :n; container=:bit)),
                                 flip(adjoint, arr(T, :n, :m; container=:bit))) : ())...)
            for dims in (1, 2, 3)
                check_case(call(f, x; dims=Known(dims)))
            end
        end
        # Base iterates a scalar into a zero-dimensional Array, not a scalar.
        check_case(call(f, scalar(T)))
        # Base requires dims for a row vector or an array of rank ≥ 2
        # (UndefKeywordError), and takes one integer, never a tuple.
        for x in (matrix_info(T, 2, 2), infer_call(adjoint, ValueInfo[vector_info(T, 2)]).value)
            @test infer_call(f, ValueInfo[x]).status === :unknown
        end
        M = matrix_info(T, 2, 2)
        for dims in (known_info((1,)), known_info(0), known_info(true), scalar_info(Int))
            @test infer_call(f, ValueInfo[M]; kwargs=[:dims => dims]).status === :unknown
        end
        @test infer_call(f, ValueInfo[M]; kwargs=[:init => known_info(zero(T))]).status === :unknown
        @test infer_call(f, ValueInfo[scalar_info(T)]; kwargs=[:dims => known_info(1)]).status === :unknown
    end
    @test infer_call(cumsum, ValueInfo[vector_info(Real, 3)]).status === :unknown
    @test infer_call(cumsum, ValueInfo[ValueInfo(type_bound=AbstractVector{Float64}, element_type=Float64, axes=(nothing,))]).status === :unknown
    # RKPPL's `m = cumsum(vcat(0.0, s))[c]; sum(m)`: every fact survives the scan.
    for s in (arr(Float64, 2), arr(Float64, :n)), c in (arr(Int, :k), arr(Int, 2, :k))
        m = call(getindex, call(cumsum, call(vcat, scalar(Float64), s)), c)
        check_case(m)
        check_case(call(sum, m))
    end
end

@testset "broadcast carries names, never assumes a name is non-singleton" begin
    for T in SHAPE_TYPES, S in (Bool, Int8, Float64)
        bc(f, xs...) = call(broadcast, Known(f), xs...)
        for (x, y) in ((arr(T, :n), arr(S, :n)), (arr(T, :n), arr(S, :p)), (arr(T, :n), arr(S, 3)),
                       (arr(T, :n), arr(S, 1)), (arr(T, :n), arr(S, Unknown(:u))),
                       (arr(T, :m, :n), arr(S, :m)), (arr(T, :m, :n), arr(S, 1, :n; container=:view)),
                       (arr(T, :m, 1), arr(S, 1, :n)), (arr(T, :m, :n), scalar(S)),
                       (flip(adjoint, arr(T, :n)), arr(S, :m)), (flip(adjoint, arr(T, :n)), arr(S, :m, :n)),
                       (flip(adjoint, arr(T, :n)), flip(transpose, arr(S, :n))),
                       (flip(adjoint, arr(T, :n, :m)), scalar(S)), (flip(transpose, arr(T, :n, :m)), arr(S, :m)))
            check_case(bc(+, x, y))
            check_case(bc(*, x, y))
        end
        check_case(bc(+, arr(T, 2, :n), arr(S, 3)); status=:incompatible)
        check_case(bc(+, flip(adjoint, arr(T, :n)), scalar(S)); status=:unknown)
        check_case(bc(+, flip(adjoint, arr(T, :n)), flip(adjoint, arr(S, :n))); status=:unknown)
    end
    # Results are conditional on success: an unknown or second name could be
    # the stretched singleton, so neither name survives.
    n, p = SymbolicAxis(:n), SymbolicAxis(:p)
    for (x, y, expected) in ((n, n, n), (n, p, nothing), (n, nothing, nothing), (n, 1, n), (n, 3, Base.OneTo(3)),
                             (n, 0, Base.OneTo(0)), (1, 1, Base.OneTo(1)))
        r = infer_call(broadcast, ValueInfo[known_info(+), vector_info(Float64, x), vector_info(Float64, y)])
        @test r.status === :supported && value_axes(r.value) == (expected,)
    end
end

@testset "log, sqrt, mean and std element types match Julia" begin
    types = (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16,
             UInt32, UInt64, UInt128, Float16, Float32, Float64)
    for T in types
        for f in (log, sqrt)
            r = infer_call(f, ValueInfo[value_info(one(T))])
            @test r.status === :supported && type_bound(r.value) === typeof(f(one(T)))
            b = infer_call(broadcast, ValueInfo[known_info(f), value_info([one(T)])])
            @test b.status === :supported && type_bound(b.value) === typeof(f.([one(T)]))
        end
        for f in (mean, std), x in (fill(one(T), 3), fill(one(T), 2, 2))
            r = infer_call(f, ValueInfo[value_info(x)])
            @test r.status === :supported && type_bound(r.value) === typeof(f(x))
            d = infer_call(f, ValueInfo[value_info(x)]; kwargs=[:dims => known_info(1)])
            @test d.status === :supported && type_bound(d.value) === typeof(f(x; dims=1))
            @test value_axes(d.value) == axes(f(x; dims=1))
        end
    end
end

@testset "cumsum and cumprod element types match Julia" begin
    types = (Bool, Int8, Int16, Int32, Int64, Int128, UInt8, UInt16,
             UInt32, UInt64, UInt128, Float16, Float32, Float64)
    for T in types, f in (cumsum, cumprod)
        x, M = fill(one(T), 3), fill(one(T), 2, 2)
        for a in (x, view(x, 1:2), one(T))
            r = infer_call(f, ValueInfo[value_info(a)])
            y = f(a)
            @test r.status === :supported && y isa type_bound(r.value) && eltype(y) === element_type(r.value)
            @test value_axes(r.value) == axes(y)
            # Only a Bool result over a view stays abstract (Array or BitArray).
            a isa SubArray && eltype(y) === Bool || @test type_bound(r.value) === typeof(y)
        end
        for (a, d) in ((M, 1), (M, 2), (x, 2), (M', 1), (x', 2), (transpose(x), 1))
            r = infer_call(f, ValueInfo[value_info(a)]; kwargs=[:dims => known_info(d)])
            y = f(a; dims=d)
            @test r.status === :supported && y isa type_bound(r.value) && eltype(y) === element_type(r.value)
            @test value_axes(r.value) == axes(y) && value_info(y).family === r.value.family
        end
    end
    for f in (cumsum, cumprod), a in (trues(3), view(trues(4), 1:3), view(fill(true, 4), 1:3))
        r = infer_call(f, ValueInfo[value_info(a)])
        y = f(a)
        @test r.status === :supported && y isa type_bound(r.value) && eltype(y) === element_type(r.value)
        a isa SubArray && eltype(y) === Bool || @test type_bound(r.value) === typeof(y)
    end
end

@testset "product element types match Julia for every machine type pair" begin
    types = (Bool, Int8, Int64, UInt8, Float16, Float32, Float64)
    for T in types, S in types
        M, N, v, w = fill(one(T), 2, 3), fill(one(S), 3, 2), fill(one(S), 3), fill(one(T), 2)
        for (a, b) in ((M, v), (M, N), (w, v'), (w', M), (v', N), (fill(one(T), 3), fill(one(S), 1, 2)),
                       (one(T), N), (M, one(S)), (one(T), v'), (M', fill(one(S), 2)), (w, transpose(v)))
            r = infer_call(*, ValueInfo[value_info(a), value_info(b)])
            y = a * b
            @test r.status === :supported
            @test y isa type_bound(r.value) && eltype(y) == element_type(r.value)
            @test value_axes(r.value) == axes(y)
            y isa AbstractArray && @test value_info(y).family === r.value.family
        end
        r = infer_call(*, ValueInfo[value_info(w'), value_info(fill(one(S), 2))])
        if T === Bool && S === Bool
            @test r.status === :unknown && value_rank(r.value) == 0
        else
            @test r.status === :supported && type_bound(r.value) === typeof(w' * fill(one(S), 2))
        end
        d = infer_call(/, ValueInfo[value_info(M), value_info(one(S))])
        @test type_bound(d.value) === typeof(M / one(S))
        h = infer_call(hcat, ValueInfo[value_info(w), value_info(fill(one(S), 2, 2))])
        @test hcat(w, fill(one(S), 2, 2)) isa type_bound(h.value) && element_type(h.value) == eltype(hcat(w, fill(one(S), 2, 2)))
        p = infer_call(+, ValueInfo[value_info(M), value_info(fill(one(S), 2, 3))])
        @test type_bound(p.value) === typeof(M + fill(one(S), 2, 3))
    end
end
