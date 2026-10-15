@testitem "OutputSignatures" begin
    using OutputSignatures, Test
    using Statistics: quantile, mean, std, var, median
    using LinearAlgebra: dot, norm, diag

    @testset "explicit preserving rules and unknown calls" begin
        x = rand(4)
        for f in (identity, copy, reverse, sort)
            @test output_signature(f, x) == (; size=size(f(x)), eltype=eltype(f(x)))
            y = alloc_output(f, x)
            @test size(y) == size(f(x)) && eltype(y) == eltype(f(x))
        end
        calls = Ref(0)
        g(x) = (calls[] += 1; length(x))
        # A valid but undeclared function may be rejected by this allocation
        # API. A later partial-metadata query can represent its result without
        # claiming that size and element type were preserved.
        sig_or_error = try output_signature(g, x) catch err; err end
        @test sig_or_error isa ArgumentError ||
              sig_or_error == (; size=(), eltype=Int)
        alloc_or_error = try alloc_output(g, x) catch err; err end
        @test alloc_or_error isa ArgumentError ||
              (size(alloc_or_error) == () && eltype(alloc_or_error) == Int)
        @test calls[] == 0
    end

    @testset "explicit preserving shapes with changing element types" begin
        for (f, x) in ((cumsum, Int8[1,2,3]),
                       (cumprod, Int8[1,2,3]),
                       (cumsum, Bool[true,false,true]),
                       (cumprod, Bool[true,false,true]),
                       (diff, Bool[true,false,true]))
            actual = f(x)
            @test output_signature(f, x) == (; size=size(actual), eltype=eltype(actual))
        end
        @test output_size(diff, (0,)) == size(diff(Float64[]))
    end

    @testset "Base reductions to scalar" begin
        x = rand(4)
        for f in (maximum, minimum, count, any, all)
            @test output_size(f, size(x)) == ()
        end
        @test output_eltype(count, eltype(x)) == Int
        @test output_eltype(any, eltype(x)) == Bool
        @test output_eltype(all, eltype(x)) == Bool

        y = alloc_output(maximum, x)
        @test size(y) == () && eltype(y) == Float64
        y = alloc_output(count, x .> 0)
        @test size(y) == () && eltype(y) == Int
    end

    @testset "sum/prod: narrow-int widening matches Base exactly" begin
        for T in (Int8, UInt8, Int16, Bool)
            v = T <: Bool ? rand(Bool, 5) : rand(T(0):T(3), 5)
            @test output_size(sum, size(v)) == ()
            @test output_eltype(sum, eltype(v)) == typeof(sum(v))
            @test output_size(prod, size(v)) == ()
            @test output_eltype(prod, eltype(v)) == typeof(prod(v))
            y = alloc_output(sum, v)
            @test size(y) == () && eltype(y) == typeof(sum(v))
        end
        v = rand(5)
        @test output_eltype(sum, eltype(v)) == Float64
    end

    @testset "Base shape transforms" begin
        v = rand(5)
        @test output_size(diff, size(v)) == (4,)
        y = alloc_output(diff, v)
        @test size(y) == (4,) && eltype(y) == Float64

        A = rand(3, 4)
        @test output_size(vec, size(A)) == (12,)
        y = alloc_output(vec, A)
        @test size(y) == (12,) && eltype(y) == Float64

        @test output_size(permutedims, size(A)) == (4, 3)
        y = alloc_output(permutedims, A)
        @test size(y) == (4, 3) && eltype(y) == Float64

        @test output_size(permutedims, size(A), (2, 1)) == (4, 3)
    end

    @testset "quantile: size from p, not |v| (Statistics ext)" begin
        v = rand(100)
        p = [0.25, 0.5, 0.75]
        @test output_size(quantile, size(v), p) == (3,)
        @test output_eltype(quantile, eltype(v), p) == Float64
        y = alloc_output(quantile, v, p)
        @test size(y) == (3,) && eltype(y) == Float64
    end

    @testset "quantile: scalar p (bug fix — used to return insize, not ())" begin
        v = rand(100)
        @test output_size(quantile, size(v), 0.5) == ()
        @test output_eltype(quantile, eltype(v), 0.5) == Float64
        y = alloc_output(quantile, v, 0.5)
        @test size(y) == () && eltype(y) == Float64
    end

    @testset "mean/std/var/median (Statistics ext)" begin
        v = rand(1:10, 20)
        for f in (mean, std, var, median)
            @test output_size(f, size(v)) == ()
            @test output_eltype(f, eltype(v)) == Float64
            y = alloc_output(f, v)
            @test size(y) == () && eltype(y) == Float64
        end
        cv = ComplexF64[1+2im, 2+3im, 3+4im]
        for f in (std, var)
            @test output_signature(f, cv) == (; size=size(f(cv)), eltype=typeof(f(cv)))
        end
    end

    @testset "LinearAlgebra ext" begin
        A = rand(3, 4)
        B = rand(4, 5)
        v = rand(4)
        @test output_size(*, size(A), B) == (3, 5)
        @test output_eltype(*, eltype(A), B) == Float64
        y = alloc_output(*, A, B)
        @test size(y) == (3, 5) && eltype(y) == Float64

        @test output_size(*, size(A), v) == (3,)
        y = alloc_output(*, A, v)
        @test size(y) == (3,) && eltype(y) == Float64

        u = rand(4)
        @test output_size(dot, size(u), v) == ()
        y = alloc_output(dot, u, v)
        @test size(y) == () && eltype(y) == Float64

        @test output_size(norm, size(v)) == ()
        y = alloc_output(norm, v)
        @test size(y) == () && eltype(y) == Float64
        cv = rand(ComplexF64, 4)
        @test output_eltype(norm, eltype(cv)) == Float64

        @test output_size(diag, size(A)) == (3,)
        y = alloc_output(diag, A)
        @test size(y) == (3,) && eltype(y) == Float64

        bools = Bool[true false; true true]
        @test output_eltype(*, Bool, bools) == eltype(bools * bools)
        @test output_eltype(dot, Bool, Bool[true, false]) ==
              typeof(dot(Bool[true, false], Bool[true, false]))
    end

    @testset "gather A[idx]: size from idx, not A" begin
        A = rand(3)
        idx = [1, 3, 2, 1, 2]
        @test output_size(getindex, size(A), idx) == (5,)
        @test output_eltype(getindex, eltype(A), idx) == Float64
        y = alloc_output(getindex, A, idx)
        @test size(y) == (5,) && eltype(y) == Float64
        @test size(y) == size(A[idx]) && eltype(y) == eltype(A[idx])

        # a matrix of indices gathers into that matrix's shape
        M = rand(5)
        midx = [1 2; 3 4]
        @test output_size(getindex, size(M), midx) == (2, 2)
        @test alloc_output(getindex, M, midx) |> size == size(M[midx])

        # scalar and whole-array linear indices have their own declarations
        @test output_size(getindex, size(A), 2) == size(A[2])
        @test output_size(getindex, size(A), :) == size(A[:])
        @test output_signature(getindex, A, 2) == (; size=size(A[2]), eltype=typeof(A[2]))
        @test size(alloc_output(getindex, A, 2)) == ()

        # A Boolean mask has a data-dependent result length; do not inspect it
        # during signature analysis or inherit the input's shape.
        mask = Bool[true, false, true]
        sig_or_error = try output_signature(getindex, A, mask) catch err; err end
        @test sig_or_error isa ArgumentError ||
              sig_or_error == (; size=size(A[mask]), eltype=eltype(A[mask]))
    end

    @testset "broadcast: shape+eltype of f.(args...), no pilot" begin
        # the PPL cases: beta[1] .+ beta[2].*x .+ b[group], exp.(log_sigma),
        # tau .* (L*z), and a scalar-first broadcast.
        x = rand(4); beta = rand(2); b = rand(3); group = [1, 2, 3, 1]
        lin_args = (beta[1], beta[2] .* x, b[group])
        @test broadcast_size(lin_args...) == (4,)
        @test broadcast_eltype(+, lin_args...) == Float64
        s = broadcast_signature(+, lin_args...)
        @test s.size == (4,) && s.eltype == Float64
        y = alloc_broadcast(+, lin_args...)
        real = broadcast(+, lin_args...)
        @test size(y) == size(real) && eltype(y) == eltype(real)

        log_sigma = rand(2)
        @test broadcast_size(log_sigma) == (2,)
        @test alloc_broadcast(exp, log_sigma) |> size == size(exp.(log_sigma))

        # scalar-first: the case the `=insize` default got wrong (3.0 .+ x)
        @test broadcast_size(3.0, x) == (4,)
        y = alloc_broadcast(+, 3.0, x)
        @test size(y) == (4,) && eltype(y) == Float64

        # integer args promote like Base's own broadcast eltype
        i = rand(1:5, 6)
        @test broadcast_eltype(+, i, 1) == eltype(i .+ 1)
        @test eltype(alloc_broadcast(+, i, 1)) == eltype(i .+ 1)
    end

    @testset "inferred_eltype (opt-in promote_op)" begin
        @test OutputSignatures.inferred_eltype(x -> x > 0, Float64) == Bool
    end
end

@testitem "OutputSignatures multi-dim gather" begin
    using OutputSignatures, Test
    using Statistics: quantile, mean, std, var, median
    using LinearAlgebra: dot, norm, diag

        @testset "multi-dim (Cartesian) gather A[i, group]: mixed scalar/vector/colon" begin
            b = reshape(1.0:15.0, 3, 5)   # 3x5
            g1 = [1, 3]                    # into dim-1 (<= 3)
            g2 = [2, 4, 1, 5]              # into dim-2 (<= 5)

            # scalar + vector: drop dim-1, keep length(g2)  (the b[i, group] case)
            @test output_size(getindex, size(b), 1, g2) == (4,)
            @test output_size(getindex, size(b), 1, g2) == size(b[1, g2])
            # colon + vector: keep dim-1 extent, keep length(g2)
            @test output_size(getindex, size(b), :, g2) == (3, 4)
            @test output_size(getindex, size(b), :, g2) == size(b[:, g2])
            # vector + scalar: keep length(g1), drop dim-2
            @test output_size(getindex, size(b), g1, 2) == (2,)
            @test output_size(getindex, size(b), g1, 2) == size(b[g1, 2])
            # vector + vector
            @test output_size(getindex, size(b), g1, g2) == (2, 4)
            @test output_size(getindex, size(b), g1, g2) == size(b[g1, g2])
            # range indices keep their length
            @test output_size(getindex, size(b), 2:3, 1:2) == (2, 2)
            @test output_size(getindex, size(b), 2:3, 1:2) == size(b[2:3, 1:2])

            # eltype preserved by an explicit rule; allocation and query agree
            @test output_eltype(getindex, eltype(b), 1, g2) == Float64
            y = alloc_output(getindex, b, 1, g2)
            @test size(y) == size(b[1, g2]) && eltype(y) == eltype(b[1, g2])
            s = output_signature(getindex, b, :, g2)
            @test s.size == size(b[:, g2]) && s.eltype == eltype(b[:, g2])

            # the single-arg linear-indexing method is untouched by the ≥2-arg method
            M = reshape(1.0:15.0, 3, 5)
            lin = [2, 6, 11]
            @test output_size(getindex, size(M), lin) == (3,)
            @test output_size(getindex, size(M), lin) == size(M[lin])
        end
end
