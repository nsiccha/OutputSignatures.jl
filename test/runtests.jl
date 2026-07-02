using OutputSignatures, Test
using Statistics: quantile, mean, std, var, median
using LinearAlgebra: dot, norm, diag

@testset "OutputSignatures" begin
    @testset "defaults (broadcast/elementwise case)" begin
        x = rand(4)
        @test output_size(identity, size(x)) == (4,)
        @test output_eltype(identity, eltype(x)) == Float64
        s = output_signature(identity, x)
        @test s.size == (4,) && s.eltype == Float64
        y = alloc_output(identity, x)
        @test size(y) == (4,) && eltype(y) == Float64
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
    end

    @testset "inferred_eltype (opt-in promote_op)" begin
        @test OutputSignatures.inferred_eltype(x -> x > 0, Float64) == Bool
    end
end
