using OutputSignatures, Test
using Statistics: quantile

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

    @testset "quantile: size from p, not |v| (Statistics ext)" begin
        v = rand(100)
        p = [0.25, 0.5, 0.75]
        @test output_size(quantile, size(v), p) == (3,)
        @test output_eltype(quantile, eltype(v), p) == Float64
        y = alloc_output(quantile, v, p)
        @test size(y) == (3,) && eltype(y) == Float64
    end

    @testset "inferred_eltype (opt-in promote_op)" begin
        @test OutputSignatures.inferred_eltype(x -> x > 0, Float64) == Bool
    end
end
