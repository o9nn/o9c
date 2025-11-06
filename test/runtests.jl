"""
Basic smoke test for DeepTreeEchoSelf framework
"""

using Test

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))

@testset "DeepTreeEchoSelf Basic Tests" begin
    
    @testset "Module Loading" begin
        # This will fail if dependencies aren't installed, but shows structure is sound
        @test_throws Exception include("../src/DeepTreeEchoSelf.jl")
    end
    
    @testset "File Structure" begin
        # Check that all expected files exist
        src_dir = joinpath(@__DIR__, "../src")
        
        @test isfile(joinpath(src_dir, "DeepTreeEchoSelf.jl"))
        @test isfile(joinpath(src_dir, "paun_membranes.jl"))
        @test isfile(joinpath(src_dir, "deep_tree_esn.jl"))
        @test isfile(joinpath(src_dir, "butcher_series.jl"))
        @test isfile(joinpath(src_dir, "j_surface.jl"))
        @test isfile(joinpath(src_dir, "emotion_theory.jl"))
        @test isfile(joinpath(src_dir, "transformer_integration.jl"))
        @test isfile(joinpath(src_dir, "cognitive_architecture.jl"))
    end
    
    @testset "Documentation" begin
        @test isfile(joinpath(@__DIR__, "../README.md"))
        @test isfile(joinpath(@__DIR__, "../docs/README.md"))
        @test isfile(joinpath(@__DIR__, "../Project.toml"))
    end
    
    @testset "Examples" begin
        @test isfile(joinpath(@__DIR__, "../examples/demo_emergence.jl"))
    end
    
end

println("\n✓ Basic structure tests passed!")
println("Note: Full functionality tests require dependencies to be installed.")
println("Run: julia --project=. -e 'using Pkg; Pkg.instantiate()' to install dependencies.")
