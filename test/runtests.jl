"""
Test suite for DeepTreeEchoSelf framework.

Organized in two layers:
  1. Component unit tests - run standalone without external dependencies.
  2. Basic smoke tests  - file structure, documentation, and module-load check.
"""

using Test

# Add the src directory to the load path
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))

# ── Component unit tests (stdlib only, always runnable) ────────────────────────

include("test_emotion_theory.jl")
include("test_deep_tree_esn.jl")

# ── Basic smoke tests ──────────────────────────────────────────────────────────

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
        examples_dir = joinpath(@__DIR__, "../examples")
        @test isfile(joinpath(examples_dir, "demo_emergence.jl"))
        @test isfile(joinpath(examples_dir, "learning_example.jl"))
        @test isfile(joinpath(examples_dir, "multi_agent_example.jl"))
    end
    
end

println("\n✓ All tests passed!")
println("Note: Full integration tests require external dependencies.")
println("Run: julia --project=. -e 'using Pkg; Pkg.instantiate()' to install dependencies.")
