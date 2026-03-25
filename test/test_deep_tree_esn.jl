"""
Unit tests for Deep Tree Echo State Network component.

These tests work standalone (no external dependencies required)
because deep_tree_esn.jl only uses Julia standard library packages.
"""

using Test
using LinearAlgebra
using Random
using Statistics

# Include standalone - only uses stdlib (LinearAlgebra, Random, Statistics)
include(joinpath(@__DIR__, "../src/deep_tree_esn.jl"))

Random.seed!(42)

@testset "Deep Tree ESN" begin

    @testset "ReservoirNode construction" begin
        node = ReservoirNode(0, 0, 10, 5)

        @test node.id == 0
        @test node.depth == 0
        @test node.reservoir_size == 10
        @test size(node.W_in) == (10, 5)
        @test size(node.W_res) == (10, 10)
        @test size(node.W_out) == (5, 10)
        @test length(node.state) == 10
        @test all(node.state .== 0.0)     # initial state is zeros
        @test isempty(node.children)
        @test node.parent === nothing
    end

    @testset "ReservoirNode echo state property" begin
        # The spectral radius of W_res should match the requested value
        node = ReservoirNode(0, 0, 10, 5; spectral_radius=0.9)
        ρ = maximum(abs.(eigvals(node.W_res)))
        @test ρ ≈ 0.9 atol=0.05

        node2 = ReservoirNode(1, 0, 10, 5; spectral_radius=0.7)
        ρ2 = maximum(abs.(eigvals(node2.W_res)))
        @test ρ2 ≈ 0.7 atol=0.05
    end

    @testset "get_persona_params completeness" begin
        for persona in [:contemplative_scholar, :dynamic_explorer,
                        :cautious_analyst, :creative_visionary, :balanced]
            params = get_persona_params(persona)
            @test haskey(params, :spectral_radius)
            @test haskey(params, :input_scaling)
            @test haskey(params, :leak_rate)
            @test 0.0 < params[:spectral_radius] < 1.1
            @test 0.0 < params[:input_scaling] <= 1.0
            @test 0.0 < params[:leak_rate] <= 1.0
        end
    end

    @testset "get_persona_params distinct styles" begin
        scholar  = get_persona_params(:contemplative_scholar)
        explorer = get_persona_params(:dynamic_explorer)
        analyst  = get_persona_params(:cautious_analyst)

        # Contemplative scholar: high memory (spectral radius), slow leak
        @test scholar[:spectral_radius] > explorer[:spectral_radius]
        @test scholar[:leak_rate] < explorer[:leak_rate]

        # Cautious analyst: highest stability
        @test analyst[:spectral_radius] >= scholar[:spectral_radius]
    end

    @testset "DeepTreeESN node count" begin
        # depth=1, branching=2 → 1 root + 2 children = 3 nodes
        esn1 = DeepTreeESN(1, 10, 5, branching=2)
        @test length(esn1.all_nodes) == 3

        # depth=2, branching=2 → 1 + 2 + 4 = 7 nodes
        esn2 = DeepTreeESN(2, 10, 5, branching=2)
        @test length(esn2.all_nodes) == 7

        # depth=0 → just root = 1 node
        esn0 = DeepTreeESN(0, 10, 5)
        @test length(esn0.all_nodes) == 1
    end

    @testset "DeepTreeESN tree structure" begin
        esn = DeepTreeESN(2, 10, 5, branching=2)

        @test esn.root.depth == 0
        @test esn.depth == 2
        @test length(esn.root.children) == 2

        # Root's children are at depth 1
        for child in esn.root.children
            @test child.depth == 1
            @test length(child.children) == 2  # each has 2 children at depth 2
        end
    end

    @testset "DeepTreeESN persona_params" begin
        esn = DeepTreeESN(1, 10, 5, persona=:dynamic_explorer)
        params = esn.persona_params
        @test haskey(params, :spectral_radius)
        @test haskey(params, :leak_rate)
    end

    @testset "update_state! changes state" begin
        node = ReservoirNode(0, 0, 10, 5)
        initial_state = copy(node.state)  # starts as zeros

        input = ones(5)
        update_state!(node, input, 0.5)

        # State should have changed from zeros
        @test norm(node.state - initial_state) > 0
    end

    @testset "update_state! bounded by tanh" begin
        node = ReservoirNode(0, 0, 10, 5)
        # Process large input for many steps
        for _ in 1:100
            update_state!(node, 100 * ones(5), 0.9)
        end
        # Leaky integration with tanh keeps values bounded
        @test all(abs.(node.state) .<= 1.0)
    end

    @testset "update_state! leaky integration" begin
        node = ReservoirNode(0, 0, 10, 5)

        # Process single input
        input = randn(5)
        update_state!(node, input, 0.0)  # leak_rate=0 → no update
        @test all(node.state .== 0.0)

        update_state!(node, input, 1.0)  # leak_rate=1 → full update
        expected = tanh.(node.W_in * input)  # W_res * zeros = 0
        @test node.state ≈ expected atol=1e-10
    end

    @testset "process_tree! runs without error" begin
        esn = DeepTreeESN(2, 10, 5)
        input = randn(5)
        @test_nowarn process_tree!(esn, input)
    end

    @testset "process_tree! updates node states" begin
        esn = DeepTreeESN(2, 10, 5)
        input = ones(5)

        process_tree!(esn, input)

        # Root state should be non-zero after processing
        @test norm(esn.root.state) > 0
    end

    @testset "collect_states dimension" begin
        esn = DeepTreeESN(2, 10, 5, branching=2)
        states = collect_states(esn)

        # 7 nodes × 10 reservoir_size = 70
        @test length(states) == 7 * 10
    end

    @testset "collect_states after processing" begin
        esn = DeepTreeESN(2, 8, 4, branching=2)
        input = randn(4)

        process_tree!(esn, input)
        states = collect_states(esn)

        @test length(states) == 7 * 8
        # Some states should be non-zero after processing
        @test norm(states) > 0
    end

    @testset "train_readout! fits data" begin
        Random.seed!(123)
        esn = DeepTreeESN(1, 10, 5, branching=2)

        n_samples = 30
        inputs  = [randn(5) for _ in 1:n_samples]
        targets = [randn(5) for _ in 1:n_samples]

        @test_nowarn train_readout!(esn, inputs, targets)

        # Output weights should now be non-zero
        @test !all(esn.root.W_out .== 0.0)
        @test size(esn.root.W_out, 1) == 5   # output_dim
    end

    @testset "multiple personas all produce valid ESNs" begin
        for persona in [:contemplative_scholar, :dynamic_explorer,
                        :cautious_analyst, :creative_visionary, :balanced]
            esn = DeepTreeESN(1, 10, 5, persona=persona)
            input = randn(5)
            @test_nowarn process_tree!(esn, input)
            states = collect_states(esn)
            @test all(isfinite.(states))
        end
    end

end
