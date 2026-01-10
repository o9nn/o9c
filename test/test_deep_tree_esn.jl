"""
Comprehensive unit tests for Deep Tree Echo State Networks
"""

@testset "Deep Tree ESN" begin

    @testset "ReservoirNode Construction" begin
        node = ReservoirNode(0, 0, 50, 10)

        @test node.id == 0
        @test node.depth == 0
        @test node.reservoir_size == 50
        @test size(node.W_in) == (50, 10)
        @test size(node.W_res) == (50, 50)
        @test size(node.W_out) == (10, 50)
        @test length(node.state) == 50
        @test all(node.state .== 0.0)  # Initial state is zeros
        @test isempty(node.children)
        @test node.parent === nothing
    end

    @testset "ReservoirNode Custom Parameters" begin
        node = ReservoirNode(1, 2, 30, 5,
            spectral_radius=0.8,
            input_scaling=0.3)

        @test node.id == 1
        @test node.depth == 2
        @test size(node.W_in) == (30, 5)

        # Verify spectral radius (for small matrices)
        eigenvalues = eigvals(node.W_res)
        actual_sr = maximum(abs.(eigenvalues))
        @test isapprox(actual_sr, 0.8, atol=0.05)
    end

    @testset "ReservoirNode Large Matrix Spectral Radius" begin
        # Test power iteration method for large matrices
        node = ReservoirNode(0, 0, 150, 20,
            spectral_radius=0.95)

        # For large matrices, spectral radius should be approximately correct
        eigenvalues = eigvals(node.W_res)
        actual_sr = maximum(abs.(eigenvalues))
        @test isapprox(actual_sr, 0.95, atol=0.1)
    end

    @testset "get_persona_params" begin
        # Test contemplative_scholar
        params = get_persona_params(:contemplative_scholar)
        @test params[:spectral_radius] == 0.95
        @test params[:input_scaling] == 0.3
        @test params[:leak_rate] == 0.2

        # Test dynamic_explorer
        params = get_persona_params(:dynamic_explorer)
        @test params[:spectral_radius] == 0.7
        @test params[:input_scaling] == 0.8
        @test params[:leak_rate] == 0.8

        # Test cautious_analyst
        params = get_persona_params(:cautious_analyst)
        @test params[:spectral_radius] == 0.99
        @test params[:input_scaling] == 0.2
        @test params[:leak_rate] == 0.3

        # Test creative_visionary
        params = get_persona_params(:creative_visionary)
        @test params[:spectral_radius] == 0.85
        @test params[:input_scaling] == 0.7
        @test params[:leak_rate] == 0.6

        # Test balanced (default)
        params = get_persona_params(:balanced)
        @test params[:spectral_radius] == 0.9
        @test params[:input_scaling] == 0.5
        @test params[:leak_rate] == 0.5

        # Test unknown persona (falls through to balanced)
        params = get_persona_params(:unknown_persona)
        @test params[:spectral_radius] == 0.9
    end

    @testset "DeepTreeESN Construction" begin
        esn = DeepTreeESN(2, 30, 10, persona=:balanced)

        @test esn.depth == 2
        @test esn.root.depth == 0

        # Count nodes: depth 2, branching 2 → 1 + 2 + 4 = 7
        @test length(esn.all_nodes) == 7

        # Verify parent-child relationships
        @test length(esn.root.children) == 2

        for child in esn.root.children
            @test child.parent === esn.root
            @test child.depth == 1
        end

        # Verify persona params are stored
        @test haskey(esn.persona_params, :spectral_radius)
        @test haskey(esn.persona_params, :input_scaling)
        @test haskey(esn.persona_params, :leak_rate)
    end

    @testset "DeepTreeESN Different Branching" begin
        esn = DeepTreeESN(2, 20, 8, branching=3)

        # Count: 1 + 3 + 9 = 13
        @test length(esn.all_nodes) == 13
        @test length(esn.root.children) == 3
    end

    @testset "DeepTreeESN Depth 0" begin
        esn = DeepTreeESN(0, 25, 5)

        @test length(esn.all_nodes) == 1
        @test isempty(esn.root.children)
    end

    @testset "update_state!" begin
        node = ReservoirNode(0, 0, 20, 5)
        input = randn(5)
        leak_rate = 0.5

        initial_state = copy(node.state)

        update_state!(node, input, leak_rate)

        # State should have changed
        @test !all(node.state .== initial_state)

        # State should be finite
        @test all(isfinite.(node.state))

        # State should be bounded (due to tanh)
        @test all(abs.(node.state) .<= 1.0)
    end

    @testset "update_state! Leak Rate Effects" begin
        node1 = ReservoirNode(0, 0, 15, 5)
        node2 = ReservoirNode(0, 0, 15, 5)

        # Use same weights
        node2.W_in .= node1.W_in
        node2.W_res .= node1.W_res

        input = randn(5)

        update_state!(node1, input, 0.1)  # Low leak rate (more memory)
        update_state!(node2, input, 0.9)  # High leak rate (less memory)

        # High leak rate should result in larger state changes from zero
        @test norm(node2.state) > norm(node1.state)
    end

    @testset "process_tree!" begin
        esn = DeepTreeESN(2, 20, 10)
        input = randn(10)

        # Ensure initial states are zeros
        for node in esn.all_nodes
            @test all(node.state .== 0.0)
        end

        process_tree!(esn, input)

        # States should have changed
        for node in esn.all_nodes
            @test !all(node.state .== 0.0)
            @test all(isfinite.(node.state))
        end
    end

    @testset "process_tree! Multiple Steps" begin
        esn = DeepTreeESN(2, 25, 8, persona=:dynamic_explorer)

        for step in 1:5
            input = randn(8)
            process_tree!(esn, input)

            # States should remain bounded
            for node in esn.all_nodes
                @test all(isfinite.(node.state))
            end
        end
    end

    @testset "collect_states" begin
        esn = DeepTreeESN(1, 20, 10)  # 1 + 2 = 3 nodes

        # Process some input
        process_tree!(esn, randn(10))

        states = collect_states(esn)

        # Total state dimension = 3 nodes × 20 reservoir_size = 60
        @test length(states) == 60
        @test all(isfinite.(states))
    end

    @testset "collect_states Concatenation Order" begin
        esn = DeepTreeESN(1, 10, 5)

        # Set known states
        for (i, node) in enumerate(esn.all_nodes)
            node.state .= fill(Float64(i), 10)
        end

        states = collect_states(esn)

        # First 10 elements should be from first node
        @test all(states[1:10] .== 1.0)
    end

    @testset "train_readout!" begin
        esn = DeepTreeESN(1, 15, 5)

        # Generate training data
        n_samples = 20
        inputs = [randn(5) for _ in 1:n_samples]
        targets = [randn(5) for _ in 1:n_samples]

        train_readout!(esn, inputs, targets)

        # Output weights should be non-zero
        @test !all(esn.root.W_out .== 0.0)
        @test all(isfinite.(esn.root.W_out))
    end

    @testset "train_readout! Dimensions" begin
        esn = DeepTreeESN(1, 20, 8)

        inputs = [randn(8) for _ in 1:15]
        targets = [randn(8) for _ in 1:15]

        train_readout!(esn, inputs, targets)

        # W_out should be output_dim × total_state_dim
        # total_state_dim = 3 nodes × 20 = 60
        @test size(esn.root.W_out) == (8, 60)
    end

    @testset "Echo State Property Stability" begin
        # Test that system is stable over many timesteps
        esn = DeepTreeESN(2, 30, 10)

        for _ in 1:100
            input = randn(10)
            process_tree!(esn, input)
        end

        # States should remain bounded after many steps
        for node in esn.all_nodes
            @test all(abs.(node.state) .<= 5.0)  # Allow some margin beyond tanh bounds
            @test all(isfinite.(node.state))
        end
    end

    @testset "Different Personas Create Different Dynamics" begin
        input = randn(10)

        esn1 = DeepTreeESN(2, 25, 10, persona=:contemplative_scholar)
        esn2 = DeepTreeESN(2, 25, 10, persona=:dynamic_explorer)

        process_tree!(esn1, input)
        process_tree!(esn2, input)

        states1 = collect_states(esn1)
        states2 = collect_states(esn2)

        # Different personas should produce different states
        @test !all(states1 .== states2)
    end

    @testset "Hierarchical Information Flow" begin
        esn = DeepTreeESN(2, 20, 10)

        # Process input
        process_tree!(esn, randn(10))

        # Root should have different state than leaves
        root_state = esn.root.state

        # Get leaf nodes
        leaf_states = [node.state for node in esn.all_nodes if isempty(node.children)]

        # States should differ (not identical)
        for leaf_state in leaf_states
            @test !all(leaf_state .== root_state)
        end
    end

    @testset "Node ID Uniqueness" begin
        esn = DeepTreeESN(3, 15, 5)

        ids = [node.id for node in esn.all_nodes]
        @test length(unique(ids)) == length(ids)
    end

    @testset "Reservoir Connectivity Sparsity" begin
        # Test that reservoir weights are random (not all same sign)
        node = ReservoirNode(0, 0, 50, 10)

        positive_count = sum(node.W_res .> 0)
        negative_count = sum(node.W_res .< 0)
        total = length(node.W_res)

        # Roughly half should be positive and half negative
        @test positive_count > total / 4
        @test negative_count > total / 4
    end

end
