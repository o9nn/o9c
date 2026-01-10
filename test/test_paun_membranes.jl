"""
Comprehensive unit tests for Paun P-System Membrane Reservoirs
"""

@testset "Paun Membranes" begin

    @testset "Membrane Construction" begin
        # Test basic membrane creation
        m = Membrane(1, 0, 10)
        @test m.id == 1
        @test m.depth == 0
        @test m.permeability == 0.5  # default
        @test length(m.state) == 10
        @test isempty(m.rules)
        @test isempty(m.children)
        @test m.parent === nothing

        # Test custom permeability
        m2 = Membrane(2, 1, 5, permeability=0.8)
        @test m2.permeability == 0.8
        @test m2.depth == 1

        # Test permeability clamping (via modulation later)
        m3 = Membrane(3, 2, 3, permeability=0.2)
        @test m3.permeability == 0.2
    end

    @testset "PaunMembraneSystem Construction" begin
        # Test basic system creation
        system = PaunMembraneSystem(2, 10)  # depth=2, dim=10

        @test system.depth == 2
        @test system.root_id == 0
        @test haskey(system.membranes, 0)  # root exists

        # Count membranes: depth 2 with branching 2 = 1 + 2 + 4 = 7
        @test length(system.membranes) == 7

        # Test root membrane
        root = system.membranes[0]
        @test root.depth == 0
        @test length(root.children) == 2  # branching factor
        @test root.parent === nothing

        # Test child membranes have correct parent
        for child_id in root.children
            child = system.membranes[child_id]
            @test child.parent == 0
            @test child.depth == 1
        end

        # Test graph structure
        @test nv(system.graph) == 7
    end

    @testset "PaunMembraneSystem Different Branching" begin
        # Test with different branching factor
        system = PaunMembraneSystem(2, 5, 3)  # depth=2, dim=5, branching=3

        # Count: 1 + 3 + 9 = 13
        @test length(system.membranes) == 13

        root = system.membranes[0]
        @test length(root.children) == 3
    end

    @testset "PaunMembraneSystem Depth 0" begin
        # Edge case: depth 0 (just root)
        system = PaunMembraneSystem(0, 8)
        @test length(system.membranes) == 1
        @test system.membranes[0].depth == 0
        @test isempty(system.membranes[0].children)
    end

    @testset "add_default_rules!" begin
        system = PaunMembraneSystem(1, 10)

        # Initially no rules
        @test isempty(system.membranes[0].rules)

        add_default_rules!(system)

        # Should have 3 rules per membrane
        for (id, membrane) in system.membranes
            @test length(membrane.rules) == 3
        end

        # Test that rules are callable
        root = system.membranes[0]
        test_state = randn(10)

        for rule in root.rules
            result = rule(test_state)
            @test length(result) == length(test_state)
            @test all(isfinite.(result))
        end
    end

    @testset "Rule 1: Tanh Activation" begin
        system = PaunMembraneSystem(0, 5)
        add_default_rules!(system)

        rule_tanh = system.membranes[0].rules[1]

        # Test tanh bounds output
        test_input = [10.0, -10.0, 0.0, 1.0, -1.0]
        result = rule_tanh(test_input)

        @test all(abs.(result) .<= 1.0)
        @test isapprox(result[3], 0.0, atol=1e-10)
    end

    @testset "Rule 2: Soft Normalization" begin
        system = PaunMembraneSystem(0, 3)
        add_default_rules!(system)

        rule_norm = system.membranes[0].rules[2]

        # Test normalization
        test_input = [3.0, 4.0, 0.0]  # norm = 5
        result = rule_norm(test_input)

        expected = test_input ./ (1.0 + 5.0)
        @test all(isapprox.(result, expected, atol=1e-10))
    end

    @testset "Rule 3: Sparse Coding" begin
        system = PaunMembraneSystem(0, 4)
        add_default_rules!(system)

        rule_sparse = system.membranes[0].rules[3]

        # Values below threshold should be zeroed
        test_input = [0.5, 0.2, -0.4, 0.1]
        result = rule_sparse(test_input)

        threshold = 0.3
        @test result[1] != 0.0  # 0.5 > 0.3
        @test result[2] == 0.0  # 0.2 < 0.3
        @test result[3] != 0.0  # |-0.4| > 0.3
        @test result[4] == 0.0  # 0.1 < 0.3
    end

    @testset "apply_membrane_rules!" begin
        system = PaunMembraneSystem(2, 5)
        add_default_rules!(system)

        # Set known initial states
        for (id, membrane) in system.membranes
            membrane.state .= randn(5)
        end

        initial_root_state = copy(system.membranes[0].state)

        # Apply rules
        apply_membrane_rules!(system, 0.1)

        # States should have changed
        @test !all(system.membranes[0].state .== initial_root_state)

        # States should be finite
        for (id, membrane) in system.membranes
            @test all(isfinite.(membrane.state))
        end
    end

    @testset "modulate_permeability! Openness" begin
        system = PaunMembraneSystem(2, 10)

        # Trigger openness emotions (wonder, curiosity, joy)
        emotion_state = Dict{Symbol,Float64}(
            :wonder => 0.8,
            :curiosity => 0.6,
            :joy => 0.4
        )

        modulate_permeability!(system, emotion_state)

        # All permeabilities should be increased
        for (id, membrane) in system.membranes
            @test membrane.permeability > 0.1
            @test membrane.permeability < 0.9
        end

        # Root (depth 0) should have higher permeability than leaves
        root_perm = system.membranes[0].permeability
        leaf_perms = [m.permeability for (id, m) in system.membranes if m.depth == system.depth]

        # Deeper membranes have lower base permeability
        @test all(root_perm >= p for p in leaf_perms)
    end

    @testset "modulate_permeability! Closure" begin
        system = PaunMembraneSystem(1, 8)

        # Trigger closure emotions (fear, anxiety, sadness)
        emotion_state = Dict{Symbol,Float64}(
            :fear => 0.9,
            :anxiety => 0.7,
            :sadness => 0.5
        )

        modulate_permeability!(system, emotion_state)

        # Permeabilities should be decreased
        for (id, membrane) in system.membranes
            @test membrane.permeability >= 0.1
            @test membrane.permeability <= 0.9
        end
    end

    @testset "modulate_permeability! Neutral" begin
        system = PaunMembraneSystem(1, 6)

        # No emotions
        emotion_state = Dict{Symbol,Float64}()

        modulate_permeability!(system, emotion_state)

        # Should be at base permeability
        root = system.membranes[0]
        expected_base = 0.7 - 0.1 * 0  # depth 0
        @test isapprox(root.permeability, expected_base, atol=0.01)
    end

    @testset "modulate_permeability! Mixed Emotions" begin
        system = PaunMembraneSystem(2, 5)

        # Balanced emotions
        emotion_state = Dict{Symbol,Float64}(
            :wonder => 0.5,
            :fear => 0.5
        )

        modulate_permeability!(system, emotion_state)

        # Net effect should be near neutral
        root = system.membranes[0]
        base = 0.7
        @test isapprox(root.permeability, base, atol=0.1)
    end

    @testset "Membrane State Dimension Consistency" begin
        dim = 15
        system = PaunMembraneSystem(3, dim, 2)

        for (id, membrane) in system.membranes
            @test length(membrane.state) == dim
        end
    end

    @testset "Multiple Rules Application" begin
        system = PaunMembraneSystem(1, 10)
        add_default_rules!(system)

        # Run multiple timesteps
        for _ in 1:10
            apply_membrane_rules!(system, 0.05)

            # States should remain bounded (due to tanh)
            for (id, membrane) in system.membranes
                @test all(isfinite.(membrane.state))
            end
        end
    end

    @testset "Child-Parent Communication" begin
        system = PaunMembraneSystem(1, 5)
        add_default_rules!(system)

        # Set child state
        root = system.membranes[0]
        child_id = root.children[1]
        child = system.membranes[child_id]

        child.state .= ones(5) * 2.0
        child.permeability = 0.9

        initial_root = copy(root.state)

        apply_membrane_rules!(system, 0.5)

        # Root should have received information from child
        # (state changed due to communication)
        @test !all(root.state .== initial_root)
    end

end
