"""
Comprehensive unit tests for Butcher B-Series and Rooted Forest Ridges
"""

@testset "Butcher B-Series" begin

    @testset "RootedTree Construction" begin
        tree = RootedTree(1, 3)

        @test tree.id == 1
        @test tree.order == 3
        @test tree.density == 1.0  # default
        @test tree.symmetry == 1
        @test isempty(tree.children)
    end

    @testset "generate_rooted_trees Order 1" begin
        trees = generate_rooted_trees(1)

        @test length(trees) == 1
        @test trees[1].order == 1
    end

    @testset "generate_rooted_trees Order 2" begin
        trees = generate_rooted_trees(2)

        # Order 2: one tree with one child of order 1
        @test length(trees) >= 1

        for tree in trees
            @test tree.order > 0
            @test tree.order <= 2
        end
    end

    @testset "generate_rooted_trees Order 3" begin
        trees = generate_rooted_trees(3)

        @test length(trees) >= 1

        # All trees should have valid orders
        for tree in trees
            @test tree.order <= 3
        end
    end

    @testset "generate_rooted_trees Order 4" begin
        trees = generate_rooted_trees(4)

        @test length(trees) >= 1

        for tree in trees
            @test tree.order <= 4
            @test tree.id >= 0
        end
    end

    @testset "compute_density Order 1" begin
        tree = RootedTree(0, 1)
        density = compute_density(tree)

        @test density == 1.0
    end

    @testset "compute_density Higher Order" begin
        # Create tree of order 2 with child of order 1
        parent = RootedTree(0, 2)
        child = RootedTree(1, 1)
        push!(parent.children, child)

        density = compute_density(parent)

        # γ(t) = 1/ρ(t) × ∏γ(tᵢ) = 1/2 × 1 = 0.5
        @test isapprox(density, 0.5, atol=1e-10)
    end

    @testset "compute_density Nested Trees" begin
        # Create tree of order 3: root with child that has grandchild
        grandchild = RootedTree(2, 1)
        child = RootedTree(1, 2)
        push!(child.children, grandchild)
        root = RootedTree(0, 3)
        push!(root.children, child)

        density = compute_density(root)

        # Should be positive and less than 1
        @test density > 0
        @test density <= 1.0
    end

    @testset "get_method_coefficients RK4" begin
        coeffs = get_method_coefficients(:rk4, 4)

        @test coeffs[1] == 1.0
        @test coeffs[2] == 0.5
        @test coeffs[3] == 0.5
        @test coeffs[4] == 1.0
    end

    @testset "get_method_coefficients Forward Euler" begin
        coeffs = get_method_coefficients(:forward_euler, 2)

        @test coeffs[1] == 1.0
        @test !haskey(coeffs, 2) || coeffs[2] != coeffs[1]
    end

    @testset "get_method_coefficients Heun" begin
        coeffs = get_method_coefficients(:heun, 2)

        @test coeffs[1] == 0.5
        @test coeffs[2] == 0.5
    end

    @testset "get_method_coefficients Default" begin
        coeffs = get_method_coefficients(:unknown_method, 4)

        # Default is 1/i
        @test coeffs[1] == 1.0
        @test coeffs[2] == 0.5
        @test isapprox(coeffs[3], 1.0/3.0, atol=1e-10)
        @test coeffs[4] == 0.25
    end

    @testset "ButcherBSeriesForest Construction" begin
        forest = ButcherBSeriesForest(3, method=:rk4)

        @test forest.max_order == 3
        @test !isempty(forest.trees)
        @test !isempty(forest.coefficients)

        # All trees should have densities computed
        for tree in forest.trees
            @test tree.density > 0
        end
    end

    @testset "ButcherBSeriesForest Different Orders" begin
        forest1 = ButcherBSeriesForest(2)
        forest2 = ButcherBSeriesForest(4)

        @test length(forest2.trees) >= length(forest1.trees)
        @test forest2.max_order > forest1.max_order
    end

    @testset "ButcherBSeriesForest Graph Structure" begin
        forest = ButcherBSeriesForest(3)

        @test nv(forest.graph) == length(forest.trees)
    end

    @testset "evaluate_tree Order 1" begin
        tree = RootedTree(0, 1)
        state = randn(5)

        f(x) = tanh.(x)
        result = evaluate_tree(tree, f, state)

        @test all(isapprox.(result, tanh.(state), atol=1e-10))
    end

    @testset "evaluate_tree Higher Order" begin
        # Create tree with child
        child = RootedTree(1, 1)
        parent = RootedTree(0, 2)
        push!(parent.children, child)

        state = randn(4)
        f(x) = sin.(x)

        result = evaluate_tree(parent, f, state)

        @test length(result) == length(state)
        @test all(isfinite.(result))
    end

    @testset "create_cognitive_dynamics" begin
        dim = 5
        f = create_cognitive_dynamics(dim)

        state = randn(dim)
        dx = f(state)

        @test length(dx) == dim
        @test all(isfinite.(dx))
    end

    @testset "create_cognitive_dynamics Parameters" begin
        dim = 4
        f = create_cognitive_dynamics(dim,
            attraction_strength=2.0,
            coupling=0.8)

        state = randn(dim)
        dx = f(state)

        @test length(dx) == dim
        @test all(isfinite.(dx))
    end

    @testset "create_cognitive_dynamics Attractor Behavior" begin
        dim = 3
        f = create_cognitive_dynamics(dim, attraction_strength=1.0, coupling=0.0)

        # States at ±1 should be stable points
        state_plus = ones(dim)
        dx_plus = f(state_plus)

        # At x=1, dx = -1*(1-1) = 0
        @test all(abs.(dx_plus) .< 0.1)
    end

    @testset "create_cognitive_dynamics Coupling" begin
        dim = 5
        f = create_cognitive_dynamics(dim, coupling=1.0)

        # Non-uniform state should have coupling effects
        state = [1.0, -1.0, 0.5, -0.5, 0.0]
        dx = f(state)

        @test all(isfinite.(dx))
    end

    @testset "apply_bseries_step" begin
        forest = ButcherBSeriesForest(2, method=:rk4)
        f = create_cognitive_dynamics(5)
        state = randn(5)
        h = 0.1

        new_state = apply_bseries_step(forest, f, state, h)

        @test length(new_state) == length(state)
        @test all(isfinite.(new_state))
        @test !all(new_state .== state)  # Should have changed
    end

    @testset "apply_bseries_step Step Size Effect" begin
        forest = ButcherBSeriesForest(3, method=:rk4)
        f = create_cognitive_dynamics(4)
        state = randn(4)

        small_step = apply_bseries_step(forest, f, state, 0.01)
        large_step = apply_bseries_step(forest, f, state, 0.5)

        # Larger step should produce bigger change
        diff_small = norm(small_step - state)
        diff_large = norm(large_step - state)

        @test diff_large > diff_small
    end

    @testset "apply_bseries_step Preserves Dimension" begin
        for dim in [3, 5, 10]
            forest = ButcherBSeriesForest(2)
            f = create_cognitive_dynamics(dim)
            state = randn(dim)

            new_state = apply_bseries_step(forest, f, state, 0.1)
            @test length(new_state) == dim
        end
    end

    @testset "Multiple Integration Steps" begin
        forest = ButcherBSeriesForest(3, method=:rk4)
        f = create_cognitive_dynamics(5)
        state = randn(5)

        trajectory = [copy(state)]
        for _ in 1:20
            state = apply_bseries_step(forest, f, state, 0.1)
            push!(trajectory, copy(state))
        end

        # All states should be finite
        for s in trajectory
            @test all(isfinite.(s))
        end
    end

    @testset "Different Integration Methods" begin
        f = create_cognitive_dynamics(4)
        state = randn(4)
        h = 0.1

        forest_rk4 = ButcherBSeriesForest(3, method=:rk4)
        forest_euler = ButcherBSeriesForest(3, method=:forward_euler)
        forest_heun = ButcherBSeriesForest(3, method=:heun)

        result_rk4 = apply_bseries_step(forest_rk4, f, state, h)
        result_euler = apply_bseries_step(forest_euler, f, state, h)
        result_heun = apply_bseries_step(forest_heun, f, state, h)

        # Different methods should give different results
        @test !all(result_rk4 .== result_euler)
        @test !all(result_heun .== result_euler)
    end

    @testset "Tree ID Uniqueness in Forest" begin
        forest = ButcherBSeriesForest(4)

        ids = [tree.id for tree in forest.trees]
        @test length(unique(ids)) == length(ids)
    end

    @testset "Forest Order Bounds" begin
        max_order = 4
        forest = ButcherBSeriesForest(max_order)

        for tree in forest.trees
            @test tree.order >= 1
            @test tree.order <= max_order
        end
    end

end
