"""
Comprehensive unit tests for J-Surface Elementary Differentials
"""

@testset "J-Surface" begin

    @testset "ElementaryDifferential Construction" begin
        diff = ElementaryDifferential(2, 5, 0.5, 3)

        @test diff.order == 2
        @test diff.tree_id == 5
        @test diff.coefficient == 0.5
        @test diff.dimension == 3
    end

    @testset "JSurfaceDifferential Construction" begin
        surface = JSurfaceDifferential(5, 3)

        @test surface.dimension == 5
        @test !isempty(surface.differentials)
        @test surface.curvature == 0.0  # initial curvature
        @test isempty(surface.critical_points)

        # Metric should be identity initially
        @test size(surface.metric) == (5, 5)
        @test surface.metric ≈ Matrix{Float64}(I, 5, 5)
    end

    @testset "JSurfaceDifferential Differential Count" begin
        dim = 4
        max_order = 3
        surface = JSurfaceDifferential(dim, max_order)

        # Should have dim × max_order differentials
        expected_count = dim * max_order
        @test length(surface.differentials) == expected_count
    end

    @testset "JSurfaceDifferential Coefficient Decay" begin
        surface = JSurfaceDifferential(3, 4)

        # Coefficients should decay with order (1/order²)
        order_1_diffs = [d for d in surface.differentials if d.order == 1]
        order_4_diffs = [d for d in surface.differentials if d.order == 4]

        @test !isempty(order_1_diffs)
        @test !isempty(order_4_diffs)

        @test order_1_diffs[1].coefficient > order_4_diffs[1].coefficient
    end

    @testset "compute_metric!" begin
        surface = JSurfaceDifferential(3, 2)

        # Generate some states
        states = [randn(3) for _ in 1:50]

        compute_metric!(surface, states)

        # Metric should be positive definite
        @test all(eigvals(surface.metric) .> 0)

        # Curvature should be computed
        @test isfinite(surface.curvature)
    end

    @testset "compute_metric! Dimension Consistency" begin
        dim = 5
        surface = JSurfaceDifferential(dim, 2)
        states = [randn(dim) for _ in 1:30]

        compute_metric!(surface, states)

        @test size(surface.metric) == (dim, dim)
    end

    @testset "geodesic_distance Basic" begin
        surface = JSurfaceDifferential(3, 2)

        x = zeros(3)
        y = ones(3)

        dist = geodesic_distance(surface, x, y)

        # With identity metric, should be Euclidean distance
        expected = sqrt(3)
        @test isapprox(dist, expected, atol=1e-10)
    end

    @testset "geodesic_distance Same Point" begin
        surface = JSurfaceDifferential(4, 2)

        x = randn(4)
        dist = geodesic_distance(surface, x, x)

        @test isapprox(dist, 0.0, atol=1e-10)
    end

    @testset "geodesic_distance Symmetry" begin
        surface = JSurfaceDifferential(3, 2)

        x = randn(3)
        y = randn(3)

        dist_xy = geodesic_distance(surface, x, y)
        dist_yx = geodesic_distance(surface, y, x)

        @test isapprox(dist_xy, dist_yx, atol=1e-10)
    end

    @testset "geodesic_distance Triangle Inequality" begin
        surface = JSurfaceDifferential(4, 2)

        x = randn(4)
        y = randn(4)
        z = randn(4)

        d_xy = geodesic_distance(surface, x, y)
        d_yz = geodesic_distance(surface, y, z)
        d_xz = geodesic_distance(surface, x, z)

        @test d_xz <= d_xy + d_yz + 1e-10  # small tolerance
    end

    @testset "geodesic_distance With Non-Identity Metric" begin
        surface = JSurfaceDifferential(3, 2)

        # Set custom metric
        surface.metric = [2.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 0.5]

        x = zeros(3)
        y = ones(3)

        dist = geodesic_distance(surface, x, y)

        # √(2×1 + 1×1 + 0.5×1) = √3.5
        expected = sqrt(3.5)
        @test isapprox(dist, expected, atol=1e-10)
    end

    @testset "project_to_surface" begin
        surface = JSurfaceDifferential(3, 2)

        x = randn(3) * 5  # Large vector
        projected = project_to_surface(surface, x)

        # With identity metric, norm should be 1
        @test isapprox(norm(projected), 1.0, atol=1e-10)
    end

    @testset "project_to_surface Zero Vector" begin
        surface = JSurfaceDifferential(3, 2)

        x = zeros(3)
        projected = project_to_surface(surface, x)

        # Zero vector should remain zero
        @test all(projected .== 0.0)
    end

    @testset "project_to_surface Direction Preservation" begin
        surface = JSurfaceDifferential(4, 2)

        x = [1.0, 2.0, 3.0, 4.0]
        projected = project_to_surface(surface, x)

        # Direction should be preserved
        x_normalized = x / norm(x)
        @test all(isapprox.(projected, x_normalized, atol=1e-10))
    end

    @testset "find_critical_points!" begin
        surface = JSurfaceDifferential(3, 2)

        # Simple vector field with known critical point at origin
        f(x) = -x

        find_critical_points!(surface, f, n_samples=50)

        # Should find at least one critical point near origin
        @test !isempty(surface.critical_points)

        # At least one should be near origin
        distances_to_origin = [norm(cp) for cp in surface.critical_points]
        @test minimum(distances_to_origin) < 0.5
    end

    @testset "find_critical_points! Multiple Attractors" begin
        surface = JSurfaceDifferential(2, 2)

        # Vector field with two attractors at (1,0) and (-1,0)
        function f(x)
            dx = similar(x)
            dx[1] = -x[1] * (x[1]^2 - 1)
            dx[2] = -x[2]
            return dx
        end

        find_critical_points!(surface, f, n_samples=100)

        # May find multiple critical points
        @test length(surface.critical_points) >= 1
    end

    @testset "parallel_transport" begin
        surface = JSurfaceDifferential(3, 2)

        vector = randn(3)
        from_point = randn(3)
        to_point = randn(3)

        transported = parallel_transport(surface, vector, from_point, to_point)

        @test length(transported) == 3
        @test all(isfinite.(transported))
    end

    @testset "parallel_transport Identity Metric" begin
        surface = JSurfaceDifferential(4, 2)

        vector = randn(4)
        from_point = randn(4)
        to_point = randn(4)

        transported = parallel_transport(surface, vector, from_point, to_point)

        # With identity metric and simplified implementation,
        # transported should be related to original vector
        @test all(isfinite.(transported))
    end

    @testset "compute_sectional_curvature" begin
        surface = JSurfaceDifferential(3, 2)

        x = randn(3)
        v1 = randn(3)
        v2 = randn(3)

        curvature = compute_sectional_curvature(surface, x, v1, v2)

        @test isfinite(curvature)
    end

    @testset "compute_sectional_curvature Parallel Vectors" begin
        surface = JSurfaceDifferential(3, 2)

        x = randn(3)
        v1 = randn(3)
        v2 = 2.0 * v1  # Parallel to v1

        curvature = compute_sectional_curvature(surface, x, v1, v2)

        # Parallel vectors span zero area, curvature should be 0 or handle gracefully
        @test isfinite(curvature)
    end

    @testset "optimize_trajectory" begin
        surface = JSurfaceDifferential(3, 2)

        start = randn(3)
        goal = randn(3)
        f(x) = -x  # Simple attractor at origin

        trajectory = optimize_trajectory(surface, start, goal, f, n_steps=30)

        @test length(trajectory) >= 2
        @test all(length(s) == 3 for s in trajectory)

        # First point should be start
        @test all(isapprox.(trajectory[1], start, atol=1e-10))
    end

    @testset "optimize_trajectory Convergence" begin
        surface = JSurfaceDifferential(2, 2)

        start = [0.0, 0.0]
        goal = [1.0, 1.0]
        f(x) = zeros(2)  # No dynamics

        trajectory = optimize_trajectory(surface, start, goal, f, n_steps=50)

        # Last point should be close to goal
        final_dist = geodesic_distance(surface, trajectory[end], goal)
        @test final_dist < 0.5
    end

    @testset "optimize_trajectory All Points Finite" begin
        surface = JSurfaceDifferential(4, 2)

        start = randn(4)
        goal = randn(4)
        f = create_cognitive_dynamics(4)

        trajectory = optimize_trajectory(surface, start, goal, f, n_steps=20)

        for point in trajectory
            @test all(isfinite.(point))
        end
    end

    @testset "Metric Update After State Collection" begin
        surface = JSurfaceDifferential(4, 3)

        # Generate trajectory
        trajectory = [randn(4) for _ in 1:40]

        compute_metric!(surface, trajectory)

        # Metric should reflect correlations in data
        @test !all(surface.metric .== I(4))
        @test all(isfinite.(surface.metric))
    end

    @testset "Curvature Bounds" begin
        surface = JSurfaceDifferential(5, 2)

        states = [randn(5) for _ in 1:100]
        compute_metric!(surface, states)

        # Curvature should be positive (trace of positive definite metric)
        @test surface.curvature > 0
        @test isfinite(surface.curvature)
    end

    @testset "Differential Tree IDs" begin
        surface = JSurfaceDifferential(3, 4)

        ids = [d.tree_id for d in surface.differentials]
        @test length(unique(ids)) == length(ids)
    end

    @testset "Large Dimension Surface" begin
        dim = 20
        surface = JSurfaceDifferential(dim, 2)

        @test size(surface.metric) == (dim, dim)
        @test length(surface.differentials) == dim * 2
    end

end
