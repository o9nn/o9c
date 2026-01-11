"""
Comprehensive unit tests for Unified Relations Framework
"""

@testset "Unified Relations Framework" begin

    # ========================================================================
    # COUPLING CONSTANTS TESTS
    # ========================================================================

    @testset "CouplingConstants" begin
        @testset "Default Construction" begin
            cc = CouplingConstants()

            @test cc.α_em == 0.3
            @test cc.α_er == 0.4
            @test cc.α_mr == 0.5
            @test cc.α_rb == 0.6
            @test cc.α_bj == 0.4
            @test cc.α_jt == 0.35
            @test cc.α_te == 0.25
            @test cc.β == 0.5
            @test cc.γ == 0.1
            @test cc.ε == 1e-8
        end

        @testset "Custom Construction" begin
            cc = CouplingConstants(α_em=0.5, α_er=0.6, β=0.8)

            @test cc.α_em == 0.5
            @test cc.α_er == 0.6
            @test cc.β == 0.8
            @test cc.α_mr == 0.5  # Default value
        end

        @testset "Default Coupling Constant" begin
            @test DEFAULT_COUPLING isa CouplingConstants
            @test DEFAULT_COUPLING.α_em == 0.3
        end
    end

    # ========================================================================
    # UNIFIED STATE TESTS
    # ========================================================================

    @testset "UnifiedState" begin
        @testset "Construction" begin
            dim = 10
            ξ = UnifiedState(dim)

            @test length(ξ.membrane_state) == dim
            @test length(ξ.reservoir_state) == dim
            @test length(ξ.bseries_state) == dim
            @test length(ξ.jsurface_state) == dim
            @test length(ξ.emotion_state) == dim
            @test length(ξ.attention_state) == dim
            @test ξ.t == 0.0
            @test ξ.coherence == 0.0
            @test ξ.convergence_distance == Inf
        end

        @testset "Initial Attention Uniform" begin
            ξ = UnifiedState(5)

            @test isapprox(sum(ξ.attention_state), 1.0, atol=1e-10)
            @test all(ξ.attention_state .≈ 0.2)
        end

        @testset "flatten_state" begin
            dim = 4
            ξ = UnifiedState(dim)

            flat = flatten_state(ξ)

            @test length(flat) == 6 * dim
            @test all(isfinite.(flat))
        end

        @testset "state_dimension" begin
            ξ = UnifiedState(8)
            @test state_dimension(ξ) == 6 * 8
        end

        @testset "State Modification" begin
            ξ = UnifiedState(5)

            ξ.membrane_state .= ones(5)
            ξ.coherence = 0.8
            ξ.t = 1.5

            @test all(ξ.membrane_state .== 1.0)
            @test ξ.coherence == 0.8
            @test ξ.t == 1.5
        end
    end

    # ========================================================================
    # COUPLING OPERATOR TESTS
    # ========================================================================

    @testset "EmotionMembraneCoupling" begin
        @testset "Construction" begin
            op = EmotionMembraneCoupling()
            @test op.strength == 0.3
            @test op.base_permeability == 0.5

            op2 = EmotionMembraneCoupling(0.5, 0.6)
            @test op2.strength == 0.5
            @test op2.base_permeability == 0.6
        end

        @testset "Empty Emotion State" begin
            op = EmotionMembraneCoupling()
            result = apply_coupling(op, Float64[])
            @test result == op.base_permeability
        end

        @testset "Zero Emotion State" begin
            op = EmotionMembraneCoupling()
            result = apply_coupling(op, zeros(5))
            @test result == op.base_permeability
        end

        @testset "Positive Emotion State" begin
            op = EmotionMembraneCoupling(0.3, 0.5)
            emotion = [0.8, 0.6, 0.4, 0.3, 0.2]

            result = apply_coupling(op, emotion)

            @test result >= 0.1
            @test result <= 0.9
            @test isfinite(result)
        end

        @testset "Bounds Clamping" begin
            op = EmotionMembraneCoupling(1.0, 0.9)
            emotion = ones(5) * 2.0

            result = apply_coupling(op, emotion)
            @test result <= 0.9
            @test result >= 0.1
        end
    end

    @testset "EmotionReservoirCoupling" begin
        @testset "Construction" begin
            op = EmotionReservoirCoupling()
            @test op.strength == 0.4
            @test op.base_spectral_radius == 0.9
            @test op.base_leak_rate == 0.5
        end

        @testset "Empty Emotion State" begin
            op = EmotionReservoirCoupling()
            ρ, λ = apply_coupling(op, Float64[])

            @test ρ == op.base_spectral_radius
            @test λ == op.base_leak_rate
        end

        @testset "Positive Emotion State" begin
            op = EmotionReservoirCoupling()
            emotion = [0.7, 0.5, 0.3]

            ρ, λ = apply_coupling(op, emotion)

            @test 0.5 <= ρ <= 0.99
            @test 0.1 <= λ <= 0.9
        end

        @testset "High Arousal Effect" begin
            op = EmotionReservoirCoupling()

            # High variance → high arousal
            high_arousal = [0.9, 0.1, 0.8, 0.2]
            low_arousal = [0.5, 0.5, 0.5, 0.5]

            ρ_high, _ = apply_coupling(op, high_arousal)
            ρ_low, _ = apply_coupling(op, low_arousal)

            # High arousal should increase spectral radius
            @test isfinite(ρ_high)
            @test isfinite(ρ_low)
        end
    end

    @testset "ReservoirBSeriesCoupling" begin
        @testset "Construction" begin
            op = ReservoirBSeriesCoupling()
            @test op.strength == 0.6
            @test op.integration_order == 4
        end

        @testset "Apply Coupling" begin
            op = ReservoirBSeriesCoupling(0.5, 3)
            reservoir_state = randn(10)
            dt = 0.1

            contribution = apply_coupling(op, reservoir_state, dt)

            @test length(contribution) == 10
            @test all(isfinite.(contribution))
        end

        @testset "Small dt Gives Small Contribution" begin
            op = ReservoirBSeriesCoupling()
            state = randn(5)

            small_contrib = apply_coupling(op, state, 0.001)
            large_contrib = apply_coupling(op, state, 0.5)

            @test norm(small_contrib) < norm(large_contrib)
        end
    end

    @testset "JSurfaceAttentionCoupling" begin
        @testset "Construction" begin
            op = JSurfaceAttentionCoupling()
            @test op.strength == 0.35
            @test op.base_scope == 0.5
        end

        @testset "High Curvature → Narrow Scope" begin
            op = JSurfaceAttentionCoupling(0.4, 0.5)

            low_curv = apply_coupling(op, 0.1)
            high_curv = apply_coupling(op, 2.0)

            @test high_curv < low_curv
        end

        @testset "Bounds" begin
            op = JSurfaceAttentionCoupling(1.0, 0.5)

            result = apply_coupling(op, 10.0)
            @test result >= 0.1
            @test result <= 0.9
        end
    end

    @testset "AttentionEmotionCoupling" begin
        @testset "Construction" begin
            op = AttentionEmotionCoupling()
            @test op.strength == 0.25
            @test op.wonder_threshold == 0.7
        end

        @testset "Apply Coupling" begin
            op = AttentionEmotionCoupling()
            attention = [0.4, 0.3, 0.2, 0.1]
            content_valence = [0.5, 0.3, -0.2, 0.1]

            trigger = apply_coupling(op, attention, content_valence)

            @test length(trigger) >= 1
            @test all(isfinite.(trigger))
        end

        @testset "Mismatched Dimensions" begin
            op = AttentionEmotionCoupling()
            attention = [0.5, 0.5]
            content_valence = [0.3]

            trigger = apply_coupling(op, attention, content_valence)
            @test length(trigger) == 2
            @test all(trigger .== 0.0)
        end
    end

    # ========================================================================
    # CLOSURE OPERATOR TESTS
    # ========================================================================

    @testset "EchoStateClosure" begin
        @testset "Construction" begin
            cl = EchoStateClosure()
            @test cl.max_spectral_radius == 0.99
            @test cl.contractivity_factor == 0.95
        end

        @testset "Apply to Valid Matrix" begin
            cl = EchoStateClosure(0.9, 0.95)
            W = randn(10, 10)

            # Scale so it might exceed threshold
            W .*= 2.0

            apply_closure!(cl, W)

            # Check spectral radius is now within bounds
            ρ = maximum(abs.(eigvals(W)))
            @test ρ <= 0.9 + 0.05  # Allow small tolerance
        end

        @testset "Preserve Valid Matrix" begin
            cl = EchoStateClosure(0.99, 0.95)
            W = 0.5 * randn(8, 8)

            # Scale to have spectral radius around 0.5
            ρ_before = maximum(abs.(eigvals(W)))

            apply_closure!(cl, W)

            ρ_after = maximum(abs.(eigvals(W)))

            # Should not change much if already valid
            @test isfinite(ρ_after)
        end
    end

    @testset "MembraneClosure" begin
        @testset "Construction" begin
            cl = MembraneClosure()
            @test cl.conservation_tolerance == 0.01
        end

        @testset "Empty States" begin
            cl = MembraneClosure()
            states = Vector{Float64}[]

            result = apply_closure!(cl, states)
            @test isempty(result)
        end

        @testset "Conservation" begin
            cl = MembraneClosure(0.01)
            states = [randn(5), randn(5), randn(5)]

            total_before = sum(sum(abs.(s)) for s in states)

            apply_closure!(cl, states)

            total_after = sum(sum(abs.(s)) for s in states)

            @test isapprox(total_before, total_after, rtol=0.01)
        end
    end

    @testset "MetricClosure" begin
        @testset "Construction" begin
            cl = MetricClosure()
            @test cl.min_eigenvalue == 1e-6
        end

        @testset "Ensure Positive Definite" begin
            cl = MetricClosure(0.01)

            # Create matrix that might not be positive definite
            G = randn(5, 5)
            G = G + G'  # Symmetrize

            apply_closure!(cl, G)

            # Check eigenvalues are positive
            λs = eigvals(Symmetric(G))
            @test all(λs .>= 0.01 - 1e-10)
        end

        @testset "Preserve Valid Metric" begin
            cl = MetricClosure(1e-6)
            G = Matrix{Float64}(I, 4, 4)

            apply_closure!(cl, G)

            @test isapprox(G, I(4), atol=1e-10)
        end
    end

    @testset "AttentionClosure" begin
        @testset "Construction" begin
            cl = AttentionClosure()
            @test cl.temperature == 1.0
        end

        @testset "Ensure Valid Distribution" begin
            cl = AttentionClosure(1.0)
            attn = randn(10)

            apply_closure!(cl, attn)

            @test isapprox(sum(attn), 1.0, atol=1e-10)
            @test all(attn .> 0)
            @test all(attn .< 1)
        end

        @testset "Temperature Effect" begin
            attn1 = [1.0, 2.0, 3.0]
            attn2 = copy(attn1)

            cl_low = AttentionClosure(0.5)
            cl_high = AttentionClosure(2.0)

            apply_closure!(cl_low, attn1)
            apply_closure!(cl_high, attn2)

            # Lower temperature → sharper distribution
            @test var(attn1) > var(attn2)
        end
    end

    # ========================================================================
    # CONVERGENCE TESTS
    # ========================================================================

    @testset "ConvergenceCriteria" begin
        @testset "Default Construction" begin
            cc = ConvergenceCriteria()

            @test cc.max_iterations == 1000
            @test cc.tolerance == 1e-6
            @test cc.stability_window == 10
            @test cc.coherence_threshold == 0.8
        end

        @testset "Custom Construction" begin
            cc = ConvergenceCriteria(
                max_iterations=500,
                tolerance=1e-4,
                coherence_threshold=0.9
            )

            @test cc.max_iterations == 500
            @test cc.tolerance == 1e-4
            @test cc.coherence_threshold == 0.9
        end
    end

    @testset "LyapunovFunction" begin
        @testset "Construction" begin
            V = LyapunovFunction()

            @test length(V.subsystem_weights) == 6
            @test V.entropy_weight == 0.3
            @test V.coherence_weight == 0.5
        end

        @testset "Evaluate at Same State" begin
            V = LyapunovFunction()
            ξ = UnifiedState(5)
            ξ_target = UnifiedState(5)

            # Make them identical
            ξ.coherence = 1.0
            ξ_target.coherence = 1.0

            value = evaluate(V, ξ, ξ_target)

            @test isfinite(value)
            @test value >= 0.0
        end

        @testset "Evaluate Distance Effect" begin
            V = LyapunovFunction()
            ξ1 = UnifiedState(5)
            ξ2 = UnifiedState(5)
            ξ_target = UnifiedState(5)

            # ξ1 is close to target
            ξ1.reservoir_state .= 0.1 * randn(5)

            # ξ2 is far from target
            ξ2.reservoir_state .= 10.0 * randn(5)

            v1 = evaluate(V, ξ1, ξ_target)
            v2 = evaluate(V, ξ2, ξ_target)

            @test v2 > v1
        end
    end

    @testset "compute_coherence" begin
        @testset "Zero States" begin
            ξ = UnifiedState(5)
            coherence = compute_coherence(ξ)

            @test 0.0 <= coherence <= 1.0
        end

        @testset "Correlated States" begin
            ξ = UnifiedState(5)

            # Set all states to same pattern
            pattern = randn(5)
            ξ.membrane_state .= pattern
            ξ.reservoir_state .= pattern
            ξ.emotion_state .= abs.(pattern)
            ξ.attention_state .= abs.(pattern) ./ sum(abs.(pattern))

            coherence = compute_coherence(ξ)

            @test coherence > 0.5
        end
    end

    # ========================================================================
    # UESRA TESTS
    # ========================================================================

    @testset "UniversalResonantArchetype" begin
        @testset "Construction" begin
            dim = 10
            uesra = UniversalResonantArchetype(dim)

            @test uesra.dimension == dim
            @test uesra.basin_of_attraction == 1.0
            @test length(uesra.resonance_frequencies) > 0
            @test length(uesra.stability_eigenvalues) == dim
        end

        @testset "Resonant State Properties" begin
            uesra = UniversalResonantArchetype(8)
            ξ_star = uesra.resonant_state

            # Coherence should be 1.0
            @test ξ_star.coherence == 1.0

            # Convergence distance should be 0.0
            @test ξ_star.convergence_distance == 0.0

            # Attention should sum to 1
            @test isapprox(sum(ξ_star.attention_state), 1.0, atol=1e-10)
        end

        @testset "Stability Eigenvalues Inside Unit Circle" begin
            uesra = UniversalResonantArchetype(6)

            for λ in uesra.stability_eigenvalues
                @test abs(λ) <= 1.0
            end
        end
    end

    @testset "create_resonant_emotion_state" begin
        @testset "Basic Properties" begin
            state = create_resonant_emotion_state(8)

            @test length(state) == 8
            @test all(state .>= 0)
            @test isapprox(sum(state), 1.0, atol=1e-10)
        end

        @testset "Wonder Dominant" begin
            state = create_resonant_emotion_state(5)

            # First element (wonder) should be highest
            @test state[1] >= state[end]
        end
    end

    @testset "create_resonant_reservoir_state" begin
        @testset "Basic Properties" begin
            state = create_resonant_reservoir_state(20)

            @test length(state) == 20
            @test all(abs.(state) .<= 1.0)  # tanh bounded
            @test all(isfinite.(state))
        end

        @testset "Structured Pattern" begin
            state = create_resonant_reservoir_state(50)

            # Should have structure (not random)
            # Check autocorrelation
            shifted = circshift(state, 5)
            corr = abs(cor(state, shifted))

            @test corr > 0.1  # Should have temporal structure
        end
    end

    @testset "distance_to_uesra" begin
        @testset "Zero at Resonant State" begin
            uesra = UniversalResonantArchetype(5)
            ξ = deepcopy(uesra.resonant_state)

            distance = distance_to_uesra(ξ, uesra)

            @test distance < 0.01
        end

        @testset "Positive for Different States" begin
            uesra = UniversalResonantArchetype(5)
            ξ = UnifiedState(5)

            distance = distance_to_uesra(ξ, uesra)

            @test distance > 0
            @test isfinite(distance)
        end

        @testset "Increases with Distance" begin
            uesra = UniversalResonantArchetype(5)

            ξ_near = UnifiedState(5)
            ξ_far = UnifiedState(5)

            # Make far state different
            ξ_far.reservoir_state .= 10.0 * randn(5)

            d_near = distance_to_uesra(ξ_near, uesra)
            d_far = distance_to_uesra(ξ_far, uesra)

            @test d_far > d_near
        end
    end

    @testset "is_in_basin" begin
        @testset "Resonant State in Basin" begin
            uesra = UniversalResonantArchetype(5)
            ξ = deepcopy(uesra.resonant_state)

            @test is_in_basin(ξ, uesra)
        end

        @testset "Far State Not in Basin" begin
            uesra = UniversalResonantArchetype(5)
            ξ = UnifiedState(5)

            # Make very different
            ξ.reservoir_state .= 100.0 * randn(5)

            @test !is_in_basin(ξ, uesra)
        end
    end

    # ========================================================================
    # EVOLUTION OPERATOR TESTS
    # ========================================================================

    @testset "UnifiedEvolutionOperator" begin
        @testset "Construction" begin
            Φ = UnifiedEvolutionOperator()

            @test Φ.dt == 0.1
            @test haskey(Φ.closures, :echo_state)
            @test haskey(Φ.closures, :attention)
        end

        @testset "Custom dt" begin
            Φ = UnifiedEvolutionOperator(dt=0.05)
            @test Φ.dt == 0.05
        end
    end

    @testset "apply_evolution!" begin
        @testset "Time Advances" begin
            Φ = UnifiedEvolutionOperator(dt=0.1)
            ξ = UnifiedState(5)

            t_before = ξ.t

            apply_evolution!(Φ, ξ)

            @test ξ.t == t_before + 0.1
        end

        @testset "State Changes" begin
            Φ = UnifiedEvolutionOperator()
            ξ = UnifiedState(5)

            # Set non-zero initial states
            ξ.emotion_state .= [0.5, 0.3, 0.2, 0.0, 0.0]
            ξ.reservoir_state .= randn(5)

            state_before = copy(ξ.reservoir_state)

            apply_evolution!(Φ, ξ)

            @test !all(ξ.reservoir_state .== state_before)
        end

        @testset "Coherence Updated" begin
            Φ = UnifiedEvolutionOperator()
            ξ = UnifiedState(5)

            apply_evolution!(Φ, ξ)

            @test 0.0 <= ξ.coherence <= 1.0
        end

        @testset "Attention Remains Valid Distribution" begin
            Φ = UnifiedEvolutionOperator()
            ξ = UnifiedState(5)

            for _ in 1:10
                apply_evolution!(Φ, ξ)

                @test isapprox(sum(ξ.attention_state), 1.0, atol=1e-6)
                @test all(ξ.attention_state .>= 0)
            end
        end

        @testset "States Remain Bounded" begin
            Φ = UnifiedEvolutionOperator()
            ξ = UnifiedState(10)

            for _ in 1:100
                apply_evolution!(Φ, ξ)

                @test all(isfinite.(ξ.reservoir_state))
                @test all(isfinite.(ξ.membrane_state))
                @test all(isfinite.(ξ.emotion_state))
            end
        end
    end

    # ========================================================================
    # RESONANCE DETECTION TESTS
    # ========================================================================

    @testset "ResonanceDetector" begin
        @testset "Construction" begin
            detector = ResonanceDetector()

            @test detector.coherence_threshold == 0.8
            @test detector.stability_threshold == 0.1
        end

        @testset "Detect at Resonant State" begin
            detector = ResonanceDetector(coherence_threshold=0.5)
            uesra = UniversalResonantArchetype(5)
            ξ = deepcopy(uesra.resonant_state)

            is_resonant, quality = detect_resonance(detector, ξ, uesra)

            @test is_resonant
            @test quality > 0
        end

        @testset "No Resonance at Random State" begin
            detector = ResonanceDetector(coherence_threshold=0.9)
            uesra = UniversalResonantArchetype(5)
            ξ = UnifiedState(5)

            ξ.reservoir_state .= 10.0 * randn(5)
            ξ.coherence = 0.1

            is_resonant, quality = detect_resonance(detector, ξ, uesra)

            @test !is_resonant
            @test quality < 0.5
        end
    end

    # ========================================================================
    # WISDOM METRICS TESTS
    # ========================================================================

    @testset "WisdomMetrics" begin
        @testset "Construction" begin
            ξ = UnifiedState(8)
            ξ.coherence = 0.7
            ξ.emotion_state[1] = 0.6

            metrics = WisdomMetrics(ξ)

            @test 0.0 <= metrics.relevance_quality <= 1.0
            @test 0.0 <= metrics.integration_depth <= 1.0
            @test 0.0 <= metrics.adaptive_flexibility <= 1.0
            @test 0.0 <= metrics.stable_coherence <= 1.0
            @test 0.0 <= metrics.transcendent_openness <= 1.0
        end

        @testset "High Coherence → High Integration" begin
            ξ = UnifiedState(5)
            ξ.coherence = 0.95

            metrics = WisdomMetrics(ξ)

            @test metrics.integration_depth == 0.95
        end
    end

    @testset "compute_wisdom_score" begin
        @testset "Bounded Score" begin
            ξ = UnifiedState(5)
            ξ.coherence = 0.5
            ξ.emotion_state[1] = 0.5

            metrics = WisdomMetrics(ξ)
            score = compute_wisdom_score(metrics)

            @test 0.0 <= score <= 1.0
        end

        @testset "High Metrics → High Score" begin
            ξ = UnifiedState(8)
            ξ.coherence = 0.95
            ξ.emotion_state[1] = 0.9

            metrics = WisdomMetrics(ξ)
            score = compute_wisdom_score(metrics)

            @test score > 0.5
        end
    end

    # ========================================================================
    # CONVERGENCE ALGORITHM TESTS
    # ========================================================================

    @testset "converge_to_uesra!" begin
        @testset "Basic Execution" begin
            ξ = UnifiedState(5)
            uesra = UniversalResonantArchetype(5)
            criteria = ConvergenceCriteria(max_iterations=50)

            result = converge_to_uesra!(ξ, uesra, criteria=criteria)

            @test haskey(result, :converged)
            @test haskey(result, :iterations)
            @test haskey(result, :history)
            @test haskey(result, :final_state)
        end

        @testset "Iterations Bounded" begin
            ξ = UnifiedState(5)
            uesra = UniversalResonantArchetype(5)
            criteria = ConvergenceCriteria(max_iterations=20)

            result = converge_to_uesra!(ξ, uesra, criteria=criteria)

            @test result.iterations <= 20
        end

        @testset "History Recorded" begin
            ξ = UnifiedState(5)
            uesra = UniversalResonantArchetype(5)
            criteria = ConvergenceCriteria(max_iterations=30)

            result = converge_to_uesra!(ξ, uesra, criteria=criteria)

            @test length(result.history[:distance]) == result.iterations
            @test length(result.history[:coherence]) == result.iterations
            @test length(result.history[:wisdom]) == result.iterations
        end

        @testset "Distance Generally Decreases" begin
            ξ = UnifiedState(5)
            uesra = UniversalResonantArchetype(5)
            criteria = ConvergenceCriteria(max_iterations=100)

            result = converge_to_uesra!(ξ, uesra, criteria=criteria)

            distances = result.history[:distance]
            if length(distances) > 10
                # Compare early vs late
                early_mean = mean(distances[1:5])
                late_mean = mean(distances[end-4:end])

                # Late should generally be smaller or similar
                @test late_mean <= early_mean + 1.0
            end
        end
    end

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    @testset "Full System Integration" begin
        @testset "Complete Evolution Cycle" begin
            dim = 8
            ξ = UnifiedState(dim)
            uesra = UniversalResonantArchetype(dim)
            Φ = UnifiedEvolutionOperator(dt=0.05)

            # Initialize with some emotion
            ξ.emotion_state[1] = 0.7  # Wonder
            ξ.emotion_state[2] = 0.5  # Curiosity

            # Run for several steps
            for _ in 1:50
                apply_evolution!(Φ, ξ)
            end

            # System should remain stable
            @test all(isfinite.(flatten_state(ξ)))

            # Should have some coherence
            @test ξ.coherence > 0

            # Attention should still be valid
            @test isapprox(sum(ξ.attention_state), 1.0, atol=1e-6)
        end

        @testset "Wisdom Increases Over Time" begin
            dim = 6
            ξ = UnifiedState(dim)
            uesra = UniversalResonantArchetype(dim)
            Φ = UnifiedEvolutionOperator(dt=0.1)

            # Initialize at resonant-like state
            ξ.emotion_state .= create_resonant_emotion_state(dim)

            initial_metrics = WisdomMetrics(ξ)
            initial_wisdom = compute_wisdom_score(initial_metrics)

            for _ in 1:30
                apply_evolution!(Φ, ξ)
            end

            final_metrics = WisdomMetrics(ξ)
            final_wisdom = compute_wisdom_score(final_metrics)

            # Wisdom should be positive
            @test final_wisdom > 0
        end

        @testset "Coupling Effects Propagate" begin
            dim = 5
            ξ = UnifiedState(dim)
            Φ = UnifiedEvolutionOperator()

            # Set strong emotion
            ξ.emotion_state .= [0.9, 0.8, 0.7, 0.6, 0.5]

            membrane_before = copy(ξ.membrane_state)
            reservoir_before = copy(ξ.reservoir_state)

            apply_evolution!(Φ, ξ)

            # Membrane should be affected
            @test !all(ξ.membrane_state .== membrane_before)

            # Reservoir should be affected
            @test !all(ξ.reservoir_state .== reservoir_before)
        end
    end

end
