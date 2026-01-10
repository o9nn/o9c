"""
Comprehensive unit tests for Cognitive Architecture Integration
"""

@testset "Cognitive Architecture" begin

    @testset "CognitiveArchitecture Construction Default" begin
        arch = CognitiveArchitecture()

        @test arch.persona == :contemplative_scholar
        @test arch.gpt_engine === nothing
        @test isempty(arch.state_history)
        @test isempty(arch.emergence_metrics)

        # Check subsystems are initialized
        @test isa(arch.membrane_system, PaunMembraneSystem)
        @test isa(arch.tree_esn, DeepTreeESN)
        @test isa(arch.bseries_forest, ButcherBSeriesForest)
        @test isa(arch.jsurface, JSurfaceDifferential)
        @test isa(arch.affective_agency, AffectiveAgency)
    end

    @testset "CognitiveArchitecture Custom Parameters" begin
        arch = CognitiveArchitecture(
            persona = :dynamic_explorer,
            depth = 3,
            reservoir_size = 40,
            input_dim = 15
        )

        @test arch.persona == :dynamic_explorer
        @test arch.membrane_system.depth == 3
        @test arch.tree_esn.depth == 3
        @test arch.tree_esn.root.reservoir_size == 40
    end

    @testset "CognitiveArchitecture With GPT" begin
        arch = CognitiveArchitecture(
            use_gpt = true,
            vocab_size = 500,
            gpt_dim = 64
        )

        @test arch.gpt_engine !== nothing
        @test arch.gpt_engine.vocab_size == 500
        @test arch.gpt_engine.dim == 64
    end

    @testset "CognitiveArchitecture Custom Emotions" begin
        custom_emotions = [:wonder, :curiosity, :joy]
        arch = CognitiveArchitecture(emotion_names = custom_emotions)

        @test length(arch.affective_agency.det.emotions) == 3
        @test haskey(arch.affective_agency.det.emotions, :wonder)
    end

    @testset "process Basic" begin
        arch = CognitiveArchitecture(
            depth = 2,
            reservoir_size = 20,
            input_dim = 10
        )

        input = randn(10)
        output = process(arch, input)

        @test length(output) == 10
        @test all(isfinite.(output))
    end

    @testset "process With Emotion Triggers" begin
        arch = CognitiveArchitecture(input_dim = 8)

        input = randn(8)
        emotion_triggers = Dict(:wonder => 0.8, :curiosity => 0.6)

        output = process(arch, input, emotion_triggers=emotion_triggers)

        @test length(output) == 8
        @test all(isfinite.(output))

        # Emotions should have been triggered
        @test arch.affective_agency.det.emotions[:wonder].intensity > 0
    end

    @testset "process State History Recording" begin
        arch = CognitiveArchitecture(depth = 2, input_dim = 10)

        @test isempty(arch.state_history)

        process(arch, randn(10))
        @test length(arch.state_history) == 1

        process(arch, randn(10))
        @test length(arch.state_history) == 2

        process(arch, randn(10))
        @test length(arch.state_history) == 3
    end

    @testset "process State Snapshot Contents" begin
        arch = CognitiveArchitecture(input_dim = 10)

        input = randn(10)
        process(arch, input)

        snapshot = arch.state_history[1]

        @test haskey(snapshot, :input)
        @test haskey(snapshot, :output)
        @test haskey(snapshot, :reservoir_states)
        @test haskey(snapshot, :emotional_landscape)
        @test haskey(snapshot, :membrane_permeability)
        @test haskey(snapshot, :jsurface_curvature)

        @test length(snapshot[:input]) == 10
    end

    @testset "process Multiple Steps" begin
        arch = CognitiveArchitecture(depth = 2, input_dim = 10)

        for step in 1:10
            input = randn(10)
            output = process(arch, input)
            @test all(isfinite.(output))
        end

        @test length(arch.state_history) == 10
    end

    @testset "process With GPT Integration" begin
        arch = CognitiveArchitecture(
            input_dim = 20,
            use_gpt = true,
            vocab_size = 100,
            gpt_dim = 32
        )

        input = randn(20)
        output = process(arch, input)

        # With GPT, output is concatenation of GPT and reservoir
        @test all(isfinite.(output))
        @test length(output) > 20  # Larger due to concatenation
    end

    @testset "record_state!" begin
        arch = CognitiveArchitecture(input_dim = 5)

        input = randn(5)
        output = randn(5)

        record_state!(arch, input, output)

        @test length(arch.state_history) == 1

        snapshot = arch.state_history[1]
        @test all(snapshot[:input] .== input)
        @test all(snapshot[:output] .== output)
    end

    @testset "compute_emergence! Insufficient History" begin
        arch = CognitiveArchitecture()

        # Only one state in history
        process(arch, randn(20))

        # Emergence metrics may not be computed with insufficient history
        compute_emergence!(arch)

        # Should handle gracefully (empty or default values)
        # Note: compute_emergence! requires at least 2 states
    end

    @testset "compute_emergence! With History" begin
        arch = CognitiveArchitecture(input_dim = 10)

        # Generate enough history
        for _ in 1:5
            process(arch, randn(10))
        end

        @test haskey(arch.emergence_metrics, :complexity)
        @test haskey(arch.emergence_metrics, :coherence)
        @test haskey(arch.emergence_metrics, :stability)
        @test haskey(arch.emergence_metrics, :adaptability)
        @test haskey(arch.emergence_metrics, :wisdom)
    end

    @testset "compute_emergence! Metric Bounds" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:10
            process(arch, randn(10))
        end

        # All metrics should be finite
        for (key, value) in arch.emergence_metrics
            @test isfinite(value)
        end

        # Most metrics should be in [0, 1] range
        @test 0.0 <= arch.emergence_metrics[:complexity] <= 1.0
        @test 0.0 <= arch.emergence_metrics[:stability] <= 1.0
        @test 0.0 <= arch.emergence_metrics[:adaptability] <= 1.0
    end

    @testset "compute_state_entropy Empty" begin
        states = Vector{Float64}[]
        entropy = compute_state_entropy(states)
        @test entropy == 0.0
    end

    @testset "compute_state_entropy Single State" begin
        states = [randn(10)]
        entropy = compute_state_entropy(states)
        @test isfinite(entropy)
    end

    @testset "compute_state_entropy Multiple States" begin
        states = [randn(20) for _ in 1:10]
        entropy = compute_state_entropy(states)

        @test isfinite(entropy)
        @test 0.0 <= entropy <= 1.0
    end

    @testset "compute_subsystem_coherence" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:5
            process(arch, randn(10))
        end

        coherence = compute_subsystem_coherence(arch.state_history)

        @test isfinite(coherence)
        @test 0.0 <= coherence <= 1.0
    end

    @testset "compute_trajectory_variance" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:5
            process(arch, randn(10))
        end

        variance = compute_trajectory_variance(arch.state_history)

        @test isfinite(variance)
        @test variance >= 0.0
    end

    @testset "compute_input_responsiveness" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:5
            process(arch, randn(10))
        end

        responsiveness = compute_input_responsiveness(arch.state_history)

        @test isfinite(responsiveness)
        @test 0.0 <= responsiveness <= 1.0
    end

    @testset "analyze_emergence" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:5
            process(arch, randn(10))
        end

        report = analyze_emergence(arch)

        @test haskey(report, :metrics)
        @test haskey(report, :emotional_state)
        @test haskey(report, :trajectory_summary)
        @test haskey(report, :recommendations)
    end

    @testset "analyze_emergence Report Contents" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:10
            process(arch, randn(10), emotion_triggers=Dict(:wonder => rand()))
        end

        report = analyze_emergence(arch)

        @test isa(report[:metrics], Dict)
        @test isa(report[:emotional_state], Dict)
        @test haskey(report[:emotional_state], :dominant_emotion)
    end

    @testset "analyze_emergence Recommendations" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:3
            process(arch, randn(10))
        end

        report = analyze_emergence(arch)

        @test isa(report[:recommendations], Vector{String})
    end

    @testset "set_persona!" begin
        arch = CognitiveArchitecture(persona = :balanced)

        @test arch.persona == :balanced

        set_persona!(arch, :contemplative_scholar)

        @test arch.persona == :contemplative_scholar

        # Should trigger wonder and curiosity
        @test arch.affective_agency.det.emotions[:wonder].intensity > 0
    end

    @testset "set_persona! Dynamic Explorer" begin
        arch = CognitiveArchitecture()

        set_persona!(arch, :dynamic_explorer)

        @test arch.persona == :dynamic_explorer
        @test arch.affective_agency.det.emotions[:curiosity].intensity > 0
        @test arch.affective_agency.det.emotions[:joy].intensity > 0
    end

    @testset "set_persona! Cautious Analyst" begin
        arch = CognitiveArchitecture()

        set_persona!(arch, :cautious_analyst)

        @test arch.persona == :cautious_analyst
        @test arch.affective_agency.det.emotions[:interest].intensity > 0
    end

    @testset "set_persona! Creative Visionary" begin
        arch = CognitiveArchitecture()

        set_persona!(arch, :creative_visionary)

        @test arch.persona == :creative_visionary
        @test arch.affective_agency.det.emotions[:wonder].intensity > 0
        @test arch.affective_agency.det.emotions[:joy].intensity > 0
    end

    @testset "set_persona! Rebuilds ESN" begin
        arch = CognitiveArchitecture(
            persona = :balanced,
            depth = 2,
            reservoir_size = 30,
            input_dim = 10
        )

        old_esn = arch.tree_esn

        set_persona!(arch, :dynamic_explorer)

        # ESN should be rebuilt
        new_esn = arch.tree_esn
        @test new_esn !== old_esn
        @test new_esn.persona_params[:spectral_radius] == 0.7  # dynamic_explorer
    end

    @testset "Different Personas Produce Different Outputs" begin
        input = randn(10)

        arch1 = CognitiveArchitecture(persona = :contemplative_scholar, input_dim = 10)
        arch2 = CognitiveArchitecture(persona = :dynamic_explorer, input_dim = 10)

        output1 = process(arch1, input)
        output2 = process(arch2, input)

        # Different personas should produce different outputs
        @test !all(output1 .== output2)
    end

    @testset "Emotion Effects on Processing" begin
        arch = CognitiveArchitecture(input_dim = 10)

        input = randn(10)

        # Process without emotions
        output1 = process(arch, input)

        # Reset and process with strong emotion
        arch2 = CognitiveArchitecture(input_dim = 10)
        output2 = process(arch2, input, emotion_triggers=Dict(:fear => 0.9))

        # Emotional state should affect output
        @test !all(output1 .== output2)
    end

    @testset "Membrane Permeability Recorded" begin
        arch = CognitiveArchitecture(input_dim = 10)

        process(arch, randn(10), emotion_triggers=Dict(:wonder => 0.8))

        snapshot = arch.state_history[1]
        permeabilities = snapshot[:membrane_permeability]

        @test !isempty(permeabilities)
        @test all(0.0 .<= permeabilities .<= 1.0)
    end

    @testset "Long Running Stability" begin
        arch = CognitiveArchitecture(
            depth = 3,
            reservoir_size = 30,
            input_dim = 15
        )

        for _ in 1:50
            input = randn(15)
            output = process(arch, input)

            @test all(isfinite.(output))
            @test all(abs.(output) .< 1e10)
        end
    end

    @testset "Wisdom Metric Computation" begin
        arch = CognitiveArchitecture(input_dim = 10)

        for _ in 1:10
            process(arch, randn(10))
        end

        @test haskey(arch.emergence_metrics, :wisdom)
        @test isfinite(arch.emergence_metrics[:wisdom])
    end

    @testset "Full Pipeline Integration" begin
        arch = CognitiveArchitecture(
            persona = :creative_visionary,
            depth = 3,
            reservoir_size = 25,
            input_dim = 12
        )

        # Run multiple steps with varying inputs and emotions
        for i in 1:10
            input = randn(12) * (1 + 0.1 * i)
            emotions = Dict(
                :wonder => 0.3 + 0.05 * i,
                :curiosity => 0.4
            )
            output = process(arch, input, emotion_triggers=emotions)

            @test all(isfinite.(output))
        end

        report = analyze_emergence(arch)

        @test !isempty(report[:metrics])
        @test haskey(report[:trajectory_summary], :n_states)
        @test report[:trajectory_summary][:n_states] == 10
    end

    @testset "Subsystem Integration" begin
        arch = CognitiveArchitecture(input_dim = 10)

        # All subsystems should be connected and functional
        input = randn(10)

        # Process through all subsystems
        output = process(arch, input, emotion_triggers=Dict(:joy => 0.7))

        # Check that each subsystem was updated
        @test !isempty(arch.affective_agency.det.history)
        @test !isempty(arch.state_history)

        # Membrane states should have changed
        for (id, membrane) in arch.membrane_system.membranes
            @test any(membrane.state .!= 0.0)
        end

        # ESN states should have changed
        for node in arch.tree_esn.all_nodes
            @test any(node.state .!= 0.0)
        end
    end

end
