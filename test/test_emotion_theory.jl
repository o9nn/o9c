"""
Unit tests for Differential Emotion Theory component.

These tests work standalone (no external dependencies required)
because emotion_theory.jl only uses Julia standard library packages.
"""

using Test
using LinearAlgebra
using Random

# Include standalone - only uses stdlib (LinearAlgebra, Random)
include(joinpath(@__DIR__, "../src/emotion_theory.jl"))

@testset "Emotion Theory" begin

    @testset "Emotion creation" begin
        # Known emotion types
        em_joy = Emotion(:joy, 0.5)
        @test em_joy.name == :joy
        @test em_joy.intensity == 0.5
        @test em_joy.valence == 1.0
        @test em_joy.arousal > 0.0
        @test em_joy.approach_avoid > 0.0   # joy is approach-oriented

        em_fear = Emotion(:fear, 0.8)
        @test em_fear.name == :fear
        @test em_fear.valence < 0.0         # fear is negative valence
        @test em_fear.approach_avoid < 0.0  # fear is avoidance-oriented

        em_wonder = Emotion(:wonder, 0.6)
        @test em_wonder.attention_scope > em_fear.attention_scope  # wonder broadens

        # Neutral default for unknown emotions
        em_custom = Emotion(:custom_emotion, 0.3)
        @test em_custom.name == :custom_emotion
        @test em_custom.intensity == 0.3
        @test em_custom.valence == 0.0

        # Test curiosity/interest share same case
        em_curiosity = Emotion(:curiosity, 0.4)
        em_interest  = Emotion(:interest, 0.4)
        @test em_curiosity.valence == em_interest.valence

        # Test awe shares wonder case
        em_awe = Emotion(:awe, 0.7)
        @test em_awe.valence == em_wonder.valence
    end

    @testset "Emotion intensity bounds" begin
        for name in [:joy, :fear, :sadness, :anger, :disgust, :surprise, :anxiety, :curiosity, :wonder]
            em = Emotion(name, 0.5)
            @test 0.0 <= em.intensity <= 1.0
            @test -1.0 <= em.valence <= 1.0
            @test 0.0 <= em.arousal <= 1.0
        end
    end

    @testset "DifferentialEmotionTheory construction" begin
        names = [:joy, :curiosity, :wonder, :fear, :sadness]
        det = DifferentialEmotionTheory(names)

        @test length(det.emotions) == 5
        @test all(em.intensity == 0.0 for em in values(det.emotions))
        @test length(det.blend) == 5
        @test all(det.blend .== 0.0)
        @test det.decay_rate == 0.1
        @test isempty(det.history)
        @test size(det.contagion_matrix) == (5, 5)
    end

    @testset "contagion_matrix diagonal" begin
        names = [:joy, :sadness, :wonder, :fear]
        det = DifferentialEmotionTheory(names)
        # Diagonal (self-reinforcement) should be 1.0
        for i in 1:4
            @test det.contagion_matrix[i, i] == 1.0
        end
    end

    @testset "trigger_emotion!" begin
        names = [:joy, :curiosity, :fear]
        det = DifferentialEmotionTheory(names)

        # Normal trigger
        trigger_emotion!(det, :joy, 0.8)
        @test det.emotions[:joy].intensity == 0.8

        # Clamped to 1.0
        trigger_emotion!(det, :joy, 1.5)
        @test det.emotions[:joy].intensity == 1.0

        # Clamped to 0.0
        trigger_emotion!(det, :joy, -0.5)
        @test det.emotions[:joy].intensity == 0.0

        # Unknown emotion: should not error and must not modify the dict
        joy_before = det.emotions[:joy].intensity
        @test_nowarn trigger_emotion!(det, :unknown_emotion, 0.5)
        @test det.emotions[:joy].intensity == joy_before  # existing emotion unchanged
        @test !haskey(det.emotions, :unknown_emotion)     # new key not created
    end

    @testset "update_emotions! decay" begin
        names = [:joy, :fear, :curiosity]
        det = DifferentialEmotionTheory(names, decay_rate=0.5)

        trigger_emotion!(det, :joy, 1.0)
        trigger_emotion!(det, :fear, 0.5)

        update_emotions!(det, 0.1)

        # History should be recorded
        @test length(det.history) == 1
        @test haskey(det.history[1], :joy)

        # All intensities should remain in [0, 1]
        for em in values(det.emotions)
            @test 0.0 <= em.intensity <= 1.0
        end

        # Blend should be updated
        @test length(det.blend) == 3
    end

    @testset "update_emotions! multiple steps" begin
        names = [:joy, :sadness]
        det = DifferentialEmotionTheory(names, decay_rate=0.3)

        trigger_emotion!(det, :joy, 0.9)
        initial_joy = det.emotions[:joy].intensity

        for _ in 1:10
            update_emotions!(det, 0.05)
        end

        @test length(det.history) == 10
        # All still in valid range
        for em in values(det.emotions)
            @test 0.0 <= em.intensity <= 1.0
        end
    end

    @testset "AffectiveAgency construction" begin
        names = [:wonder, :curiosity, :joy, :interest, :surprise, :sadness, :fear, :anxiety]
        agency = AffectiveAgency(names)

        @test agency.attention_modulation == 0.5
        @test agency.memory_modulation == 0.5
        @test agency.threshold_modulation == 0.5
        @test agency.learning_rate_modulation == 0.5
        @test length(agency.det.emotions) == 8
    end

    @testset "compute_cognitive_modulation! neutral" begin
        names = [:wonder, :curiosity, :joy, :interest, :surprise, :sadness, :fear, :anxiety]
        agency = AffectiveAgency(names)

        # No emotions active → neutral default values
        compute_cognitive_modulation!(agency)
        @test agency.attention_modulation == 0.5
        @test agency.memory_modulation == 0.5
        @test agency.threshold_modulation == 0.5
        @test agency.learning_rate_modulation == 0.5
    end

    @testset "compute_cognitive_modulation! with emotion" begin
        names = [:wonder, :curiosity, :joy, :interest, :surprise, :sadness, :fear, :anxiety]
        agency = AffectiveAgency(names)

        # Trigger joy: broad attention, high arousal → learning
        trigger_emotion!(agency.det, :joy, 1.0)
        compute_cognitive_modulation!(agency)

        # Joy has broad attention scope (0.8)
        @test agency.attention_modulation > 0.5

        # Fear: narrow attention, avoidance
        names2 = [:fear, :anxiety]
        agency2 = AffectiveAgency(names2)
        trigger_emotion!(agency2.det, :fear, 1.0)
        compute_cognitive_modulation!(agency2)
        @test agency2.threshold_modulation < 0.5  # avoidance lowers threshold
    end

    @testset "get_emotional_landscape neutral" begin
        names = [:wonder, :curiosity, :joy, :interest, :surprise, :sadness, :fear, :anxiety]
        agency = AffectiveAgency(names)

        landscape = get_emotional_landscape(agency)
        @test haskey(landscape, :dominant_emotion)
        @test haskey(landscape, :valence)
        @test haskey(landscape, :arousal)
        @test haskey(landscape, :approach_avoid)
        @test landscape[:dominant_emotion] == :neutral
        @test landscape[:valence] == 0.0
    end

    @testset "get_emotional_landscape with dominant emotion" begin
        names = [:joy, :wonder, :fear, :curiosity, :interest, :surprise, :sadness, :anxiety]
        agency = AffectiveAgency(names)

        trigger_emotion!(agency.det, :joy, 0.9)
        landscape = get_emotional_landscape(agency)

        @test landscape[:dominant_emotion] == :joy
        @test landscape[:valence] > 0.0    # joy is positive
        @test landscape[:approach_avoid] > 0.0  # joy is approach-oriented
    end

    @testset "modulate_attention broad scope" begin
        names = [:joy]
        agency = AffectiveAgency(names)

        # Trigger joy → broad attention (scope > 0.5)
        trigger_emotion!(agency.det, :joy, 1.0)
        compute_cognitive_modulation!(agency)
        @test agency.attention_modulation > 0.5

        base_attn = [0.2, 0.3, 0.5]
        modulated = modulate_attention(agency, base_attn)

        @test length(modulated) == 3
        # Normalization: sum ≈ 1
        @test sum(modulated) ≈ 1.0 atol=1e-5
        # Broad attention should flatten distribution
        @test maximum(modulated) - minimum(modulated) <= maximum(base_attn) - minimum(base_attn)
    end

    @testset "modulate_learning_rate" begin
        names = [:fear, :anxiety]
        agency = AffectiveAgency(names)

        trigger_emotion!(agency.det, :fear, 1.0)
        compute_cognitive_modulation!(agency)

        base_rate = 0.01
        modulated_rate = modulate_learning_rate(agency, base_rate)
        @test modulated_rate > 0.0
        @test modulated_rate >= base_rate * 0.5  # at minimum half base rate
    end

end
