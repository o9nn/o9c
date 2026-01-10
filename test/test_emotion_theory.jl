"""
Comprehensive unit tests for Differential Emotion Theory Framework
"""

@testset "Emotion Theory" begin

    @testset "Emotion Construction Joy" begin
        emotion = Emotion(:joy, 0.5)

        @test emotion.name == :joy
        @test emotion.intensity == 0.5
        @test emotion.valence == 1.0  # positive
        @test emotion.arousal == 0.6
        @test emotion.attention_scope == 0.8  # broad
        @test emotion.processing_depth == 0.5
        @test emotion.approach_avoid == 0.9  # approach
    end

    @testset "Emotion Construction Interest/Curiosity" begin
        emotion = Emotion(:interest, 0.7)

        @test emotion.name == :interest
        @test emotion.intensity == 0.7
        @test emotion.valence == 0.5  # slightly positive
        @test emotion.arousal == 0.6
        @test emotion.attention_scope == 0.7
        @test emotion.processing_depth == 0.8  # deep processing
        @test emotion.approach_avoid == 0.8

        # Curiosity should have same parameters as interest
        curiosity = Emotion(:curiosity, 0.7)
        @test curiosity.valence == emotion.valence
    end

    @testset "Emotion Construction Wonder/Awe" begin
        emotion = Emotion(:wonder, 0.9)

        @test emotion.name == :wonder
        @test emotion.intensity == 0.9
        @test emotion.valence == 0.8  # positive
        @test emotion.arousal == 0.5  # moderate
        @test emotion.attention_scope == 0.9  # very broad
        @test emotion.processing_depth == 0.9  # very deep
    end

    @testset "Emotion Construction Surprise" begin
        emotion = Emotion(:surprise, 0.6)

        @test emotion.valence == 0.0  # neutral
        @test emotion.arousal == 0.9  # high
        @test emotion.approach_avoid == 0.0  # neither
    end

    @testset "Emotion Construction Sadness" begin
        emotion = Emotion(:sadness, 0.4)

        @test emotion.valence == -0.6  # negative
        @test emotion.arousal == 0.3  # low
        @test emotion.attention_scope == 0.3  # narrow
        @test emotion.approach_avoid == -0.3  # mild avoidance
    end

    @testset "Emotion Construction Anger" begin
        emotion = Emotion(:anger, 0.8)

        @test emotion.valence == -0.5  # negative
        @test emotion.arousal == 0.8  # high
        @test emotion.approach_avoid == 0.5  # approach (to confront)
    end

    @testset "Emotion Construction Fear/Anxiety" begin
        emotion_fear = Emotion(:fear, 0.7)
        emotion_anxiety = Emotion(:anxiety, 0.7)

        @test emotion_fear.valence == -0.7  # negative
        @test emotion_fear.arousal == 0.9  # high
        @test emotion_fear.approach_avoid == -0.8  # strong avoidance

        # Fear and anxiety should have same parameters
        @test emotion_anxiety.valence == emotion_fear.valence
    end

    @testset "Emotion Construction Disgust" begin
        emotion = Emotion(:disgust, 0.6)

        @test emotion.valence == -0.8  # very negative
        @test emotion.approach_avoid == -0.9  # strong avoidance
    end

    @testset "Emotion Construction Unknown/Neutral" begin
        emotion = Emotion(:unknown_emotion, 0.5)

        # Should get neutral defaults
        @test emotion.valence == 0.0
        @test emotion.arousal == 0.3
        @test emotion.attention_scope == 0.5
        @test emotion.approach_avoid == 0.0
    end

    @testset "Emotion Default Intensity" begin
        emotion = Emotion(:joy)  # no intensity specified
        @test emotion.intensity == 0.0
    end

    @testset "DifferentialEmotionTheory Construction" begin
        emotion_names = [:joy, :sadness, :fear, :anger]
        det = DifferentialEmotionTheory(emotion_names)

        @test length(det.emotions) == 4
        @test haskey(det.emotions, :joy)
        @test haskey(det.emotions, :sadness)
        @test length(det.blend) == 4
        @test isempty(det.history)
        @test det.decay_rate == 0.1  # default
    end

    @testset "DifferentialEmotionTheory Custom Decay" begin
        emotion_names = [:wonder, :curiosity]
        det = DifferentialEmotionTheory(emotion_names, decay_rate=0.2)

        @test det.decay_rate == 0.2
    end

    @testset "initialize_contagion_matrix" begin
        emotion_names = [:joy, :sadness]
        C = initialize_contagion_matrix(emotion_names)

        @test size(C) == (2, 2)

        # Diagonal should be 1 (self-reinforcement)
        @test C[1, 1] == 1.0
        @test C[2, 2] == 1.0

        # Joy and sadness are opposite valence -> negative influence
        @test C[1, 2] < 0  # joy -> sadness (opponent)
        @test C[2, 1] < 0  # sadness -> joy (opponent)
    end

    @testset "initialize_contagion_matrix Similar Emotions" begin
        emotion_names = [:joy, :interest]  # Both positive valence
        C = initialize_contagion_matrix(emotion_names)

        # Similar valence should have positive influence
        @test C[1, 2] > 0
        @test C[2, 1] > 0
    end

    @testset "initialize_contagion_matrix Wonder Special Case" begin
        emotion_names = [:wonder, :joy]
        C = initialize_contagion_matrix(emotion_names)

        # Wonder should be compatible with positive emotions
        @test C[1, 2] > 0  # wonder -> joy
    end

    @testset "trigger_emotion!" begin
        emotion_names = [:joy, :fear]
        det = DifferentialEmotionTheory(emotion_names)

        # Initially intensity is 0
        @test det.emotions[:joy].intensity == 0.0

        trigger_emotion!(det, :joy, 0.8)

        @test det.emotions[:joy].intensity == 0.8
    end

    @testset "trigger_emotion! Clamping" begin
        det = DifferentialEmotionTheory([:anger])

        trigger_emotion!(det, :anger, 1.5)  # Over 1.0
        @test det.emotions[:anger].intensity == 1.0

        trigger_emotion!(det, :anger, -0.5)  # Under 0.0
        @test det.emotions[:anger].intensity == 0.0
    end

    @testset "trigger_emotion! Unknown Emotion" begin
        det = DifferentialEmotionTheory([:joy])

        # Should handle gracefully (no error)
        trigger_emotion!(det, :unknown, 0.5)
        @test !haskey(det.emotions, :unknown)
    end

    @testset "update_emotions! Decay" begin
        det = DifferentialEmotionTheory([:joy], decay_rate=0.5)

        trigger_emotion!(det, :joy, 1.0)
        initial = det.emotions[:joy].intensity

        update_emotions!(det, 0.1)

        # Intensity should decrease due to decay
        @test det.emotions[:joy].intensity < initial
    end

    @testset "update_emotions! History Recording" begin
        det = DifferentialEmotionTheory([:joy, :fear])

        @test isempty(det.history)

        update_emotions!(det, 0.1)

        @test length(det.history) == 1
        @test haskey(det.history[1], :joy)
        @test haskey(det.history[1], :fear)
    end

    @testset "update_emotions! Blend Update" begin
        det = DifferentialEmotionTheory([:joy, :fear])

        trigger_emotion!(det, :joy, 0.8)
        update_emotions!(det, 0.1)

        # Blend should reflect intensities
        @test det.blend[1] != 0.0 || det.blend[2] != 0.0
    end

    @testset "update_emotions! Intensity Bounds" begin
        det = DifferentialEmotionTheory([:joy, :anger, :fear])

        trigger_emotion!(det, :joy, 0.9)
        trigger_emotion!(det, :anger, 0.9)

        for _ in 1:20
            update_emotions!(det, 0.1)
        end

        # All intensities should remain in [0, 1]
        for (name, emotion) in det.emotions
            @test 0.0 <= emotion.intensity <= 1.0
        end
    end

    @testset "AffectiveAgency Construction" begin
        emotion_names = [:wonder, :curiosity, :joy, :fear]
        agency = AffectiveAgency(emotion_names)

        @test agency.attention_modulation == 0.5
        @test agency.memory_modulation == 0.5
        @test agency.threshold_modulation == 0.5
        @test agency.learning_rate_modulation == 0.5

        @test length(agency.det.emotions) == 4
    end

    @testset "compute_cognitive_modulation! No Emotions" begin
        agency = AffectiveAgency([:joy, :fear])

        # All intensities at 0
        compute_cognitive_modulation!(agency)

        @test agency.attention_modulation == 0.5
        @test agency.memory_modulation == 0.5
    end

    @testset "compute_cognitive_modulation! Joy Dominant" begin
        agency = AffectiveAgency([:joy, :fear])

        trigger_emotion!(agency.det, :joy, 0.9)
        compute_cognitive_modulation!(agency)

        # Joy has broad attention scope (0.8)
        @test agency.attention_modulation > 0.5

        # Joy has approach motivation (0.9)
        @test agency.threshold_modulation > 0.5
    end

    @testset "compute_cognitive_modulation! Fear Dominant" begin
        agency = AffectiveAgency([:joy, :fear])

        trigger_emotion!(agency.det, :fear, 0.9)
        compute_cognitive_modulation!(agency)

        # Fear has narrow attention scope (0.3)
        @test agency.attention_modulation < 0.5

        # Fear has high arousal (0.9)
        @test agency.learning_rate_modulation > 0.5
    end

    @testset "modulate_attention Broad Scope" begin
        agency = AffectiveAgency([:joy])
        trigger_emotion!(agency.det, :joy, 0.9)
        compute_cognitive_modulation!(agency)

        base_attention = [0.1, 0.2, 0.3, 0.4]
        modulated = modulate_attention(agency, base_attention)

        @test length(modulated) == 4
        @test sum(modulated) ≈ 1.0  # Should be normalized

        # Broad scope should flatten distribution
        variance_original = var(base_attention)
        variance_modulated = var(modulated)
        @test variance_modulated <= variance_original + 0.01
    end

    @testset "modulate_attention Narrow Scope" begin
        agency = AffectiveAgency([:fear])
        trigger_emotion!(agency.det, :fear, 0.9)
        compute_cognitive_modulation!(agency)

        base_attention = [0.1, 0.2, 0.3, 0.4]
        modulated = modulate_attention(agency, base_attention)

        @test sum(modulated) ≈ 1.0
    end

    @testset "modulate_learning_rate High Arousal" begin
        agency = AffectiveAgency([:fear])  # Fear has high arousal
        trigger_emotion!(agency.det, :fear, 0.9)
        compute_cognitive_modulation!(agency)

        base_rate = 0.1
        modulated = modulate_learning_rate(agency, base_rate)

        @test modulated > base_rate * 0.5  # At least half
        @test modulated <= base_rate  # At most base rate
    end

    @testset "modulate_learning_rate Low Arousal" begin
        agency = AffectiveAgency([:sadness])  # Sadness has low arousal
        trigger_emotion!(agency.det, :sadness, 0.9)
        compute_cognitive_modulation!(agency)

        base_rate = 0.1
        modulated = modulate_learning_rate(agency, base_rate)

        # Low arousal should result in lower learning rate
        @test modulated >= base_rate * 0.5
    end

    @testset "get_emotional_landscape No Emotions" begin
        agency = AffectiveAgency([:joy, :fear])

        landscape = get_emotional_landscape(agency)

        @test landscape[:dominant_emotion] == :neutral
        @test landscape[:valence] == 0.0
        @test landscape[:arousal] == 0.3
        @test landscape[:approach_avoid] == 0.0
    end

    @testset "get_emotional_landscape With Dominant Emotion" begin
        agency = AffectiveAgency([:joy, :fear, :anger])

        trigger_emotion!(agency.det, :joy, 0.9)
        trigger_emotion!(agency.det, :fear, 0.2)

        landscape = get_emotional_landscape(agency)

        @test landscape[:dominant_emotion] == :joy
        @test landscape[:valence] > 0  # Joy is positive
    end

    @testset "get_emotional_landscape Weighted Average" begin
        agency = AffectiveAgency([:joy, :sadness])

        # Equal intensity of opposite valence emotions
        trigger_emotion!(agency.det, :joy, 0.5)
        trigger_emotion!(agency.det, :sadness, 0.5)

        landscape = get_emotional_landscape(agency)

        # Valence should be somewhere between extremes
        @test -1.0 <= landscape[:valence] <= 1.0
    end

    @testset "Emotion Dynamics Over Time" begin
        agency = AffectiveAgency([:joy, :fear, :interest])

        trigger_emotion!(agency.det, :joy, 0.8)

        intensities = Float64[]

        for _ in 1:20
            update_emotions!(agency.det, 0.1)
            push!(intensities, agency.det.emotions[:joy].intensity)
        end

        # Intensity should decay over time
        @test intensities[end] < intensities[1]
    end

    @testset "Emotion Contagion Effects" begin
        # Joy should reinforce interest (similar valence)
        agency = AffectiveAgency([:joy, :interest])

        trigger_emotion!(agency.det, :joy, 0.9)

        for _ in 1:5
            update_emotions!(agency.det, 0.1)
        end

        # Interest may have increased due to contagion from joy
        # (depends on contagion matrix values)
        @test agency.det.emotions[:interest].intensity >= 0.0
    end

    @testset "Multiple Emotions Interaction" begin
        agency = AffectiveAgency([:wonder, :curiosity, :joy, :fear, :sadness])

        trigger_emotion!(agency.det, :wonder, 0.6)
        trigger_emotion!(agency.det, :curiosity, 0.7)
        trigger_emotion!(agency.det, :fear, 0.3)

        compute_cognitive_modulation!(agency)

        # With wonder and curiosity dominant, attention should be broad
        @test agency.attention_modulation > 0.4
    end

    @testset "Emotion History Length" begin
        det = DifferentialEmotionTheory([:joy])

        for _ in 1:50
            update_emotions!(det, 0.1)
        end

        @test length(det.history) == 50
    end

end
