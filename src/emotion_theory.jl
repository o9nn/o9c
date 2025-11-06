"""
# Differential Emotion Theory Framework

Implements discrete emotion systems based on Differential Emotion Theory (DET).
Emotions are not mere epiphenomena but constitute a fundamental way of knowing -
participatory knowing that transforms the knower.

## Theoretical Foundation

Drawing on Izard's Differential Emotion Theory and affective neuroscience:

1. **Discrete Emotions**: Basic emotions are evolutionarily ancient, neurally distinct
2. **Affective-Cognitive Integration**: Emotions modulate attention, memory, action
3. **Motivational Properties**: Emotions energize and direct behavior
4. **Phenomenological Quality**: Each emotion has distinct subjective feel

## Cognitive-Affective Interaction

Emotions shape cognition by:
- **Modulating salience**: What becomes relevant
- **Biasing processing**: How information is interpreted  
- **Energizing action**: What we're motivated to do
- **Coloring qualia**: What experience feels like

This embodies participatory knowing - we know through being affected.

## Basic Emotions

Following DET, we implement these fundamental emotions:
- Interest/Curiosity: Opens attention, exploration
- Joy: Broadens scope, integration
- Surprise: Interrupts, reorients
- Sadness: Narrows focus, conservation
- Anger: Focuses on obstacles, assertion
- Disgust: Rejection, boundary formation
- Fear/Anxiety: Heightens vigilance, protection
- Wonder/Awe: Expansive, self-transcendent
"""

using LinearAlgebra
using Random

"""
    Emotion

Represents a discrete emotion state with its cognitive effects.

# Fields
- `name::Symbol`: Name of the emotion
- `intensity::Float64`: Current intensity (0-1)
- `valence::Float64`: Positive/negative quality (-1 to +1)
- `arousal::Float64`: Activation level (0-1)
- `attention_scope::Float64`: How much it broadens/narrows attention
- `processing_depth::Float64`: Shallow/deep processing bias
- `approach_avoid::Float64`: Approach (+) vs avoidance (-) motivation
"""
mutable struct Emotion
    name::Symbol
    intensity::Float64
    valence::Float64
    arousal::Float64
    attention_scope::Float64      # Narrow (0) to Broad (1)
    processing_depth::Float64     # Shallow (0) to Deep (1)
    approach_avoid::Float64       # Avoid (-1) to Approach (+1)
    
    function Emotion(name::Symbol, intensity::Float64=0.0)
        # Set parameters based on emotion type
        if name == :joy
            new(name, intensity, 1.0, 0.6, 0.8, 0.5, 0.9)
        elseif name == :interest || name == :curiosity
            new(name, intensity, 0.5, 0.6, 0.7, 0.8, 0.8)
        elseif name == :wonder || name == :awe
            new(name, intensity, 0.8, 0.5, 0.9, 0.9, 0.7)
        elseif name == :surprise
            new(name, intensity, 0.0, 0.9, 0.6, 0.3, 0.0)
        elseif name == :sadness
            new(name, intensity, -0.6, 0.3, 0.3, 0.7, -0.3)
        elseif name == :anger
            new(name, intensity, -0.5, 0.8, 0.4, 0.4, 0.5)
        elseif name == :fear || name == :anxiety
            new(name, intensity, -0.7, 0.9, 0.3, 0.5, -0.8)
        elseif name == :disgust
            new(name, intensity, -0.8, 0.5, 0.2, 0.3, -0.9)
        else
            # Neutral default
            new(name, intensity, 0.0, 0.3, 0.5, 0.5, 0.0)
        end
    end
end

"""
    DifferentialEmotionTheory

Container for the complete emotion system with dynamics.

# Fields
- `emotions::Dict{Symbol,Emotion}`: All active emotions
- `blend::Vector{Float64}`: Current emotional blend vector
- `history::Vector{Dict{Symbol,Float64}}`: Emotion trajectory over time
- `decay_rate::Float64`: How quickly emotions fade
- `contagion_matrix::Matrix{Float64}`: How emotions influence each other
"""
mutable struct DifferentialEmotionTheory
    emotions::Dict{Symbol,Emotion}
    blend::Vector{Float64}
    history::Vector{Dict{Symbol,Float64}}
    decay_rate::Float64
    contagion_matrix::Matrix{Float64}
    
    function DifferentialEmotionTheory(emotion_names::Vector{Symbol}; decay_rate::Float64=0.1)
        emotions = Dict{Symbol,Emotion}()
        for name in emotion_names
            emotions[name] = Emotion(name, 0.0)
        end
        
        n = length(emotion_names)
        blend = zeros(n)
        history = Dict{Symbol,Float64}[]
        
        # Emotion contagion/interaction matrix
        # How activation of one emotion affects others
        contagion = initialize_contagion_matrix(emotion_names)
        
        new(emotions, blend, history, decay_rate, contagion)
    end
end

"""
    initialize_contagion_matrix(emotion_names::Vector{Symbol})

Initialize matrix describing how emotions influence each other.
Captures opponent processing and emotional dynamics.

# Principles
- Similar valence emotions reinforce (joy → interest)
- Opposite valence emotions oppose (joy ↔ sadness)
- High arousal emotions suppress low arousal
- Wonder/awe transcends other emotions
"""
function initialize_contagion_matrix(emotion_names::Vector{Symbol})
    n = length(emotion_names)
    C = zeros(n, n)
    
    for (i, name_i) in enumerate(emotion_names)
        for (j, name_j) in enumerate(emotion_names)
            if i == j
                C[i, j] = 1.0  # Self-reinforcement
            else
                # Get emotional qualities
                em_i = Emotion(name_i)
                em_j = Emotion(name_j)
                
                # Valence similarity
                val_sim = 1.0 - abs(em_i.valence - em_j.valence) / 2.0
                
                # Arousal compatibility  
                arousal_diff = abs(em_i.arousal - em_j.arousal)
                
                # Similar valence and compatible arousal → positive influence
                # Opposite valence → negative influence (opponent processing)
                if em_i.valence * em_j.valence > 0
                    C[i, j] = 0.3 * val_sim * (1.0 - 0.5 * arousal_diff)
                else
                    C[i, j] = -0.2 * (1.0 - val_sim)
                end
                
                # Wonder/awe is transcendent - compatible with positive emotions
                if (name_i == :wonder || name_i == :awe) && em_j.valence > 0
                    C[i, j] = 0.4
                end
            end
        end
    end
    
    return C
end

"""
    update_emotions!(det::DifferentialEmotionTheory, dt::Float64)

Update emotion dynamics with decay and contagion effects.
Emotions naturally decay over time but also influence each other.

# Dynamics
dI/dt = -λI + ∑ C_ij I_j

where I is intensity, λ is decay, C is contagion matrix
"""
function update_emotions!(det::DifferentialEmotionTheory, dt::Float64)
    n = length(det.emotions)
    emotion_names = collect(keys(det.emotions))
    
    # Get current intensities
    current_intensities = [det.emotions[name].intensity for name in emotion_names]
    
    # Compute changes
    dI = -det.decay_rate * current_intensities
    dI .+= det.contagion_matrix * current_intensities
    
    # Update intensities
    for (i, name) in enumerate(emotion_names)
        new_intensity = current_intensities[i] + dt * dI[i]
        det.emotions[name].intensity = clamp(new_intensity, 0.0, 1.0)
    end
    
    # Update blend vector
    det.blend = [det.emotions[name].intensity for name in emotion_names]
    
    # Record history
    history_entry = Dict{Symbol,Float64}()
    for name in emotion_names
        history_entry[name] = det.emotions[name].intensity
    end
    push!(det.history, history_entry)
end

"""
    trigger_emotion!(det::DifferentialEmotionTheory, emotion_name::Symbol, intensity::Float64)

Trigger an emotion with specified intensity.
This is how external events affect the emotional system.
"""
function trigger_emotion!(det::DifferentialEmotionTheory, emotion_name::Symbol, intensity::Float64)
    if haskey(det.emotions, emotion_name)
        det.emotions[emotion_name].intensity = clamp(intensity, 0.0, 1.0)
    end
end

"""
    AffectiveAgency

The affective agency system that modulates cognitive processing based on emotion.
This is where emotion and cognition integrate - participatory knowing in action.

# Fields
- `det::DifferentialEmotionTheory`: The emotion system
- `attention_modulation::Float64`: Current effect on attention scope
- `memory_modulation::Float64`: Current effect on memory encoding
- `threshold_modulation::Float64`: Current effect on decision thresholds
- `learning_rate_modulation::Float64`: Current effect on learning
"""
mutable struct AffectiveAgency
    det::DifferentialEmotionTheory
    attention_modulation::Float64
    memory_modulation::Float64
    threshold_modulation::Float64
    learning_rate_modulation::Float64
    
    function AffectiveAgency(emotion_names::Vector{Symbol})
        det = DifferentialEmotionTheory(emotion_names)
        new(det, 0.5, 0.5, 0.5, 0.5)
    end
end

"""
    compute_cognitive_modulation!(agency::AffectiveAgency)

Compute how current emotional state modulates cognitive parameters.
Different emotions have different effects on cognition.
"""
function compute_cognitive_modulation!(agency::AffectiveAgency)
    det = agency.det
    
    # Weighted sum of emotional effects
    total_intensity = sum(em.intensity for em in values(det.emotions))
    
    if total_intensity < 1e-6
        # No emotion - default neutral values
        agency.attention_modulation = 0.5
        agency.memory_modulation = 0.5
        agency.threshold_modulation = 0.5
        agency.learning_rate_modulation = 0.5
        return
    end
    
    # Compute weighted averages of emotion parameters
    attention_scope = 0.0
    processing_depth = 0.0
    approach_strength = 0.0
    arousal_level = 0.0
    
    for emotion in values(det.emotions)
        w = emotion.intensity / total_intensity
        attention_scope += w * emotion.attention_scope
        processing_depth += w * emotion.processing_depth
        approach_strength += w * emotion.approach_avoid
        arousal_level += w * emotion.arousal
    end
    
    # Map to cognitive parameters
    agency.attention_modulation = attention_scope
    agency.memory_modulation = processing_depth
    agency.threshold_modulation = 0.5 + 0.3 * approach_strength  # Approach lowers thresholds
    agency.learning_rate_modulation = arousal_level  # High arousal increases learning
end

"""
    modulate_attention(agency::AffectiveAgency, base_attention::Vector{Float64})

Modulate attention based on emotional state.
Emotions reshape the salience landscape - what becomes relevant.

# Effect
- Broad scope (joy, wonder) → distributed attention
- Narrow scope (fear, sadness) → focused attention
"""
function modulate_attention(agency::AffectiveAgency, base_attention::Vector{Float64})
    scope = agency.attention_modulation
    
    # Broad scope → flatten distribution (distributed attention)
    # Narrow scope → sharpen distribution (focused attention)
    if scope > 0.5
        # Broaden: reduce differences
        mean_attn = mean(base_attention)
        modulated = scope * base_attention + (1 - scope) * mean_attn
    else
        # Narrow: amplify differences  
        modulated = base_attention .^ (1.0 / (scope + 0.1))
    end
    
    # Renormalize
    return modulated ./ (sum(modulated) + 1e-10)
end

"""
    modulate_learning_rate(agency::AffectiveAgency, base_rate::Float64)

Modulate learning rate based on emotional arousal.
High arousal emotions enhance learning (emotional tagging).
"""
function modulate_learning_rate(agency::AffectiveAgency, base_rate::Float64)
    return base_rate * (0.5 + 0.5 * agency.learning_rate_modulation)
end

"""
    get_emotional_landscape(agency::AffectiveAgency)

Get current emotional landscape as a structured summary.
Useful for understanding the affective state.

# Returns
Dictionary with:
- dominant_emotion: Most intense emotion
- valence: Overall positive/negative
- arousal: Overall activation
- approach_avoid: Overall motivation direction
"""
function get_emotional_landscape(agency::AffectiveAgency)
    det = agency.det
    
    # Find dominant emotion
    max_intensity = 0.0
    dominant = :neutral
    for (name, emotion) in det.emotions
        if emotion.intensity > max_intensity
            max_intensity = emotion.intensity
            dominant = name
        end
    end
    
    # Compute weighted averages
    total_intensity = sum(em.intensity for em in values(det.emotions))
    
    if total_intensity < 1e-6
        return Dict(
            :dominant_emotion => :neutral,
            :valence => 0.0,
            :arousal => 0.3,
            :approach_avoid => 0.0
        )
    end
    
    overall_valence = 0.0
    overall_arousal = 0.0
    overall_approach = 0.0
    
    for emotion in values(det.emotions)
        w = emotion.intensity / total_intensity
        overall_valence += w * emotion.valence
        overall_arousal += w * emotion.arousal
        overall_approach += w * emotion.approach_avoid
    end
    
    return Dict(
        :dominant_emotion => dominant,
        :valence => overall_valence,
        :arousal => overall_arousal,
        :approach_avoid => overall_approach
    )
end

