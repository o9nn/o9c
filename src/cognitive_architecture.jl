"""
# Cognitive Architecture Integration

This is where all components integrate into a unified cognitive architecture
that exhibits emergent wisdom through the optimization of relevance realization.

## The Deep Tree Echo Self

The "self" emerges from the recursive interplay of:

1. **Hierarchical Structure** (Paun membranes + Deep Tree ESN)
   - Multi-scale processing
   - Nested abstraction levels
   - Compositional representation

2. **Temporal Dynamics** (Butcher B-series + J-surface)
   - Differential unfolding
   - Trajectory optimization
   - Historical embedding

3. **Affective Modulation** (Differential Emotion Theory)
   - Participatory knowing
   - Value-laden processing
   - Motivational orientation

4. **Attention-based Relevance** (Transformer GPT)
   - Dynamic salience
   - Context-sensitive filtering
   - Propositional content

The self is not a thing but a process - the ongoing optimization of relevance
realization across all these dimensions simultaneously.

## Persona as Cognitive Style

Different personas instantiate different patterns of hyperparameters across
all subsystems, creating distinct cognitive styles that embody different
ways of being-in-the-world.
"""

using LinearAlgebra
using Statistics
using Random

"""
    CognitiveArchitecture

The complete integrated cognitive architecture.

# Fields
- `membrane_system::PaunMembraneSystem`: Hierarchical filtering
- `tree_esn::DeepTreeESN`: Reservoir computing dynamics
- `bseries_forest::ButcherBSeriesForest`: Temporal integration
- `jsurface::JSurfaceDifferential`: Geometric trajectory space
- `affective_agency::AffectiveAgency`: Emotion-cognition integration
- `gpt_engine::Union{GPTInferenceEngine,Nothing}`: Transformer attention
- `persona::Symbol`: Current persona/character
- `state_history::Vector{Dict{Symbol,Any}}`: Complete state trajectory
- `emergence_metrics::Dict{Symbol,Float64}`: Self-organization measures
"""
mutable struct CognitiveArchitecture
    membrane_system::PaunMembraneSystem
    tree_esn::DeepTreeESN
    bseries_forest::ButcherBSeriesForest
    jsurface::JSurfaceDifferential
    affective_agency::AffectiveAgency
    gpt_engine::Union{GPTInferenceEngine,Nothing}
    persona::Symbol
    state_history::Vector{Dict{Symbol,Any}}
    emergence_metrics::Dict{Symbol,Float64}
    
    function CognitiveArchitecture(;
        persona::Symbol = :contemplative_scholar,
        emotion_names::Vector{Symbol} = [:wonder, :curiosity, :joy, :interest, :surprise, :sadness, :fear, :anxiety],
        depth::Int = 4,
        reservoir_size::Int = 50,
        input_dim::Int = 20,
        use_gpt::Bool = false,
        vocab_size::Int = 1000,
        gpt_dim::Int = 128
    )
        # Initialize subsystems
        membrane_system = PaunMembraneSystem(depth, input_dim)
        add_default_rules!(membrane_system)
        
        tree_esn = DeepTreeESN(depth, reservoir_size, input_dim, persona=persona)
        
        bseries_forest = ButcherBSeriesForest(4, method=:rk4)
        
        jsurface = JSurfaceDifferential(input_dim, 3)
        
        affective_agency = AffectiveAgency(emotion_names)
        
        # Optional GPT integration
        gpt_engine = nothing
        if use_gpt
            gpt_engine = GPTInferenceEngine(vocab_size, gpt_dim, 2, 4)
        end
        
        state_history = Dict{Symbol,Any}[]
        emergence_metrics = Dict{Symbol,Float64}()
        
        new(membrane_system, tree_esn, bseries_forest, jsurface, affective_agency,
            gpt_engine, persona, state_history, emergence_metrics)
    end
end

"""
    process(arch::CognitiveArchitecture, input::Vector{Float64}; 
           emotion_triggers::Dict{Symbol,Float64} = Dict{Symbol,Float64}(),
           dt::Float64 = 0.1)

Process input through the complete cognitive architecture.
This is where the magic happens - the integration of all subsystems.

# Processing Flow
1. Trigger emotions (if any)
2. Update emotion dynamics
3. Compute affective modulation
4. Modulate membrane permeability
5. Process through membranes (filtering)
6. Process through tree ESN (dynamics)
7. Integrate via B-series (temporal)
8. Project onto J-surface (geometry)
9. If GPT available, integrate attention
10. Record state and compute emergence

# Returns
Processed output vector
"""
function process(arch::CognitiveArchitecture, input::Vector{Float64}; 
                emotion_triggers::Dict{Symbol,Float64} = Dict{Symbol,Float64}(),
                dt::Float64 = 0.1)
    
    # Step 1-3: Affective processing
    for (emotion, intensity) in emotion_triggers
        trigger_emotion!(arch.affective_agency.det, emotion, intensity)
    end
    
    update_emotions!(arch.affective_agency.det, dt)
    compute_cognitive_modulation!(arch.affective_agency)
    
    # Step 4: Modulate membranes based on emotion
    emotion_state = Dict(name => em.intensity for (name, em) in arch.affective_agency.det.emotions)
    modulate_permeability!(arch.membrane_system, emotion_state)
    
    # Step 5: Process through membrane system (hierarchical filtering)
    # Inject input at root
    arch.membrane_system.membranes[arch.membrane_system.root_id].state .= input
    apply_membrane_rules!(arch.membrane_system, dt)
    
    # Extract filtered input from membranes
    filtered_input = arch.membrane_system.membranes[arch.membrane_system.root_id].state
    
    # Step 6: Process through tree ESN (reservoir dynamics)
    process_tree!(arch.tree_esn, filtered_input)
    reservoir_state = collect_states(arch.tree_esn)
    
    # Step 7: Temporal integration via B-series
    # Create cognitive dynamics vector field
    vector_field = create_cognitive_dynamics(
        length(filtered_input),
        attraction_strength = 1.0,
        coupling = 0.5
    )
    
    integrated_state = apply_bseries_step(
        arch.bseries_forest,
        vector_field,
        filtered_input,
        dt
    )
    
    # Step 8: Project onto J-surface (geometric manifold)
    projected_state = project_to_surface(arch.jsurface, integrated_state)
    
    # Step 9: GPT integration (if available)
    if arch.gpt_engine !== nothing
        # For demonstration, use projected_state as token sequence
        # In practice, would need proper tokenization
        token_ids = [clamp(round(Int, (x + 1) * 500), 1, arch.gpt_engine.vocab_size) 
                    for x in projected_state[1:min(10, length(projected_state))]]
        
        gpt_output, attention_maps = forward_gpt(arch.gpt_engine, token_ids)
        
        # Extract relevance landscape
        relevance = extract_relevance_landscape(attention_maps)
        
        # Integrate GPT with reservoir
        final_output = integrate_with_reservoir(gpt_output, reservoir_state)
    else
        # Without GPT, use projected state as output
        final_output = projected_state
    end
    
    # Step 10: Record state and compute emergence metrics
    record_state!(arch, input, final_output)
    compute_emergence!(arch)
    
    return final_output
end

"""
    record_state!(arch::CognitiveArchitecture, input::Vector{Float64}, output::Vector{Float64})

Record complete state snapshot for analysis.
"""
function record_state!(arch::CognitiveArchitecture, input::Vector{Float64}, output::Vector{Float64})
    snapshot = Dict{Symbol,Any}()
    
    snapshot[:input] = copy(input)
    snapshot[:output] = copy(output)
    snapshot[:reservoir_states] = collect_states(arch.tree_esn)
    snapshot[:emotional_landscape] = get_emotional_landscape(arch.affective_agency)
    snapshot[:membrane_permeability] = [m.permeability for (id, m) in arch.membrane_system.membranes]
    snapshot[:jsurface_curvature] = arch.jsurface.curvature
    
    push!(arch.state_history, snapshot)
end

"""
    compute_emergence!(arch::CognitiveArchitecture)

Compute metrics of self-organization and emergence.

# Metrics
- complexity: How rich/structured the dynamics are
- coherence: How integrated the subsystems are
- stability: How robust to perturbations
- adaptability: How responsive to change
- wisdom: Overall optimization of relevance realization
"""
function compute_emergence!(arch::CognitiveArchitecture)
    if length(arch.state_history) < 2
        return
    end
    
    # Get recent history
    n_recent = min(10, length(arch.state_history))
    recent = arch.state_history[end-n_recent+1:end]
    
    # Compute complexity (entropy of reservoir states)
    reservoir_states = [s[:reservoir_states] for s in recent]
    complexity = compute_state_entropy(reservoir_states)
    
    # Compute coherence (correlation between subsystems)
    coherence = compute_subsystem_coherence(recent)
    
    # Compute stability (variance of trajectory)
    stability = 1.0 / (1.0 + compute_trajectory_variance(recent))
    
    # Compute adaptability (responsiveness to input)
    adaptability = compute_input_responsiveness(recent)
    
    # Wisdom = balanced optimization
    # High on all metrics, but especially coherence and stability
    wisdom = (coherence + stability + 0.5 * complexity + 0.5 * adaptability) / 3.0
    
    arch.emergence_metrics[:complexity] = complexity
    arch.emergence_metrics[:coherence] = coherence
    arch.emergence_metrics[:stability] = stability
    arch.emergence_metrics[:adaptability] = adaptability
    arch.emergence_metrics[:wisdom] = wisdom
end

"""
    compute_state_entropy(states::Vector)

Compute entropy of state sequence (complexity measure).
"""
function compute_state_entropy(states::Vector)
    if isempty(states)
        return 0.0
    end
    
    # Concatenate states
    state_matrix = hcat(states...)'
    
    # Compute covariance
    C = cov(state_matrix)
    
    # Entropy ≈ log det(C)
    # Add regularization for numerical stability
    ε = 1e-6
    C_reg = C + ε * I(size(C, 1))
    
    entropy = 0.5 * log(det(C_reg) + 1e-10)
    
    return clamp(entropy / 10.0, 0.0, 1.0)  # Normalize
end

"""
    compute_subsystem_coherence(history::Vector)

Measure how integrated the subsystems are (correlation).
"""
function compute_subsystem_coherence(history::Vector)
    if length(history) < 2
        return 0.5
    end
    
    # Extract metrics from different subsystems
    permeabilities = [mean(s[:membrane_permeability]) for s in history]
    emotions = [s[:emotional_landscape][:valence] for s in history]
    curvatures = [s[:jsurface_curvature] for s in history]
    
    # Compute pairwise correlations
    correlations = []
    
    if var(permeabilities) > 1e-6 && var(emotions) > 1e-6
        push!(correlations, abs(cor(permeabilities, emotions)))
    end
    if var(emotions) > 1e-6 && var(curvatures) > 1e-6
        push!(correlations, abs(cor(emotions, curvatures)))
    end
    if var(permeabilities) > 1e-6 && var(curvatures) > 1e-6
        push!(correlations, abs(cor(permeabilities, curvatures)))
    end
    
    return isempty(correlations) ? 0.5 : mean(correlations)
end

"""
    compute_trajectory_variance(history::Vector)

Measure variance of output trajectory.
"""
function compute_trajectory_variance(history::Vector)
    outputs = [s[:output] for s in history]
    output_matrix = hcat(outputs...)'
    return mean(var(output_matrix, dims=1))
end

"""
    compute_input_responsiveness(history::Vector)

Measure how responsive system is to input changes.
"""
function compute_input_responsiveness(history::Vector)
    if length(history) < 2
        return 0.5
    end
    
    # Compute correlation between input changes and output changes
    input_diffs = []
    output_diffs = []
    
    for i in 2:length(history)
        input_diff = norm(history[i][:input] - history[i-1][:input])
        output_diff = norm(history[i][:output] - history[i-1][:output])
        push!(input_diffs, input_diff)
        push!(output_diffs, output_diff)
    end
    
    if var(input_diffs) < 1e-6 || var(output_diffs) < 1e-6
        return 0.5
    end
    
    return clamp(abs(cor(input_diffs, output_diffs)), 0.0, 1.0)
end

"""
    analyze_emergence(arch::CognitiveArchitecture)

Analyze and report on emergent properties of the system.

# Returns
Dictionary with:
- metrics: Current emergence metrics
- emotional_state: Current emotional landscape
- trajectory_summary: Summary of state trajectory
- recommendations: Suggestions for optimization
"""
function analyze_emergence(arch::CognitiveArchitecture)
    report = Dict{Symbol,Any}()
    
    # Current metrics
    report[:metrics] = copy(arch.emergence_metrics)
    
    # Emotional state
    report[:emotional_state] = get_emotional_landscape(arch.affective_agency)
    
    # Trajectory summary
    if !isempty(arch.state_history)
        recent_complexity = [compute_state_entropy([s[:reservoir_states]]) 
                           for s in arch.state_history[max(1,end-9):end]]
        report[:trajectory_summary] = Dict(
            :mean_complexity => mean(recent_complexity),
            :complexity_trend => length(recent_complexity) > 1 ? 
                                sign(recent_complexity[end] - recent_complexity[1]) : 0,
            :n_states => length(arch.state_history)
        )
    end
    
    # Recommendations based on metrics
    recommendations = String[]
    
    wisdom = get(arch.emergence_metrics, :wisdom, 0.5)
    complexity = get(arch.emergence_metrics, :complexity, 0.5)
    coherence = get(arch.emergence_metrics, :coherence, 0.5)
    stability = get(arch.emergence_metrics, :stability, 0.5)
    
    if wisdom < 0.4
        push!(recommendations, "System showing low wisdom - consider adjusting persona or emotional triggers")
    end
    if complexity < 0.3
        push!(recommendations, "Low complexity - system may be under-stimulated or over-regularized")
    elseif complexity > 0.8
        push!(recommendations, "High complexity - system may be chaotic, consider increasing stability")
    end
    if coherence < 0.4
        push!(recommendations, "Low coherence - subsystems not well integrated")
    end
    if stability < 0.3
        push!(recommendations, "Low stability - system highly variable, may need stronger attractors")
    end
    
    report[:recommendations] = recommendations
    
    return report
end

"""
    set_persona!(arch::CognitiveArchitecture, persona::Symbol)

Change the persona, which modulates hyperparameters across all subsystems.
This changes the cognitive style - how the system processes information.
"""
function set_persona!(arch::CognitiveArchitecture, persona::Symbol)
    arch.persona = persona
    
    # Rebuild tree ESN with new persona
    arch.tree_esn = DeepTreeESN(
        arch.tree_esn.depth,
        arch.tree_esn.root.reservoir_size,
        size(arch.tree_esn.root.W_in, 2),
        persona=persona
    )
    
    # Adjust emotional baseline for persona
    if persona == :contemplative_scholar
        trigger_emotion!(arch.affective_agency.det, :wonder, 0.6)
        trigger_emotion!(arch.affective_agency.det, :curiosity, 0.5)
    elseif persona == :dynamic_explorer
        trigger_emotion!(arch.affective_agency.det, :curiosity, 0.8)
        trigger_emotion!(arch.affective_agency.det, :joy, 0.6)
    elseif persona == :cautious_analyst
        trigger_emotion!(arch.affective_agency.det, :interest, 0.6)
        trigger_emotion!(arch.affective_agency.det, :anxiety, 0.3)
    elseif persona == :creative_visionary
        trigger_emotion!(arch.affective_agency.det, :wonder, 0.7)
        trigger_emotion!(arch.affective_agency.det, :joy, 0.7)
    end
end

end  # of included file scope
