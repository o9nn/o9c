"""
# Example: Multi-Agent Cognitive Architecture

Demonstrates two (or more) DeepTreeEchoSelf agents interacting.
Each agent:
  - Has its own persona and emotional baseline
  - Processes the same shared stimulus
  - Can "broadcast" its reservoir state to influence its peers

The inter-agent coupling models:
  - Social emotion contagion (one agent's valence affects another's)
  - Shared attention (agents co-orient toward salient inputs)
  - Emergent consensus (metrics converge under coupling)

Steps:
  1. Create two agents with contrasting personas
  2. Run a shared stimulus sequence
  3. Apply social emotion coupling between agents
  4. Compare and analyze emergence metrics for both agents
"""

push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using DeepTreeEchoSelf

using Random
using Statistics
using LinearAlgebra

println("=" ^ 70)
println("Deep Tree Echo Self - Multi-Agent Example")
println("=" ^ 70)
println()

Random.seed!(99)

# ── 1. Create two agents ──────────────────────────────────────────────────

emotion_names = [:wonder, :curiosity, :joy, :interest,
                 :surprise, :sadness, :fear, :anxiety]

agent_a = CognitiveArchitecture(
    persona       = :contemplative_scholar,
    emotion_names = emotion_names,
    depth         = 3,
    reservoir_size = 25,
    input_dim     = 8,
)

agent_b = CognitiveArchitecture(
    persona       = :dynamic_explorer,
    emotion_names = emotion_names,
    depth         = 3,
    reservoir_size = 25,
    input_dim     = 8,
)

println("Agent A: contemplative_scholar")
println("Agent B: dynamic_explorer")
println()

# ── 2. Social emotion coupling helper ────────────────────────────────────

"""
    couple_emotions!(source, target; coupling_strength=0.2)

Broadcast the dominant emotion from `source` to `target` with a given
coupling strength, modeling social emotion contagion.
"""
function couple_emotions!(source::CognitiveArchitecture,
                          target::CognitiveArchitecture;
                          coupling_strength::Float64 = 0.2)
    landscape = get_emotional_landscape(source.affective_agency)
    dom_emotion = landscape[:dominant_emotion]
    dom_intensity = get(source.affective_agency.det.emotions, dom_emotion,
                        nothing)
    if dom_intensity !== nothing && dom_emotion != :neutral
        # Weakly trigger the same emotion in the target agent
        trigger_emotion!(target.affective_agency.det, dom_emotion,
                         dom_intensity.intensity * coupling_strength)
    end
end

# ── 3. Shared stimulus processing with coupling ───────────────────────────

n_steps   = 30
input_dim = 8

println("Running $n_steps interaction steps with emotion coupling...")
println()

metrics_a = Dict{Symbol, Vector{Float64}}(
    :wisdom => Float64[], :coherence => Float64[], :adaptability => Float64[])
metrics_b = Dict{Symbol, Vector{Float64}}(
    :wisdom => Float64[], :coherence => Float64[], :adaptability => Float64[])

for step in 1:n_steps
    # Shared environmental stimulus (same for both agents)
    input = sin.(2π .* (1:input_dim) .* step / n_steps) .+ 0.1 * randn(input_dim)

    # Emotion triggers at specific moments
    triggers_a = Dict{Symbol, Float64}()
    triggers_b = Dict{Symbol, Float64}()

    if step == 5
        triggers_a[:wonder]    = 0.7   # Scholar wonders first
    elseif step == 10
        triggers_b[:curiosity] = 0.8   # Explorer acts on it
    elseif step == 20
        triggers_a[:joy]       = 0.6
        triggers_b[:joy]       = 0.5
    end

    # Process each agent
    process(agent_a, input, emotion_triggers=triggers_a, dt=0.1)
    process(agent_b, input, emotion_triggers=triggers_b, dt=0.1)

    # Social coupling: each agent's state mildly influences the other
    couple_emotions!(agent_a, agent_b, coupling_strength=0.15)
    couple_emotions!(agent_b, agent_a, coupling_strength=0.15)

    # Collect emergence metrics
    for (key, vec) in metrics_a
        push!(vec, get(agent_a.emergence_metrics, key, 0.0))
    end
    for (key, vec) in metrics_b
        push!(vec, get(agent_b.emergence_metrics, key, 0.0))
    end

    if step % 10 == 0
        la = get_emotional_landscape(agent_a.affective_agency)
        lb = get_emotional_landscape(agent_b.affective_agency)
        println("Step $step")
        println("  Agent A dominant: $(la[:dominant_emotion]) " *
                "(valence=$(round(la[:valence], digits=2)))")
        println("  Agent B dominant: $(lb[:dominant_emotion]) " *
                "(valence=$(round(lb[:valence], digits=2)))")
        println()
    end
end

# ── 4. Compare final emergence metrics ───────────────────────────────────

println("=" ^ 70)
println("Final Emergence Metrics Comparison")
println("=" ^ 70)
println()

analysis_a = analyze_emergence(agent_a)
analysis_b = analyze_emergence(agent_b)

header = rpad("Metric", 18)
col_a  = rpad("Agent A (Scholar)", 22)
col_b  = "Agent B (Explorer)"
println("  $header $col_a $col_b")
println("  " * "-"^62)

for metric in [:wisdom, :complexity, :coherence, :stability, :adaptability]
    val_a = round(get(analysis_a[:metrics], metric, 0.0), digits=3)
    val_b = round(get(analysis_b[:metrics], metric, 0.0), digits=3)
    println("  $(rpad(string(metric), 18)) $(rpad(string(val_a), 22)) $val_b")
end

println()

# ── 5. Convergence of wisdom under coupling ───────────────────────────────

if !isempty(metrics_a[:wisdom]) && !isempty(metrics_b[:wisdom])
    final_wisdom_a = last(metrics_a[:wisdom])
    final_wisdom_b = last(metrics_b[:wisdom])
    wisdom_diff = abs(final_wisdom_a - final_wisdom_b)
    println("Wisdom gap (final): $(round(wisdom_diff, digits=4))")
    println("(Lower gap indicates convergence through social coupling)")
    println()
end

println("=" ^ 70)
println("Key Observations")
println("=" ^ 70)
println()
println("1. Contrasting personas produce distinct emergence profiles:")
println("   - Scholar: higher coherence & stability (deep processing)")
println("   - Explorer: higher adaptability (fast, broad responses)")
println()
println("2. Social emotion coupling gradually aligns emotional baselines,")
println("   modeling how shared experiences create shared meaning.")
println()
println("3. Despite different starting points, collective processing of the")
println("   same stimulus leads the two agents toward similar wisdom metrics,")
println("   illustrating emergent consensus without central coordination.")
println()
