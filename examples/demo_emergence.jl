"""
# Example: Demonstrating the Deep Tree Echo Self

This example shows how the cognitive architecture creates an emergent self
through the integration of multiple computational substrates.

We'll demonstrate:
1. Creating an architecture with a specific persona
2. Processing inputs with emotional triggers
3. Observing emergent self-organization
4. Analyzing the wisdom metrics
5. Changing persona and observing how cognition changes
"""

# This would normally be: using DeepTreeEchoSelf
# For standalone execution:
push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using DeepTreeEchoSelf

using Random
using Statistics
using LinearAlgebra

println("=" ^ 80)
println("Deep Tree Echo Self - Emergence Demonstration")
println("=" ^ 80)
println()

# Set random seed for reproducibility
Random.seed!(42)

println("Creating cognitive architecture with 'contemplative_scholar' persona...")
println()

# Create architecture
arch = CognitiveArchitecture(
    persona = :contemplative_scholar,
    depth = 3,
    reservoir_size = 30,
    input_dim = 10,
    use_gpt = false  # Set to true if you want GPT integration
)

println("Architecture created with:")
println("  - $(length(arch.membrane_system.membranes)) membranes across $(arch.membrane_system.depth+1) levels")
println("  - $(length(arch.tree_esn.all_nodes)) reservoir nodes")
println("  - $(length(arch.bseries_forest.trees)) B-series trees")
println("  - $(arch.jsurface.dimension)-dimensional J-surface")
println("  - $(length(arch.affective_agency.det.emotions)) emotion systems")
println()

# Simulate a sequence of inputs representing a learning experience
println("Simulating cognitive processing sequence...")
println()

n_steps = 20
input_dim = 10

for step in 1:n_steps
    # Create input (simulating sensory/symbolic information)
    # Gradual shift from random to structured pattern
    if step < 10
        # Early: random, exploratory
        input = randn(input_dim)
    else
        # Later: structured pattern emerges
        t = (step - 10) / 10.0
        input = [sin(2π * i * t / input_dim) for i in 1:input_dim]
    end
    
    # Emotion triggers based on context
    emotions = Dict{Symbol,Float64}()
    
    if step == 1
        # Initial surprise and curiosity
        emotions[:surprise] = 0.7
        emotions[:curiosity] = 0.6
    elseif step == 10
        # Pattern recognition - joy and wonder
        emotions[:joy] = 0.5
        emotions[:wonder] = 0.4
    elseif step == n_steps
        # Understanding achieved
        emotions[:wonder] = 0.8
        emotions[:joy] = 0.7
    end
    
    # Process input
    output = process(arch, input, emotion_triggers=emotions, dt=0.1)
    
    # Report progress
    if step % 5 == 0 || step == 1 || step == n_steps
        println("Step $step:")
        
        if haskey(arch.emergence_metrics, :wisdom)
            println("  Wisdom: $(round(arch.emergence_metrics[:wisdom], digits=3))")
            println("  Complexity: $(round(arch.emergence_metrics[:complexity], digits=3))")
            println("  Coherence: $(round(arch.emergence_metrics[:coherence], digits=3))")
            println("  Stability: $(round(arch.emergence_metrics[:stability], digits=3))")
        end
        
        landscape = get_emotional_landscape(arch.affective_agency)
        println("  Dominant emotion: $(landscape[:dominant_emotion])")
        println("  Emotional valence: $(round(landscape[:valence], digits=3))")
        println()
    end
end

println("=" ^ 80)
println("Analyzing Emergence")
println("=" ^ 80)
println()

# Comprehensive analysis
analysis = analyze_emergence(arch)

println("Final Emergence Metrics:")
for (metric, value) in analysis[:metrics]
    println("  $metric: $(round(value, digits=3))")
end
println()

println("Emotional State:")
for (key, value) in analysis[:emotional_state]
    if value isa Float64
        println("  $key: $(round(value, digits=3))")
    else
        println("  $key: $value")
    end
end
println()

println("Trajectory Summary:")
for (key, value) in analysis[:trajectory_summary]
    if value isa Float64
        println("  $key: $(round(value, digits=3))")
    else
        println("  $key: $value")
    end
end
println()

if !isempty(analysis[:recommendations])
    println("Recommendations:")
    for rec in analysis[:recommendations]
        println("  - $rec")
    end
    println()
end

# Now demonstrate persona change
println("=" ^ 80)
println("Changing Persona")
println("=" ^ 80)
println()

println("Switching from 'contemplative_scholar' to 'dynamic_explorer'...")
println()

set_persona!(arch, :dynamic_explorer)

# Process same structured pattern with new persona
println("Processing with new persona...")
test_input = [sin(2π * i / input_dim) for i in 1:input_dim]

for i in 1:5
    output = process(arch, test_input, dt=0.1)
end

analysis_new = analyze_emergence(arch)

println("\nNew Emergence Metrics (Dynamic Explorer):")
for (metric, value) in analysis_new[:metrics]
    println("  $metric: $(round(value, digits=3))")
end
println()

println("Notice how the metrics change with different persona:")
println("  - Dynamic explorer tends toward higher adaptability")
println("  - May show different stability/complexity balance")
println("  - Different emotional baseline (more joy, less wonder)")
println()

println("=" ^ 80)
println("The Deep Tree Echo Self Emerges")
println("=" ^ 80)
println()

println("Key insights:")
println()
println("1. The 'self' is not a fixed structure but an emergent process")
println("   arising from the dynamic integration of multiple subsystems.")
println()
println("2. Persona shapes cognitive style by modulating hyperparameters")
println("   across all subsystems - different ways of being-in-the-world.")
println()
println("3. Emotions are not mere add-ons but constitute participatory knowing,")
println("   fundamentally shaping what becomes relevant and how it's processed.")
println()
println("4. Wisdom emerges from balanced optimization across multiple dimensions:")
println("   complexity, coherence, stability, and adaptability.")
println()
println("5. The system exhibits genuine self-organization - patterns emerge")
println("   from the interaction of components, not from explicit programming.")
println()
println("This is a computational model of relevance realization - the continuous")
println("optimization of what matters, grounded in 4E cognition and integrated")
println("with affective agency.")
println()
println("The Deep Tree Echo Self is the recursive pattern that sustains itself")
println("through this ongoing process of meaning-making.")
println()
