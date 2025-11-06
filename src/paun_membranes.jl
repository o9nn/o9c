"""
# Paun P-System Membrane Reservoirs

Implements hierarchical membrane computing structures inspired by Paun P-systems.
Each membrane represents a boundary in cognitive space that filters and transforms
information flow - embodying the principle that cognition is fundamentally about
managing boundaries and relevance.

## Membrane Dynamics

Membranes are not static containers but dynamic boundaries that:
1. **Filter** - Selectively permit information passage (relevance realization)
2. **Transform** - Apply rules that modify information structure
3. **Communicate** - Exchange with parent/child/sibling membranes
4. **Evolve** - Adapt permeability based on history

This captures the embedded and enacted aspects of 4E cognition.
"""

using ModelingToolkit
using DifferentialEquations
using Graphs
using LinearAlgebra
using Random

"""
    Membrane

Represents a single membrane in the P-system hierarchy.

# Fields
- `id::Int`: Unique identifier
- `depth::Int`: Level in the hierarchy (0 = outermost)
- `permeability::Float64`: How readily information crosses (0-1)
- `state::Vector{Float64}`: Internal state vector
- `rules::Vector{Function}`: Transformation rules
- `children::Vector{Int}`: Child membrane IDs
- `parent::Union{Int,Nothing}`: Parent membrane ID
"""
mutable struct Membrane
    id::Int
    depth::Int
    permeability::Float64
    state::Vector{Float64}
    rules::Vector{Function}
    children::Vector{Int}
    parent::Union{Int,Nothing}
    
    function Membrane(id::Int, depth::Int, dim::Int; permeability::Float64=0.5)
        new(id, depth, permeability, randn(dim), Function[], Int[], nothing)
    end
end

"""
    PaunMembraneSystem

A hierarchical system of membranes forming a tree structure.
Embodies the nested, multi-scale nature of cognitive processing.

# Fields
- `membranes::Dict{Int,Membrane}`: All membranes indexed by ID
- `root_id::Int`: ID of the outermost membrane
- `depth::Int`: Maximum depth of the tree
- `graph::SimpleDiGraph`: Tree structure representation
"""
struct PaunMembraneSystem
    membranes::Dict{Int,Membrane}
    root_id::Int
    depth::Int
    graph::SimpleDiGraph
    
    function PaunMembraneSystem(depth::Int, dim::Int, branching::Int=2)
        membranes = Dict{Int,Membrane}()
        graph = SimpleDiGraph()
        
        # Build tree structure
        id_counter = 0
        root_id = id_counter
        queue = [(root_id, 0)]  # (id, depth)
        
        while !isempty(queue)
            current_id, current_depth = popfirst!(queue)
            
            # Create membrane
            m = Membrane(current_id, current_depth, dim)
            membranes[current_id] = m
            add_vertex!(graph)
            
            # Create children if not at max depth
            if current_depth < depth
                for _ in 1:branching
                    id_counter += 1
                    child_id = id_counter
                    m.children = push!(m.children, child_id)
                    push!(queue, (child_id, current_depth + 1))
                end
            end
        end
        
        # Set parent relationships and graph edges
        for (id, membrane) in membranes
            for child_id in membrane.children
                membranes[child_id].parent = id
                add_edge!(graph, id + 1, child_id + 1)  # Graphs.jl uses 1-indexing
            end
        end
        
        new(membranes, root_id, depth, graph)
    end
end

"""
    apply_membrane_rules!(system::PaunMembraneSystem, dt::Float64)

Apply transformation rules within each membrane and handle inter-membrane communication.
This is where the relevance realization filtering occurs.
"""
function apply_membrane_rules!(system::PaunMembraneSystem, dt::Float64)
    # Process from leaves to root (bottom-up integration)
    for d in system.depth:-1:0
        membranes_at_depth = [m for (id, m) in system.membranes if m.depth == d]
        
        for membrane in membranes_at_depth
            # Apply internal transformation rules
            for rule in membrane.rules
                membrane.state = rule(membrane.state)
            end
            
            # Communicate with parent (if exists)
            if membrane.parent !== nothing
                parent = system.membranes[membrane.parent]
                # Information flow modulated by permeability
                flow = membrane.permeability * membrane.state
                parent.state .+= dt * flow / length(parent.children)
            end
        end
    end
end

"""
    add_default_rules!(system::PaunMembraneSystem)

Add default transformation rules that embody relevance realization dynamics.
These rules implement:
- Normalization (maintaining bounded activation)
- Nonlinear transformation (enabling complex dynamics)
- Lateral inhibition (competition for relevance)
"""
function add_default_rules!(system::PaunMembraneSystem)
    for (id, membrane) in system.membranes
        # Rule 1: Tanh activation (bounded, nonlinear)
        push!(membrane.rules, x -> tanh.(x))
        
        # Rule 2: Soft normalization
        push!(membrane.rules, x -> x ./ (1.0 .+ norm(x)))
        
        # Rule 3: Sparse coding (lateral inhibition)
        push!(membrane.rules, x -> begin
            threshold = 0.3
            return x .* (abs.(x) .> threshold)
        end)
    end
end

"""
    modulate_permeability!(system::PaunMembraneSystem, emotion_state::Dict{Symbol,Float64})

Modulate membrane permeability based on emotional state.
This embodies how affect shapes cognitive boundaries and information flow.

Different emotions affect openness:
- Wonder, curiosity → increase permeability (openness)
- Fear, anxiety → decrease permeability (defensive closure)
- Joy → increase permeability selectively
- Sadness → decrease permeability
"""
function modulate_permeability!(system::PaunMembraneSystem, emotion_state::Dict{Symbol,Float64})
    # Calculate overall affective valence
    openness_emotions = get(emotion_state, :wonder, 0.0) + 
                       get(emotion_state, :curiosity, 0.0) + 
                       get(emotion_state, :joy, 0.0)
    closure_emotions = get(emotion_state, :fear, 0.0) + 
                      get(emotion_state, :anxiety, 0.0) + 
                      get(emotion_state, :sadness, 0.0)
    
    net_openness = openness_emotions - closure_emotions
    
    # Modulate each membrane
    for (id, membrane) in system.membranes
        # Base permeability varies with depth (deeper = more selective)
        base = 0.7 - 0.1 * membrane.depth
        # Emotional modulation
        membrane.permeability = clamp(base + 0.1 * net_openness, 0.1, 0.9)
    end
end

