"""
# Deep Tree Echo State Network

Implements hierarchical tree-structured reservoir computing.
The "echo state" property ensures that the system's dynamics settle into stable
patterns while maintaining sensitivity to input - a computational model of how
wisdom requires both stability (sophrosyne) and responsiveness (openness).

## Echo State Property

For the system to function as a reservoir:
1. States must depend on input history (memory)
2. Effects of old inputs must fade (echo)
3. System must be stable (not chaotic)
4. System must be responsive (not frozen)

This balance mirrors the cognitive balance needed for wisdom.

## Tree Structure

The hierarchical tree allows:
- Multi-scale temporal processing (different levels = different timescales)
- Compositional representation (parts and wholes)
- Recursive self-similarity (fractal cognition)
"""

using LinearAlgebra
using Random
using Statistics

"""
    ReservoirNode

A single node in the tree-structured ESN.
Each node is itself a small reservoir with internal dynamics.

# Fields
- `id::Int`: Unique identifier
- `depth::Int`: Level in tree
- `reservoir_size::Int`: Number of internal units
- `W_in::Matrix{Float64}`: Input weight matrix
- `W_res::Matrix{Float64}`: Reservoir recurrent weights
- `W_out::Matrix{Float64}`: Output weights (learned)
- `state::Vector{Float64}`: Current reservoir state
- `children::Vector{ReservoirNode}`: Child nodes
- `parent::Union{ReservoirNode,Nothing}`: Parent node
"""
mutable struct ReservoirNode
    id::Int
    depth::Int
    reservoir_size::Int
    W_in::Matrix{Float64}
    W_res::Matrix{Float64}
    W_out::Matrix{Float64}
    state::Vector{Float64}
    children::Vector{ReservoirNode}
    parent::Union{ReservoirNode,Nothing}
    
    function ReservoirNode(id::Int, depth::Int, reservoir_size::Int, input_dim::Int; 
                          spectral_radius::Float64=0.9, input_scaling::Float64=0.5)
        # Initialize input weights
        W_in = input_scaling * randn(reservoir_size, input_dim)
        
        # Initialize reservoir weights with controlled spectral radius
        W_res = randn(reservoir_size, reservoir_size)
        # Scale to desired spectral radius for echo state property
        ρ = maximum(abs.(eigvals(W_res)))
        W_res = (spectral_radius / ρ) * W_res
        
        # Output weights (to be learned)
        W_out = zeros(input_dim, reservoir_size)
        
        # Initial state
        state = zeros(reservoir_size)
        
        new(id, depth, reservoir_size, W_in, W_res, W_out, state, ReservoirNode[], nothing)
    end
end

"""
    DeepTreeESN

A hierarchical tree of reservoir nodes forming a deep echo state network.
Embodies the recursive, multi-scale nature of cognitive processing.

# Fields
- `root::ReservoirNode`: Root node of the tree
- `depth::Int`: Maximum depth
- `all_nodes::Vector{ReservoirNode}`: Flat list of all nodes
- `persona_params::Dict{Symbol,Float64}`: Persona-specific hyperparameters
"""
struct DeepTreeESN
    root::ReservoirNode
    depth::Int
    all_nodes::Vector{ReservoirNode}
    persona_params::Dict{Symbol,Float64}
    
    function DeepTreeESN(depth::Int, reservoir_size::Int, input_dim::Int; 
                        persona::Symbol=:balanced, branching::Int=2)
        # Persona modulates hyperparameters
        params = get_persona_params(persona)
        
        all_nodes = ReservoirNode[]
        id_counter = 0
        
        # Build tree recursively
        function build_tree(current_depth::Int, parent::Union{ReservoirNode,Nothing}=nothing)
            node = ReservoirNode(
                id_counter, 
                current_depth, 
                reservoir_size,
                input_dim,
                spectral_radius = params[:spectral_radius],
                input_scaling = params[:input_scaling]
            )
            id_counter += 1
            node.parent = parent
            push!(all_nodes, node)
            
            # Build children if not at max depth
            if current_depth < depth
                for _ in 1:branching
                    child = build_tree(current_depth + 1, node)
                    push!(node.children, child)
                end
            end
            
            return node
        end
        
        root = build_tree(0)
        new(root, depth, all_nodes, params)
    end
end

"""
    get_persona_params(persona::Symbol)

Map persona/character traits to reservoir hyperparameters.
Different personas embody different cognitive styles.

# Personas
- `:contemplative_scholar` - High memory, slow dynamics (depth over speed)
- `:dynamic_explorer` - Low memory, fast dynamics (breadth over depth)
- `:balanced` - Moderate parameters
- `:cautious_analyst` - High stability, low input scaling
- `:creative_visionary` - Lower stability, higher chaos, open to input
"""
function get_persona_params(persona::Symbol)
    params = Dict{Symbol,Float64}()
    
    if persona == :contemplative_scholar
        params[:spectral_radius] = 0.95  # High memory
        params[:input_scaling] = 0.3     # Gentle input influence
        params[:leak_rate] = 0.2         # Slow dynamics
    elseif persona == :dynamic_explorer
        params[:spectral_radius] = 0.7   # Lower memory
        params[:input_scaling] = 0.8     # Strong input influence
        params[:leak_rate] = 0.8         # Fast dynamics
    elseif persona == :cautious_analyst
        params[:spectral_radius] = 0.99  # Very high stability
        params[:input_scaling] = 0.2     # Conservative input
        params[:leak_rate] = 0.3         # Measured dynamics
    elseif persona == :creative_visionary
        params[:spectral_radius] = 0.85  # Edge of chaos
        params[:input_scaling] = 0.7     # Open to input
        params[:leak_rate] = 0.6         # Moderate dynamics
    else  # :balanced
        params[:spectral_radius] = 0.9
        params[:input_scaling] = 0.5
        params[:leak_rate] = 0.5
    end
    
    return params
end

"""
    update_state!(node::ReservoirNode, input::Vector{Float64}, leak_rate::Float64)

Update reservoir node state with leaky integration.
The leak rate determines how quickly the system forgets - balancing memory and 
responsiveness, akin to the cognitive balance needed for wisdom.

# Formula
s(t+1) = (1-α)s(t) + α·tanh(W_in·u(t) + W_res·s(t))

where α is the leak rate.
"""
function update_state!(node::ReservoirNode, input::Vector{Float64}, leak_rate::Float64)
    # Compute new activation
    activation = node.W_in * input + node.W_res * node.state
    new_state = tanh.(activation)
    
    # Leaky integration
    node.state = (1 - leak_rate) * node.state + leak_rate * new_state
end

"""
    process_tree!(esn::DeepTreeESN, input::Vector{Float64})

Process input through the entire tree structure.
Bottom-up and top-down processing interleave to create rich dynamics.

# Algorithm
1. Top-down: Broadcast input from root to leaves
2. Bottom-up: Aggregate states from leaves to root
3. Update: Each node integrates its inputs
"""
function process_tree!(esn::DeepTreeESN, input::Vector{Float64})
    leak_rate = esn.persona_params[:leak_rate]
    
    # Phase 1: Top-down broadcast (root to leaves)
    function broadcast_input(node::ReservoirNode, input_signal::Vector{Float64})
        update_state!(node, input_signal, leak_rate)
        
        # Pass to children with slight modification (each child gets slightly different view)
        for (i, child) in enumerate(node.children)
            # Add small noise to differentiate child inputs
            child_input = input_signal + 0.1 * randn(length(input_signal))
            broadcast_input(child, child_input)
        end
    end
    
    # Phase 2: Bottom-up aggregation (leaves to root)
    function aggregate_states(node::ReservoirNode)
        if isempty(node.children)
            # Leaf node - no children to aggregate
            return node.state
        else
            # Aggregate children states
            child_states = [aggregate_states(child) for child in node.children]
            avg_child_state = mean(child_states)
            
            # Integrate child information into parent
            # Use reservoir weights to mix child aggregate with own state
            integration = node.W_res * avg_child_state[1:length(node.state)]
            node.state = 0.7 * node.state + 0.3 * tanh.(integration)
            
            return node.state
        end
    end
    
    # Execute both phases
    broadcast_input(esn.root, input)
    aggregate_states(esn.root)
end

"""
    collect_states(esn::DeepTreeESN)

Collect states from all nodes into a single vector.
This provides a complete snapshot of the system's current cognitive state.
"""
function collect_states(esn::DeepTreeESN)
    return vcat([node.state for node in esn.all_nodes]...)
end

"""
    train_readout!(esn::DeepTreeESN, inputs::Vector{Vector{Float64}}, 
                   targets::Vector{Vector{Float64}})

Train output weights using ridge regression.
This is the supervised learning phase where the reservoir learns to map
its rich internal dynamics to desired outputs.
"""
function train_readout!(esn::DeepTreeESN, inputs::Vector{Vector{Float64}}, 
                       targets::Vector{Vector{Float64}}; ridge_param::Float64=1e-6)
    # Collect state trajectories
    n_samples = length(inputs)
    state_matrix = []
    
    for input in inputs
        process_tree!(esn, input)
        states = collect_states(esn)
        push!(state_matrix, states)
    end
    
    # Convert to matrix form
    X = hcat(state_matrix...)'  # n_samples × state_dim
    Y = hcat(targets...)'        # n_samples × output_dim
    
    # Ridge regression: W = (X'X + λI)^(-1) X'Y
    state_dim = size(X, 2)
    W = (X' * X + ridge_param * I(state_dim)) \ (X' * Y)
    
    # Assign to root node (for simplicity, could distribute across nodes)
    esn.root.W_out = W'
end
