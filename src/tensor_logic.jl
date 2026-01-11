"""
# Tensor Logic: Unified Neural-Symbolic Reasoning

Implementation of Tensor Logic based on the framework by Pedro Domingos (arXiv:2510.12269).
This module provides a unified language for neural and symbolic AI, where the tensor equation
is the sole construct, based on the equivalence of logical rules and Einstein summation.

## Core Principle

Logical rules and Einstein summation are essentially the same operation:
- Relations → sparse Boolean tensors
- Logical joins → tensor products
- Projection/marginalization → summation over indices
- Inference → forward/backward chaining through tensor operations

## Temperature-Controlled Reasoning

The temperature parameter T controls the reasoning modality:
- T = 0: Pure deductive reasoning (sound, no hallucination)
- T > 0: Analogical reasoning (similarity-based generalization)

## Integration with Cognitive Architecture

Tensor Logic provides:
- Propositional reasoning in embedding space
- Gradient-based learning of relations
- Sound inference with neural scalability
- Unification with transformer attention mechanisms

## References

- Domingos, P. (2025). Tensor Logic: The Language of AI. arXiv:2510.12269
- https://tensor-logic.org/
"""

using LinearAlgebra
using Statistics
using Random
using SparseArrays

# ============================================================================
# SECTION 1: TENSOR RELATIONS
# ============================================================================

"""
    TensorRelation

Represents a logical relation as a sparse Boolean tensor.
An n-ary relation becomes an n-rank tensor where element is 1 if relation holds.

# Fields
- `name::Symbol`: Relation name (e.g., :Parent, :Sister)
- `arity::Int`: Number of arguments
- `domain_sizes::Vector{Int}`: Size of each domain
- `tensor::AbstractArray`: The actual tensor (sparse or dense)
- `is_boolean::Bool`: Whether this is a pure Boolean relation
"""
mutable struct TensorRelation
    name::Symbol
    arity::Int
    domain_sizes::Vector{Int}
    tensor::AbstractArray
    is_boolean::Bool

    function TensorRelation(name::Symbol, domain_sizes::Vector{Int}; sparse::Bool=true)
        arity = length(domain_sizes)

        if sparse && arity == 2
            tensor = spzeros(Float64, domain_sizes...)
        else
            tensor = zeros(Float64, domain_sizes...)
        end

        new(name, arity, domain_sizes, tensor, true)
    end
end

"""
    set_tuple!(rel::TensorRelation, indices::Vector{Int}, value::Float64=1.0)

Assert that a tuple holds in the relation.
"""
function set_tuple!(rel::TensorRelation, indices::Vector{Int}, value::Float64=1.0)
    @assert length(indices) == rel.arity "Indices must match relation arity"
    rel.tensor[indices...] = value
end

"""
    get_tuple(rel::TensorRelation, indices::Vector{Int})

Query whether a tuple holds in the relation.
"""
function get_tuple(rel::TensorRelation, indices::Vector{Int})
    @assert length(indices) == rel.arity "Indices must match relation arity"
    return rel.tensor[indices...]
end

"""
    from_tuples(name::Symbol, domain_sizes::Vector{Int}, tuples::Vector{Vector{Int}})

Create a TensorRelation from a list of tuples.
"""
function from_tuples(name::Symbol, domain_sizes::Vector{Int}, tuples::Vector{Vector{Int}})
    rel = TensorRelation(name, domain_sizes, sparse=false)

    for tuple in tuples
        set_tuple!(rel, tuple, 1.0)
    end

    return rel
end

# ============================================================================
# SECTION 2: EINSTEIN SUMMATION OPERATIONS
# ============================================================================

"""
    EinsumSpec

Specification for an Einstein summation operation.

# Fields
- `input_indices::Vector{Vector{Symbol}}`: Indices for each input tensor
- `output_indices::Vector{Symbol}`: Indices in the result
- `contraction_indices::Vector{Symbol}`: Indices summed over (implicit)
"""
struct EinsumSpec
    input_indices::Vector{Vector{Symbol}}
    output_indices::Vector{Symbol}
    contraction_indices::Vector{Symbol}

    function EinsumSpec(inputs::Vector{Vector{Symbol}}, output::Vector{Symbol})
        # Find contraction indices (appear in inputs but not output)
        all_input_indices = unique(vcat(inputs...))
        contraction = setdiff(all_input_indices, output)

        new(inputs, output, contraction)
    end
end

"""
    tensor_project(T::AbstractArray, keep_dims::Vector{Int})

Project (marginalize) tensor by summing over dimensions not in keep_dims.
Implements: π_α(T) = Σ_β T_{αβ}
"""
function tensor_project(T::AbstractArray, keep_dims::Vector{Int})
    all_dims = collect(1:ndims(T))
    sum_dims = setdiff(all_dims, keep_dims)

    if isempty(sum_dims)
        return T
    end

    result = T
    # Sum over dimensions in reverse order to maintain indexing
    for dim in sort(sum_dims, rev=true)
        result = dropdims(sum(result, dims=dim), dims=dim)
    end

    return result
end

"""
    tensor_join(U::AbstractArray, V::AbstractArray,
                u_shared::Vector{Int}, v_shared::Vector{Int})

Natural join of two tensors over shared indices.
Implements: (U ⨝ V)_{αβγ} = U_{αβ} · V_{βγ}

# Arguments
- `U`, `V`: Input tensors
- `u_shared`: Dimensions of U that are shared
- `v_shared`: Dimensions of V that are shared (must align with u_shared)
"""
function tensor_join(U::AbstractArray, V::AbstractArray,
                    u_shared::Vector{Int}, v_shared::Vector{Int})
    @assert length(u_shared) == length(v_shared) "Shared dimensions must match"

    if isempty(u_shared)
        # No shared dimensions - outer product
        return reshape(U, size(U)..., ones(Int, ndims(V))...) .*
               reshape(V, ones(Int, ndims(U))..., size(V)...)
    end

    # For 2D case (most common), use matrix multiplication
    if ndims(U) == 2 && ndims(V) == 2 && u_shared == [2] && v_shared == [1]
        return U * V
    end

    # General einsum-style contraction
    # This is a simplified implementation; full einsum would be more general
    result_size = [size(U)..., size(V)...]
    # Remove contracted dimensions
    for (ui, vi) in zip(reverse(u_shared), reverse(v_shared))
        deleteat!(result_size, ndims(U) + vi)
    end

    # Compute via tensor contraction
    # Simplified: assume contraction over last dim of U and first dim of V
    if u_shared == [ndims(U)] && v_shared == [1]
        # Standard matrix-style multiplication generalized
        return _contract_last_first(U, V)
    end

    # Fallback: explicit loop (inefficient but correct)
    return _explicit_contraction(U, V, u_shared, v_shared)
end

"""
    _contract_last_first(U::AbstractArray, V::AbstractArray)

Contract last dimension of U with first dimension of V.
"""
function _contract_last_first(U::AbstractArray, V::AbstractArray)
    @assert size(U, ndims(U)) == size(V, 1) "Contraction dimensions must match"

    u_shape = size(U)
    v_shape = size(V)

    # Reshape for matrix multiplication
    U_2d = reshape(U, :, u_shape[end])
    V_2d = reshape(V, v_shape[1], :)

    result_2d = U_2d * V_2d

    # Reshape back
    result_shape = (u_shape[1:end-1]..., v_shape[2:end]...)
    return reshape(result_2d, result_shape)
end

"""
    _explicit_contraction(U, V, u_shared, v_shared)

Fallback explicit contraction (slow but general).
"""
function _explicit_contraction(U::AbstractArray, V::AbstractArray,
                               u_shared::Vector{Int}, v_shared::Vector{Int})
    # For now, handle the 2D binary relation case
    if ndims(U) == 2 && ndims(V) == 2
        m, n = size(U)
        n2, p = size(V)
        @assert n == n2 "Inner dimensions must match"

        result = zeros(m, p)
        for i in 1:m
            for j in 1:p
                for k in 1:n
                    result[i, j] += U[i, k] * V[k, j]
                end
            end
        end
        return result
    end

    error("General tensor contraction not yet implemented for ndims > 2")
end

# ============================================================================
# SECTION 3: LOGICAL OPERATIONS AS TENSOR OPS
# ============================================================================

"""
    heaviside_step(x::Float64)

Heaviside step function: 1 if x > 0, else 0.
Used to convert continuous to Boolean in tensor logic.
"""
heaviside_step(x::Float64) = x > 0 ? 1.0 : 0.0
heaviside_step(x::Real) = heaviside_step(Float64(x))

"""
    soft_step(x::Float64, temperature::Float64)

Soft (differentiable) step function with temperature control.
- T → 0: Approaches hard step
- T > 0: Smooth sigmoid

σ(x, T) = 1 / (1 + e^{-x/T})
"""
function soft_step(x::Float64, temperature::Float64)
    if temperature < 1e-10
        return heaviside_step(x)
    end
    return 1.0 / (1.0 + exp(-x / temperature))
end

"""
    apply_rule(premises::Vector{TensorRelation},
               shared_indices::Vector{Tuple{Int,Int,Int,Int}},
               temperature::Float64=0.0)

Apply a logical rule by joining premises and applying step function.

Example: Aunt[x,z] = step(Sister[x,y] · Parent[y,z])

# Arguments
- `premises`: Vector of relations to join
- `shared_indices`: Tuples of (rel1_idx, dim1, rel2_idx, dim2) for joins
- `temperature`: Reasoning temperature (0 = pure deduction)
"""
function apply_rule(premises::Vector{TensorRelation},
                   shared_indices::Vector{Tuple{Int,Int,Int,Int}},
                   temperature::Float64=0.0)

    if length(premises) == 0
        error("At least one premise required")
    end

    if length(premises) == 1
        # Single premise, just apply step
        result = copy(premises[1].tensor)
        if temperature < 1e-10
            return heaviside_step.(result)
        else
            return soft_step.(result, temperature)
        end
    end

    # Binary case: join two relations
    if length(premises) == 2
        rel1, rel2 = premises
        # Find shared dimensions from shared_indices
        u_shared = Int[]
        v_shared = Int[]
        for (r1, d1, r2, d2) in shared_indices
            if r1 == 1 && r2 == 2
                push!(u_shared, d1)
                push!(v_shared, d2)
            elseif r1 == 2 && r2 == 1
                push!(u_shared, d2)
                push!(v_shared, d1)
            end
        end

        if isempty(u_shared)
            # Default: join on last dim of first, first dim of second
            u_shared = [rel1.arity]
            v_shared = [1]
        end

        joined = tensor_join(rel1.tensor, rel2.tensor, u_shared, v_shared)

        # Apply step function
        if temperature < 1e-10
            return heaviside_step.(joined)
        else
            return soft_step.(joined, temperature)
        end
    end

    # Multi-way join: chain binary joins
    result = premises[1].tensor
    for i in 2:length(premises)
        # Find shared indices for this pair
        pair_shared = [(s[2], s[4]) for s in shared_indices
                       if (s[1] == i-1 && s[3] == i) || (s[1] == i && s[3] == i-1)]

        u_shared = isempty(pair_shared) ? [ndims(result)] : [p[1] for p in pair_shared]
        v_shared = isempty(pair_shared) ? [1] : [p[2] for p in pair_shared]

        result = tensor_join(result, premises[i].tensor, u_shared, v_shared)
    end

    # Apply step function
    if temperature < 1e-10
        return heaviside_step.(result)
    else
        return soft_step.(result, temperature)
    end
end

# ============================================================================
# SECTION 4: EMBEDDING SPACE REASONING
# ============================================================================

"""
    EmbeddingSpace

Embedding space for entities enabling similarity-based reasoning.

# Fields
- `dimension::Int`: Embedding dimension D
- `entity_embeddings::Matrix{Float64}`: Embeddings (n_entities × D)
- `entity_names::Vector{Symbol}`: Entity names
- `gram_matrix::Matrix{Float64}`: Similarity matrix (cached)
"""
mutable struct EmbeddingSpace
    dimension::Int
    entity_embeddings::Matrix{Float64}
    entity_names::Vector{Symbol}
    gram_matrix::Matrix{Float64}

    function EmbeddingSpace(n_entities::Int, dimension::Int;
                           init_scale::Float64=0.1)
        embeddings = init_scale * randn(n_entities, dimension)
        # Normalize embeddings
        for i in 1:n_entities
            embeddings[i, :] ./= norm(embeddings[i, :]) + 1e-10
        end

        names = [Symbol("e$i") for i in 1:n_entities]
        gram = embeddings * embeddings'

        new(dimension, embeddings, names, gram)
    end
end

"""
    set_entity_name!(space::EmbeddingSpace, idx::Int, name::Symbol)

Set the name for an entity.
"""
function set_entity_name!(space::EmbeddingSpace, idx::Int, name::Symbol)
    space.entity_names[idx] = name
end

"""
    get_entity_idx(space::EmbeddingSpace, name::Symbol)

Get entity index by name.
"""
function get_entity_idx(space::EmbeddingSpace, name::Symbol)
    return findfirst(==(name), space.entity_names)
end

"""
    update_gram_matrix!(space::EmbeddingSpace)

Recompute the Gram (similarity) matrix after embedding changes.
"""
function update_gram_matrix!(space::EmbeddingSpace)
    space.gram_matrix = space.entity_embeddings * space.entity_embeddings'
end

"""
    similarity(space::EmbeddingSpace, i::Int, j::Int)

Get similarity between entities i and j.
"""
function similarity(space::EmbeddingSpace, i::Int, j::Int)
    return space.gram_matrix[i, j]
end

"""
    embed_relation(rel::TensorRelation, space::EmbeddingSpace)

Embed a relation into the embedding space.
EmbR[i,j] = Σ_{x,y} R(x,y) · Emb[x,i] · Emb[y,j]

This creates a "superposition" of all tuples in embedding space.
"""
function embed_relation(rel::TensorRelation, space::EmbeddingSpace)
    @assert rel.arity == 2 "Currently only binary relations supported"

    E = space.entity_embeddings
    R = rel.tensor

    # EmbR = E' * R * E (for binary relations)
    return E' * R * E
end

"""
    query_embedded_relation(embR::Matrix{Float64}, space::EmbeddingSpace,
                           query_entity::Int, position::Int)

Query an embedded relation for a specific entity.
Returns similarity scores for all possible answers.
"""
function query_embedded_relation(embR::Matrix{Float64}, space::EmbeddingSpace,
                                query_entity::Int, position::Int)
    E = space.entity_embeddings
    query_emb = E[query_entity, :]

    if position == 1
        # Query: R(query_entity, ?)
        # D[y] = embR[i,j] · Emb[query,i] · Emb[y,j]
        intermediate = query_emb' * embR
        return vec(E * intermediate')
    else
        # Query: R(?, query_entity)
        # D[x] = embR[i,j] · Emb[x,i] · Emb[query,j]
        intermediate = embR * query_emb
        return vec(E * intermediate)
    end
end

"""
    analogical_inference(rel::TensorRelation, space::EmbeddingSpace,
                        query::Vector{Int}, temperature::Float64)

Perform analogical inference: similar entities share inferences.

At T=0: Pure deductive lookup
At T>0: Weighted by similarity from Gram matrix
"""
function analogical_inference(rel::TensorRelation, space::EmbeddingSpace,
                             query::Vector{Int}, temperature::Float64)
    @assert rel.arity == length(query) "Query must match relation arity"

    # Direct lookup
    direct_result = get_tuple(rel, query)

    if temperature < 1e-10
        # Pure deduction
        return direct_result
    end

    # Analogical reasoning: weight by similarity
    result = direct_result

    # For each query position, consider similar entities
    for pos in 1:rel.arity
        similar_entities = space.gram_matrix[query[pos], :]

        for other in 1:length(similar_entities)
            if other != query[pos]
                alt_query = copy(query)
                alt_query[pos] = other

                alt_result = get_tuple(rel, alt_query)
                # Weight by similarity and temperature
                weight = soft_step(similar_entities[other], temperature)
                result += weight * alt_result * (1.0 - exp(-temperature))
            end
        end
    end

    # Apply soft step to result
    return soft_step(result, temperature)
end

# ============================================================================
# SECTION 5: INFERENCE ENGINES
# ============================================================================

"""
    Rule

A logical rule in tensor logic.

# Fields
- `head::Symbol`: Conclusion relation name
- `premises::Vector{Symbol}`: Premise relation names
- `join_spec::Vector{Tuple{Int,Int,Int,Int}}`: How premises join
- `temperature::Float64`: Inference temperature
"""
struct Rule
    head::Symbol
    premises::Vector{Symbol}
    join_spec::Vector{Tuple{Int,Int,Int,Int}}
    temperature::Float64

    Rule(head::Symbol, premises::Vector{Symbol};
         join_spec::Vector{Tuple{Int,Int,Int,Int}}=Tuple{Int,Int,Int,Int}[],
         temperature::Float64=0.0) = new(head, premises, join_spec, temperature)
end

"""
    KnowledgeBase

A knowledge base of relations and rules.

# Fields
- `relations::Dict{Symbol,TensorRelation}`: Named relations
- `rules::Vector{Rule}`: Inference rules
- `embedding_space::Union{EmbeddingSpace,Nothing}`: Optional embeddings
- `default_temperature::Float64`: Default reasoning temperature
"""
mutable struct KnowledgeBase
    relations::Dict{Symbol,TensorRelation}
    rules::Vector{Rule}
    embedding_space::Union{EmbeddingSpace,Nothing}
    default_temperature::Float64

    KnowledgeBase(temperature::Float64=0.0) =
        new(Dict{Symbol,TensorRelation}(), Rule[], nothing, temperature)
end

"""
    add_relation!(kb::KnowledgeBase, rel::TensorRelation)

Add a relation to the knowledge base.
"""
function add_relation!(kb::KnowledgeBase, rel::TensorRelation)
    kb.relations[rel.name] = rel
end

"""
    add_rule!(kb::KnowledgeBase, rule::Rule)

Add an inference rule to the knowledge base.
"""
function add_rule!(kb::KnowledgeBase, rule::Rule)
    push!(kb.rules, rule)
end

"""
    forward_chain!(kb::KnowledgeBase; max_iterations::Int=100)

Forward chaining inference: derive all consequences.
Iterates until fixpoint or max iterations.
"""
function forward_chain!(kb::KnowledgeBase; max_iterations::Int=100)
    for iter in 1:max_iterations
        changed = false

        for rule in kb.rules
            # Get premise relations
            premises = [kb.relations[p] for p in rule.premises if haskey(kb.relations, p)]

            if length(premises) != length(rule.premises)
                continue  # Missing premises
            end

            # Apply rule
            temp = rule.temperature > 0 ? rule.temperature : kb.default_temperature
            result = apply_rule(premises, rule.join_spec, temp)

            # Create or update head relation
            if !haskey(kb.relations, rule.head)
                # Infer domain sizes from result
                domain_sizes = collect(size(result))
                kb.relations[rule.head] = TensorRelation(rule.head, domain_sizes, sparse=false)
                kb.relations[rule.head].tensor = result
                changed = true
            else
                old_tensor = kb.relations[rule.head].tensor
                # Union (max) with existing
                new_tensor = max.(old_tensor, result)
                if !all(new_tensor .== old_tensor)
                    kb.relations[rule.head].tensor = new_tensor
                    changed = true
                end
            end
        end

        if !changed
            break  # Fixpoint reached
        end
    end

    return kb
end

"""
    backward_chain(kb::KnowledgeBase, query_rel::Symbol, query_tuple::Vector{Int};
                  depth::Int=10)

Backward chaining inference: derive specific query.
"""
function backward_chain(kb::KnowledgeBase, query_rel::Symbol, query_tuple::Vector{Int};
                       depth::Int=10)
    if depth <= 0
        return 0.0
    end

    # Direct lookup
    if haskey(kb.relations, query_rel)
        rel = kb.relations[query_rel]
        if all(1 .<= query_tuple .<= rel.domain_sizes)
            direct = get_tuple(rel, query_tuple)
            if direct > 0
                return direct
            end
        end
    end

    # Try to derive via rules
    for rule in kb.rules
        if rule.head != query_rel
            continue
        end

        # This is a simplified backward chaining
        # Full implementation would need unification
        premises = [kb.relations[p] for p in rule.premises if haskey(kb.relations, p)]

        if length(premises) == length(rule.premises)
            temp = rule.temperature > 0 ? rule.temperature : kb.default_temperature
            result = apply_rule(premises, rule.join_spec, temp)

            if all(1 .<= query_tuple .<= size(result))
                derived = result[query_tuple...]
                if derived > 0
                    return derived
                end
            end
        end
    end

    return 0.0
end

# ============================================================================
# SECTION 6: NEURAL NETWORK OPERATIONS IN TENSOR LOGIC
# ============================================================================

"""
    tensor_mlp_layer(input::Vector{Float64}, weights::Matrix{Float64},
                    bias::Vector{Float64}, activation::Function)

Single MLP layer in tensor logic notation:
Y = activation(W[i,j] · X[j] + B[i])
"""
function tensor_mlp_layer(input::Vector{Float64}, weights::Matrix{Float64},
                         bias::Vector{Float64}, activation::Function)
    pre_activation = weights * input + bias
    return activation.(pre_activation)
end

"""
    tensor_attention(query::Matrix{Float64}, key::Matrix{Float64},
                    value::Matrix{Float64}, temperature::Float64=1.0)

Attention mechanism in tensor logic:
Attn[p,d_v] = softmax(Q[p,d_k] · K[p',d_k] / √D_k) · V[p',d_v]
"""
function tensor_attention(query::Matrix{Float64}, key::Matrix{Float64},
                         value::Matrix{Float64}, temperature::Float64=1.0)
    d_k = size(key, 2)
    scale = 1.0 / sqrt(d_k)

    # Compute attention scores
    scores = (query * key') * scale

    # Apply temperature
    if temperature != 1.0
        scores ./= temperature
    end

    # Softmax over key positions
    scores_max = maximum(scores, dims=2)
    exp_scores = exp.(scores .- scores_max)
    attention_weights = exp_scores ./ sum(exp_scores, dims=2)

    # Apply to values
    output = attention_weights * value

    return output, attention_weights
end

"""
    tensor_layer_norm(x::Vector{Float64}; ε::Float64=1e-5)

Layer normalization as tensor operation.
"""
function tensor_layer_norm(x::Vector{Float64}; ε::Float64=1e-5)
    μ = mean(x)
    σ² = var(x)
    return (x .- μ) ./ sqrt(σ² + ε)
end

"""
    tensor_softmax(x::Vector{Float64}, temperature::Float64=1.0)

Softmax with temperature control.
"""
function tensor_softmax(x::Vector{Float64}, temperature::Float64=1.0)
    scaled = x ./ temperature
    shifted = scaled .- maximum(scaled)
    exp_x = exp.(shifted)
    return exp_x ./ sum(exp_x)
end

# ============================================================================
# SECTION 7: GRADIENT COMPUTATION (AUTODIFF SUPPORT)
# ============================================================================

"""
    TensorGradient

Structure for tracking gradients through tensor operations.
"""
mutable struct TensorGradient
    tensor::AbstractArray
    gradient::AbstractArray
    requires_grad::Bool

    TensorGradient(tensor::AbstractArray; requires_grad::Bool=true) =
        new(tensor, zeros(size(tensor)), requires_grad)
end

"""
    backward!(grad::TensorGradient, upstream::AbstractArray)

Accumulate gradient from upstream.
"""
function backward!(grad::TensorGradient, upstream::AbstractArray)
    if grad.requires_grad
        grad.gradient .+= upstream
    end
end

"""
    tensor_mse_loss(predicted::Vector{Float64}, target::Vector{Float64})

Mean squared error loss.
Loss = (Y[i] - Target[i])²
"""
function tensor_mse_loss(predicted::Vector{Float64}, target::Vector{Float64})
    return mean((predicted .- target).^2)
end

"""
    tensor_mse_gradient(predicted::Vector{Float64}, target::Vector{Float64})

Gradient of MSE loss with respect to predictions.
"""
function tensor_mse_gradient(predicted::Vector{Float64}, target::Vector{Float64})
    n = length(predicted)
    return 2.0 * (predicted .- target) / n
end

# ============================================================================
# SECTION 8: INTEGRATION WITH COGNITIVE ARCHITECTURE
# ============================================================================

"""
    TensorLogicReasoner

Integration of tensor logic with the cognitive architecture.
Provides symbolic reasoning capabilities with neural compatibility.

# Fields
- `kb::KnowledgeBase`: The knowledge base
- `embedding_dim::Int`: Embedding dimension
- `temperature::Float64`: Current reasoning temperature
- `inference_history::Vector{Dict{Symbol,Any}}`: History of inferences
"""
mutable struct TensorLogicReasoner
    kb::KnowledgeBase
    embedding_dim::Int
    temperature::Float64
    inference_history::Vector{Dict{Symbol,Any}}

    function TensorLogicReasoner(;
        embedding_dim::Int=64,
        temperature::Float64=0.0,
        n_entities::Int=100
    )
        kb = KnowledgeBase(temperature)
        kb.embedding_space = EmbeddingSpace(n_entities, embedding_dim)

        new(kb, embedding_dim, temperature, Dict{Symbol,Any}[])
    end
end

"""
    define_relation!(reasoner::TensorLogicReasoner, name::Symbol,
                    domain_sizes::Vector{Int})

Define a new relation in the reasoner.
"""
function define_relation!(reasoner::TensorLogicReasoner, name::Symbol,
                         domain_sizes::Vector{Int})
    rel = TensorRelation(name, domain_sizes)
    add_relation!(reasoner.kb, rel)
    return rel
end

"""
    assert_fact!(reasoner::TensorLogicReasoner, rel_name::Symbol,
                tuple::Vector{Int})

Assert a fact (ground atom) in the knowledge base.
"""
function assert_fact!(reasoner::TensorLogicReasoner, rel_name::Symbol,
                     tuple::Vector{Int})
    if haskey(reasoner.kb.relations, rel_name)
        set_tuple!(reasoner.kb.relations[rel_name], tuple, 1.0)
    end
end

"""
    define_rule!(reasoner::TensorLogicReasoner, head::Symbol,
                premises::Vector{Symbol};
                join_spec::Vector{Tuple{Int,Int,Int,Int}}=Tuple{Int,Int,Int,Int}[])

Define an inference rule.
"""
function define_rule!(reasoner::TensorLogicReasoner, head::Symbol,
                     premises::Vector{Symbol};
                     join_spec::Vector{Tuple{Int,Int,Int,Int}}=Tuple{Int,Int,Int,Int}[])
    rule = Rule(head, premises, join_spec=join_spec, temperature=reasoner.temperature)
    add_rule!(reasoner.kb, rule)
end

"""
    reason!(reasoner::TensorLogicReasoner; mode::Symbol=:forward)

Perform inference in the knowledge base.
"""
function reason!(reasoner::TensorLogicReasoner; mode::Symbol=:forward)
    if mode == :forward
        forward_chain!(reasoner.kb)
    end

    # Record inference step
    snapshot = Dict{Symbol,Any}(
        :mode => mode,
        :temperature => reasoner.temperature,
        :n_relations => length(reasoner.kb.relations),
        :n_rules => length(reasoner.kb.rules)
    )
    push!(reasoner.inference_history, snapshot)

    return reasoner
end

"""
    query(reasoner::TensorLogicReasoner, rel_name::Symbol, tuple::Vector{Int})

Query a relation for a specific tuple.
"""
function query(reasoner::TensorLogicReasoner, rel_name::Symbol, tuple::Vector{Int})
    if reasoner.temperature > 0 && reasoner.kb.embedding_space !== nothing
        # Analogical reasoning
        rel = reasoner.kb.relations[rel_name]
        return analogical_inference(rel, reasoner.kb.embedding_space,
                                   tuple, reasoner.temperature)
    else
        # Pure deduction
        return backward_chain(reasoner.kb, rel_name, tuple)
    end
end

"""
    set_temperature!(reasoner::TensorLogicReasoner, temperature::Float64)

Set the reasoning temperature.
T=0: Pure deduction
T>0: Analogical reasoning
"""
function set_temperature!(reasoner::TensorLogicReasoner, temperature::Float64)
    reasoner.temperature = temperature
    reasoner.kb.default_temperature = temperature
end

"""
    integrate_with_attention(reasoner::TensorLogicReasoner,
                            attention_weights::Matrix{Float64},
                            rel_name::Symbol)

Integrate tensor logic reasoning with attention mechanism.
Uses attention to weight relation queries.
"""
function integrate_with_attention(reasoner::TensorLogicReasoner,
                                 attention_weights::Matrix{Float64},
                                 rel_name::Symbol)
    if !haskey(reasoner.kb.relations, rel_name)
        return zeros(size(attention_weights, 1))
    end

    rel = reasoner.kb.relations[rel_name]

    # Weight relation by attention
    weighted_rel = attention_weights' * rel.tensor

    return weighted_rel
end

# ============================================================================
# SECTION 9: EXPORTS
# ============================================================================

export TensorRelation, set_tuple!, get_tuple, from_tuples
export EinsumSpec, tensor_project, tensor_join
export heaviside_step, soft_step, apply_rule
export EmbeddingSpace, set_entity_name!, get_entity_idx
export update_gram_matrix!, similarity, embed_relation
export query_embedded_relation, analogical_inference
export Rule, KnowledgeBase, add_relation!, add_rule!
export forward_chain!, backward_chain
export tensor_mlp_layer, tensor_attention, tensor_layer_norm, tensor_softmax
export TensorGradient, backward!, tensor_mse_loss, tensor_mse_gradient
export TensorLogicReasoner, define_relation!, assert_fact!, define_rule!
export reason!, query, set_temperature!, integrate_with_attention
