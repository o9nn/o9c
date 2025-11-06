"""
# J-Surface Elementary Differentials

Implements geometric structures representing the space of possible cognitive 
trajectories. The J-surface is a manifold in the space of differential operations
that captures the landscape of relevance realization possibilities.

## Geometric Interpretation

The J-surface provides a geometric view of cognition:
- **Points on surface**: Possible cognitive states
- **Tangent vectors**: Directions of change
- **Curvature**: Constraints on transformation
- **Geodesics**: Optimal paths (wisdom trajectories)

This embodies the perspectival knowing - how we see depends on where we stand
in this space.

## Elementary Differentials

Elementary differentials are the basic building blocks of change, corresponding
to rooted trees in Butcher series. The J-surface organizes these into a coherent
geometric structure.
"""

using LinearAlgebra
using Statistics
using ModelingToolkit
using Symbolics

"""
    ElementaryDifferential

Represents a single elementary differential operation.
These are the atomic units of cognitive change.

# Fields
- `order::Int`: Order of the differential
- `tree_id::Int`: Associated rooted tree ID
- `coefficient::Float64`: Weight in the differential
- `dimension::Int`: Which dimension it operates on
"""
struct ElementaryDifferential
    order::Int
    tree_id::Int
    coefficient::Float64
    dimension::Int
end

"""
    JSurfaceDifferential

The J-surface structure organizing elementary differentials into a geometric manifold.
This is the landscape through which cognitive trajectories flow.

# Fields
- `dimension::Int`: Dimensionality of the state space
- `differentials::Vector{ElementaryDifferential}`: All elementary differentials
- `metric::Matrix{Float64}`: Riemannian metric on the surface
- `curvature::Float64`: Mean curvature of the surface
- `critical_points::Vector{Vector{Float64}}`: Attractors/repellers
"""
mutable struct JSurfaceDifferential
    dimension::Int
    differentials::Vector{ElementaryDifferential}
    metric::Matrix{Float64}
    curvature::Float64
    critical_points::Vector{Vector{Float64}}
    
    function JSurfaceDifferential(dimension::Int, max_order::Int)
        # Generate elementary differentials up to max_order
        differentials = ElementaryDifferential[]
        tree_id = 0
        
        for order in 1:max_order
            for dim in 1:dimension
                # Create differential for this order and dimension
                coeff = 1.0 / factorial(order)
                diff = ElementaryDifferential(order, tree_id, coeff, dim)
                push!(differentials, diff)
                tree_id += 1
            end
        end
        
        # Initialize Riemannian metric (identity for now, can be learned)
        metric = Matrix{Float64}(I, dimension, dimension)
        
        # Curvature starts at zero (flat space)
        curvature = 0.0
        
        # Critical points (attractors) - to be discovered
        critical_points = Vector{Float64}[]
        
        new(dimension, differentials, metric, curvature, critical_points)
    end
end

"""
    compute_metric!(surface::JSurfaceDifferential, states::Vector{Vector{Float64}})

Compute the Riemannian metric from observed state trajectories.
The metric encodes how "distance" is measured in cognitive space.

States that co-occur frequently are "closer" in this metric, even if
Euclidean distance is large. This captures the participatory nature of knowing.
"""
function compute_metric!(surface::JSurfaceDifferential, states::Vector{Vector{Float64}})
    n_states = length(states)
    dim = surface.dimension
    
    # Compute covariance matrix of states
    state_matrix = hcat(states...)'
    μ = mean(state_matrix, dims=1)
    centered = state_matrix .- μ
    cov_matrix = (centered' * centered) / n_states
    
    # Metric is inverse of covariance (Fisher information metric)
    # Add regularization for numerical stability
    ε = 1e-6
    surface.metric = inv(cov_matrix + ε * I(dim))
    
    # Compute mean curvature (trace of second fundamental form)
    # Simplified: use trace of metric tensor
    surface.curvature = tr(surface.metric) / dim
end

"""
    geodesic_distance(surface::JSurfaceDifferential, x::Vector{Float64}, y::Vector{Float64})

Compute geodesic distance between two points on the J-surface.
This is the "true" distance accounting for the geometry of cognitive space.

# Formula
d(x,y) = √((x-y)ᵀ G (x-y))

where G is the metric tensor.
"""
function geodesic_distance(surface::JSurfaceDifferential, x::Vector{Float64}, y::Vector{Float64})
    diff = x - y
    return sqrt(diff' * surface.metric * diff)
end

"""
    find_critical_points!(surface::JSurfaceDifferential, vector_field::Function; 
                         n_samples::Int=100)

Find critical points (attractors/repellers) in the cognitive dynamics.
These are the stable states toward which the system tends to evolve.

Critical points represent:
- **Attractors**: Stable patterns of meaning (schemas, concepts)
- **Repellers**: Unstable configurations to avoid
- **Saddle points**: Decision points between different interpretations
"""
function find_critical_points!(surface::JSurfaceDifferential, vector_field::Function; 
                              n_samples::Int=100)
    dim = surface.dimension
    candidates = Vector{Float64}[]
    
    # Sample random initial points
    for _ in 1:n_samples
        x0 = randn(dim)
        
        # Perform gradient descent to find nearby critical point
        x = copy(x0)
        for iter in 1:50
            dx = vector_field(x)
            
            # Check if we're at a critical point (dx ≈ 0)
            if norm(dx) < 1e-3
                # Found a critical point
                # Check if it's new (not already in list)
                is_new = true
                for cp in candidates
                    if geodesic_distance(surface, x, cp) < 0.1
                        is_new = false
                        break
                    end
                end
                
                if is_new
                    push!(candidates, copy(x))
                end
                break
            end
            
            # Move toward critical point
            x .- = 0.1 * dx
        end
    end
    
    surface.critical_points = candidates
end

"""
    project_to_surface(surface::JSurfaceDifferential, x::Vector{Float64})

Project a point onto the J-surface manifold.
This ensures states remain on the valid cognitive manifold.

Uses the metric to define the projection.
"""
function project_to_surface(surface::JSurfaceDifferential, x::Vector{Float64})
    # Normalize using the metric
    # x_projected = x / √(xᵀGx)
    norm_squared = x' * surface.metric * x
    if norm_squared > 1e-10
        return x / sqrt(norm_squared)
    else
        return x
    end
end

"""
    parallel_transport(surface::JSurfaceDifferential, 
                      vector::Vector{Float64}, 
                      from::Vector{Float64}, 
                      to::Vector{Float64})

Parallel transport a vector along the surface from one point to another.
This preserves the "direction" of a cognitive transformation as we move
through state space.

Philosophically: How does the "meaning" of a change depend on context?
Parallel transport captures context-dependent transformation.
"""
function parallel_transport(surface::JSurfaceDifferential, 
                           vector::Vector{Float64}, 
                           from::Vector{Float64}, 
                           to::Vector{Float64})
    # Simplified parallel transport using metric
    # Full implementation would integrate Christoffel symbols along a path
    
    # Project vector onto tangent space at destination
    G = surface.metric
    
    # Compute connection coefficients (simplified)
    transported = G \ (G * vector)
    
    # Project onto tangent space at 'to'
    return project_to_surface(surface, transported)
end

"""
    compute_sectional_curvature(surface::JSurfaceDifferential, 
                                x::Vector{Float64},
                                v1::Vector{Float64}, 
                                v2::Vector{Float64})

Compute sectional curvature at a point in two given directions.
Curvature measures how much the surface bends - how constrained cognition is.

High positive curvature: Highly constrained, convergent thinking
High negative curvature: Divergent, exploratory thinking  
Near-zero curvature: Flexible, balanced cognition
"""
function compute_sectional_curvature(surface::JSurfaceDifferential, 
                                    x::Vector{Float64},
                                    v1::Vector{Float64}, 
                                    v2::Vector{Float64})
    # Simplified sectional curvature computation
    # Full implementation would use Riemann curvature tensor
    
    G = surface.metric
    
    # Compute Gram determinant
    gram_matrix = [v1'*G*v1 v1'*G*v2; v2'*G*v1 v2'*G*v2]
    area_squared = det(gram_matrix)
    
    if area_squared < 1e-10
        return 0.0
    end
    
    # Return approximate curvature
    return surface.curvature / area_squared
end

"""
    optimize_trajectory(surface::JSurfaceDifferential,
                       start::Vector{Float64},
                       goal::Vector{Float64},
                       vector_field::Function)

Find optimal trajectory (geodesic) from start to goal state.
This is the "wisest" path - the transformation that minimizes
disruption while achieving the desired change.

# Returns
Vector of states along the optimal path
"""
function optimize_trajectory(surface::JSurfaceDifferential,
                            start::Vector{Float64},
                            goal::Vector{Float64},
                            vector_field::Function;
                            n_steps::Int=50)
    trajectory = Vector{Float64}[]
    push!(trajectory, copy(start))
    
    current = copy(start)
    
    for step in 1:n_steps
        # Direction toward goal in Riemannian metric
        direction = surface.metric * (goal - current)
        direction = direction / (norm(direction) + 1e-10)
        
        # Also consider vector field (dynamics)
        dynamics = vector_field(current)
        
        # Combine goal-directed and dynamic components
        total_direction = 0.7 * direction + 0.3 * dynamics
        
        # Step size decreases as we approach goal
        dist_to_goal = geodesic_distance(surface, current, goal)
        step_size = 0.1 * min(1.0, dist_to_goal)
        
        # Update position
        current = current + step_size * total_direction
        current = project_to_surface(surface, current)
        
        push!(trajectory, copy(current))
        
        # Stop if close enough to goal
        if dist_to_goal < 0.01
            break
        end
    end
    
    return trajectory
end

