"""
Comprehensive unit tests for Tensor Logic Framework
Based on arXiv:2510.12269 - Tensor Logic: The Language of AI
"""

@testset "Tensor Logic Framework" begin

    # ========================================================================
    # TENSOR RELATIONS TESTS
    # ========================================================================

    @testset "TensorRelation" begin
        @testset "Construction" begin
            rel = TensorRelation(:Parent, [10, 10])

            @test rel.name == :Parent
            @test rel.arity == 2
            @test rel.domain_sizes == [10, 10]
            @test rel.is_boolean == true
            @test size(rel.tensor) == (10, 10)
        end

        @testset "Sparse Construction" begin
            rel = TensorRelation(:Friend, [100, 100], sparse=true)

            @test rel.name == :Friend
            @test issparse(rel.tensor)
        end

        @testset "Dense Construction" begin
            rel = TensorRelation(:Knows, [5, 5], sparse=false)

            @test !issparse(rel.tensor)
        end

        @testset "set_tuple!" begin
            rel = TensorRelation(:Parent, [5, 5], sparse=false)

            set_tuple!(rel, [1, 2], 1.0)
            set_tuple!(rel, [3, 4], 1.0)

            @test rel.tensor[1, 2] == 1.0
            @test rel.tensor[3, 4] == 1.0
            @test rel.tensor[1, 1] == 0.0
        end

        @testset "get_tuple" begin
            rel = TensorRelation(:Parent, [5, 5], sparse=false)
            set_tuple!(rel, [2, 3], 1.0)

            @test get_tuple(rel, [2, 3]) == 1.0
            @test get_tuple(rel, [1, 1]) == 0.0
        end

        @testset "from_tuples" begin
            tuples = [[1, 2], [2, 3], [3, 4]]
            rel = from_tuples(:Chain, [5, 5], tuples)

            @test get_tuple(rel, [1, 2]) == 1.0
            @test get_tuple(rel, [2, 3]) == 1.0
            @test get_tuple(rel, [3, 4]) == 1.0
            @test get_tuple(rel, [1, 3]) == 0.0
        end

        @testset "Arity Mismatch Error" begin
            rel = TensorRelation(:Binary, [5, 5])

            @test_throws AssertionError set_tuple!(rel, [1, 2, 3], 1.0)
            @test_throws AssertionError get_tuple(rel, [1])
        end
    end

    # ========================================================================
    # EINSTEIN SUMMATION TESTS
    # ========================================================================

    @testset "Einstein Summation" begin
        @testset "EinsumSpec Construction" begin
            spec = EinsumSpec([[:i, :j], [:j, :k]], [:i, :k])

            @test spec.input_indices == [[:i, :j], [:j, :k]]
            @test spec.output_indices == [:i, :k]
            @test :j in spec.contraction_indices
        end

        @testset "tensor_project Single Dimension" begin
            T = [1.0 2.0; 3.0 4.0; 5.0 6.0]  # 3x2

            # Project to keep only first dimension (sum over columns)
            result = tensor_project(T, [1])

            @test length(result) == 3
            @test result[1] ≈ 3.0  # 1+2
            @test result[2] ≈ 7.0  # 3+4
            @test result[3] ≈ 11.0 # 5+6
        end

        @testset "tensor_project Keep Both" begin
            T = randn(4, 5)

            result = tensor_project(T, [1, 2])

            @test result ≈ T
        end

        @testset "tensor_project Total Sum" begin
            T = ones(3, 4)

            result = tensor_project(T, Int[])

            @test result ≈ 12.0
        end

        @testset "tensor_join Matrix Multiplication" begin
            U = [1.0 2.0; 3.0 4.0]
            V = [5.0 6.0; 7.0 8.0]

            # Join on shared dimension (standard matrix mult)
            result = tensor_join(U, V, [2], [1])

            expected = U * V
            @test result ≈ expected
        end

        @testset "tensor_join Dimension Preservation" begin
            U = randn(4, 3)
            V = randn(3, 5)

            result = tensor_join(U, V, [2], [1])

            @test size(result) == (4, 5)
        end

        @testset "_contract_last_first" begin
            U = randn(3, 4)
            V = randn(4, 5)

            result = _contract_last_first(U, V)

            @test size(result) == (3, 5)
            @test result ≈ U * V
        end
    end

    # ========================================================================
    # LOGICAL OPERATIONS TESTS
    # ========================================================================

    @testset "Logical Operations" begin
        @testset "heaviside_step" begin
            @test heaviside_step(0.5) == 1.0
            @test heaviside_step(0.0) == 0.0
            @test heaviside_step(-0.5) == 0.0
            @test heaviside_step(100.0) == 1.0
        end

        @testset "soft_step T=0" begin
            # At T=0, should behave like hard step
            @test soft_step(0.5, 0.0) == 1.0
            @test soft_step(-0.5, 0.0) == 0.0
        end

        @testset "soft_step T>0" begin
            # At T>0, should be smooth
            result = soft_step(0.0, 1.0)
            @test result ≈ 0.5  # sigmoid(0) = 0.5

            # Higher T → more spread
            low_t = soft_step(1.0, 0.1)
            high_t = soft_step(1.0, 10.0)

            @test low_t > high_t  # Lower T → sharper
        end

        @testset "soft_step Bounds" begin
            for x in [-10.0, -1.0, 0.0, 1.0, 10.0]
                for T in [0.1, 1.0, 10.0]
                    result = soft_step(x, T)
                    @test 0.0 <= result <= 1.0
                end
            end
        end

        @testset "apply_rule Single Premise" begin
            rel = from_tuples(:A, [3, 3], [[1, 2], [2, 3]])

            result = apply_rule([rel], Tuple{Int,Int,Int,Int}[], 0.0)

            @test result[1, 2] == 1.0
            @test result[2, 3] == 1.0
            @test result[1, 3] == 0.0
        end

        @testset "apply_rule Two Premises Join" begin
            # Sister[x,y] and Parent[y,z] → Aunt[x,z]
            sister = from_tuples(:Sister, [4, 4], [[1, 2]])  # 1 is sister of 2
            parent = from_tuples(:Parent, [4, 4], [[2, 3]]) # 2 is parent of 3

            result = apply_rule([sister, parent], [(1, 2, 2, 1)], 0.0)

            # Should infer Aunt(1, 3)
            @test result[1, 3] == 1.0
            @test result[1, 1] == 0.0
        end

        @testset "apply_rule with Temperature" begin
            rel = from_tuples(:A, [3, 3], [[1, 2]])

            hard_result = apply_rule([rel], Tuple{Int,Int,Int,Int}[], 0.0)
            soft_result = apply_rule([rel], Tuple{Int,Int,Int,Int}[], 0.5)

            # Hard should be exactly 0 or 1
            @test hard_result[1, 2] == 1.0
            @test hard_result[1, 1] == 0.0

            # Soft should be in (0, 1)
            @test 0.0 < soft_result[1, 2] < 1.0 || soft_result[1, 2] ≈ 1.0
        end
    end

    # ========================================================================
    # EMBEDDING SPACE TESTS
    # ========================================================================

    @testset "EmbeddingSpace" begin
        @testset "Construction" begin
            space = EmbeddingSpace(10, 32)

            @test space.dimension == 32
            @test size(space.entity_embeddings) == (10, 32)
            @test length(space.entity_names) == 10
            @test size(space.gram_matrix) == (10, 10)
        end

        @testset "Normalized Embeddings" begin
            space = EmbeddingSpace(5, 16)

            for i in 1:5
                emb_norm = norm(space.entity_embeddings[i, :])
                @test isapprox(emb_norm, 1.0, atol=0.01)
            end
        end

        @testset "set_entity_name!" begin
            space = EmbeddingSpace(5, 8)

            set_entity_name!(space, 1, :Alice)
            set_entity_name!(space, 2, :Bob)

            @test space.entity_names[1] == :Alice
            @test space.entity_names[2] == :Bob
        end

        @testset "get_entity_idx" begin
            space = EmbeddingSpace(5, 8)

            set_entity_name!(space, 3, :Charlie)

            @test get_entity_idx(space, :Charlie) == 3
            @test get_entity_idx(space, :Unknown) === nothing
        end

        @testset "update_gram_matrix!" begin
            space = EmbeddingSpace(4, 16)

            # Modify embeddings
            space.entity_embeddings[1, :] .= 1.0 / sqrt(16)
            space.entity_embeddings[2, :] .= 1.0 / sqrt(16)

            update_gram_matrix!(space)

            # Self-similarity should be 1
            @test isapprox(space.gram_matrix[1, 1], 1.0, atol=0.01)

            # Same embeddings → similarity = 1
            @test isapprox(space.gram_matrix[1, 2], 1.0, atol=0.01)
        end

        @testset "similarity" begin
            space = EmbeddingSpace(3, 8)

            # Self-similarity
            self_sim = similarity(space, 1, 1)
            @test isapprox(self_sim, 1.0, atol=0.01)

            # Different entities
            other_sim = similarity(space, 1, 2)
            @test isfinite(other_sim)
        end

        @testset "embed_relation" begin
            space = EmbeddingSpace(5, 16)
            rel = from_tuples(:Knows, [5, 5], [[1, 2], [2, 3]])

            embedded = embed_relation(rel, space)

            @test size(embedded) == (16, 16)
            @test all(isfinite.(embedded))
        end

        @testset "query_embedded_relation" begin
            space = EmbeddingSpace(5, 32)
            rel = from_tuples(:Parent, [5, 5], [[1, 2]])

            embedded = embed_relation(rel, space)

            # Query: Parent(1, ?)
            scores = query_embedded_relation(embedded, space, 1, 1)

            @test length(scores) == 5
            @test all(isfinite.(scores))
        end
    end

    # ========================================================================
    # ANALOGICAL INFERENCE TESTS
    # ========================================================================

    @testset "Analogical Inference" begin
        @testset "Pure Deduction T=0" begin
            space = EmbeddingSpace(5, 16)
            rel = from_tuples(:Likes, [5, 5], [[1, 2]])

            result_exists = analogical_inference(rel, space, [1, 2], 0.0)
            result_absent = analogical_inference(rel, space, [1, 3], 0.0)

            @test result_exists == 1.0
            @test result_absent == 0.0
        end

        @testset "Analogical T>0" begin
            space = EmbeddingSpace(5, 16)
            rel = from_tuples(:Likes, [5, 5], [[1, 2]])

            # Make entities 1 and 3 similar
            space.entity_embeddings[3, :] .= space.entity_embeddings[1, :]
            update_gram_matrix!(space)

            # With temperature, similar entities should share inferences
            result = analogical_inference(rel, space, [3, 2], 1.0)

            # Should be positive due to similarity with entity 1
            @test result >= 0.0
        end

        @testset "Temperature Effect" begin
            space = EmbeddingSpace(4, 8)
            rel = from_tuples(:R, [4, 4], [[1, 2]])

            low_t = analogical_inference(rel, space, [1, 3], 0.1)
            high_t = analogical_inference(rel, space, [1, 3], 2.0)

            # Both should be non-negative
            @test low_t >= 0.0
            @test high_t >= 0.0
        end
    end

    # ========================================================================
    # INFERENCE ENGINE TESTS
    # ========================================================================

    @testset "Rule" begin
        @testset "Construction" begin
            rule = Rule(:Grandparent, [:Parent, :Parent])

            @test rule.head == :Grandparent
            @test rule.premises == [:Parent, :Parent]
            @test rule.temperature == 0.0
        end

        @testset "With Join Spec" begin
            rule = Rule(:Aunt, [:Sister, :Parent],
                       join_spec=[(1, 2, 2, 1)],
                       temperature=0.5)

            @test rule.join_spec == [(1, 2, 2, 1)]
            @test rule.temperature == 0.5
        end
    end

    @testset "KnowledgeBase" begin
        @testset "Construction" begin
            kb = KnowledgeBase()

            @test isempty(kb.relations)
            @test isempty(kb.rules)
            @test kb.default_temperature == 0.0
        end

        @testset "Custom Temperature" begin
            kb = KnowledgeBase(0.5)
            @test kb.default_temperature == 0.5
        end

        @testset "add_relation!" begin
            kb = KnowledgeBase()
            rel = TensorRelation(:Test, [5, 5])

            add_relation!(kb, rel)

            @test haskey(kb.relations, :Test)
            @test kb.relations[:Test] === rel
        end

        @testset "add_rule!" begin
            kb = KnowledgeBase()
            rule = Rule(:C, [:A, :B])

            add_rule!(kb, rule)

            @test length(kb.rules) == 1
            @test kb.rules[1] === rule
        end
    end

    @testset "Forward Chaining" begin
        @testset "Single Rule" begin
            kb = KnowledgeBase()

            # Add base relation
            parent = from_tuples(:Parent, [4, 4], [[1, 2], [2, 3]])
            add_relation!(kb, parent)

            # Add rule: Ancestor(x,z) = Parent(x,y) · Parent(y,z)
            # (simplified as just applying the join)
            rule = Rule(:Ancestor, [:Parent, :Parent],
                       join_spec=[(1, 2, 2, 1)])
            add_rule!(kb, rule)

            forward_chain!(kb, max_iterations=10)

            # Should derive Ancestor
            @test haskey(kb.relations, :Ancestor)
        end

        @testset "Fixpoint" begin
            kb = KnowledgeBase()

            rel = from_tuples(:A, [3, 3], [[1, 2]])
            add_relation!(kb, rel)

            # Rule that derives nothing new
            rule = Rule(:B, [:A])
            add_rule!(kb, rule)

            forward_chain!(kb, max_iterations=100)

            # Should reach fixpoint quickly
            @test haskey(kb.relations, :B)
        end

        @testset "Missing Premise" begin
            kb = KnowledgeBase()

            # Rule with premise not in KB
            rule = Rule(:C, [:NonExistent])
            add_rule!(kb, rule)

            forward_chain!(kb)

            # Should not derive C
            @test !haskey(kb.relations, :C)
        end
    end

    @testset "Backward Chaining" begin
        @testset "Direct Lookup" begin
            kb = KnowledgeBase()

            rel = from_tuples(:Known, [5, 5], [[1, 2], [3, 4]])
            add_relation!(kb, rel)

            result = backward_chain(kb, :Known, [1, 2])

            @test result == 1.0
        end

        @testset "Not Found" begin
            kb = KnowledgeBase()

            rel = from_tuples(:Known, [5, 5], [[1, 2]])
            add_relation!(kb, rel)

            result = backward_chain(kb, :Known, [3, 4])

            @test result == 0.0
        end

        @testset "Unknown Relation" begin
            kb = KnowledgeBase()

            result = backward_chain(kb, :Unknown, [1, 2])

            @test result == 0.0
        end
    end

    # ========================================================================
    # NEURAL NETWORK OPERATIONS TESTS
    # ========================================================================

    @testset "Neural Network Operations" begin
        @testset "tensor_mlp_layer" begin
            input = randn(10)
            weights = randn(5, 10)
            bias = randn(5)

            output = tensor_mlp_layer(input, weights, bias, tanh)

            @test length(output) == 5
            @test all(abs.(output) .<= 1.0)  # tanh bounded
        end

        @testset "tensor_mlp_layer ReLU" begin
            input = randn(8)
            weights = randn(4, 8)
            bias = zeros(4)

            output = tensor_mlp_layer(input, weights, bias, x -> max(0, x))

            @test length(output) == 4
            @test all(output .>= 0)  # ReLU non-negative
        end

        @testset "tensor_attention" begin
            seq_len = 5
            d_k = 8
            d_v = 8

            query = randn(seq_len, d_k)
            key = randn(seq_len, d_k)
            value = randn(seq_len, d_v)

            output, weights = tensor_attention(query, key, value)

            @test size(output) == (seq_len, d_v)
            @test size(weights) == (seq_len, seq_len)

            # Attention weights should sum to 1 per row
            for i in 1:seq_len
                @test isapprox(sum(weights[i, :]), 1.0, atol=1e-6)
            end
        end

        @testset "tensor_attention Temperature" begin
            query = randn(3, 4)
            key = randn(3, 4)
            value = randn(3, 4)

            _, weights_low = tensor_attention(query, key, value, 0.1)
            _, weights_high = tensor_attention(query, key, value, 10.0)

            # Low temperature → sharper (lower entropy)
            # High temperature → more uniform
            entropy_low = -sum(weights_low .* log.(weights_low .+ 1e-10))
            entropy_high = -sum(weights_high .* log.(weights_high .+ 1e-10))

            @test entropy_low < entropy_high
        end

        @testset "tensor_layer_norm" begin
            x = randn(20)

            normalized = tensor_layer_norm(x)

            @test isapprox(mean(normalized), 0.0, atol=1e-6)
            @test isapprox(var(normalized), 1.0, atol=0.1)
        end

        @testset "tensor_softmax" begin
            x = randn(10)

            probs = tensor_softmax(x)

            @test isapprox(sum(probs), 1.0, atol=1e-10)
            @test all(probs .> 0)
            @test all(probs .< 1)
        end

        @testset "tensor_softmax Temperature" begin
            x = [1.0, 2.0, 3.0]

            sharp = tensor_softmax(x, 0.1)
            flat = tensor_softmax(x, 10.0)

            # Low temp → more peaked
            @test sharp[3] > flat[3]
        end
    end

    # ========================================================================
    # GRADIENT COMPUTATION TESTS
    # ========================================================================

    @testset "Gradient Computation" begin
        @testset "TensorGradient Construction" begin
            tensor = randn(5, 5)
            grad = TensorGradient(tensor)

            @test grad.tensor === tensor
            @test size(grad.gradient) == (5, 5)
            @test all(grad.gradient .== 0)
            @test grad.requires_grad == true
        end

        @testset "backward!" begin
            tensor = randn(3, 3)
            grad = TensorGradient(tensor)

            upstream = ones(3, 3)
            backward!(grad, upstream)

            @test all(grad.gradient .== 1.0)

            # Accumulate
            backward!(grad, upstream)
            @test all(grad.gradient .== 2.0)
        end

        @testset "tensor_mse_loss" begin
            predicted = [1.0, 2.0, 3.0]
            target = [1.0, 2.0, 3.0]

            loss = tensor_mse_loss(predicted, target)

            @test loss == 0.0
        end

        @testset "tensor_mse_loss Non-Zero" begin
            predicted = [1.0, 2.0, 3.0]
            target = [2.0, 3.0, 4.0]

            loss = tensor_mse_loss(predicted, target)

            @test loss == 1.0  # Mean of [1, 1, 1]
        end

        @testset "tensor_mse_gradient" begin
            predicted = [1.0, 2.0, 3.0]
            target = [2.0, 3.0, 4.0]

            grad = tensor_mse_gradient(predicted, target)

            @test length(grad) == 3
            # Gradient should be 2*(pred - target)/n
            expected = 2.0 * (predicted .- target) / 3
            @test grad ≈ expected
        end
    end

    # ========================================================================
    # TENSOR LOGIC REASONER TESTS
    # ========================================================================

    @testset "TensorLogicReasoner" begin
        @testset "Construction" begin
            reasoner = TensorLogicReasoner()

            @test reasoner.embedding_dim == 64
            @test reasoner.temperature == 0.0
            @test isempty(reasoner.inference_history)
            @test reasoner.kb.embedding_space !== nothing
        end

        @testset "Custom Parameters" begin
            reasoner = TensorLogicReasoner(
                embedding_dim=32,
                temperature=0.5,
                n_entities=50
            )

            @test reasoner.embedding_dim == 32
            @test reasoner.temperature == 0.5
        end

        @testset "define_relation!" begin
            reasoner = TensorLogicReasoner()

            rel = define_relation!(reasoner, :Loves, [10, 10])

            @test haskey(reasoner.kb.relations, :Loves)
            @test reasoner.kb.relations[:Loves].arity == 2
        end

        @testset "assert_fact!" begin
            reasoner = TensorLogicReasoner()

            define_relation!(reasoner, :Friend, [5, 5])
            assert_fact!(reasoner, :Friend, [1, 2])

            @test get_tuple(reasoner.kb.relations[:Friend], [1, 2]) == 1.0
        end

        @testset "define_rule!" begin
            reasoner = TensorLogicReasoner()

            define_relation!(reasoner, :Parent, [5, 5])
            define_rule!(reasoner, :Ancestor, [:Parent, :Parent])

            @test length(reasoner.kb.rules) == 1
            @test reasoner.kb.rules[1].head == :Ancestor
        end

        @testset "reason!" begin
            reasoner = TensorLogicReasoner()

            define_relation!(reasoner, :A, [3, 3])
            assert_fact!(reasoner, :A, [1, 2])
            define_rule!(reasoner, :B, [:A])

            reason!(reasoner, mode=:forward)

            @test length(reasoner.inference_history) == 1
            @test haskey(reasoner.kb.relations, :B)
        end

        @testset "query Pure Deduction" begin
            reasoner = TensorLogicReasoner(temperature=0.0)

            define_relation!(reasoner, :Knows, [5, 5])
            assert_fact!(reasoner, :Knows, [1, 2])

            result_yes = query(reasoner, :Knows, [1, 2])
            result_no = query(reasoner, :Knows, [3, 4])

            @test result_yes == 1.0
            @test result_no == 0.0
        end

        @testset "set_temperature!" begin
            reasoner = TensorLogicReasoner()

            @test reasoner.temperature == 0.0

            set_temperature!(reasoner, 0.5)

            @test reasoner.temperature == 0.5
            @test reasoner.kb.default_temperature == 0.5
        end

        @testset "integrate_with_attention" begin
            reasoner = TensorLogicReasoner()

            define_relation!(reasoner, :R, [5, 5])
            assert_fact!(reasoner, :R, [1, 2])
            assert_fact!(reasoner, :R, [2, 3])

            attention_weights = ones(3, 5) / 5

            result = integrate_with_attention(reasoner, attention_weights, :R)

            @test size(result) == (3, 5)
            @test all(isfinite.(result))
        end
    end

    # ========================================================================
    # INTEGRATION TESTS
    # ========================================================================

    @testset "Integration Tests" begin
        @testset "Family Relations Example" begin
            reasoner = TensorLogicReasoner(n_entities=10)

            # Define relations
            define_relation!(reasoner, :Parent, [10, 10])
            define_relation!(reasoner, :Sibling, [10, 10])

            # Assert facts
            # Alice(1) is parent of Bob(2)
            # Bob(2) is parent of Charlie(3)
            # David(4) is sibling of Alice(1)
            assert_fact!(reasoner, :Parent, [1, 2])
            assert_fact!(reasoner, :Parent, [2, 3])
            assert_fact!(reasoner, :Sibling, [4, 1])

            # Define rules
            # Grandparent: Parent(x,y) ∧ Parent(y,z) → Grandparent(x,z)
            define_rule!(reasoner, :Grandparent, [:Parent, :Parent],
                        join_spec=[(1, 2, 2, 1)])

            # Uncle/Aunt: Sibling(x,y) ∧ Parent(y,z) → UncleAunt(x,z)
            define_rule!(reasoner, :UncleAunt, [:Sibling, :Parent],
                        join_spec=[(1, 2, 2, 1)])

            # Reason
            reason!(reasoner, mode=:forward)

            # Check derived relations
            @test haskey(reasoner.kb.relations, :Grandparent)
            @test haskey(reasoner.kb.relations, :UncleAunt)
        end

        @testset "Transitive Closure" begin
            reasoner = TensorLogicReasoner(n_entities=5)

            define_relation!(reasoner, :Connected, [5, 5])

            # Chain: 1 → 2 → 3 → 4
            assert_fact!(reasoner, :Connected, [1, 2])
            assert_fact!(reasoner, :Connected, [2, 3])
            assert_fact!(reasoner, :Connected, [3, 4])

            # Self-join rule for transitive closure
            define_rule!(reasoner, :Reachable, [:Connected, :Connected],
                        join_spec=[(1, 2, 2, 1)])

            reason!(reasoner, mode=:forward)

            # Should derive some reachability
            @test haskey(reasoner.kb.relations, :Reachable)
        end

        @testset "Temperature Spectrum" begin
            # Test reasoning at different temperatures
            for temp in [0.0, 0.1, 0.5, 1.0, 2.0]
                reasoner = TensorLogicReasoner(temperature=temp, n_entities=5)

                define_relation!(reasoner, :R, [5, 5])
                assert_fact!(reasoner, :R, [1, 2])

                result = query(reasoner, :R, [1, 2])

                @test result >= 0.0
                @test isfinite(result)
            end
        end

        @testset "Embedding Space Consistency" begin
            reasoner = TensorLogicReasoner(embedding_dim=32, n_entities=10)

            space = reasoner.kb.embedding_space

            # Check Gram matrix is symmetric
            @test space.gram_matrix ≈ space.gram_matrix'

            # Check diagonal is approximately 1 (normalized embeddings)
            for i in 1:10
                @test isapprox(space.gram_matrix[i, i], 1.0, atol=0.02)
            end
        end
    end

end
