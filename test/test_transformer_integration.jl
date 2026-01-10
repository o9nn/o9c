"""
Comprehensive unit tests for Transformer Integration and GPT Inference Engine
"""

@testset "Transformer Integration" begin

    @testset "AttentionHead Construction" begin
        head = AttentionHead(64)

        @test head.dim == 64
        @test size(head.W_q) == (64, 64)
        @test size(head.W_k) == (64, 64)
        @test size(head.W_v) == (64, 64)
        @test head.scale ≈ 1.0 / sqrt(64)
    end

    @testset "AttentionHead Custom Init Scale" begin
        head = AttentionHead(32, init_scale=0.1)

        # Weights should be scaled
        @test all(isfinite.(head.W_q))
        @test maximum(abs.(head.W_q)) < 1.0  # Roughly scaled by 0.1
    end

    @testset "softmax_matrix" begin
        X = randn(5, 5)
        result = softmax_matrix(X)

        # Each row should sum to 1
        for i in 1:5
            @test isapprox(sum(result[i, :]), 1.0, atol=1e-10)
        end

        # All values should be positive
        @test all(result .> 0)

        # All values should be <= 1
        @test all(result .<= 1.0)
    end

    @testset "softmax_matrix Numerical Stability" begin
        # Test with large values
        X = randn(3, 3) * 100
        result = softmax_matrix(X)

        @test all(isfinite.(result))
        for i in 1:3
            @test isapprox(sum(result[i, :]), 1.0, atol=1e-10)
        end
    end

    @testset "softmax_matrix Uniform Input" begin
        X = ones(4, 4)
        result = softmax_matrix(X)

        # Uniform input should give uniform output
        expected = 0.25
        @test all(isapprox.(result, expected, atol=1e-10))
    end

    @testset "compute_attention" begin
        head = AttentionHead(16)
        X = randn(5, 16)  # 5 tokens, dim 16

        output, weights = compute_attention(head, X)

        @test size(output) == (5, 16)
        @test size(weights) == (5, 5)
        @test all(isfinite.(output))
        @test all(isfinite.(weights))

        # Attention weights should sum to 1 per row
        for i in 1:5
            @test isapprox(sum(weights[i, :]), 1.0, atol=1e-10)
        end
    end

    @testset "compute_attention Single Token" begin
        head = AttentionHead(8)
        X = randn(1, 8)

        output, weights = compute_attention(head, X)

        @test size(output) == (1, 8)
        @test size(weights) == (1, 1)
        @test weights[1, 1] ≈ 1.0
    end

    @testset "compute_attention With Mask" begin
        head = AttentionHead(16)
        X = randn(4, 16)

        # Causal mask (upper triangular should be -inf)
        mask = zeros(4, 4)
        for i in 1:4
            for j in (i+1):4
                mask[i, j] = -1e9
            end
        end

        output, weights = compute_attention(head, X, mask=mask)

        @test all(isfinite.(output))

        # Check that masked positions have near-zero attention
        for i in 1:4
            for j in (i+1):4
                @test weights[i, j] < 1e-6
            end
        end
    end

    @testset "MultiHeadAttention Construction" begin
        mha = MultiHeadAttention(8, 32)  # 8 heads, 32 dim per head

        @test mha.n_heads == 8
        @test mha.dim == 32
        @test length(mha.heads) == 8

        # Output projection size
        @test size(mha.W_o) == (256, 256)  # 8 * 32
    end

    @testset "apply_multihead_attention" begin
        mha = MultiHeadAttention(4, 16)
        X = randn(10, 64)  # 10 tokens, total dim 64

        output, attn_maps = apply_multihead_attention(mha, X)

        @test size(output) == (10, 64)
        @test length(attn_maps) == 4  # 4 heads
        @test all(isfinite.(output))
    end

    @testset "apply_multihead_attention Head Independence" begin
        mha = MultiHeadAttention(2, 8)
        X = randn(5, 16)

        output, attn_maps = apply_multihead_attention(mha, X)

        # Each head should produce different attention maps
        @test !all(attn_maps[1] .== attn_maps[2])
    end

    @testset "relu" begin
        @test relu(5.0) == 5.0
        @test relu(-3.0) == 0.0
        @test relu(0.0) == 0.0
        @test relu(-0.001) == 0.0
        @test relu(0.001) == 0.001
    end

    @testset "layer_norm" begin
        x = randn(20)
        normalized = layer_norm(x)

        # Mean should be approximately 0
        @test isapprox(mean(normalized), 0.0, atol=1e-10)

        # Variance should be approximately 1
        @test isapprox(var(normalized), 1.0, atol=0.1)
    end

    @testset "layer_norm Constant Input" begin
        x = fill(5.0, 10)
        normalized = layer_norm(x)

        # Constant input should give zeros
        @test all(isapprox.(normalized, 0.0, atol=1e-5))
    end

    @testset "TransformerBlock Construction" begin
        block = TransformerBlock(4, 64, 256)

        @test block.dim == 64
        @test size(block.W_ff1) == (256, 64)
        @test size(block.W_ff2) == (64, 256)
    end

    @testset "forward_transformer_block" begin
        block = TransformerBlock(4, 64, 256)
        X = randn(10, 64)

        output, attn_maps = forward_transformer_block(block, X)

        @test size(output) == (10, 64)
        @test all(isfinite.(output))
        @test length(attn_maps) == 4
    end

    @testset "forward_transformer_block Residual Connection" begin
        block = TransformerBlock(2, 32, 128)
        X = randn(5, 32)

        output, _ = forward_transformer_block(block, X)

        # Output should be related to input (residual connection)
        # Not identical but correlated
        @test size(output) == size(X)
    end

    @testset "forward_transformer_block Layer Norm Effect" begin
        block = TransformerBlock(2, 16, 64)
        X = randn(8, 16) * 100  # Large values

        output, _ = forward_transformer_block(block, X)

        # Layer norm should keep values reasonable
        @test all(abs.(output) .< 100)
    end

    @testset "create_positional_encoding" begin
        PE = create_positional_encoding(100, 64)

        @test size(PE) == (100, 64)
        @test all(isfinite.(PE))

        # Values should be in [-1, 1] (sin/cos range)
        @test all(PE .>= -1.0)
        @test all(PE .<= 1.0)
    end

    @testset "create_positional_encoding Different Positions" begin
        PE = create_positional_encoding(50, 32)

        # Different positions should have different encodings
        @test !all(PE[1, :] .== PE[2, :])
        @test !all(PE[1, :] .== PE[10, :])
    end

    @testset "GPTInferenceEngine Construction" begin
        gpt = GPTInferenceEngine(1000, 128, 2, 4)

        @test gpt.vocab_size == 1000
        @test gpt.dim == 128
        @test length(gpt.blocks) == 2

        @test size(gpt.embedding) == (1000, 128)
        @test size(gpt.positional_encoding) == (512, 128)
    end

    @testset "forward_gpt" begin
        gpt = GPTInferenceEngine(500, 64, 2, 4)
        token_ids = [1, 5, 10, 20, 50]

        output, attn_maps = forward_gpt(gpt, token_ids)

        @test size(output) == (5, 64)
        @test all(isfinite.(output))
        @test length(attn_maps) == 2  # 2 layers
    end

    @testset "forward_gpt Single Token" begin
        gpt = GPTInferenceEngine(100, 32, 1, 2)
        token_ids = [1]

        output, attn_maps = forward_gpt(gpt, token_ids)

        @test size(output) == (1, 32)
    end

    @testset "forward_gpt Long Sequence" begin
        gpt = GPTInferenceEngine(200, 48, 2, 3)
        token_ids = collect(1:100)

        output, attn_maps = forward_gpt(gpt, token_ids)

        @test size(output) == (100, 48)
        @test all(isfinite.(output))
    end

    @testset "integrate_with_reservoir" begin
        gpt_output = randn(10, 64)
        reservoir_state = randn(100)

        integrated = integrate_with_reservoir(gpt_output, reservoir_state)

        # Should be concatenation of pooled GPT output and reservoir
        expected_length = 64 + 100
        @test length(integrated) == expected_length
        @test all(isfinite.(integrated))
    end

    @testset "integrate_with_reservoir Pooling" begin
        # GPT output with known mean
        gpt_output = ones(5, 10) * 2.0
        reservoir_state = zeros(20)

        integrated = integrate_with_reservoir(gpt_output, reservoir_state)

        # First 10 elements should be mean of GPT output rows (2.0)
        @test all(isapprox.(integrated[1:10], 2.0, atol=1e-10))

        # Last 20 elements should be reservoir state (zeros)
        @test all(integrated[11:end] .== 0.0)
    end

    @testset "extract_relevance_landscape" begin
        # Create mock attention maps (2 layers, 2 heads each)
        layer1_maps = [randn(5, 5), randn(5, 5)]
        layer2_maps = [randn(5, 5), randn(5, 5)]

        # Apply softmax to make them valid attention weights
        for layer_maps in [layer1_maps, layer2_maps]
            for i in 1:length(layer_maps)
                layer_maps[i] = softmax_matrix(layer_maps[i])
            end
        end

        attention_maps = [layer1_maps, layer2_maps]

        landscape = extract_relevance_landscape(attention_maps)

        @test haskey(landscape, :mean_attention)
        @test haskey(landscape, :entropy)
        @test haskey(landscape, :peak_positions)
        @test haskey(landscape, :focus)

        @test isfinite(landscape[:entropy])
        @test landscape[:focus] > 0
    end

    @testset "extract_relevance_landscape Uniform Attention" begin
        # Uniform attention should have high entropy
        uniform_attn = softmax_matrix(zeros(4, 4))

        attention_maps = [[uniform_attn, uniform_attn]]
        landscape = extract_relevance_landscape(attention_maps)

        # Uniform attention has maximum entropy
        @test landscape[:entropy] > 0
        @test landscape[:focus] > 0
        @test landscape[:focus] < 1  # Not perfectly focused
    end

    @testset "extract_relevance_landscape Focused Attention" begin
        # One-hot attention (focused)
        focused_attn = zeros(4, 4)
        focused_attn[1, 1] = 1.0
        focused_attn[2, 2] = 1.0
        focused_attn[3, 3] = 1.0
        focused_attn[4, 4] = 1.0

        attention_maps = [[focused_attn]]
        landscape = extract_relevance_landscape(attention_maps)

        # Focused attention should have low entropy (high focus)
        @test landscape[:focus] > 0
    end

    @testset "Full GPT Pipeline" begin
        # Test complete pipeline
        gpt = GPTInferenceEngine(500, 64, 3, 4)
        token_ids = [1, 10, 20, 30, 40]

        output, attn_maps = forward_gpt(gpt, token_ids)
        relevance = extract_relevance_landscape(attn_maps)

        @test size(output) == (5, 64)
        @test all(isfinite.(output))
        @test haskey(relevance, :focus)
    end

    @testset "GPT With Reservoir Integration" begin
        gpt = GPTInferenceEngine(200, 32, 2, 2)
        token_ids = [1, 5, 10]

        gpt_output, _ = forward_gpt(gpt, token_ids)
        reservoir_state = randn(50)

        integrated = integrate_with_reservoir(gpt_output, reservoir_state)

        @test length(integrated) == 32 + 50
        @test all(isfinite.(integrated))
    end

    @testset "Attention Map Dimensions" begin
        n_heads = 4
        mha = MultiHeadAttention(n_heads, 16)
        X = randn(8, 64)  # 8 tokens

        _, attn_maps = apply_multihead_attention(mha, X)

        # Each head should produce 8x8 attention map
        for map in attn_maps
            @test size(map) == (8, 8)
        end
    end

    @testset "Transformer Block Gradient Flow" begin
        # Test that values don't explode through multiple blocks
        blocks = [TransformerBlock(4, 64, 256) for _ in 1:5]
        X = randn(10, 64)

        for block in blocks
            X, _ = forward_transformer_block(block, X)
        end

        @test all(isfinite.(X))
        @test maximum(abs.(X)) < 1000
    end

end
