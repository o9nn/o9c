"""
# Example: Learning with the Deep Tree Echo Self

Demonstrates supervised learning using the reservoir readout.
A DeepTreeESN learns to predict a target time-series by training
output weights via ridge regression on collected reservoir states.

Steps:
  1. Generate a sinusoidal target signal
  2. Drive the reservoir with lagged inputs
  3. Train the readout (W_out) via train_readout!
  4. Evaluate prediction accuracy
  5. Compare performance across personas
"""

push!(LOAD_PATH, joinpath(@__DIR__, "../src"))
using DeepTreeEchoSelf

using Random
using Statistics
using LinearAlgebra

println("=" ^ 70)
println("Deep Tree Echo Self - Learning Example")
println("=" ^ 70)
println()

Random.seed!(7)

# ── 1. Generate training data: predict sin(t) one step ahead ─────────────

n_total   = 200
input_dim = 5
dt        = 0.05

# Input: random noise driving the system
inputs = [randn(input_dim) for _ in 1:n_total]

# Target: sinusoidal pattern (system should learn to produce this)
targets = [[sin(2π * k * dt) for k in 1:input_dim] for _ in 1:n_total]

n_train = 150
n_test  = n_total - n_train

train_inputs  = inputs[1:n_train]
train_targets = targets[1:n_train]
test_inputs   = inputs[n_train+1:end]
test_targets  = targets[n_train+1:end]

println("Dataset: $n_train training samples, $n_test test samples")
println("Input dim: $input_dim,  Output dim: $input_dim")
println()

# ── 2. Train readout for each persona ─────────────────────────────────────

personas = [:contemplative_scholar, :dynamic_explorer, :cautious_analyst, :creative_visionary]

for persona in personas
    esn = DeepTreeESN(2, 30, input_dim, persona=persona)

    # Train output weights
    train_readout!(esn, train_inputs, train_targets)

    # Evaluate: drive with test inputs and predict via W_out
    errors = Float64[]
    for (inp, tgt) in zip(test_inputs, test_targets)
        process_tree!(esn, inp)
        state  = collect_states(esn)
        pred   = esn.root.W_out * state[1:size(esn.root.W_out, 2)]
        push!(errors, norm(pred - tgt))
    end

    mae = mean(errors)
    println("Persona: $(rpad(string(persona), 25)) | Test MAE: $(round(mae, digits=4))")
end

println()
println("Note: MAE measures average prediction error across the test set.")
println("Different personas (reservoir hyperparameters) give different fits.")
println()

# ── 3. Affective modulation of learning ───────────────────────────────────

println("=" ^ 70)
println("Affective Learning Rate Modulation")
println("=" ^ 70)
println()

agency_names = [:wonder, :curiosity, :joy, :interest, :surprise,
                :sadness, :fear, :anxiety]

agency = AffectiveAgency(agency_names)
base_lr = 0.01

println("Base learning rate: $base_lr")
println()

for (emotion, intensity) in [(:wonder, 0.9), (:curiosity, 0.7), (:fear, 0.8), (:sadness, 0.6)]
    # Reset and trigger single emotion
    for name in agency_names
        trigger_emotion!(agency.det, name, 0.0)
    end
    trigger_emotion!(agency.det, emotion, intensity)
    compute_cognitive_modulation!(agency)

    modulated_lr = modulate_learning_rate(agency, base_lr)
    println("  Emotion: $(rpad(string(emotion), 12)) (intensity=$intensity) " *
            "→ effective LR = $(round(modulated_lr, digits=5))")
end

println()
println("Emotions with high arousal (fear, wonder) amplify the learning signal.")
println()

# ── 4. Online learning loop (illustrative) ────────────────────────────────

println("=" ^ 70)
println("Conceptual Online Learning Loop")
println("=" ^ 70)
println()

println("""
In an online (incremental) setting the readout can be updated after each
new observation using recursive least squares or a simple gradient step:

    W_out += lr * (target - W_out * state) * state'

This allows the reservoir to continually adapt as the environment evolves.
To enable this, replace train_readout! with an incremental update function
that is called inside the processing loop.
""")
