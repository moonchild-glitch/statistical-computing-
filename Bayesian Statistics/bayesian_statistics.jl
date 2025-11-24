# ============================================================================
# Bayesian Statistics
# ============================================================================

using Random
using Statistics
using Distributions
using Plots

# ============================================================================
# Bayes Factors
# ============================================================================

function bayes_factor(data, mu_null=0.0)
    """
    Calculate Bayes Factor for a one-sample t-test.
    """
    n = length(data)
    sample_mean = mean(data)
    sample_std = std(data)
    t_stat = (sample_mean - mu_null) / (sample_std / sqrt(n))

    # Calculate p-value
    p_value = 2 * (1 - cdf(TDist(n - 1), abs(t_stat)))

    # Approximate Bayes Factor (using BIC approximation)
    bic_null = n * log(sample_std^2) + 2 * log(n)
    bic_alt = n * log(sample_std^2) + 2 * log(n) + t_stat^2
    bf = exp((bic_null - bic_alt) / 2)

    return bf, p_value
end

# ============================================================================
# Example: Bayes Factor Calculation
# ============================================================================

Random.seed!(123)
n = 50
mu_null = 0.0
mu_alt = 0.5
sigma = 1.0
data = rand(Normal(mu_alt, sigma), n)

println("Simulated data:")
println("  Sample size: n = $n")
println("  True mean: $mu_alt")
println("  True SD: $sigma")
println("  Sample mean: $(round(mean(data), digits=4))")
println("  Sample SD: $(round(std(data), digits=4))\n")

# Calculate Bayes Factor and p-value
bf, p_value = bayes_factor(data, mu_null=mu_null)

println("Frequentist t-test:")
println("  H0: μ = $mu_null")
println("  Ha: μ ≠ 0")
println("  p-value: $(round(p_value, digits=4))\n")

println("Bayesian t-test (Bayes Factor):")
println("  H0: μ = $mu_null")
println("  H1: μ ≠ 0")
println("  Bayes Factor (BF10): $(round(bf, digits=4))\n")

# ============================================================================
# Visualization: BF vs p-value relationship
# ============================================================================

println("Creating visualization of Bayes Factor vs p-value relationship...\n")

# Simulate multiple datasets to show relationship
n_sims = 100
sample_sizes = [20, 50, 100, 200]
results = DataFrame(n=Int[], pvalue=Float64[], bayes_factor=Float64[])

for n_size in sample_sizes
    for _ in 1:n_sims
        sim_data = rand(Normal(0.3, 1.0), n_size)
        bf, p = bayes_factor(sim_data, mu_null=0.0)
        push!(results, (n_size, p, bf))
    end
end

# Plot 1: Scatter plot of BF vs p-value
scatter(results.pvalue, results.bayes_factor, alpha=0.5, color=:blue, 
    xlabel="p-value", ylabel="Bayes Factor", title="Bayes Factor vs p-value", 
    xscale=:log10, yscale=:log10)
hline!([1], color=:red, linestyle=:dash, label="BF = 1 (no evidence)")
vline!([0.05], color=:green, linestyle=:dash, label="p = 0.05")

# ============================================================================
# Summary
# ============================================================================

println("Summary: Bayesian Statistics\n")
println("Key concepts covered:\n")
println("1. Bayesian Inference\n")
println("2. Priors\n")
println("3. Bayesian Point Estimates\n")
println("4. Bayesian Hypothesis Testing\n")
println("5. Bayes Factors\n")
println("Applications: Pattern recognition, spam detection, etc.\n")