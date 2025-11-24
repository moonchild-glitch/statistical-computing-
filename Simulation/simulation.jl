"""
SIMULATION AND RANDOM NUMBER GENERATION
Statistical Computing Tutorial - Julia Version

Topics covered:
1. Random number generation theory
2. Box-Muller transformation
3. Inverse CDF method
4. Rejection sampling
5. Statistical validation techniques
"""

using Random
using Distributions
using Plots
using Statistics
using StatsBase
using HypothesisTests
using Printf

# Create plots directory if it doesn't exist
if !isdir("../plots")
    mkdir("../plots")
end

println("="^70)
println("SIMULATION AND RANDOM NUMBER GENERATION TUTORIAL")
println("="^70)

# =============================================================================
# PART 1: RANDOM NUMBER GENERATION BASICS
# =============================================================================

println("\n", "="^70)
println("PART 1: RANDOM NUMBER GENERATION BASICS")
println("="^70)

# Set seed for reproducibility
Random.seed!(42)

# Basic uniform random numbers
println("\nBasic uniform random numbers U(0,1):")
u = rand(10)
println(u[1:5])

# Transform to other distributions using inverse CDF
println("\nInverse CDF method - Exponential(rate=2):")
rate = 2
x_exp = -log.(1 .- u) ./ rate
println("First 5 values: ", x_exp[1:5])
println("Compare with built-in: ", rand(Exponential(1/rate), 5))

# =============================================================================
# PART 2: BOX-MULLER TRANSFORMATION
# =============================================================================

println("\n", "="^70)
println("PART 2: BOX-MULLER TRANSFORMATION")
println("="^70)

println("\nThe Box-Muller transformation converts uniform random variables")
println("to normal random variables using the transformation:")
println("Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)")
println("Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)")
println("where U1, U2 ~ Uniform(0,1) and Z1, Z2 ~ Normal(0,1)")

"""
    bmnormal(n::Int, mu::Float64=0.0, sd::Float64=1.0)

Generate n draws from Normal(mu, sd) using Box-Muller transformation.

# Arguments
- `n::Int`: Number of samples to generate
- `mu::Float64`: Mean of the normal distribution (default: 0.0)
- `sd::Float64`: Standard deviation (default: 1.0)

# Returns
- `Vector{Float64}`: Array of n samples from Normal(mu, sd)

# Notes
The Box-Muller transformation converts pairs of uniform random variables
into pairs of independent standard normal variables. We generate ceiling(n/2)
pairs and return the first n values.
"""
function bmnormal(n::Int, mu::Float64=0.0, sd::Float64=1.0)
    # Number of pairs needed
    n_pairs = ceil(Int, n / 2)
    
    # Generate uniform random variables
    u1 = rand(n_pairs)
    u2 = rand(n_pairs)
    
    # Box-Muller transformation
    r = sqrt.(-2 .* log.(u1))
    theta = 2 .* π .* u2
    
    z1 = r .* cos.(theta)
    z2 = r .* sin.(theta)
    
    # Combine and take first n values
    z = vcat(z1, z2)[1:n]
    
    # Transform to desired mean and sd
    x = mu .+ sd .* z
    
    return x
end

# Test the function
println("\nTesting bmnormal function:")
samples = bmnormal(5, 0.0, 1.0)
println("5 samples from N(0,1): ", samples)

samples = bmnormal(5, 10.0, 3.0)
println("5 samples from N(10,3): ", samples)

# =============================================================================
# EXERCISE: BOX-MULLER VALIDATION
# =============================================================================

println("\n", "="^70)
println("EXERCISE: VALIDATE BOX-MULLER IMPLEMENTATION")
println("="^70)

# Generate 2000 samples from Normal(10, 3)
n = 2000
mu = 10.0
sd = 3.0

println("\nGenerating $n samples from Normal($mu, $sd) using Box-Muller...")
Random.seed!(123)
samples_bm = bmnormal(n, mu, sd)

# Calculate sample statistics
sample_mean = mean(samples_bm)
sample_sd = std(samples_bm)

println("\nSample Statistics:")
@printf("Sample mean: %.4f (expected: %.0f)\n", sample_mean, mu)
@printf("Sample SD:   %.4f (expected: %.0f)\n", sample_sd, sd)

# =============================================================================
# VISUAL VALIDATION
# =============================================================================

println("\n", "="^70)
println("VISUAL VALIDATION")
println("="^70)

# Create comprehensive validation plot
p1 = histogram(samples_bm, bins=50, normalize=:pdf, alpha=0.7, 
               color=:skyblue, label="Box-Muller samples",
               xlabel="Value", ylabel="Density",
               title="Histogram vs Theoretical Density")
x_range = range(minimum(samples_bm), maximum(samples_bm), length=200)
plot!(p1, x_range, pdf.(Normal(mu, sd), x_range), 
      linewidth=2, color=:red, label="N($mu, $sd) density")
vline!(p1, [sample_mean], linestyle=:dash, linewidth=2, 
       color=:blue, label="Sample mean: $(round(sample_mean, digits=2))")
vline!(p1, [mu], linestyle=:dash, linewidth=2, 
       color=:red, label="True mean: $mu")

# Q-Q plot
theoretical_quantiles = quantile.(Normal(mu, sd), (1:n) ./ (n + 1))
sample_quantiles = sort(samples_bm)
p2 = scatter(theoretical_quantiles, sample_quantiles, 
             markersize=2, alpha=0.5, color=:blue,
             xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
             title="Q-Q Plot", label="")
plot!(p2, [minimum(theoretical_quantiles), maximum(theoretical_quantiles)],
      [minimum(theoretical_quantiles), maximum(theoretical_quantiles)],
      linewidth=2, color=:red, label="")

# ECDF comparison
sorted_samples = sort(samples_bm)
ecdf_y = (1:length(sorted_samples)) ./ length(sorted_samples)
p3 = plot(sorted_samples, ecdf_y, linewidth=2, color=:blue,
          xlabel="Value", ylabel="Cumulative Probability",
          title="ECDF vs Theoretical CDF", label="Empirical CDF")
plot!(p3, sorted_samples, cdf.(Normal(mu, sd), sorted_samples),
      linewidth=2, color=:red, linestyle=:dash, label="Theoretical CDF")

# Box plot
p4 = boxplot(["Samples"], samples_bm, fillalpha=0.7, color=:lightblue,
             ylabel="Value", title="Box Plot", legend=false)
hline!(p4, [mu], linestyle=:dash, linewidth=2, color=:red, 
       label="True mean: $mu")
hline!(p4, [sample_mean], linestyle=:dash, linewidth=2, color=:blue,
       label="Sample mean: $(round(sample_mean, digits=2))")

# Running mean convergence
running_mean = cumsum(samples_bm) ./ (1:length(samples_bm))
p5 = plot(running_mean, linewidth=1, color=:blue,
          xlabel="Sample size", ylabel="Mean",
          title="Convergence of Sample Mean", label="Running mean")
hline!(p5, [mu], linestyle=:dash, linewidth=2, color=:red, label="True mean: $mu")
ci_lower = mu .- 1.96 .* sd ./ sqrt.(1:length(samples_bm))
ci_upper = mu .+ 1.96 .* sd ./ sqrt.(1:length(samples_bm))
plot!(p5, 1:length(samples_bm), ci_lower, fillrange=ci_upper,
      fillalpha=0.3, color=:red, label="95% CI", linewidth=0)

# Variance convergence
running_var = [var(samples_bm[1:i]) for i in 2:length(samples_bm)]
p6 = plot(2:length(samples_bm), running_var, linewidth=1, color=:blue,
          xlabel="Sample size", ylabel="Variance",
          title="Convergence of Sample Variance", label="Running variance")
hline!(p6, [sd^2], linestyle=:dash, linewidth=2, color=:red,
       label="True variance: $(sd^2)")

# Combine plots
plot_combined = plot(p1, p2, p3, p4, p5, p6, 
                     layout=(2, 3), size=(1500, 1000),
                     plot_title="Box-Muller Transformation Validation: N(10, 3) with n=2000")
savefig(plot_combined, "../plots/simulation_boxmuller_validation.png")
println("\nSaved: ../plots/simulation_boxmuller_validation.png")

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

println("\n", "="^70)
println("STATISTICAL TESTS")
println("="^70)

# 1. Shapiro-Wilk test for normality
println("\n1. Shapiro-Wilk test for normality:")
println("   H0: Data comes from a normal distribution")
# Note: Julia's ShapiroWilkTest is in HypothesisTests
sw_test = ShapiroWilkTest(samples_bm)
@printf("   Test statistic W = %.6f\n", sw_test.W)
@printf("   P-value = %.4f\n", pvalue(sw_test))
if pvalue(sw_test) > 0.05
    println("   ✓ Cannot reject H0: Data appears normally distributed (p > 0.05)")
else
    println("   ✗ Reject H0: Data does not appear normally distributed (p ≤ 0.05)")
end

# 2. Kolmogorov-Smirnov test
println("\n2. Kolmogorov-Smirnov test:")
println("   H0: Data comes from Normal($mu, $sd)")
ks_test = ExactOneSampleKSTest(samples_bm, Normal(mu, sd))
@printf("   Test statistic D = %.6f\n", ks_test.δ)
@printf("   P-value = %.4f\n", pvalue(ks_test))
if pvalue(ks_test) > 0.05
    println("   ✓ Cannot reject H0: Data consistent with N($mu,$sd) (p > 0.05)")
else
    println("   ✗ Reject H0: Data not consistent with N($mu,$sd) (p ≤ 0.05)")
end

# 3. t-test for mean
println("\n3. One-sample t-test for mean:")
println("   H0: Population mean = $mu")
t_test = OneSampleTTest(samples_bm, mu)
@printf("   Test statistic t = %.4f\n", t_test.t)
@printf("   P-value = %.4f\n", pvalue(t_test))
if pvalue(t_test) > 0.05
    println("   ✓ Cannot reject H0: Mean consistent with $mu (p > 0.05)")
else
    println("   ✗ Reject H0: Mean different from $mu (p ≤ 0.05)")
end

# 4. Chi-squared test for variance
println("\n4. Chi-squared test for variance:")
println("   H0: Population variance = $(sd^2)")
chi_stat = (n - 1) * sample_sd^2 / sd^2
p_chi = 2 * min(cdf(Chisq(n-1), chi_stat), 1 - cdf(Chisq(n-1), chi_stat))
@printf("   Test statistic χ² = %.4f\n", chi_stat)
@printf("   P-value = %.4f\n", p_chi)
if p_chi > 0.05
    println("   ✓ Cannot reject H0: Variance consistent with $(sd^2) (p > 0.05)")
else
    println("   ✗ Reject H0: Variance different from $(sd^2) (p ≤ 0.05)")
end

# 5. Independence test for Z1 and Z2
println("\n5. Testing independence of Z1 and Z2 pairs:")
println("   H0: Z1 and Z2 are independent (correlation = 0)")
# Generate pairs
Random.seed!(123)
n_pairs = 1000
u1 = rand(n_pairs)
u2 = rand(n_pairs)
r = sqrt.(-2 .* log.(u1))
theta = 2 .* π .* u2
z1 = r .* cos.(theta)
z2 = r .* sin.(theta)

corr_val = cor(z1, z2)
# Approximate p-value using Fisher transformation
z_fisher = 0.5 * log((1 + corr_val) / (1 - corr_val))
p_corr = 2 * (1 - cdf(Normal(0, 1/sqrt(n_pairs-3)), abs(z_fisher)))
@printf("   Correlation coefficient r = %.6f\n", corr_val)
@printf("   P-value = %.4f\n", p_corr)
if p_corr > 0.05
    println("   ✓ Cannot reject H0: Z1 and Z2 appear independent (p > 0.05)")
else
    println("   ✗ Reject H0: Z1 and Z2 appear correlated (p ≤ 0.05)")
end

# Create scatter plot of Z1 vs Z2
p_independence = scatter(z1, z2, alpha=0.5, markersize=2,
                         xlabel="Z1", ylabel="Z2",
                         title="Independence of Z1 and Z2\n(Correlation: r=$(round(corr_val, digits=4)), p=$(round(p_corr, digits=4)))",
                         label="", aspect_ratio=:equal)
hline!(p_independence, [0], color=:gray, linestyle=:dash, alpha=0.5, label="")
vline!(p_independence, [0], color=:gray, linestyle=:dash, alpha=0.5, label="")
savefig(p_independence, "../plots/simulation_boxmuller_independence.png")
println("\nSaved: ../plots/simulation_boxmuller_independence.png")

# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

println("\n", "="^70)
println("PERFORMANCE COMPARISON")
println("="^70)

n_test = 10000
n_reps = 100

# Time Box-Muller implementation
time_bm = @elapsed for _ in 1:n_reps
    bmnormal(n_test, mu, sd)
end

# Time built-in normal generator
time_builtin = @elapsed for _ in 1:n_reps
    rand(Normal(mu, sd), n_test)
end

println("\nGenerating $n_test samples, $n_reps repetitions:")
@printf("Box-Muller implementation: %.4f seconds\n", time_bm)
@printf("Built-in randn: %.4f seconds\n", time_builtin)
@printf("Ratio (Box-Muller / Built-in): %.2fx\n", time_bm/time_builtin)

# =============================================================================
# ADDITIONAL METHODS: REJECTION SAMPLING
# =============================================================================

println("\n", "="^70)
println("ADDITIONAL METHOD: REJECTION SAMPLING")
println("="^70)

println("\nRejection sampling is useful when:")
println("1. The inverse CDF is difficult to compute")
println("2. We can bound the target density with a proposal density")
println("\nExample: Sampling from Beta(2, 5) using Uniform(0, 1) proposal")

"""
    rejection_beta(n::Int, a::Float64=2.0, b::Float64=5.0)

Sample from Beta(a, b) using rejection sampling with Uniform(0,1) proposal.

The Beta(2,5) density is f(x) = 30*x*(1-x)^4 for x in [0,1].
Maximum occurs at x = 1/5, where f(1/5) ≈ 2.4576.
We use M = 2.5 to ensure M*g(x) ≥ f(x) for all x.
"""
function rejection_beta(n::Int, a::Float64=2.0, b::Float64=5.0)
    samples = Float64[]
    attempts = 0
    
    # Upper bound on Beta(a,b) density
    M = 2.5
    
    while length(samples) < n
        # Propose from Uniform(0, 1)
        u = rand()
        
        # Accept/reject
        acceptance_prob = pdf(Beta(a, b), u) / M
        if rand() < acceptance_prob
            push!(samples, u)
        end
        
        attempts += 1
    end
    
    acceptance_rate = n / attempts
    return samples, acceptance_rate
end

# Generate samples
Random.seed!(456)
samples_beta, acc_rate = rejection_beta(1000)

println("\nGenerated 1000 samples from Beta(2, 5)")
@printf("Acceptance rate: %.4f (%.2f%%)\n", acc_rate, acc_rate*100)
@printf("Sample mean: %.4f (expected: %.4f)\n", mean(samples_beta), 2/(2+5))
@printf("Sample variance: %.4f (expected: %.4f)\n", var(samples_beta), 2*5/((2+5)^2*(2+5+1)))

# Visualize rejection sampling
p_beta1 = histogram(samples_beta, bins=40, normalize=:pdf, alpha=0.7,
                    color=:skyblue, label="Rejection samples",
                    xlabel="Value", ylabel="Density",
                    title="Rejection Sampling: Beta(2, 5)")
x_range = range(0, 1, length=200)
plot!(p_beta1, x_range, pdf.(Beta(2, 5), x_range),
      linewidth=2, color=:red, label="Beta(2, 5) density")
hline!(p_beta1, [2.5], linestyle=:dash, linewidth=2,
       color=:green, label="Proposal bound M*g(x)")

# Q-Q plot
theoretical_quantiles_beta = quantile.(Beta(2, 5), (1:1000) ./ 1001)
sample_quantiles_beta = sort(samples_beta)
p_beta2 = scatter(theoretical_quantiles_beta, sample_quantiles_beta,
                  markersize=2, alpha=0.5, color=:blue,
                  xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
                  title="Q-Q Plot: Beta(2, 5)", label="")
plot!(p_beta2, [0, 1], [0, 1], linewidth=2, color=:red, label="")

plot_rejection = plot(p_beta1, p_beta2, layout=(1, 2), size=(1400, 500))
savefig(plot_rejection, "../plots/simulation_rejection_sampling.png")
println("Saved: ../plots/simulation_rejection_sampling.png")

# =============================================================================
# POLAR BOX-MULLER METHOD
# =============================================================================

println("\n", "="^70)
println("POLAR BOX-MULLER METHOD")
println("="^70)

println("\nThe polar Box-Muller method avoids trigonometric functions:")
println("1. Generate U1, U2 ~ Uniform(-1, 1)")
println("2. Calculate S = U1² + U2²")
println("3. If S ≥ 1, reject and try again")
println("4. Z1 = U1 * sqrt(-2*ln(S)/S)")
println("5. Z2 = U2 * sqrt(-2*ln(S)/S)")

"""
    bmnormal_polar(n::Int, mu::Float64=0.0, sd::Float64=1.0)

Generate n draws from Normal(mu, sd) using polar Box-Muller method.
This method avoids computing sine and cosine.
"""
function bmnormal_polar(n::Int, mu::Float64=0.0, sd::Float64=1.0)
    samples = Float64[]
    
    while length(samples) < n
        # Generate uniform in (-1, 1) x (-1, 1)
        u1 = 2 * rand() - 1
        u2 = 2 * rand() - 1
        s = u1^2 + u2^2
        
        # Reject if outside unit circle
        if s >= 1 || s == 0
            continue
        end
        
        # Polar transformation
        factor = sqrt(-2 * log(s) / s)
        z1 = u1 * factor
        z2 = u2 * factor
        
        push!(samples, z1, z2)
    end
    
    # Take first n samples and transform
    z = samples[1:n]
    x = mu .+ sd .* z
    
    return x
end

# Test polar method
println("\nTesting polar Box-Muller:")
Random.seed!(789)
samples_polar = bmnormal_polar(2000, 10.0, 3.0)
@printf("Sample mean: %.4f (expected: 10)\n", mean(samples_polar))
@printf("Sample SD:   %.4f (expected: 3)\n", std(samples_polar))

# Compare methods visually
p_polar1 = histogram(samples_bm, bins=50, normalize=:pdf, alpha=0.6,
                     color=:skyblue, label="Standard Box-Muller",
                     xlabel="Value", ylabel="Density",
                     title="Comparison of Box-Muller Methods")
histogram!(p_polar1, samples_polar, bins=50, normalize=:pdf, alpha=0.6,
           color=:lightcoral, label="Polar Box-Muller")
x_range = range(min(minimum(samples_bm), minimum(samples_polar)),
                max(maximum(samples_bm), maximum(samples_polar)), length=200)
plot!(p_polar1, x_range, pdf.(Normal(10, 3), x_range),
      linewidth=2, color=:black, label="N(10, 3) density")

# Q-Q plot for polar method
theoretical_quantiles_polar = quantile.(Normal(10, 3), (1:2000) ./ 2001)
sample_quantiles_polar = sort(samples_polar)
p_polar2 = scatter(theoretical_quantiles_polar, sample_quantiles_polar,
                   markersize=2, alpha=0.5, color=:blue,
                   xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
                   title="Q-Q Plot: Polar Box-Muller", label="")
plot!(p_polar2, [minimum(theoretical_quantiles_polar), maximum(theoretical_quantiles_polar)],
      [minimum(theoretical_quantiles_polar), maximum(theoretical_quantiles_polar)],
      linewidth=2, color=:red, label="")

plot_polar = plot(p_polar1, p_polar2, layout=(1, 2), size=(1400, 500))
savefig(plot_polar, "../plots/simulation_polar_comparison.png")
println("Saved: ../plots/simulation_polar_comparison.png")

# =============================================================================
# SUMMARY
# =============================================================================

println("\n", "="^70)
println("SUMMARY: RANDOM NUMBER GENERATION METHODS")
println("="^70)

println("\n1. INVERSE CDF METHOD")
println("   - Best when inverse CDF has closed form")
println("   - Example: Exponential, Uniform, Pareto")
println("   - X = F^(-1)(U) where U ~ Uniform(0,1)")

println("\n2. BOX-MULLER TRANSFORMATION")
println("   - Converts uniform to normal random variables")
println("   - Two variants: standard (uses trig) and polar (avoids trig)")
println("   - Generates pairs of independent normals")
println("   - Exact method (not approximate)")

println("\n3. REJECTION SAMPLING")
println("   - Useful when density is known up to a constant")
println("   - Requires proposal distribution and bound M")
println("   - Acceptance rate = 1/M (want M close to 1)")
println("   - Example: Beta, truncated distributions")

println("\n4. TRANSFORMATION METHOD")
println("   - Use properties of distributions")
println("   - Example: If Z ~ N(0,1), then X = μ + σZ ~ N(μ, σ²)")
println("   - Example: If Z ~ N(0,1), then Z² ~ χ²(1)")

println("\n", "="^70)
println("VALIDATION CHECKLIST")
println("="^70)

println("\n✓ Visual checks:")
println("  - Histogram matches theoretical density")
println("  - Q-Q plot shows points near diagonal")
println("  - ECDF matches theoretical CDF")
println("  - Running mean converges to true mean")

println("\n✓ Statistical tests:")
println("  - Shapiro-Wilk for normality")
println("  - Kolmogorov-Smirnov for distribution fit")
println("  - t-test for mean")
println("  - Chi-squared test for variance")
println("  - Correlation test for independence")

println("\n", "="^70)
println("Generated plots:")
println("  1. simulation_boxmuller_validation.png - 6-panel validation")
println("  2. simulation_boxmuller_independence.png - Z1 vs Z2 scatter")
println("  3. simulation_rejection_sampling.png - Beta distribution example")
println("  4. simulation_polar_comparison.png - Standard vs Polar methods")
println("="^70)

println("\nSIMULATION TUTORIAL COMPLETE!")
println("All methods validated and working correctly.")
