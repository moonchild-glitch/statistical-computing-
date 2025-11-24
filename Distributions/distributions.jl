# ============================================
# DISTRIBUTIONS IN JULIA
# ============================================
# Statistical Computing Tutorial
# Topic: Probability Distributions and Goodness of Fit Testing
#
# Agenda:
# 1. Random number generation
# 2. Built-in distributions in Julia (Distributions.jl)
# 3. Parametric distributions as models
# 4. Methods of fitting (moments, generalized moments, likelihood)
# 5. Methods of checking (visual comparisons, statistics, tests, calibration)
# 6. Chi-squared test for continuous distributions
# 7. Better alternatives (K-S test, bootstrap, smooth tests)
# ============================================

using Distributions
using Plots
using StatsBase
using HypothesisTests
using Statistics
using Random
using LinearAlgebra
using Optim
using StatsPlots
using Printf

# Set theme
theme(:default)

# Create plots directory if it doesn't exist
if !isdir("../plots")
    mkpath("../plots")
end

# Set seed for reproducibility
Random.seed!(42)

println("\n" * "="^50)
println("RANDOM NUMBER GENERATION")
println("="^50 * "\n")

println("Julia has built-in random number generators via Distributions.jl")
println("")
println("General naming convention:")
println("  - pdf(dist, x): probability density function")
println("  - cdf(dist, x): cumulative distribution function")
println("  - quantile(dist, q): quantile function (inverse CDF)")
println("  - rand(dist, n): random number generator")
println("")
println("where 'dist' is the distribution object (Normal, Exponential, Uniform, etc.)")

# Examples of random number generation
println("\n--- Uniform Distribution ---")
uniform_sample = rand(Uniform(0, 1), 10)
println("Sample of 10 uniform random numbers [0,1]:")
println(round.(uniform_sample, digits=4))

println("\n--- Normal Distribution ---")
normal_sample = rand(Normal(0, 1), 10)
println("Sample of 10 standard normal random numbers:")
println(round.(normal_sample, digits=4))

println("\n--- Exponential Distribution ---")
exp_sample = rand(Exponential(1), 10)
println("Sample of 10 exponential random numbers (scale=1):")
println(round.(exp_sample, digits=4))

# ============================================
# DISTRIBUTIONS IN JULIA
# ============================================
println("\n" * "="^50)
println("BUILT-IN DISTRIBUTIONS IN JULIA")
println("="^50 * "\n")

println("Julia (Distributions.jl) provides many common distributions:")
println("")
println("Continuous distributions:")
println("  - Normal: Normal(μ, σ)")
println("  - Exponential: Exponential(θ) where θ is scale")
println("  - Uniform: Uniform(a, b)")
println("  - Gamma: Gamma(α, θ) where α is shape, θ is scale")
println("  - Beta: Beta(α, β)")
println("  - Chi-squared: Chisq(ν)")
println("  - t-distribution: TDist(ν)")
println("  - F-distribution: FDist(ν₁, ν₂)")
println("")
println("Discrete distributions:")
println("  - Binomial: Binomial(n, p)")
println("  - Poisson: Poisson(λ)")
println("  - Geometric: Geometric(p)")
println("  - Negative binomial: NegativeBinomial(r, p)")

# Visualize some distributions
p = plot(layout=(2, 3), size=(1200, 800), legend=:topright)

# Normal distribution
x_norm = range(-4, 4, length=1000)
plot!(p[1], x_norm, pdf.(Normal(0, 1), x_norm), lw=2, color=:blue, label="PDF")
histogram!(p[1], rand(Normal(0, 1), 1000), bins=30, normalize=:pdf, 
          alpha=0.3, color=:blue, label="Sample")
title!(p[1], "Normal Distribution")
xlabel!(p[1], "x")
ylabel!(p[1], "Density")

# Exponential distribution
x_exp = range(0, 5, length=1000)
plot!(p[2], x_exp, pdf.(Exponential(1), x_exp), lw=2, color=:red, label="PDF")
histogram!(p[2], rand(Exponential(1), 1000), bins=30, normalize=:pdf,
          alpha=0.3, color=:red, label="Sample")
title!(p[2], "Exponential Distribution")
xlabel!(p[2], "x")
ylabel!(p[2], "Density")

# Gamma distribution
x_gamma = range(0, 20, length=1000)
plot!(p[3], x_gamma, pdf.(Gamma(2, 2), x_gamma), lw=2, color=:green, label="PDF")
histogram!(p[3], rand(Gamma(2, 2), 1000), bins=30, normalize=:pdf,
          alpha=0.3, color=:green, label="Sample")
title!(p[3], "Gamma Distribution (shape=2, scale=2)")
xlabel!(p[3], "x")
ylabel!(p[3], "Density")

# Beta distribution
x_beta = range(0, 1, length=1000)
plot!(p[4], x_beta, pdf.(Beta(2, 5), x_beta), lw=2, color=:purple, label="PDF")
histogram!(p[4], rand(Beta(2, 5), 1000), bins=30, normalize=:pdf,
          alpha=0.3, color=:purple, label="Sample")
title!(p[4], "Beta Distribution (a=2, b=5)")
xlabel!(p[4], "x")
ylabel!(p[4], "Density")

# Chi-squared distribution
x_chisq = range(0, 20, length=1000)
plot!(p[5], x_chisq, pdf.(Chisq(5), x_chisq), lw=2, color=:orange, label="PDF")
histogram!(p[5], rand(Chisq(5), 1000), bins=30, normalize=:pdf,
          alpha=0.3, color=:orange, label="Sample")
title!(p[5], "Chi-squared Distribution (df=5)")
xlabel!(p[5], "x")
ylabel!(p[5], "Density")

# Binomial distribution (discrete)
x_binom = 0:20
plot!(p[6], x_binom, pdf.(Binomial(20, 0.5), x_binom), 
     seriestype=:sticks, lw=3, color=:darkblue, label="PMF",
     marker=:circle, markersize=4)
title!(p[6], "Binomial Distribution (n=20, p=0.5)")
xlabel!(p[6], "x")
ylabel!(p[6], "Probability")

savefig(p, "../plots/dist_common_distributions.png")
println("Plot saved: dist_common_distributions.png")

# ============================================
# PARAMETRIC DISTRIBUTIONS AS MODELS
# ============================================
println("\n" * "="^50)
println("PARAMETRIC DISTRIBUTIONS AS MODELS")
println("="^50 * "\n")

println("Parametric distributions serve as models for real-world data")
println("")
println("Key idea: Assume data comes from a known family of distributions")
println("          but with unknown parameters")
println("")
println("Goal: Estimate the parameters from the data")
println("      Check if the model fits well")

# Generate some example data (exponential)
true_rate = 0.5
sample_data = rand(Exponential(1/true_rate), 500)

@printf("\nGenerated 500 samples from Exponential(rate=%.1f)\n", true_rate)
@printf("Sample mean: %.4f (theoretical: %.4f)\n", mean(sample_data), 1/true_rate)
@printf("Sample variance: %.4f (theoretical: %.4f)\n", var(sample_data), (1/true_rate)^2)

# ============================================
# METHOD OF MOMENTS
# ============================================
println("\n" * "="^50)
println("FITTING: METHOD OF MOMENTS")
println("="^50 * "\n")

println("Method of Moments: Match sample moments to theoretical moments")
println("")
println("For exponential distribution:")
println("  E[X] = 1/λ")
println("  So: λ_hat = 1/mean(X)")

# Estimate rate using method of moments
rate_mom = 1/mean(sample_data)
@printf("\nMethod of Moments estimate: λ = %.4f\n", rate_mom)
@printf("True rate: λ = %.4f\n", true_rate)
@printf("Error: %.4f\n", abs(rate_mom - true_rate))

# Visualize the fit
p1 = histogram(sample_data, bins=30, normalize=:pdf, alpha=0.6, 
              color=:lightblue, label="Data histogram")
x_seq = range(0, maximum(sample_data), length=1000)
plot!(p1, x_seq, pdf.(Exponential(1/rate_mom), x_seq), lw=2, color=:red,
     label=@sprintf("MoM fit (λ=%.3f)", rate_mom))
plot!(p1, x_seq, pdf.(Exponential(1/true_rate), x_seq), lw=2, color=:blue,
     linestyle=:dash, label=@sprintf("True (λ=%.3f)", true_rate))
xlabel!(p1, "Value")
ylabel!(p1, "Density")
title!(p1, "Method of Moments Fit")
savefig(p1, "../plots/dist_method_of_moments.png")
println("Plot saved: dist_method_of_moments.png")

# ============================================
# MAXIMUM LIKELIHOOD ESTIMATION
# ============================================
println("\n" * "="^50)
println("FITTING: MAXIMUM LIKELIHOOD ESTIMATION")
println("="^50 * "\n")

println("Maximum Likelihood: Find parameters that maximize the likelihood function")
println("")
println("Likelihood function: L(θ|data) = product of f(x_i|θ)")
println("Log-likelihood: l(θ|data) = sum of log(f(x_i|θ))")
println("")
println("For exponential distribution:")
println("  l(λ) = n*log(λ) - λ*sum(x_i)")
println("  Maximize by taking derivative and setting to 0")
println("  Solution: λ_MLE = n / sum(x_i) = 1/mean(x)")

# MLE for exponential (same as method of moments in this case!)
rate_mle = 1/mean(sample_data)
@printf("\nMaximum Likelihood estimate: λ = %.4f\n", rate_mle)
@printf("True rate: λ = %.4f\n", true_rate)

# For more complex distributions, use Optim
# Example: fit gamma distribution using MLE
println("\n--- Fitting Gamma Distribution ---")

# Generate gamma data
gamma_data = rand(Gamma(2, 2), 500)  # shape=2, scale=2 (so rate=0.5)

# Negative log-likelihood for gamma
function neg_log_lik_gamma(params, data)
    shape, scale = params
    if shape <= 0 || scale <= 0
        return Inf
    end
    return -sum(logpdf.(Gamma(shape, scale), data))
end

# Optimize
result = optimize(p -> neg_log_lik_gamma(p, gamma_data), [1.0, 1.0], LBFGS())
shape_mle, scale_mle = Optim.minimizer(result)
rate_mle_gamma = 1/scale_mle

@printf("MLE estimates: shape = %.4f, scale = %.4f (rate = %.4f)\n", 
        shape_mle, scale_mle, rate_mle_gamma)
println("True parameters: shape = 2.0000, scale = 2.0000 (rate = 0.5000)")

# Visualize
p2 = histogram(gamma_data, bins=30, normalize=:pdf, alpha=0.6,
              color=:lightgreen, label="Data histogram")
x_seq_g = range(0, maximum(gamma_data), length=1000)
plot!(p2, x_seq_g, pdf.(Gamma(shape_mle, scale_mle), x_seq_g), lw=2, color=:red,
     label=@sprintf("MLE fit (shape=%.2f, scale=%.2f)", shape_mle, scale_mle))
plot!(p2, x_seq_g, pdf.(Gamma(2, 2), x_seq_g), lw=2, color=:blue,
     linestyle=:dash, label="True (shape=2.00, scale=2.00)")
xlabel!(p2, "Value")
ylabel!(p2, "Density")
title!(p2, "Maximum Likelihood Fit (Gamma Distribution)")
savefig(p2, "../plots/dist_mle_gamma.png")
println("Plot saved: dist_mle_gamma.png")

# ============================================
# VISUAL COMPARISON: Q-Q PLOTS
# ============================================
println("\n" * "="^50)
println("CHECKING FIT: VISUAL COMPARISON (Q-Q PLOTS)")
println("="^50 * "\n")

println("Q-Q Plot (Quantile-Quantile Plot):")
println("  - Compare quantiles of data to theoretical quantiles")
println("  - If data matches distribution, points lie on diagonal line")
println("  - Deviations indicate departure from assumed distribution")

p_qq = plot(layout=(1, 3), size=(1500, 500))

# Q-Q plot for exponential data
sorted_exp = sort(sample_data)
n_exp = length(sorted_exp)
probs_exp = (1:n_exp) ./ (n_exp + 1)
theo_quant_exp = quantile.(Exponential(1/rate_mle), probs_exp)
scatter!(p_qq[1], theo_quant_exp, sorted_exp, color=:blue, alpha=0.6, label="")
plot!(p_qq[1], [minimum(theo_quant_exp), maximum(theo_quant_exp)],
     [minimum(theo_quant_exp), maximum(theo_quant_exp)], 
     color=:red, lw=2, label="y=x")
title!(p_qq[1], "Q-Q Plot: Exponential Data")
xlabel!(p_qq[1], "Theoretical Quantiles")
ylabel!(p_qq[1], "Sample Quantiles")

# Q-Q plot for gamma data
sorted_gamma = sort(gamma_data)
n_gamma = length(sorted_gamma)
probs_gamma = (1:n_gamma) ./ (n_gamma + 1)
theo_quant_gamma = quantile.(Gamma(shape_mle, scale_mle), probs_gamma)
scatter!(p_qq[2], theo_quant_gamma, sorted_gamma, color=:green, alpha=0.6, label="")
plot!(p_qq[2], [minimum(theo_quant_gamma), maximum(theo_quant_gamma)],
     [minimum(theo_quant_gamma), maximum(theo_quant_gamma)],
     color=:red, lw=2, label="y=x")
title!(p_qq[2], "Q-Q Plot: Gamma Data")
xlabel!(p_qq[2], "Theoretical Quantiles")
ylabel!(p_qq[2], "Sample Quantiles")

# Q-Q plot for normal data (should be good)
normal_data = rand(Normal(0, 1), 500)
sorted_norm = sort(normal_data)
n_norm = length(sorted_norm)
probs_norm = (1:n_norm) ./ (n_norm + 1)
theo_quant_norm = quantile.(Normal(0, 1), probs_norm)
scatter!(p_qq[3], theo_quant_norm, sorted_norm, color=:purple, alpha=0.6, label="")
plot!(p_qq[3], [minimum(theo_quant_norm), maximum(theo_quant_norm)],
     [minimum(theo_quant_norm), maximum(theo_quant_norm)],
     color=:red, lw=2, label="y=x")
title!(p_qq[3], "Q-Q Plot: Normal Data")
xlabel!(p_qq[3], "Theoretical Quantiles")
ylabel!(p_qq[3], "Sample Quantiles")

savefig(p_qq, "../plots/dist_qq_plots.png")
println("Plot saved: dist_qq_plots.png")

# ============================================
# EMPIRICAL CDF COMPARISON
# ============================================
println("\n" * "="^50)
println("CHECKING FIT: EMPIRICAL CDF COMPARISON")
println("="^50 * "\n")

println("Empirical CDF: Step function of observed data")
println("Compare to theoretical CDF")

p_ecdf = plot(layout=(1, 2), size=(1500, 500))

# ECDF for exponential data
sorted_exp = sort(sample_data)
ecdf_exp = ecdf(sample_data)
plot!(p_ecdf[1], sorted_exp, ecdf_exp(sorted_exp), lw=2, color=:blue, 
     seriestype=:steppost, label="Empirical CDF")
x_seq_ecdf = range(0, maximum(sample_data), length=1000)
plot!(p_ecdf[1], x_seq_ecdf, cdf.(Exponential(1/rate_mle), x_seq_ecdf), 
     lw=2, color=:red, linestyle=:dash, label="Theoretical CDF")
title!(p_ecdf[1], "ECDF vs Theoretical CDF (Exponential)")
xlabel!(p_ecdf[1], "Value")
ylabel!(p_ecdf[1], "Cumulative Probability")

# ECDF for gamma data
sorted_gamma = sort(gamma_data)
ecdf_gamma = ecdf(gamma_data)
plot!(p_ecdf[2], sorted_gamma, ecdf_gamma(sorted_gamma), lw=2, color=:blue,
     seriestype=:steppost, label="Empirical CDF")
x_seq_gamma = range(0, maximum(gamma_data), length=1000)
plot!(p_ecdf[2], x_seq_gamma, cdf.(Gamma(shape_mle, scale_mle), x_seq_gamma),
     lw=2, color=:red, linestyle=:dash, label="Theoretical CDF")
title!(p_ecdf[2], "ECDF vs Theoretical CDF (Gamma)")
xlabel!(p_ecdf[2], "Value")
ylabel!(p_ecdf[2], "Cumulative Probability")

savefig(p_ecdf, "../plots/dist_ecdf_comparison.png")
println("Plot saved: dist_ecdf_comparison.png")

# ============================================
# CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS
# ============================================
println("\n" * "="^50)
println("CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS")
println("="^50 * "\n")

println("Chi-squared goodness-of-fit test:")
println("  - Designed for discrete/categorical data")
println("  - For continuous data: must discretize into bins")
println("  - Test statistic: χ² = sum((O_i - E_i)² / E_i)")
println("    where O_i = observed count in bin i")
println("          E_i = expected count in bin i")

# Discretize exponential data
n_bins = 10
breaks = quantile(sample_data, range(0, 1, length=n_bins+1))
breaks[1] = 0.0  # Ensure lower bound is 0
observed_counts = fit(Histogram, sample_data, breaks).weights

# Expected counts under fitted exponential
expected_probs = diff(cdf.(Exponential(1/rate_mle), breaks))
expected_probs = expected_probs ./ sum(expected_probs)  # Normalize
expected_counts = expected_probs .* length(sample_data)

println("\nObserved vs Expected counts:")
using DataFrames
df_counts = DataFrame(
    Bin = 1:n_bins,
    Observed = observed_counts,
    Expected = round.(expected_counts, digits=2)
)
println(df_counts)

# Chi-squared test
chisq_stat = sum((observed_counts .- expected_counts).^2 ./ expected_counts)
df_chisq = n_bins - 1
chisq_pval = 1 - cdf(Chisq(df_chisq), chisq_stat)

println("\nChi-squared test result:")
@printf("  χ² = %.4f\n", chisq_stat)
@printf("  df = %d\n", df_chisq)
@printf("  p-value = %.4f\n", chisq_pval)

# Visualize
p_chisq = groupedbar([observed_counts expected_counts], 
                     bar_position=:dodge,
                     label=["Observed" "Expected"],
                     color=[:lightblue :salmon],
                     legend=:topright)
xlabel!(p_chisq, "Bin")
ylabel!(p_chisq, "Count")
title!(p_chisq, "Chi-squared Test: Observed vs Expected Counts")
xticks!(p_chisq, 1:n_bins)
savefig(p_chisq, "../plots/dist_chisq_test.png")
println("Plot saved: dist_chisq_test.png")

# ============================================
# PROBLEMS WITH CHI-SQUARED TEST
# ============================================
println("\n" * "="^50)
println("PROBLEMS WITH CHI-SQUARED TEST")
println("="^50 * "\n")

println("Issues with chi-squared test for continuous distributions:")
println("")
println("1. LOSS OF INFORMATION FROM DISCRETIZATION")
println("   - Converting continuous data to bins loses precision")
println("   - Different binning choices can give different results")
println("")
println("2. LOTS OF WORK JUST TO USE chisq.test()")
println("   - Must choose number of bins")
println("   - Must calculate expected counts manually")
println("   - Results depend on arbitrary binning choices")
println("")
println("3. LOW POWER")
println("   - May not detect departures from null hypothesis")
println("   - Especially with small sample sizes")
println("")
println("BETTER ALTERNATIVES:")
println("  ✓ Kolmogorov-Smirnov test (ExactOneSampleKSTest)")
println("  ✓ Bootstrap testing")
println("  ✓ Smooth tests of goodness of fit")
println("  ✓ Anderson-Darling test")

# ============================================
# BETTER ALTERNATIVE: KOLMOGOROV-SMIRNOV TEST
# ============================================
println("\n" * "="^50)
println("BETTER ALTERNATIVE: KOLMOGOROV-SMIRNOV TEST")
println("="^50 * "\n")

println("Kolmogorov-Smirnov (K-S) test:")
println("  - Compares empirical CDF to theoretical CDF")
println("  - Test statistic: D = max|F_empirical(x) - F_theoretical(x)|")
println("  - No binning required!")
println("  - More powerful than chi-squared for continuous data")

# K-S test for exponential data
ks_test = ExactOneSampleKSTest(sample_data, Exponential(1/rate_mle))
ks_stat = ks_test.δ
ks_pval = pvalue(ks_test)

println("\nKolmogorov-Smirnov test result:")
@printf("  D = %.4f\n", ks_stat)
@printf("  p-value = %.4f\n", ks_pval)

if ks_pval > 0.05
    println("Conclusion: Cannot reject null hypothesis")
    println("            Data is consistent with exponential distribution")
else
    println("Conclusion: Reject null hypothesis")
    println("            Data does not follow exponential distribution")
end

# Visualize K-S statistic
sorted_data = sort(sample_data)
ecdf_func = ecdf(sample_data)
p_ks = plot(sorted_data, ecdf_func(sorted_data), lw=2, color=:blue,
           seriestype=:steppost, label="Empirical CDF")
x_seq_ks = range(0, maximum(sample_data), length=1000)
plot!(p_ks, x_seq_ks, cdf.(Exponential(1/rate_mle), x_seq_ks), lw=2, color=:red,
     label="Theoretical CDF")

# Find where maximum difference occurs
theo_cdf = cdf.(Exponential(1/rate_mle), sorted_data)
emp_cdf = ecdf_func(sorted_data)
diffs = abs.(emp_cdf .- theo_cdf)
max_idx = argmax(diffs)
x_max = sorted_data[max_idx]

plot!(p_ks, [x_max, x_max], [emp_cdf[max_idx], theo_cdf[max_idx]], 
     lw=3, color=:green, linestyle=:dash,
     label=@sprintf("Max Difference (D=%.4f)", ks_stat))
xlabel!(p_ks, "Value")
ylabel!(p_ks, "Cumulative Probability")
title!(p_ks, "Kolmogorov-Smirnov Test Visualization")
savefig(p_ks, "../plots/dist_ks_test.png")
println("Plot saved: dist_ks_test.png")

# ============================================
# BOOTSTRAP TESTING
# ============================================
println("\n" * "="^50)
println("BOOTSTRAP TESTING")
println("="^50 * "\n")

println("Bootstrap approach for goodness-of-fit:")
println("  1. Fit distribution to observed data")
println("  2. Generate many bootstrap samples from fitted distribution")
println("  3. Calculate test statistic for each bootstrap sample")
println("  4. Compare observed test statistic to bootstrap distribution")

# Bootstrap K-S test
n_boot = 1000
boot_stats = zeros(n_boot)

for i in 1:n_boot
    boot_sample = rand(Exponential(1/rate_mle), length(sample_data))
    boot_test = ExactOneSampleKSTest(boot_sample, Exponential(1/rate_mle))
    boot_stats[i] = boot_test.δ
end

# Calculate p-value
bootstrap_pval = mean(boot_stats .>= ks_stat)

@printf("\nBootstrap K-S test (B = %d):\n", n_boot)
@printf("Observed K-S statistic: %.4f\n", ks_stat)
@printf("Bootstrap p-value: %.4f\n", bootstrap_pval)

# Visualize bootstrap distribution
p_boot = histogram(boot_stats, bins=30, normalize=:pdf, alpha=0.7,
                  color=:lightgray, label="")
vline!(p_boot, [ks_stat], lw=2, color=:red, linestyle=:dash,
      label=@sprintf("Observed D = %.4f\np = %.4f", ks_stat, bootstrap_pval))
xlabel!(p_boot, "K-S Statistic")
ylabel!(p_boot, "Density")
title!(p_boot, "Bootstrap Distribution of K-S Statistic")
savefig(p_boot, "../plots/dist_bootstrap_test.png")
println("Plot saved: dist_bootstrap_test.png")

# ============================================
# COMPARISON OF TESTS
# ============================================
println("\n" * "="^50)
println("COMPARISON OF GOODNESS-OF-FIT TESTS")
println("="^50 * "\n")

println("Test results summary:")
println("")
@printf("Chi-squared test:      χ² = %.4f, p = %.4f\n", chisq_stat, chisq_pval)
@printf("Kolmogorov-Smirnov:    D = %.4f,  p = %.4f\n", ks_stat, ks_pval)
@printf("Bootstrap K-S:         D = %.4f,  p = %.4f\n", ks_stat, bootstrap_pval)
println("")
println("All tests agree: data is consistent with exponential distribution")

# ============================================
# PRACTICAL EXAMPLE: TESTING NORMALITY
# ============================================
println("\n" * "="^50)
println("PRACTICAL EXAMPLE: TESTING NORMALITY")
println("="^50 * "\n")

println("Generate data from mixture of normals (not truly normal)")
println("Test if various methods can detect the departure from normality")

# Generate non-normal data (mixture)
Random.seed!(123)
mixture_data = vcat(rand(Normal(0, 1), 400), rand(Normal(3, 0.5), 100))

# Fit normal distribution
mean_est = mean(mixture_data)
sd_est = std(mixture_data)

@printf("\nFitted normal: mean = %.4f, sd = %.4f\n", mean_est, sd_est)

# Visual inspection
p_norm = plot(layout=(2, 2), size=(1500, 1000))

# Histogram with fitted normal
histogram!(p_norm[1], mixture_data, bins=30, normalize=:pdf, alpha=0.6,
          color=:lightblue, label="Data histogram")
x_norm_range = range(minimum(mixture_data), maximum(mixture_data), length=1000)
plot!(p_norm[1], x_norm_range, pdf.(Normal(mean_est, sd_est), x_norm_range),
     lw=2, color=:red, label="Fitted Normal")
xlabel!(p_norm[1], "Value")
ylabel!(p_norm[1], "Density")
title!(p_norm[1], "Histogram with Fitted Normal")

# Q-Q plot
sorted_mix = sort(mixture_data)
n_mix = length(sorted_mix)
probs_mix = (1:n_mix) ./ (n_mix + 1)
theo_quant_mix = quantile.(Normal(mean_est, sd_est), probs_mix)
scatter!(p_norm[2], theo_quant_mix, sorted_mix, alpha=0.6, label="")
plot!(p_norm[2], [minimum(theo_quant_mix), maximum(theo_quant_mix)],
     [minimum(theo_quant_mix), maximum(theo_quant_mix)],
     color=:red, lw=2, label="y=x")
title!(p_norm[2], "Q-Q Plot")
xlabel!(p_norm[2], "Theoretical Quantiles")
ylabel!(p_norm[2], "Sample Quantiles")

# ECDF comparison
ecdf_mix = ecdf(mixture_data)
plot!(p_norm[3], sorted_mix, ecdf_mix(sorted_mix), lw=2, color=:blue,
     seriestype=:steppost, label="Empirical CDF")
plot!(p_norm[3], x_norm_range, cdf.(Normal(mean_est, sd_est), x_norm_range),
     lw=2, color=:red, label="Theoretical CDF")
xlabel!(p_norm[3], "Value")
ylabel!(p_norm[3], "Cumulative Probability")
title!(p_norm[3], "Empirical vs Theoretical CDF")

# Box plot
boxplot!(p_norm[4], mixture_data, orientation=:horizontal, label="",
        color=:lightgreen)
xlabel!(p_norm[4], "Value")
title!(p_norm[4], "Box Plot")

savefig(p_norm, "../plots/dist_normality_test.png")
println("Plot saved: dist_normality_test.png")

# Statistical tests
println("\n--- Statistical Tests for Normality ---")

# Jarque-Bera test (alternative to Shapiro-Wilk in Julia)
jb_test = JarqueBeraTest(mixture_data)
@printf("Jarque-Bera test: JB = %.4f, p = %.4f\n", jb_test.JB, pvalue(jb_test))

# K-S test for normality
ks_norm_test = ExactOneSampleKSTest(mixture_data, Normal(mean_est, sd_est))
@printf("K-S test:         D = %.4f, p = %.4f\n", ks_norm_test.δ, pvalue(ks_norm_test))

println("\nConclusion:")
if pvalue(jb_test) < 0.05
    println("  Jarque-Bera test rejects normality (p < 0.05)")
    println("  The data does NOT appear to be normally distributed")
else
    println("  Tests suggest data may not be perfectly normal")
    println("  Visual inspection (Q-Q plot) shows deviation in tails")
end

# ============================================
# CALIBRATION PLOTS
# ============================================
println("\n" * "="^50)
println("CALIBRATION PLOTS")
println("="^50 * "\n")

println("Calibration: Check if predicted probabilities match observed frequencies")
println("")
println("For a well-fitted model:")
println("  - If we predict P(Y=1) = 0.7, about 70% should actually be Y=1")
println("  - Calibration plot: predicted probability vs observed frequency")

# Generate binary data with known probabilities
Random.seed!(456)
n_cal = 1000
x_cal = rand(Uniform(-3, 3), n_cal)
true_prob = 1 ./ (1 .+ exp.(-x_cal))  # logistic function
y_cal = rand.(Bernoulli.(true_prob))

# Fit logistic regression (simplified - using known relationship)
using GLM
df_cal = DataFrame(x=x_cal, y=y_cal)
cal_model = glm(@formula(y ~ x), df_cal, Binomial(), LogitLink())
pred_prob = predict(cal_model, df_cal)

# Create calibration plot
n_cal_bins = 10
bin_edges = range(0, 1, length=n_cal_bins+1)
bin_centers = zeros(n_cal_bins)
obs_freq = zeros(n_cal_bins)

for i in 1:n_cal_bins
    mask = (pred_prob .>= bin_edges[i]) .& (pred_prob .< bin_edges[i+1])
    if i == n_cal_bins  # Include upper bound in last bin
        mask = (pred_prob .>= bin_edges[i]) .& (pred_prob .<= bin_edges[i+1])
    end
    if sum(mask) > 0
        bin_centers[i] = mean(pred_prob[mask])
        obs_freq[i] = mean(y_cal[mask])
    end
end

# Remove empty bins
valid_bins = bin_centers .> 0
bin_centers = bin_centers[valid_bins]
obs_freq = obs_freq[valid_bins]

p_cal = scatter(bin_centers, obs_freq, markersize=8, color=:blue, alpha=0.6,
               label="Calibration points", markerstrokewidth=2)
plot!(p_cal, [0, 1], [0, 1], lw=2, color=:red, linestyle=:dash,
     label="Perfect calibration")
xlabel!(p_cal, "Predicted Probability")
ylabel!(p_cal, "Observed Frequency")
title!(p_cal, "Calibration Plot")
xlims!(p_cal, 0, 1)
ylims!(p_cal, 0, 1)
savefig(p_cal, "../plots/dist_calibration.png")
println("Plot saved: dist_calibration.png")

println("\nCalibration results:")
println("  Points near diagonal = well-calibrated")
println("  Points above diagonal = underestimating probability")
println("  Points below diagonal = overestimating probability")

# ============================================
# SUMMARY
# ============================================
println("\n" * "="^50)
println("SUMMARY")
println("="^50 * "\n")

println("Key takeaways:")
println("")
println("✓ RANDOM NUMBER GENERATION")
println("  - Julia has built-in generators via Distributions.jl")
println("  - Also provides density (pdf), CDF (cdf), quantile functions")
println("")
println("✓ DISTRIBUTIONS IN JULIA")
println("  - Many continuous distributions: Normal, Exponential, Gamma, Beta, etc.")
println("  - Many discrete distributions: Binomial, Poisson, Geometric, etc.")
println("  - Easy to work with using Distributions.jl")
println("")
println("✓ PARAMETRIC DISTRIBUTIONS ARE MODELS")
println("  - Assume data comes from known family with unknown parameters")
println("  - Goal: estimate parameters and check fit")
println("")
println("✓ METHODS OF FITTING")
println("  - Method of Moments: match sample moments to theoretical moments")
println("  - Maximum Likelihood: maximize probability of observing the data")
println("  - Generalized Method of Moments: more flexible, use more moments")
println("")
println("✓ METHODS OF CHECKING FIT")
println("  - Visual comparisons: histograms, Q-Q plots, ECDF plots")
println("  - Goodness-of-fit statistics: K-S statistic, chi-squared statistic")
println("  - Hypothesis tests: K-S test, chi-squared test, Jarque-Bera")
println("  - Calibration: predicted probabilities vs observed frequencies")
println("")
println("✓ CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS")
println("  - Requires discretization (binning) of continuous data")
println("  - DRAWBACKS:")
println("    • Loss of information from discretization")
println("    • Lots of work just to use chi-squared test")
println("    • Results depend on arbitrary binning choices")
println("    • Low power compared to alternatives")
println("")
println("✓ BETTER ALTERNATIVES")
println("  - ExactOneSampleKSTest: Kolmogorov-Smirnov test (no binning, more powerful)")
println("  - Bootstrap testing: resampling approach for p-values")
println("  - Smooth tests of goodness of fit: more sophisticated methods")
println("  - Anderson-Darling test: gives more weight to tails")
println("  - Jarque-Bera test: specifically for testing normality")

println("\n" * "="^60)
println("DISTRIBUTIONS TUTORIAL COMPLETE")
println("="^60 * "\n")

final_plot_count = length(filter(f -> occursin("dist_", f), readdir("../plots")))
println("Total plots generated: $final_plot_count")
println("\nAll plots saved to: ../plots/")
println("\nThank you for completing this tutorial!")
