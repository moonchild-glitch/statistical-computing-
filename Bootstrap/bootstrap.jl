# ============================================================================
# Bootstrap Methods in Julia
# ============================================================================

using Statistics
using Random
using Distributions
using StatsBase
using DataFrames
using Plots
using StatsPlots
using LinearAlgebra
using Printf

# Set random seed for reproducibility
Random.seed!(42)

# Set plotting defaults
default(size=(1200, 800), dpi=300)

println("="^80)
println("Bootstrap Methods - Julia Implementation")
println("="^80)
println()

# ============================================================================
# Exercise: Toy Collector Problem
# ============================================================================

println("\n", "="^80)
println("Exercise: Toy Collector Problem")
println("="^80)
println()

println("Children (and some adults) are frequently enticed to buy breakfast cereal")
println("in an effort to collect all the action figures. Assume there are 15 action")
println("figures and each cereal box contains exactly one with each figure being")
println("equally likely.\n")

println("Questions:")
println("1. Find the expected number of boxes needed to collect all 15 action figures.")
println("2. Find the standard deviation of the number of boxes needed to collect all")
println("   15 action figures.")
println("3. Now suppose we no longer have equal probabilities...\n")

# ============================================================================
# Part 1 & 2: Equal Probabilities (Theoretical Solution)
# ============================================================================

println("="^80)
println("Part 1 & 2: Equal Probabilities (Theoretical Solution)")
println("="^80)
println()

n = 15  # number of action figures

# Geometric Approach
println("=== Equal Probabilities (Theoretical - Geometric Approach) ===")
println("Expected number of boxes:")
println("  E[T] = 15/15 + 15/14 + 15/13 + ... + 15/1")

expected_boxes = sum(n / (n - i + 1) for i in 1:n)
@printf("  E[T] = %.5f\n", expected_boxes)
@printf("  E[T] ≈ %.2f\n\n", expected_boxes)

println("Variance calculation:")
println("  Var[T] = 15*(1-15/15)/(15/15)^2 + 15*(1-14/15)/(14/15)^2 + ... + 15*(1-1/15)/(1/15)^2")

variance_boxes = sum(n * (1 - (n - i + 1)/n) / ((n - i + 1)/n)^2 for i in 1:n)
sd_boxes_geometric = sqrt(variance_boxes)
@printf("  Var[T] = %.3f\n", variance_boxes)
@printf("  Var[T] ≈ %.2f\n\n", variance_boxes)
@printf("Standard deviation: %.5f\n", sd_boxes_geometric)
@printf("SD ≈ %.2f\n\n", sd_boxes_geometric)

# Harmonic Number Approach
println("=== Equal Probabilities (Theoretical - Harmonic Number Approach) ===")
harmonic_n = sum(1/i for i in 1:n)
expected_harmonic = n * harmonic_n
println("Expected number of boxes: E[T] = n * H_n")
@printf("  where H_n = sum(1/i) for i=1 to %d\n", n)
@printf("  H_%d = %.6f\n", n, harmonic_n)
@printf("  E[T] = %.5f\n\n", expected_harmonic)

variance_harmonic = n^2 * sum(1/i^2 for i in 1:n) - n * harmonic_n
sd_boxes_harmonic = sqrt(variance_harmonic)
println("Variance: Var[T] = n^2 * sum(1/i^2) - n * H_n")
@printf("  Var[T] = %.4f\n", variance_harmonic)
@printf("Standard deviation: %.5f\n\n", sd_boxes_harmonic)

# Verification
println("=== Verification: Both Methods Agree ===")
@printf("Geometric approach - Expected: %.4f\n", expected_boxes)
@printf("Harmonic approach  - Expected: %.4f\n", expected_harmonic)
@printf("Difference: %g\n\n", abs(expected_boxes - expected_harmonic))

@printf("Geometric approach - Variance: %.3f\n", variance_boxes)
@printf("Harmonic approach  - Variance: %.4f\n", variance_harmonic)
@printf("Difference: %.3f\n\n", abs(variance_boxes - variance_harmonic))

# ============================================================================
# Simulation with Equal Probabilities
# ============================================================================

println("\n", "="^80)
println("Simulation with Equal Probabilities")
println("="^80)
println()

function count_boxes_equal_prob(n_toys=15)
    """Simulate collecting all toys with equal probabilities."""
    collected = Set{Int}()
    boxes = 0
    while length(collected) < n_toys
        boxes += 1
        push!(collected, rand(1:n_toys))
    end
    return boxes
end

# Run simulation
trials_equal = 10000
sim_boxes_equal = [count_boxes_equal_prob(n) for _ in 1:trials_equal]

println("=== Equal Probabilities (Simulation) ===")
@printf("Expected number of boxes (simulated): %.4f\n", mean(sim_boxes_equal))
@printf("Standard deviation (simulated): %.5f\n\n", std(sim_boxes_equal))

println("Comparison with theory:")
@printf("Expected value difference: %.7f\n", abs(mean(sim_boxes_equal) - expected_boxes))
@printf("SD difference: %.8f\n\n", abs(std(sim_boxes_equal) - sd_boxes_harmonic))

# ============================================================================
# Part 3: Unequal Probabilities
# ============================================================================

println("="^80)
println("Part 3: Unequal Probabilities")
println("="^80)
println()

# Define probability table
prob_table = [0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 
              0.02, 0.02, 0.02, 0.02, 0.02]

println("=== Unequal Probabilities ===")
@printf("Sum of probabilities: %g\n", sum(prob_table))

# Create probability dataframe
figures = collect('A':'O')
prob_df = DataFrame(Figure = figures, Probability = prob_table)
println(prob_df)
println()

function box_count(prob_table)
    """Count boxes needed with unequal probabilities."""
    check = falses(length(prob_table))
    count = 0
    while !all(check)
        count += 1
        toy = sample(1:length(prob_table), Weights(prob_table))
        check[toy] = true
    end
    return count
end

# Part 3a: Expected number of boxes
trials = 1000
sim_boxes = [box_count(prob_table) for _ in 1:trials]

println("3a. Expected number of boxes (unequal probabilities):")
@printf("    Point estimate: %.3f\n", mean(sim_boxes))
println("    Example output: est = 115.468\n")

# Part 3b: Uncertainty estimate
mcse = std(sim_boxes) / sqrt(trials)
ci_lower = mean(sim_boxes) - 1.96 * mcse
ci_upper = mean(sim_boxes) + 1.96 * mcse

println("3b. Uncertainty of estimate:")
@printf("    Monte Carlo Standard Error (MCSE): %.6f\n", mcse)
@printf("    95%% Confidence interval: %.4f to %.4f\n", ci_lower, ci_upper)
println("    Example output: interval = [112.0715, 118.8645]\n")

# More precise estimate with 10000 simulations
trials_precise = 10000
sim_boxes_precise = [box_count(prob_table) for _ in 1:trials_precise]

@printf("More precise estimate (with %d simulations):\n", trials_precise)
@printf("    Expected number of boxes: %.4f\n", mean(sim_boxes_precise))
@printf("    Standard deviation of boxes: %.5f\n", std(sim_boxes_precise))
@printf("    Standard error of the mean: %.7f\n", std(sim_boxes_precise) / sqrt(trials_precise))
mcse_precise = std(sim_boxes_precise) / sqrt(trials_precise)
@printf("    95%% Confidence interval: %.4f to %.4f\n\n", 
        mean(sim_boxes_precise) - 1.96*mcse_precise, 
        mean(sim_boxes_precise) + 1.96*mcse_precise)

# Part 3c: Probabilities
@printf("3c. Probabilities (from %d simulations):\n", trials)
@printf("    P(boxes > 300): %.3f\n", mean(sim_boxes .> 300))
@printf("    P(boxes > 500): %.3f\n", mean(sim_boxes .> 500))
@printf("    P(boxes > 800): %.3f\n\n", mean(sim_boxes .> 800))

@printf("Probabilities (from %d simulations - more precise):\n", trials_precise)
@printf("    P(boxes > 300): %.3f\n", mean(sim_boxes_precise .> 300))
@printf("    P(boxes > 500): %.4f\n", mean(sim_boxes_precise .> 500))
@printf("    P(boxes > 800): %.3f\n", mean(sim_boxes_precise .> 800))
@printf("    P(boxes > 800): %.3f\n\n", mean(sim_boxes_precise .> 800))

# Visualizations
p1 = histogram(sim_boxes_equal, bins=30, color=:lightblue, 
               xlabel="Number of Boxes", ylabel="Frequency",
               title="Equal Probabilities Simulation", legend=false)
vline!([expected_boxes], color=:red, linestyle=:dash, linewidth=2, 
       label="Theoretical: $(round(expected_boxes, digits=2))")

p2 = histogram(sim_boxes_precise, bins=50, color=:lightgreen,
               xlabel="Number of Boxes", ylabel="Frequency",
               title="Unequal Probabilities (10,000 trials)", legend=false)
vline!([300], color=:red, linestyle=:dash, linewidth=2, label="300 boxes")

p3 = boxplot(["Equal", "Unequal"], [sim_boxes_equal[1:1000], sim_boxes_precise[1:1000]],
             ylabel="Number of Boxes", title="Comparison: Equal vs Unequal",
             legend=false)

probs = [mean(sim_boxes_precise .> 300), mean(sim_boxes_precise .> 500), 
         mean(sim_boxes_precise .> 800)]
p4 = bar(["> 300", "> 500", "> 800"], probs, color=[:red, :orange, :yellow],
         ylabel="Probability", title="Probability of Exceeding Thresholds",
         legend=false, ylims=(0, maximum(probs) * 1.2))

plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000))
savefig("toy_collector_analysis.png")
println("Saved plot: toy_collector_analysis.png\n")

# ============================================================================
# Speed of Light Example - Bootstrap Hypothesis Testing
# ============================================================================

println("\n", "="^80)
println("Speed of Light Example - Bootstrap Hypothesis Testing")
println("="^80)
println()

# Newcomb's speed of light data (1882)
speed = [28, -44, 29, 30, 26, 27, 22, 23, 33, 16, 24, 29, 24, 40, 21, 31, 34, -2, 25, 19]

println("Dataset: Newcomb's Speed of Light Measurements (1882)")
@printf("Number of observations: %d\n", length(speed))
println("Measurements (passage time in nanoseconds above 24,800):")
println(speed)
println()

println("Summary statistics:")
@printf("  Mean: %.2f\n", mean(speed))
@printf("  Median: %.2f\n", median(speed))
@printf("  Standard deviation: %.2f\n", std(speed))
@printf("  Min: %d, Max: %d\n\n", minimum(speed), maximum(speed))

# Hypothesis Testing
println("="^80)
println("Hypothesis Testing: Does the data support the accepted speed?")
println("="^80)
println()

println("Step 1: State the Hypotheses")
println("  H₀: μ = 33.02  (The true mean speed equals the accepted value)")
println("  Hₐ: μ ≠ 33.02  (The true mean speed differs from the accepted value)")
println("  (This is a two-sided test)\n")

alpha = 0.05
@printf("Step 2: Significance Level\n")
@printf("  α = %.2f\n\n", alpha)

println("Step 3: Test Statistic")
println("  Test statistic: X̄ (sample mean)\n")

observed_mean = mean(speed)
@printf("Step 4: Observed Test Statistic\n")
@printf("  Observed sample mean: %.2f\n", observed_mean)
@printf("  Difference from H₀: %.2f\n\n", observed_mean - 33.02)

# Shift data to satisfy H0
newspeed = speed .- mean(speed) .+ 33.02

println("Bootstrap Solution: Shift the Data to Satisfy H₀")
println("  newspeed = speed - mean(speed) + 33.02")
@printf("  Original mean: %.2f\n", mean(speed))
@printf("  Shifted mean: %.2f\n\n", mean(newspeed))

# Bootstrap resampling
n_bootstrap = 1000
bstrap = zeros(n_bootstrap)

@printf("Performing %d bootstrap replications...\n", n_bootstrap)
for i in 1:n_bootstrap
    newsample = sample(newspeed, 20, replace=true)
    bstrap[i] = mean(newsample)
end
println("Bootstrap complete!\n")

# Calculate p-value
distance_from_null = abs(observed_mean - 33.02)
extreme_lower = observed_mean
extreme_upper = 33.02 + distance_from_null

lower_tail = sum(bstrap .< extreme_lower)
upper_tail = sum(bstrap .> extreme_upper)
extreme_count = lower_tail + upper_tail
pvalue = extreme_count / n_bootstrap

println("Step 6: Calculate the p-value")
@printf("  Distance from null: %.2f\n", distance_from_null)
@printf("  Count in lower tail (< %.2f): %d\n", extreme_lower, lower_tail)
@printf("  Count in upper tail (> %.2f): %d\n", extreme_upper, upper_tail)
@printf("  p-value = %.3f\n\n", pvalue)

# Decision
println("Step 7: Make a Conclusion")
if pvalue < alpha
    @printf("Result: p-value = %.3f < %.2f\n", pvalue, alpha)
    println("Decision: REJECT H₀")
    println("\nConclusion:")
    println("  We have sufficient evidence to conclude that Newcomb's measurements")
    println("  were NOT consistent with the currently accepted figure.\n")
else
    @printf("Result: p-value = %.3f >= %.2f\n", pvalue, alpha)
    println("Decision: FAIL TO REJECT H₀\n")
end

# Visualization
p1 = histogram(speed, bins=10, color=:lightblue, 
               xlabel="Speed", ylabel="Frequency", title="Original Data",
               legend=:topleft)
vline!([mean(speed)], color=:blue, linestyle=:dash, linewidth=2, 
       label="Mean: $(round(mean(speed), digits=2))")
vline!([33.02], color=:red, linestyle=:dash, linewidth=2, label="H₀: μ = 33.02")

p2 = histogram(newspeed, bins=10, color=:lightgreen,
               xlabel="Speed", ylabel="Frequency", title="Shifted Data (H₀ is True)",
               legend=:topleft)
vline!([mean(newspeed)], color=:darkgreen, linestyle=:dash, linewidth=2,
       label="Mean: $(round(mean(newspeed), digits=2))")
vline!([33.02], color=:red, linestyle=:dash, linewidth=2, label="H₀: μ = 33.02")

p3 = histogram(bstrap, bins=30, color=:lightblue, normalize=:pdf,
               xlabel="Sample Mean", ylabel="Density",
               title="Bootstrap Sampling Distribution Under H₀",
               legend=:topleft)
vline!([33.02], color=:darkgreen, linestyle=:dash, linewidth=2, label="H₀: μ = 33.02")
vline!([observed_mean], color=:red, linewidth=3, label="Observed: $(round(observed_mean, digits=2))")
density!(bstrap, color=:blue, linewidth=2, label="Bootstrap density")

# ECDF
sorted_bstrap = sort(bstrap)
ecdf_vals = (1:length(sorted_bstrap)) ./ length(sorted_bstrap)
p4 = plot(sorted_bstrap, ecdf_vals, color=:blue, linewidth=2,
          xlabel="Sample Mean", ylabel="Cumulative Probability",
          title="Empirical CDF", legend=:bottomright)
vline!([33.02], color=:darkgreen, linestyle=:dash, linewidth=2, label="H₀: μ = 33.02")
vline!([observed_mean], color=:red, linewidth=2, label="Observed: $(round(observed_mean, digits=2))")
hline!([0.025, 0.975], color=:gray, linestyle=:dot, linewidth=1, label="")

plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000))
savefig("speed_of_light_bootstrap.png")
println("Saved plot: speed_of_light_bootstrap.png\n")

# ============================================================================
# Sleep Study Example - Two-Sample Bootstrap
# ============================================================================

println("\n", "="^80)
println("Sleep Study Example - Two-Sample Bootstrap")
println("="^80)
println()

# Sleep data
sleep_data = DataFrame(
    extra = [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0,
             1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4],
    group = repeat([1, 2], inner=10)
)

println("Dataset: Student's Sleep Data")
println("Description: Effect of two soporific drugs on sleep increase")
println("\nSleep data:")
println(sleep_data)
println()

group1_data = sleep_data[sleep_data.group .== 1, :extra]
group2_data = sleep_data[sleep_data.group .== 2, :extra]

println("Summary by group:")
@printf("Group 1: Mean = %.3f, SD = %.3f\n", mean(group1_data), std(group1_data))
@printf("Group 2: Mean = %.3f, SD = %.3f\n", mean(group2_data), std(group2_data))

observed_diff = mean(group1_data) - mean(group2_data)
@printf("\nObserved difference (Group 1 - Group 2): %.3f\n\n", observed_diff)

# Bootstrap functions
function bootstrap_resample(data)
    """Resample data with replacement."""
    return sample(data, length(data), replace=true)
end

function diff_in_means(df)
    """Calculate difference in means."""
    group1 = df[df.group .== 1, :extra]
    group2 = df[df.group .== 2, :extra]
    return mean(group1) - mean(group2)
end

# Perform bootstrap
n_resamples = 2000
resample_diffs = zeros(n_resamples)

@printf("Performing %d bootstrap replications...\n", n_resamples)
for i in 1:n_resamples
    indices = sample(1:nrow(sleep_data), nrow(sleep_data), replace=true)
    boot_sample = sleep_data[indices, :]
    resample_diffs[i] = diff_in_means(boot_sample)
end
println("Bootstrap complete!\n")

println("Bootstrap Distribution Summary:")
@printf("  Mean: %.3f\n", mean(resample_diffs))
@printf("  SD: %.3f\n\n", std(resample_diffs))

# Bootstrap confidence interval
boot_ci_lower = quantile(resample_diffs, 0.025)
boot_ci_upper = quantile(resample_diffs, 0.975)

println("95% Bootstrap Confidence Interval (Percentile Method)")
@printf("  Lower bound: %.3f\n", boot_ci_lower)
@printf("  Upper bound: %.3f\n\n", boot_ci_upper)

if boot_ci_lower > 0
    println("  Since the entire CI is above 0, Drug 1 increases sleep more than Drug 2.\n")
elseif boot_ci_upper < 0
    println("  Since the entire CI is below 0, Drug 2 increases sleep more than Drug 1.\n")
else
    println("  Since the CI includes 0, we cannot conclude a significant difference.\n")
end

# Visualization
p1 = @df sleep_data boxplot(:group, :extra, xlabel="Group", 
                             ylabel="Increase in Hours of Sleep",
                             title="Sleep Increase by Drug", legend=false)

p2 = histogram(resample_diffs, bins=40, color=:lightblue, normalize=:pdf,
               xlabel="Difference in Means", ylabel="Density",
               title="Bootstrap Sampling Distribution", legend=:topright)
vline!([observed_diff], color=:red, linewidth=2, label="Observed: $(round(observed_diff, digits=2))")
vline!([0], color=:darkgreen, linestyle=:dash, linewidth=2, label="No difference (0)")

# CI visualization
in_ci = (resample_diffs .>= boot_ci_lower) .& (resample_diffs .<= boot_ci_upper)
p3 = histogram(resample_diffs, bins=40, color=:lightgray, normalize=:pdf,
               xlabel="Difference in Means", ylabel="Density",
               title="95% Bootstrap Confidence Interval", legend=:topright)
histogram!(resample_diffs[in_ci], bins=40, color=:lightblue, normalize=:pdf)
vline!([boot_ci_lower, boot_ci_upper], color=:blue, linestyle=:dash, linewidth=2, label="95% CI bounds")
vline!([observed_diff], color=:red, linewidth=2, label="Observed")
vline!([0], color=:darkgreen, linestyle=:dash, linewidth=2, label="No diff")

# Q-Q plot
using StatsPlots
p4 = qqplot(Normal(), resample_diffs, xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles", title="Q-Q Plot: Bootstrap Differences",
            legend=false, markersize=3)

plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000))
savefig("sleep_study_bootstrap.png")
println("Saved plot: sleep_study_bootstrap.png\n")

# ============================================================================
# R-squared Bootstrap Example
# ============================================================================

println("\n", "="^80)
println("Bootstrapping a Single Statistic: R-squared")
println("="^80)
println()

# mtcars data
mtcars_data = DataFrame(
    mpg = [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
           17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
           21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
    wt = [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
          3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
          2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
    disp = [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6,
            167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1,
            120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0]
)

println("Dataset: mtcars")
println("Variables: mpg (miles per gallon), wt (weight), disp (displacement)")
println("\nFirst few rows:")
println(first(mtcars_data, 5))
println()

# Fit original model
using GLM
X = hcat(ones(nrow(mtcars_data)), Matrix(mtcars_data[:, [:wt, :disp]]))
y = mtcars_data.mpg
beta = X \ y
y_pred = X * beta
ss_res = sum((y .- y_pred).^2)
ss_tot = sum((y .- mean(y)).^2)
original_r2 = 1 - ss_res / ss_tot

@printf("Original R²: %.4f\n", original_r2)
@printf("This means %.2f%% of variance in mpg is explained by wt and disp.\n\n", original_r2*100)

# Bootstrap R-squared
function rsq_bootstrap(data, indices)
    """Calculate R² for bootstrap sample."""
    boot_data = data[indices, :]
    X_boot = hcat(ones(nrow(boot_data)), Matrix(boot_data[:, [:wt, :disp]]))
    y_boot = boot_data.mpg
    
    beta_boot = X_boot \ y_boot
    y_pred_boot = X_boot * beta_boot
    ss_res_boot = sum((y_boot .- y_pred_boot).^2)
    ss_tot_boot = sum((y_boot .- mean(y_boot)).^2)
    return 1 - ss_res_boot / ss_tot_boot
end

n_boot = 1000
@printf("Performing %d bootstrap replications for R²...\n", n_boot)
boot_r2 = zeros(n_boot)
for i in 1:n_boot
    indices = sample(1:nrow(mtcars_data), nrow(mtcars_data), replace=true)
    boot_r2[i] = rsq_bootstrap(mtcars_data, indices)
end
println("Bootstrap complete!\n")

println("Bootstrap Results:")
@printf("  Original R²: %.4f\n", original_r2)
@printf("  Bootstrap mean R²: %.4f\n", mean(boot_r2))
@printf("  Bootstrap bias: %.4f\n", mean(boot_r2) - original_r2)
@printf("  Bootstrap SE: %.4f\n\n", std(boot_r2))

# Bootstrap CI for R²
r2_ci_lower = quantile(boot_r2, 0.025)
r2_ci_upper = quantile(boot_r2, 0.975)

println("95% Bootstrap Confidence Interval for R²:")
@printf("  [%.4f, %.4f]\n", r2_ci_lower, r2_ci_upper)
@printf("  We are 95%% confident that between %.2f%% and %.2f%%\n", r2_ci_lower*100, r2_ci_upper*100)
println("  of variance in MPG is explained by weight and displacement.\n")

# Visualization
p1 = histogram(boot_r2, bins=30, color=:lightgreen, normalize=:pdf,
               xlabel="R-squared", ylabel="Density",
               title="Bootstrap Distribution of R²", legend=:topleft)
vline!([original_r2], color=:red, linewidth=3, label="Original: $(round(original_r2, digits=4))")
density!(boot_r2, color=:darkgreen, linewidth=2, label="Bootstrap density")

in_ci_r2 = (boot_r2 .>= r2_ci_lower) .& (boot_r2 .<= r2_ci_upper)
p2 = histogram(boot_r2, bins=30, color=:lightgray, normalize=:pdf,
               xlabel="R-squared", ylabel="Density",
               title="95% Percentile CI for R²", legend=:topleft)
histogram!(boot_r2[in_ci_r2], bins=30, color=:lightgreen, normalize=:pdf)
vline!([r2_ci_lower, r2_ci_upper], color=:blue, linestyle=:dash, linewidth=2, label="CI bounds")
vline!([original_r2], color=:red, linewidth=2, label="Original")

p3 = qqplot(Normal(), boot_r2, xlabel="Theoretical Quantiles",
            ylabel="Sample Quantiles", title="Q-Q Plot: Bootstrap R²",
            legend=false, markersize=3)

p4 = plot(boot_r2, color=:darkgreen, alpha=0.5, linewidth=0.5,
          xlabel="Bootstrap Replicate", ylabel="R-squared",
          title="Bootstrap R² Sequence", legend=false)
hline!([original_r2], color=:red, linestyle=:dash, linewidth=2)
hline!([mean(boot_r2)], color=:blue, linestyle=:dot, linewidth=1)

plot(p1, p2, p3, p4, layout=(2,2), size=(1400, 1000))
savefig("rsquared_bootstrap.png")
println("Saved plot: rsquared_bootstrap.png\n")

println("="^80)
println("Bootstrap Analysis Complete!")
println("="^80)
println("\nGenerated plots:")
println("  1. toy_collector_analysis.png")
println("  2. speed_of_light_bootstrap.png")
println("  3. sleep_study_bootstrap.png")
println("  4. rsquared_bootstrap.png")
