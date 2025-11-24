"""
MONTE CARLO SIMULATIONS
Statistical Computing Tutorial - Julia Version

Topics covered:
1. Ordinary Monte Carlo (OMC) theory
2. Monte Carlo integration examples
3. Approximating distributions
4. Bootstrap and permutation methods
5. Toy collector exercise (Coupon Collector Problem)
"""

using Statistics
using Distributions
using Plots
using Random
using Printf
using StatsBase
using DataFrames

# Create plots directory if it doesn't exist
mkpath("../plots")

println("=" ^ 70)
println("MONTE CARLO SIMULATIONS TUTORIAL")
println("=" ^ 70)
println()

# Set seed for reproducibility
Random.seed!(42)

# =============================================================================
# PART 1: ORDINARY MONTE CARLO (OMC) - THEORY
# =============================================================================

println("\n", "=" ^ 70)
println("PART 1: ORDINARY MONTE CARLO - THEORY")
println("=" ^ 70)
println()

println("The 'Monte Carlo method' refers to the theory and practice of learning")
println("about probability distributions by simulation rather than calculus.\n")

println("In Ordinary Monte Carlo (OMC) we use IID simulations from the")
println("distribution of interest.\n")

println("Setup:")
println("-" ^ 60)
println("Suppose X₁, X₂, ... are IID simulations from some distribution,")
println("and we want to know an expectation:\n")
println("  θ = E[Y₁] = E[g(X₁)]\n")

println("Law of Large Numbers (LLN):")
println("-" ^ 60)
println("  ȳₙ = (1/n) Σᵢ Yᵢ = (1/n) Σᵢ g(Xᵢ)\n")
println("converges in probability to θ.\n")

println("Central Limit Theorem (CLT):")
println("-" ^ 60)
println("  √n(ȳₙ - θ)/σ →ᵈ N(0,1)\n")
println("That is, for sufficiently large n:")
println("  ȳₙ ~ N(θ, σ²/n)\n")

println("Standard Error Estimation:")
println("-" ^ 60)
println("We can estimate the standard error σ/√n with sₙ/√n")
println("where sₙ is the sample standard deviation.\n")

println("KEY INSIGHT:")
println("-" ^ 60)
println("The theory of OMC is just the theory of frequentist statistical inference.")
println("The only differences are that:\n")
println("1. The 'data' X₁,...,Xₙ are computer simulations rather than")
println("   measurements on objects in the real world\n")
println("2. The 'sample size' n is the number of computer simulations")
println("   rather than the size of some real world data\n")
println("3. The unknown parameter θ is in principle completely known,")
println("   given by some integral, which we are unable to do.\n")

println("VECTOR CASE:")
println("-" ^ 60)
println("Everything works just the same when the data X₁, X₂, ...")
println("(which are computer simulations) are vectors.")
println("But the functions of interest g(X₁), g(X₂), ... are scalars.\n")

println("LIMITATION:")
println("-" ^ 60)
println("OMC works great, but it can be very difficult to simulate IID")
println("simulations of random variables or random vectors whose")
println("distribution is not brand name distributions.\n")

# =============================================================================
# PART 2: APPROXIMATING THE BINOMIAL DISTRIBUTION
# =============================================================================

println("\n", "=" ^ 70)
println("PART 2: APPROXIMATING THE BINOMIAL DISTRIBUTION")
println("=" ^ 70)
println()

println("Problem: Flip a coin 10 times. What is P(more than 3 heads)?")
println("-" ^ 60)
println()

println("This is trivial for the Binomial distribution, but we'll use")
println("Monte Carlo simulation to demonstrate the method.\n")

# Monte Carlo simulation
runs = 10000

function one_trial()
    return sum(rand(0:1, 10)) > 3
end

@printf("Running %d Monte Carlo simulations...\n\n", runs)

Random.seed!(123)
mc_results = [one_trial() for _ in 1:runs]
mc_binom = mean(mc_results)

# Exact probability
exact_prob = 1 - cdf(Binomial(10, 0.5), 3)

# Calculate Monte Carlo standard error
mc_se = sqrt(mc_binom * (1 - mc_binom) / runs)

println("RESULTS:")
@printf("Monte Carlo estimate: %.6f\n", mc_binom)
@printf("Exact probability:    %.6f\n", exact_prob)
@printf("Absolute error:       %.6f\n", abs(mc_binom - exact_prob))
@printf("\nMonte Carlo standard error: %.6f\n", mc_se)
@printf("95%% Confidence Interval: [%.6f, %.6f]\n", 
        mc_binom - 1.96*mc_se, mc_binom + 1.96*mc_se)

in_ci = (exact_prob >= mc_binom - 1.96*mc_se) && (exact_prob <= mc_binom + 1.96*mc_se)
println("Exact value in CI: ", in_ci ? "YES ✓" : "NO ✗")

# =============================================================================
# EXERCISE SOLUTION: MONTE CARLO STANDARD ERROR
# =============================================================================

println("\n\n", "=" ^ 70)
println("EXERCISE: ESTIMATING MONTE CARLO STANDARD ERROR")
println("=" ^ 70)
println()

println("For a binary outcome (success/failure), the standard error is:")
println("  SE = √[p(1-p)/n]\n")

println("where:")
println("  p = estimated probability (proportion of successes)")
println("  n = number of Monte Carlo simulations\n")

println("In our case:")
@printf("  p = %.6f\n", mc_binom)
@printf("  n = %d\n", runs)
@printf("  SE = √[%.6f × %.6f / %d] = %.6f\n", mc_binom, 1-mc_binom, runs, mc_se)

# Demonstrate convergence
println("\n\nDemonstrating convergence with different sample sizes:")
println("-" ^ 60)
println()

sample_sizes = [100, 1000, 10000, 100000]
results_table = DataFrame(
    n = Int[],
    estimate = Float64[],
    se = Float64[],
    ci_lower = Float64[],
    ci_upper = Float64[],
    in_ci = Bool[]
)

Random.seed!(456)
for n in sample_sizes
    mc_trials = [one_trial() for _ in 1:n]
    p_hat = mean(mc_trials)
    se = sqrt(p_hat * (1 - p_hat) / n)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    in_ci_check = (exact_prob >= ci_lower) && (exact_prob <= ci_upper)
    
    push!(results_table, (n, p_hat, se, ci_lower, ci_upper, in_ci_check))
end

println(results_table)

@printf("\nExact probability: %.6f\n", exact_prob)
println("\nNote: Standard error decreases as O(1/√n)")

# =============================================================================
# PART 3: APPROXIMATING π
# =============================================================================

println("\n\n", "=" ^ 70)
println("PART 3: APPROXIMATING π USING MONTE CARLO")
println("=" ^ 70)
println()

println("Geometric Approach to Estimating π")
println("-" ^ 60)
println()

println("Key insight:")
println("  Area of a circle = πr²")
println("  Area of square containing the circle = (2r)² = 4r²\n")

println("Therefore, the ratio of areas is:")
println("  πr² / 4r² = π/4\n")

println("If we can empirically determine the ratio of the area of the")
println("circle to the area of the square, we can multiply by 4 to get π.\n")

println("Method:")
println("-" ^ 60)
println("1. Randomly sample (x, y) points on the unit square centered at 0")
println("   (i.e., x, y ∈ [-0.5, 0.5])")
println("2. Check if x² + y² ≤ 0.5² (point is inside the circle)")
println("3. Ratio of points in circle × 4 = estimate of π\n")

# Monte Carlo estimation of π
runs = 100000
Random.seed!(2024)

@printf("Running %d Monte Carlo simulations...\n\n", runs)

xs = rand(Uniform(-0.5, 0.5), runs)
ys = rand(Uniform(-0.5, 0.5), runs)
in_circle = (xs.^2 .+ ys.^2) .<= 0.5^2
mc_pi = mean(in_circle) * 4

# Calculate standard error
p = mean(in_circle)
se_p = sqrt(p * (1 - p) / runs)
se_pi = 4 * se_p

println("RESULTS:")
@printf("Monte Carlo estimate of π: %.6f\n", mc_pi)
@printf("True value of π:           %.6f\n", π)
@printf("Absolute error:            %.6f\n", abs(mc_pi - π))
@printf("Relative error:            %.4f%%\n", 100 * abs(mc_pi - π) / π)
@printf("\nProportion in circle:      %.6f\n", p)
@printf("Standard error of π:       %.6f\n", se_pi)
@printf("95%% CI for π:              [%.6f, %.6f]\n", 
        mc_pi - 1.96*se_pi, mc_pi + 1.96*se_pi)

in_ci_pi = (π >= mc_pi - 1.96*se_pi) && (π <= mc_pi + 1.96*se_pi)
println("True π in CI: ", in_ci_pi ? "YES ✓" : "NO ✗")

# Convergence analysis
println("\n\nConvergence analysis with different sample sizes:")
println("-" ^ 60)
println()

sample_sizes_pi = [100, 1000, 10000, 100000, 1000000]
pi_results = DataFrame(
    n = Int[],
    estimate = Float64[],
    error = Float64[],
    rel_error_pct = Float64[],
    se = Float64[]
)

Random.seed!(12345)
for n in sample_sizes_pi
    xs_temp = rand(Uniform(-0.5, 0.5), n)
    ys_temp = rand(Uniform(-0.5, 0.5), n)
    in_circle_temp = (xs_temp.^2 .+ ys_temp.^2) .<= 0.5^2
    pi_est = mean(in_circle_temp) * 4
    
    p_temp = mean(in_circle_temp)
    se_temp = 4 * sqrt(p_temp * (1 - p_temp) / n)
    
    push!(pi_results, (n, pi_est, abs(pi_est - π), 
                       100 * abs(pi_est - π) / π, se_temp))
end

println(pi_results)

@printf("\nTrue π = %.10f\n", π)

# Visualization
println("\n\nCreating π approximation visualization...")

p1 = scatter(xs[in_circle], ys[in_circle], 
            c=:blue, ms=1, alpha=0.5, label="", 
            xlabel="x", ylabel="y",
            title="MC π Estimation (n=$runs)")
scatter!(p1, xs[.!in_circle], ys[.!in_circle], 
         c=:red, ms=1, alpha=0.5, label="")
theta = range(0, 2π, length=200)
plot!(p1, 0.5 .* cos.(theta), 0.5 .* sin.(theta), 
      c=:black, lw=2, label="")
plot!(p1, [-0.5, 0.5, 0.5, -0.5, -0.5], 
      [-0.5, -0.5, 0.5, 0.5, -0.5], 
      c=:black, lw=2, label="", aspect_ratio=:equal)

# Convergence plot
Random.seed!(111)
n_conv = 50000
xs_conv = rand(Uniform(-0.5, 0.5), n_conv)
ys_conv = rand(Uniform(-0.5, 0.5), n_conv)
in_circle_conv = (xs_conv.^2 .+ ys_conv.^2) .<= 0.5^2
cumulative_pi = cumsum(in_circle_conv) ./ (1:n_conv) .* 4

p2 = plot(1:n_conv, cumulative_pi, 
         c=:blue, lw=2, label="MC estimate",
         xlabel="Number of simulations",
         ylabel="Estimated π",
         title="Convergence of π Estimate",
         ylim=(2.8, 3.5))
hline!(p2, [π], c=:red, lw=2, ls=:dash, label="True π")

# Error vs sample size
p3 = plot([r.n for r in eachrow(pi_results)], 
         [r.error for r in eachrow(pi_results)],
         c=:red, marker=:circle, ms=8, lw=2,
         xscale=:log10, yscale=:log10,
         xlabel="Sample size (n)",
         ylabel="Absolute error",
         title="Error vs Sample Size",
         label="", grid=true)

# Distribution of π estimates
Random.seed!(222)
n_reps = 1000
n_per_rep = 10000
pi_estimates = Float64[]
for _ in 1:n_reps
    xs_temp = rand(Uniform(-0.5, 0.5), n_per_rep)
    ys_temp = rand(Uniform(-0.5, 0.5), n_per_rep)
    push!(pi_estimates, mean(xs_temp.^2 .+ ys_temp.^2 .<= 0.5^2) * 4)
end

p4 = histogram(pi_estimates, bins=30, normalize=true,
              c=:lightgreen, alpha=0.7, label="",
              xlabel="Estimated π",
              title="Distribution of π Estimates\n(1000 reps, n=$n_per_rep each)")
vline!(p4, [π], c=:red, lw=3, ls=:dash, label="True π")
vline!(p4, [mean(pi_estimates)], c=:blue, lw=2, ls=:dash, label="Mean estimate")

# Relative error
p5 = plot([r.n for r in eachrow(pi_results)], 
         [r.rel_error_pct for r in eachrow(pi_results)],
         c=:purple, marker=:circle, ms=8, lw=2,
         xscale=:log10,
         xlabel="Sample size (n)",
         ylabel="Relative error (%)",
         title="Relative Error vs Sample Size",
         label="", grid=true)

# Standard error
p6 = plot([r.n for r in eachrow(pi_results)], 
         [r.se for r in eachrow(pi_results)],
         c=:orange, marker=:circle, ms=8, lw=2,
         xscale=:log10, yscale=:log10,
         xlabel="Sample size (n)",
         ylabel="Standard error",
         title="Standard Error vs Sample Size",
         label="", grid=true)

plot(p1, p2, p3, p4, p5, p6, layout=(2, 3), size=(1500, 1000))
savefig("../plots/monte_carlo_pi_approximation.png")
println("Saved: ../plots/monte_carlo_pi_approximation.png")

# Detailed π plot
println("Creating detailed π visualization...")
p_detail = scatter(xs[in_circle], ys[in_circle], 
                  c=:blue, ms=0.1, alpha=0.5, label="Inside",
                  xlabel="", ylabel="",
                  title="MC Approximation of π = $(round(mc_pi, digits=4))")
scatter!(p_detail, xs[.!in_circle], ys[.!in_circle], 
         c=:grey, ms=0.1, alpha=0.5, label="Outside")
plot!(p_detail, 0.5 .* cos.(theta), 0.5 .* sin.(theta), 
      c=:black, lw=2, label="")
plot!(p_detail, [-0.5, 0.5, 0.5, -0.5, -0.5], 
      [-0.5, -0.5, 0.5, 0.5, -0.5], 
      c=:black, lw=2, label="", aspect_ratio=:equal, size=(800, 800))
savefig("../plots/monte_carlo_pi_detailed.png")
println("Saved: ../plots/monte_carlo_pi_detailed.png")

# =============================================================================
# PART 4: MONTE CARLO INTEGRATION WITH SEQUENTIAL STOPPING
# =============================================================================

println("\n\n", "=" ^ 70)
println("PART 4: MONTE CARLO INTEGRATION")
println("=" ^ 70)
println()

println("Example: Intractable Expectation")
println("-" ^ 60)
println()

println("Let X ~ Gamma(3/2, 1), i.e.")
println("  f(x) = (2/√π) √x e^(-x) I(x > 0)\n")

println("Suppose we want to find:")
println("  θ = E[1/((X+1)log(X+3))]")
println("    = ∫₀^∞ 1/((x+1)log(x+3)) * (2/√π) √x e^(-x) dx\n")

println("The expectation (or integral) θ is intractable - we don't know")
println("how to compute it analytically.\n")

println("GOAL: Estimate θ such that the 95% CI length is less than 0.002\n")

# Initial Monte Carlo estimation
n = 1000
Random.seed!(4040)

@printf("Initial estimation with n = %d:\n", n)
println("-" ^ 40)

x = rand(Gamma(3/2, 1), n)
@printf("Mean of X (theoretical = 3/2 = 1.5): %.6f\n", mean(x))

y = 1 ./ ((x .+ 1) .* log.(x .+ 3))
est = mean(y)
@printf("\nInitial estimate of θ: %.7f\n", est)

mcse = std(y) / sqrt(length(y))
interval = est .+ [-1, 1] .* 1.96 .* mcse
@printf("Monte Carlo SE: %.7f\n", mcse)
@printf("95%% CI: [%.7f, %.7f]\n", interval[1], interval[2])
@printf("CI length: %.7f\n", interval[2] - interval[1])

# Sequential stopping rule
println("\n\nApplying Sequential Stopping Rule:")
println("-" ^ 60)
println("Target CI length: 0.002")
println("Adding samples in batches of 1000 until target is reached...\n")

eps = 0.002
len_ci = interval[2] - interval[1]
plotting_var = [[est, interval[1], interval[2]]]
iteration = 1

while len_ci > eps
    new_x = rand(Gamma(3/2, 1), n)
    new_y = 1 ./ ((new_x .+ 1) .* log.(new_x .+ 3))
    y = vcat(y, new_y)
    est = mean(y)
    mcse = std(y) / sqrt(length(y))
    interval = est .+ [-1, 1] .* 1.96 .* mcse
    len_ci = interval[2] - interval[1]
    push!(plotting_var, [est, interval[1], interval[2]])
    iteration += 1
    
    if iteration % 20 == 0
        @printf("  Iteration %3d: n = %6d, CI length = %.6f\n", 
                iteration, length(y), len_ci)
    end
end

plotting_var = hcat(plotting_var...)'

println("\nSequential stopping complete!")
@printf("Final sample size: %d\n", length(y))
@printf("Final estimate: %.7f\n", est)
@printf("Final 95%% CI: [%.7f, %.7f]\n", interval[1], interval[2])
@printf("Final CI length: %.7f (target: %.3f)\n", len_ci, eps)
@printf("Final SE: %.7f\n", mcse)

# Visualization
println("\n\nCreating sequential stopping visualization...")

temp = 1000:1000:length(y)
p_seq1 = plot(temp, plotting_var[:, 1], 
             c=:black, lw=2, label="Estimate",
             xlabel="Sample size (n)",
             ylabel="Estimate of θ",
             title="Sequential Estimation with 95% CI")
plot!(p_seq1, temp, plotting_var[:, 2], c=:red, lw=2, label="95% CI")
plot!(p_seq1, temp, plotting_var[:, 3], c=:red, lw=2, label="")
hline!(p_seq1, [est], c=:blue, lw=1, ls=:dash, label="Final estimate")

ci_lengths = plotting_var[:, 3] .- plotting_var[:, 2]
p_seq2 = plot(temp, ci_lengths, 
             c=:purple, lw=2, label="",
             xlabel="Sample size (n)",
             ylabel="CI length",
             title="Convergence of CI Length")
hline!(p_seq2, [eps], c=:red, lw=2, ls=:dash, label="Target = $eps")

plot(p_seq1, p_seq2, layout=(1, 2), size=(1400, 600))
savefig("../plots/monte_carlo_integration_sequential.png")
println("Saved: ../plots/monte_carlo_integration_sequential.png")

# Additional analysis
println("\n\nAdditional Analysis:")
println("-" ^ 60)
@printf("Sample size increase: 1000 → %d (%.1fx)\n", 
        length(y), length(y) / 1000)
@printf("CI length reduction: %.6f → %.6f (%.1fx)\n", 
        ci_lengths[1], len_ci, ci_lengths[1] / len_ci)

# Distribution plots
println("\n\nCreating integration distribution plots...")

p_dist1 = histogram(x[1:10000], bins=50, normalize=true,
                   c=:lightblue, alpha=0.7, label="",
                   xlabel="X",
                   title="Distribution of X ~ Gamma(3/2, 1)")
x_range = range(0, maximum(x[1:10000]), length=200)
plot!(p_dist1, x_range, pdf.(Gamma(3/2, 1), x_range), 
      c=:red, lw=2, label="True PDF")

y_plot = y[y .< quantile(y, 0.99)]
p_dist2 = histogram(y_plot, bins=100, normalize=true,
                   c=:lightgreen, alpha=0.7, label="",
                   xlabel="Y",
                   title="Distribution of Y = 1/((X+1)log(X+3))")
vline!(p_dist2, [mean(y)], c=:red, lw=2, ls=:dash, label="Mean")

p_dist3 = scatter(x[1:5000], y[1:5000], 
                 ms=1, alpha=0.3, c=:blue, label="",
                 xlabel="X ~ Gamma(3/2, 1)",
                 ylabel="Y = 1/((X+1)log(X+3))",
                 title="Relationship between X and Y")
hline!(p_dist3, [mean(y)], c=:red, lw=2, ls=:dash, label="Mean Y")

plot(p_dist1, p_dist2, p_dist3, layout=(1, 3), size=(1200, 400))
savefig("../plots/monte_carlo_integration_distribution.png")
println("Saved: ../plots/monte_carlo_integration_distribution.png")

# =============================================================================
# PART 5: BOOTSTRAP AND PERMUTATION METHODS
# =============================================================================

println("\n\n", "=" ^ 70)
println("PART 5: BOOTSTRAP AND PERMUTATION METHODS")
println("=" ^ 70)
println()

println("HIGH-DIMENSIONAL EXAMPLES:")
println("Monte Carlo methods are essential for complex, high-dimensional problems:")
println("  - FiveThirtyEight's Election Forecast")
println("  - FiveThirtyEight's NBA Predictions")
println("  - Vanguard's Retirement Nest Egg Calculator")
println("  - Fisher's Exact Test in Julia\n")

# Permutations
println("PERMUTATIONS WITH Julia:")
println("-" ^ 60)
println("shuffle() works on any array-like object\n")

println("Example 1: Simple permutations")
Random.seed!(5050)
println("shuffle(0:4):")
println(shuffle(0:4))

println("\nshuffle(1:6):")
println(shuffle(1:6))

println("\nExample 2: Permuting arrays")
println("Multiple permutations of ['Curly', 'Larry', 'Moe', 'Shemp']:")
Random.seed!(6060)
stooges = ["Curly", "Larry", "Moe", "Shemp"]
stooges_perms = hcat([shuffle(stooges) for _ in 1:3]...)
println(stooges_perms)

# Bootstrap
println("\n\nRESAMPLING WITH Julia - BOOTSTRAP:")
println("-" ^ 60)
println("Resampling from any existing distribution gives bootstrap estimators\n")

println("Key difference from jackknife:")
println("  - Jackknife: removes one point and recalculates")
println("  - Bootstrap: resamples same length WITH REPLACEMENT\n")

function bootstrap_resample(arr)
    """Bootstrap resample with replacement"""
    return sample(arr, length(arr), replace=true)
end

println("Example: Bootstrap resampling")
Random.seed!(7070)
println("Bootstrap resamples of [6, 7, 8, 9, 10]:")
bootstrap_example = hcat([bootstrap_resample(6:10) for _ in 1:5]...)
println(bootstrap_example)

println("\nNote: Values can (and do) repeat due to replacement")

# Bootstrap two-sample test
println("\n\nBOOTSTRAP TEST: TWO-SAMPLE DIFFERENCE IN MEANS")
println("-" ^ 60)
println()

println("The 2-sample t-test checks for differences in means according to")
println("a known null distribution. Let's use bootstrap to generate the")
println("sampling distribution under the bootstrap assumption.\n")

println("Example: Simulated cat heart weights by sex\n")

# Simulate cat data
Random.seed!(123)
n_males = 97
n_females = 47
male_hwt = rand(Normal(11.32, 2.54), n_males)
female_hwt = rand(Normal(9.20, 1.36), n_females)

cats_data = DataFrame(
    Sex = vcat(fill("M", n_males), fill("F", n_females)),
    Hwt = vcat(male_hwt, female_hwt)
)

function diff_in_means(df)
    """Calculate difference in means (M - F)"""
    return mean(df[df.Sex .== "M", :Hwt]) - mean(df[df.Sex .== "F", :Hwt])
end

obs_diff = diff_in_means(cats_data)
@printf("Observed difference in means (M - F): %.4f g\n", obs_diff)

println("\nSummary by sex:")
@printf("Males:   n=%d, mean=%.2f g, sd=%.2f g\n", 
        n_males, mean(male_hwt), std(male_hwt))
@printf("Females: n=%d, mean=%.2f g, sd=%.2f g\n", 
        n_females, mean(female_hwt), std(female_hwt))

# Bootstrap resampling
println("\nGenerating bootstrap distribution (1000 replicates)...")
Random.seed!(8080)
n_boot = 1000
resample_diffs = zeros(n_boot)

for i in 1:n_boot
    boot_indices = sample(1:nrow(cats_data), nrow(cats_data), replace=true)
    boot_sample = cats_data[boot_indices, :]
    resample_diffs[i] = diff_in_means(boot_sample)
end

println("\nBootstrap results:")
@printf("Mean of bootstrap diffs: %.4f g\n", mean(resample_diffs))
@printf("SD of bootstrap diffs:   %.4f g\n", std(resample_diffs))
@printf("95%% CI (percentile):     [%.4f, %.4f] g\n", 
        quantile(resample_diffs, 0.025), quantile(resample_diffs, 0.975))

# Compare with t-test (manual calculation)
se_diff = sqrt(var(male_hwt)/n_males + var(female_hwt)/n_females)
t_crit = quantile(TDist(n_males + n_females - 2), 0.975)
ci_ttest = [obs_diff - t_crit * se_diff, obs_diff + t_crit * se_diff]

println("\nComparison with t-test:")
@printf("t-test 95%% CI: [%.4f, %.4f] g\n", ci_ttest[1], ci_ttest[2])

# Visualization
println("\nCreating bootstrap test visualization...")

p_boot1 = histogram(resample_diffs, bins=40, normalize=true,
                   c=:lightblue, alpha=0.7, label="",
                   xlabel="Difference in heart weight (M - F, grams)",
                   title="Bootstrap Distribution of Difference in Means")
vline!(p_boot1, [obs_diff], c=:red, lw=3, ls=:dash, label="Observed")
vline!(p_boot1, [mean(resample_diffs)], c=:blue, lw=2, ls=:dash, label="Bootstrap mean")
ci_boot = quantile(resample_diffs, [0.025, 0.975])
vline!(p_boot1, [ci_boot[1], ci_boot[2]], c=:darkgreen, lw=2, ls=:dot, label="95% CI")

p_boot2 = boxplot(["F", "M"], [female_hwt, male_hwt],
                 c=[:pink :lightblue], label="",
                 xlabel="Sex", ylabel="Heart weight (grams)",
                 title="Cat Heart Weights by Sex")
scatter!(p_boot2, [1, 2], [mean(female_hwt), mean(male_hwt)],
        c=:red, ms=8, marker=:diamond, label="Mean")

p_boot3 = scatter(Normal(), resample_diffs,
                 xlabel="Theoretical Quantiles", ylabel="Sample Quantiles",
                 title="Q-Q Plot: Bootstrap Distribution",
                 c=:blue, ms=3, label="")

sorted_diffs = sort(resample_diffs)
ecdf_y = (1:length(sorted_diffs)) ./ length(sorted_diffs)
p_boot4 = plot(sorted_diffs, ecdf_y, 
              c=:blue, lw=2, label="",
              xlabel="Difference in heart weight (grams)",
              ylabel="Cumulative probability",
              title="ECDF of Bootstrap Differences")
vline!(p_boot4, [obs_diff], c=:red, lw=2, ls=:dash, label="Observed")
vline!(p_boot4, [0], c=:gray, lw=1, ls=:dot, label="Zero")

plot(p_boot1, p_boot2, p_boot3, p_boot4, layout=(2, 2), size=(1200, 800))
savefig("../plots/monte_carlo_bootstrap_test.png")
println("Saved: ../plots/monte_carlo_bootstrap_test.png")

# =============================================================================
# PART 6: TOY COLLECTOR EXERCISE
# =============================================================================

println("\n\n", "=" ^ 70)
println("PART 6: TOY COLLECTOR EXERCISE (COUPON COLLECTOR PROBLEM)")
println("=" ^ 70)
println()

println("Problem: Children are enticed to buy cereal to collect action figures.")
println("Assume there are 15 action figures and each box contains exactly one,")
println("with each figure being equally likely initially.\n")

function simulate_collection(n_toys, probs=nothing, max_boxes=10000)
    """
    Simulate collecting all n_toys with given probabilities.
    
    Parameters:
    - n_toys: number of unique toys
    - probs: probability of each toy (nothing = equal probability)
    - max_boxes: maximum boxes to try
    
    Returns: number of boxes needed to collect all toys
    """
    if probs === nothing
        probs = fill(1/n_toys, n_toys)
    end
    
    collected = falses(n_toys)
    n_boxes = 0
    
    while !all(collected) && n_boxes < max_boxes
        n_boxes += 1
        toy = sample(1:n_toys, Weights(probs))
        collected[toy] = true
    end
    
    return n_boxes
end

# Questions 1 & 2: Equal probabilities
println("QUESTIONS 1 & 2: EQUAL PROBABILITIES (1/15 each)")
println("-" ^ 70)
println()

n_toys = 15
n_simulations = 10000

@printf("Running %d simulations...\n", n_simulations)
Random.seed!(9090)
boxes_needed_equal = [simulate_collection(n_toys) for _ in 1:n_simulations]

mean_boxes_equal = mean(boxes_needed_equal)
sd_boxes_equal = std(boxes_needed_equal)
se_boxes_equal = sd_boxes_equal / sqrt(n_simulations)

# Theoretical expectation
harmonic_number = sum(1 ./ (1:n_toys))
theoretical_mean = n_toys * harmonic_number

println("\nRESULTS FOR EQUAL PROBABILITIES:")
@printf("Q1. Expected number of boxes (simulated):   %.2f\n", mean_boxes_equal)
@printf("    Expected number of boxes (theoretical): %.2f\n", theoretical_mean)
@printf("Q2. Standard deviation:                     %.2f\n", sd_boxes_equal)
@printf("    Standard error of estimate:             %.4f\n", se_boxes_equal)
@printf("    Median: %.0f, Range: [%d, %d]\n", 
        median(boxes_needed_equal), minimum(boxes_needed_equal), 
        maximum(boxes_needed_equal))

quantiles_equal = quantile(boxes_needed_equal, [0.25, 0.5, 0.75, 0.9, 0.95])
println("\nQuantiles:")
for (q, val) in zip([25, 50, 75, 90, 95], quantiles_equal)
    @printf("  %d%%: %d boxes\n", q, Int(round(val)))
end

# Questions 3, 4, 5: Unequal probabilities
println("\n\nQUESTIONS 3, 4, 5: UNEQUAL PROBABILITIES")
println("-" ^ 70)
println()

toy_names = [string(Char(64 + i)) for i in 1:15]  # A-O
toy_probs = [.2, .1, .1, .1, .1, .1, .05, .05, .05, .05, .02, .02, .02, .02, .02]

println("Figure probabilities:")
prob_df = DataFrame(Figure = toy_names, Probability = toy_probs)
println(prob_df)
@printf("\nSum of probabilities: %.2f (must equal 1.0)\n", sum(toy_probs))

@printf("\nRunning %d simulations...\n", n_simulations)
Random.seed!(10101)
boxes_needed_unequal = [simulate_collection(n_toys, toy_probs) 
                        for _ in 1:n_simulations]

mean_boxes_unequal = mean(boxes_needed_unequal)
sd_boxes_unequal = std(boxes_needed_unequal)
se_boxes_unequal = sd_boxes_unequal / sqrt(n_simulations)

println("\nRESULTS FOR UNEQUAL PROBABILITIES:")
@printf("Q3. Expected number of boxes: %.2f\n", mean_boxes_unequal)
println("Q4. Uncertainty of estimate:")
@printf("    Standard deviation: %.2f\n", sd_boxes_unequal)
@printf("    Standard error:     %.4f\n", se_boxes_unequal)
@printf("    95%% CI: [%.2f, %.2f]\n", 
        mean_boxes_unequal - 1.96*se_boxes_unequal, 
        mean_boxes_unequal + 1.96*se_boxes_unequal)
@printf("    Relative error: %.2f%%\n", 100*se_boxes_unequal/mean_boxes_unequal)

println("\nQ5. Probability of buying more than X boxes:")
for threshold in [50, 100, 200]
    prob = mean(boxes_needed_unequal .> threshold)
    count = sum(boxes_needed_unequal .> threshold)
    @printf("    P(boxes > %3d) = %.4f (%.2f%%) - %d/%d simulations\n", 
            threshold, prob, 100*prob, count, n_simulations)
end

@printf("\nMedian: %.0f, Range: [%d, %d]\n", 
        median(boxes_needed_unequal), minimum(boxes_needed_unequal), 
        maximum(boxes_needed_unequal))

quantiles_unequal = quantile(boxes_needed_unequal, [0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
println("\nQuantiles:")
for (q, val) in zip([25, 50, 75, 90, 95, 99], quantiles_unequal)
    @printf("  %d%%: %d boxes\n", q, Int(round(val)))
end

# Visualization
println("\n\nCreating toy collector visualizations...")

p_toy1 = histogram(boxes_needed_equal, bins=50,
                  c=:lightblue, alpha=0.7, label="",
                  xlabel="Number of boxes needed",
                  ylabel="Frequency",
                  title="Equal Probabilities\n(each figure: 1/15)")
vline!(p_toy1, [mean_boxes_equal], c=:red, lw=3, ls=:dash, 
       label="Simulated: $(round(mean_boxes_equal, digits=1))")
vline!(p_toy1, [theoretical_mean], c=:darkgreen, lw=3, ls=:dot, 
       label="Theoretical: $(round(theoretical_mean, digits=1))")

p_toy2 = histogram(boxes_needed_unequal, bins=50,
                  c=:lightcoral, alpha=0.7, label="",
                  xlabel="Number of boxes needed",
                  ylabel="Frequency",
                  title="Unequal Probabilities\n(rare figures: 0.02)")
vline!(p_toy2, [mean_boxes_unequal], c=:red, lw=3, ls=:dash, 
       label="Mean: $(round(mean_boxes_unequal, digits=1))")

sorted_equal = sort(boxes_needed_equal)
ecdf_equal = (1:length(sorted_equal)) ./ length(sorted_equal)
sorted_unequal = sort(boxes_needed_unequal)
ecdf_unequal = (1:length(sorted_unequal)) ./ length(sorted_unequal)

p_toy3 = plot(sorted_equal, ecdf_equal, c=:blue, lw=2, label="Equal probs",
             xlabel="Number of boxes", ylabel="Cumulative Probability",
             title="Cumulative Distribution Comparison")
plot!(p_toy3, sorted_unequal, ecdf_unequal, c=:red, lw=2, label="Unequal probs")
for threshold in [50, 100, 200]
    vline!(p_toy3, [threshold], c=:gray, ls=:dot, alpha=0.5, label="")
end

p_toy4 = boxplot(["Equal" "Unequal"], 
                [boxes_needed_equal boxes_needed_unequal],
                c=[:lightblue :lightcoral], label="",
                ylabel="Number of boxes needed",
                title="Distribution Comparison")
scatter!(p_toy4, [1, 2], [mean_boxes_equal, mean_boxes_unequal],
        c=:red, ms=8, marker=:diamond, label="Mean")

threshold_seq = 0:10:400
prob_exceed_equal = [mean(boxes_needed_equal .> x) for x in threshold_seq]
prob_exceed_unequal = [mean(boxes_needed_unequal .> x) for x in threshold_seq]

p_toy5 = plot(threshold_seq, prob_exceed_equal, c=:blue, lw=2, 
             label="Equal probs",
             xlabel="Number of boxes", ylabel="P(boxes > x)",
             title="Probability of Exceeding Threshold",
             ylim=(0, 1))
plot!(p_toy5, threshold_seq, prob_exceed_unequal, c=:red, lw=2, 
      label="Unequal probs")
for h in [0.5, 0.9, 0.95]
    hline!(p_toy5, [h], c=:gray, ls=:dot, alpha=0.5, label="")
end
for v in [50, 100, 200]
    vline!(p_toy5, [v], c=:gray, ls=:dot, alpha=0.5, label="")
end

p_toy6 = bar(toy_names, toy_probs, c=:lightcoral, label="",
            xlabel="Figure", ylabel="Probability",
            title="Figure Probabilities\n(Unequal Case)",
            ylim=(0, maximum(toy_probs) * 1.1))
hline!(p_toy6, [1/15], c=:blue, ls=:dash, lw=2, 
       label="Equal prob = 1/15")

plot(p_toy1, p_toy2, p_toy3, p_toy4, p_toy5, p_toy6, 
     layout=(2, 3), size=(1500, 1000))
savefig("../plots/monte_carlo_toy_collector.png")
println("Saved: ../plots/monte_carlo_toy_collector.png")

# Key insights
println("\n\nKEY INSIGHTS:")
println("-" ^ 70)
println("1. Impact of unequal probabilities:")
@printf("   Equal case:   %.1f boxes expected\n", mean_boxes_equal)
@printf("   Unequal case: %.1f boxes expected\n", mean_boxes_unequal)
@printf("   Increase: %.1f boxes (%.0f%%)\n", 
        mean_boxes_unequal - mean_boxes_equal, 
        100*(mean_boxes_unequal - mean_boxes_equal)/mean_boxes_equal)

println("\n2. Rare items dominate collection time:")
println("   Rarest figures have probability 0.02 (vs 1/15=0.067)")
println("   Expected wait for a specific rare item: 1/0.02 = 50 boxes")

@printf("\n3. High variability in unequal case:\n")
@printf("   95th percentile: %d boxes (%.1f%% above mean)\n", 
        Int(round(quantiles_unequal[5])), 
        100*(quantiles_unequal[5] - mean_boxes_unequal)/mean_boxes_unequal)

println("\n4. Practical implications:")
@printf("   P(> 100 boxes) = %.1f%% - significant risk of extreme cases\n", 
        100*mean(boxes_needed_unequal .> 100))
@printf("   P(> 200 boxes) = %.1f%% - rare but possible\n", 
        100*mean(boxes_needed_unequal .> 200))

# =============================================================================
# SUMMARY
# =============================================================================

println("\n\n", "=" ^ 70)
println("SUMMARY: MONTE CARLO SIMULATIONS")
println("=" ^ 70)
println()

println("KEY PRINCIPLES:")
println("1. OMC uses IID simulations to estimate expectations: θ = E[g(X)]")
println("2. Law of Large Numbers: ȳₙ converges to θ")
println("3. Central Limit Theorem: ȳₙ ~ N(θ, σ²/n) for large n")
println("4. Standard error decreases as O(1/√n)")
println("5. We can construct confidence intervals using SE = s/√n\n")

println("METHODS COVERED:")
println("1. Ordinary Monte Carlo - basic estimation with IID samples")
println("2. Approximating π - geometric probability method")
println("3. Monte Carlo integration - sequential stopping rule")
println("4. Bootstrap methods - resampling with replacement")
println("5. Permutation tests - resampling without replacement")
println("6. Coupon collector problem - complex probability estimation\n")

println("PRACTICAL APPLICATIONS:")
println("- High-dimensional problems (election forecasts, sports predictions)")
println("- Intractable integrals and expectations")
println("- Hypothesis testing without parametric assumptions")
println("- Uncertainty quantification in complex systems")
println("- Sequential decision making with stopping rules\n")

println("=" ^ 70)
println("Generated plots:")
println("  1. monte_carlo_binomial.png - Binomial approximation analysis")
println("  2. monte_carlo_pi_approximation.png - π estimation convergence")
println("  3. monte_carlo_pi_detailed.png - π scatter plot visualization")
println("  4. monte_carlo_integration_sequential.png - Sequential stopping")
println("  5. monte_carlo_integration_distribution.png - Integration distributions")
println("  6. monte_carlo_bootstrap_test.png - Bootstrap hypothesis test")
println("  7. monte_carlo_toy_collector.png - Coupon collector analysis")
println("=" ^ 70)

println("\nMONTE CARLO SIMULATIONS TUTORIAL COMPLETE!")
println("All methods demonstrated with practical examples.")
