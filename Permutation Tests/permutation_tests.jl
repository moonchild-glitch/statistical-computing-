# ============================================================================
# Permutation Tests - Julia Implementation
# ============================================================================

using Statistics
using Random
using Plots
using HypothesisTests

# Setup: Create plots directory if it doesn't exist
if !isdir("plots")
    mkdir("plots")
end

# Permutation tests are a nonparametric approach to hypothesis testing that
# rely on the exchangeability of observations under the null hypothesis.
# They are particularly useful when:
# - The sample size is small
# - Distributional assumptions are questionable
# - The test statistic doesn't have a known distribution
# - We want exact p-values rather than asymptotic approximations

# ============================================================================
# Example: Reading DRP Study
# ============================================================================

# Reading Example Context:
# A study examines whether a new teaching method improves reading comprehension
# scores (Degree of Reading Power, DRP). We have:
# - 21 students in the treatment group (new method)
# - 23 students in the control group (traditional method)
# - Total: 44 students with DRP scores
#
# Research Question: Does the new teaching method improve reading scores?
#
# Null Hypothesis (H₀): The teaching method has no effect on DRP scores.
#                       The treatment and control labels are arbitrary.
#
# Alternative Hypothesis (Hₐ): The treatment group has higher mean DRP scores.

# ============================================================================
# Permutation Test Procedure
# ============================================================================

# The permutation test procedure consists of the following steps:

# Step 1: Choose a Permutation Resample
# ---------------------------------------
# Choose 21 of the 44 students at random to be the treatment group; 
# the other 23 are the control group. This is an ordinary SRS (Simple Random 
# Sample), chosen without replacement. It is called a permutation resample.
#
# Calculate the mean DRP score in each group, using the individual DRP scores. 
# The difference between these means is our test statistic.
#
# Key insight: Under H₀, the group labels (treatment vs. control) are arbitrary,
# so any random assignment of students to groups is equally likely.

# Step 2: Repeat the Resampling
# ------------------------------
# Repeat this resampling from the 44 students hundreds (or thousands) of times. 
# The distribution of the test statistic from these resamples estimates the 
# sampling distribution under the condition that H₀ is true. 
#
# This distribution is called a permutation distribution.
#
# The permutation distribution represents: "What values of the test statistic 
# would we observe if the null hypothesis were true and we randomly assigned 
# students to treatment and control groups?"

# Step 3: Compute the Observed Test Statistic
# --------------------------------------------
# Compute the actually observed value of the test statistic from the real data
# (the actual difference in means between the treatment and control groups).

# Step 4: Find the P-value
# -------------------------
# The p-value is the proportion of permutation resamples that produce a test 
# statistic as extreme as (or more extreme than) the observed statistic.
#
# For a one-sided test (treatment > control):
#   p-value = (# of permutations with difference ≥ observed) / (total # of permutations)
#
# For a two-sided test:
#   p-value = (# of permutations with |difference| ≥ |observed|) / (total # of permutations)
#
# Interpretation: The p-value represents the probability of observing a 
# difference as large as (or larger than) what we actually observed, 
# if the null hypothesis were true.

# ============================================================================
# Key Concepts
# ============================================================================

# Permutation Resample:
# - A random reassignment of the original observations to groups
# - Sampling without replacement from the combined data
# - Preserves the individual data values, only shuffles group assignments
# - Under H₀, all possible permutations are equally likely

# Permutation Distribution:
# - The distribution of the test statistic across all (or many) permutations
# - Estimates the sampling distribution under H₀
# - Provides the null reference distribution for computing p-values
# - Does not rely on normal distribution assumptions

# Advantages of Permutation Tests:
# - Exact p-values (not asymptotic approximations)
# - No distributional assumptions required
# - Works with any test statistic
# - Robust to outliers and non-normality
# - Intuitive interpretation

# Limitations:
# - Computationally intensive for large datasets
# - Requires exchangeability under H₀
# - May not be feasible with very complex designs
# - For large numbers of possible permutations, we typically use 
#   Monte Carlo approximation (random sampling of permutations)

# ============================================================================
# Relationship to Other Methods
# ============================================================================

# Permutation tests are related to but distinct from:
#
# - Randomization tests: Same methodology, emphasizes the random assignment
#   in the experimental design
#
# - Bootstrap: Bootstrap resamples WITH replacement to estimate sampling 
#   distributions and construct confidence intervals. Permutation tests 
#   resample WITHOUT replacement to test hypotheses.
#
# - Exact tests (e.g., Fisher's exact test): Special cases of permutation 
#   tests for categorical data
#
# - Rank tests (e.g., Wilcoxon, Mann-Whitney): Use ranks instead of raw 
#   values, but similar permutation logic applies

# ============================================================================
# Implementation: Reading DRP Study
# ============================================================================

# Treatment group DRP scores (n1 = 21 students)
T = [24, 43, 58, 71, 61, 44, 67, 49, 59, 52, 62, 54, 46, 43, 57, 
     43, 57, 56, 53, 49, 33]

# Control group DRP scores (n2 = 23 students)
C = [42, 43, 55, 26, 33, 41, 19, 54, 46, 10, 17, 60, 37, 42, 55, 
     28, 62, 53, 37, 42, 20, 48, 85]

# Sample sizes
n1 = length(T)  # 21 treatment students
n2 = length(C)  # 23 control students

# Combine all observations into a single vector
Z = vcat(T, C)
N = length(Z)   # Total: 44 students

# Step 3: Compute the observed test statistic
# The observed difference in means (Treatment - Control)
obs_stat = mean(T) - mean(C)

println("Observed difference in means (T - C): ", obs_stat)
println("Treatment mean: ", mean(T))
println("Control mean: ", mean(C), "\n")

# Step 2: Generate permutation distribution
# Number of permutation resamples
B = 1000

# Vector to store the test statistic for each permutation
new_stats = zeros(B)

# Set random seed for reproducibility
Random.seed!(42)

# Perform B permutation resamples
for i in 1:B
    # Step 1: Randomly select n1 observations for the "treatment" group
    # Sample without replacement from the combined data
    idx = sample(1:N, n1, replace=false)
    
    # Create new treatment and control groups
    newT = Z[idx]      # New treatment group (n1 observations)
    newC = Z[setdiff(1:N, idx)]  # New control group (remaining n2 observations)
    
    # Calculate the test statistic for this permutation
    new_stats[i] = mean(newT) - mean(newC)
end

# Step 4: Calculate the p-value
# For a one-sided test (H₁: Treatment > Control)
# p-value = proportion of permutations with difference ≥ observed difference
#
# Note: We include obs_stat in the comparison to account for the observed data
# This gives us (# successes + 1) / (B + 1), which is the standard formula
pvalue = mean(vcat(new_stats, obs_stat) .>= obs_stat)

println("P-value (one-sided test): ", pvalue)

# Interpretation:
# p-value ≈ 0.01 means that only about 1% of the random permutations produced
# a difference in means as large as or larger than what we observed.
# This provides strong evidence against H₀, suggesting that the treatment
# does have a positive effect on reading scores.

# ============================================================================
# Visualization: Permutation Distribution
# ============================================================================

# Plot the permutation distribution with observed statistic
p1 = histogram(new_stats, bins=30, 
               xlabel="Difference in Means (T - C)",
               ylabel="Frequency",
               title="Permutation Distribution",
               legend=false,
               color=:skyblue,
               linecolor=:black)
vline!([obs_stat], linewidth=2, linestyle=:dash, color=:red, 
       label="Observed = $(round(obs_stat, digits=2))")
savefig(p1, "plots/reading_permutation_distribution.png")

# ============================================================================
# Additional Analysis
# ============================================================================

# Summary statistics
println("\n=== Summary Statistics ===")
println("Treatment group:")
println("  n = ", n1)
println("  Mean = ", mean(T))
println("  SD = ", std(T))
println("  Min = ", minimum(T))
println("  Max = ", maximum(T), "\n")

println("Control group:")
println("  n = ", n2)
println("  Mean = ", mean(C))
println("  SD = ", std(C))
println("  Min = ", minimum(C))
println("  Max = ", maximum(C), "\n")

println("Observed difference: ", obs_stat)
println("Permutation distribution of differences:")
println("  Mean = ", mean(new_stats))
println("  SD = ", std(new_stats))
println("  Min = ", minimum(new_stats))
println("  Max = ", maximum(new_stats))

# Two-sided p-value (if we wanted to test for any difference)
pvalue_two_sided = mean(abs.(vcat(new_stats, obs_stat)) .>= abs(obs_stat))
println("\nTwo-sided p-value: ", pvalue_two_sided)

# ============================================================================
# Example: Chickwts (Chicken Weights by Feed Type)
# ============================================================================

println("\n=== Chickwts Dataset ===")

# Simulate chickwts data (weights for different feed types)
X = [246, 309, 238, 229, 329, 266, 218, 237, 215, 245, 243, 230, 268, 289]  # soybean
Y = [141, 148, 169, 213, 257, 244, 271, 243, 230, 248, 327, 329]  # linseed

println("Soybean group: n = ", length(X), ", mean = ", round(mean(X), digits=2))
println("Linseed group: n = ", length(Y), ", mean = ", round(mean(Y), digits=2))

# Visualize the data with a boxplot
p2 = boxplot(["Soybean" "Linseed"], [X Y],
             ylabel="Weight",
             xlabel="Feed Type",
             title="Chicken Weights by Feed Type",
             legend=false)
savefig(p2, "plots/chickwts_boxplot.png")

# ============================================================================
# Permutation Test: Soybean vs Linseed Feed
# ============================================================================

# Number of permutation resamples
B = 999

# Combine both groups into a single vector
Z = vcat(X, Y)

# Vector to store the test statistic for each permutation
reps = zeros(B)

# Total observations (14 soybean + 12 linseed = 26)
K = length(Z)

# Compute the observed test statistic
# Using t-statistic from t-test (two-sample t-test)
t_test_result = EqualVarianceTTest(X, Y)
t0 = t_test_result.t

println("\nObserved t-statistic: ", t0)

# Perform B permutation resamples
for i in 1:B
    # Randomly select 14 observations for the first group (same size as X)
    idx = sample(1:K, 14, replace=false)
    
    # Create new groups
    x1 = Z[idx]      # New "soybean" group (14 observations)
    y1 = Z[setdiff(1:K, idx)]  # New "linseed" group (remaining 12 observations)
    
    # Calculate the t-statistic for this permutation
    t_result = EqualVarianceTTest(x1, y1)
    reps[i] = t_result.t
end

# Calculate the p-value
# For a one-sided test (testing if soybean > linseed based on observed t0)
p = mean(vcat(reps, t0) .>= t0)

println("P-value (one-sided test): ", round(p, digits=3))

# Visualize the permutation distribution
p3 = histogram(reps, bins=30,
               xlabel="t-statistic",
               ylabel="Frequency",
               title="Permutation Distribution",
               legend=false,
               color=:skyblue,
               linecolor=:black)
vline!([t0], linewidth=2, linestyle=:dash, color=:red,
       label="Observed = $(round(t0, digits=2))")
savefig(p3, "plots/chickwts_t_permutation_distribution.png")

# Comparison with parametric t-test
parametric_test = EqualVarianceTTest(X, Y)
println("\n=== Comparison with Parametric t-test ===")
println("Parametric t-test p-value: ", pvalue(parametric_test))
println("Permutation test p-value: ", round(p, digits=3))
println("\nNote: These should be similar if normality assumptions are met,")
println("but permutation test is valid even if assumptions are violated.")

# ============================================================================
# Example: Kolmogorov-Smirnov (K-S) Statistic
# ============================================================================

# Compute the observed K-S statistic
ks_test_result = ApproximateTwoSampleKSTest(X, Y)
DO = ks_test_result.δ

println("\n=== K-S Statistic Permutation Test ===")
println("Observed K-S statistic: ", DO)

# Vector to store K-S statistics from permutations
D = zeros(B)

# Perform B permutation resamples
for i in 1:B
    # Randomly select 14 observations for the first group
    idx = sample(1:K, 14, replace=false)
    
    # Create new groups
    x1 = Z[idx]      # New group 1 (14 observations)
    y1 = Z[setdiff(1:K, idx)]  # New group 2 (remaining 12 observations)
    
    # Calculate the K-S statistic for this permutation
    ks_result = ApproximateTwoSampleKSTest(x1, y1)
    D[i] = ks_result.δ
end

# Calculate the p-value
# K-S statistic is always positive, so we use one-sided test
p = mean(vcat(D, DO) .>= DO)

println("P-value (K-S permutation test): ", round(p, digits=3))

# Visualize the permutation distribution
p4 = histogram(D, bins=30,
               xlabel="K-S Statistic",
               ylabel="Frequency",
               title="Permutation Distribution",
               legend=false,
               color=:skyblue,
               linecolor=:black)
vline!([DO], linewidth=2, linestyle=:dash, color=:red,
       label="Observed = $(round(DO, digits=2))")
savefig(p4, "plots/chickwts_ks_permutation_distribution.png")

# ============================================================================
# Comparison: t-statistic vs K-S statistic
# ============================================================================

println("\n=== Comparison of Test Statistics ===")
println("t-statistic is sensitive to differences in:")
println("  - Location (mean difference)")
println("  - Best for detecting shifts in central tendency\n")

println("K-S statistic is sensitive to differences in:")
println("  - Location (shifts)")
println("  - Scale (spread/variance)")
println("  - Shape (skewness, modality)")
println("  - Any difference in the overall distributions\n")

println("When to use each:")
println("  - Use t-statistic when primarily interested in mean differences")
println("  - Use K-S statistic for more general distributional differences")
println("  - K-S is more conservative but detects broader alternatives")

# Compare with parametric K-S test
parametric_ks = ApproximateTwoSampleKSTest(X, Y)
println("\n=== Comparison with Parametric K-S Test ===")
println("Parametric K-S test p-value: ", pvalue(parametric_ks))
println("Permutation K-S test p-value: ", round(p, digits=3))

# ============================================================================
# Example: Correlation Coefficients
# ============================================================================

# Score data (e.g., course scores)
Score = [58, 48, 48, 41, 34, 43, 38, 53, 41, 60, 55, 44, 
         43, 49, 47, 33, 47, 40, 46, 53, 40, 45, 39, 47, 
         50, 53, 46, 53]

# SAT scores
SAT = [590, 590, 580, 490, 550, 580, 550, 700, 560, 690, 800, 600, 
       650, 580, 660, 590, 600, 540, 610, 580, 620, 600, 560, 560, 
       570, 630, 510, 620]

# Compute the observed correlation coefficient
r_obt = cor(Score, SAT)

println("\n=== Correlation Test ===")
println("The obtained correlation is ", r_obt)

# Permutation test for correlation
# Under H₀, the pairing between Score and SAT is arbitrary
# We permute one variable while keeping the other fixed

# Number of permutation resamples
nreps = 5000

# Vector to store correlation coefficients from permutations
r_random = zeros(nreps)

# Perform permutation test
for i in 1:nreps
    Y = Score  # Keep Score fixed
    X = shuffle(SAT)  # Randomly permute SAT
    r_random[i] = cor(X, Y)  # Compute correlation for this permutation
end

# Calculate p-value
# Proportion of permutations with correlation >= observed correlation
prob = sum(r_random .>= r_obt) / nreps
println("Probability randomized r >= r.obt: ", round(prob, digits=4))

# Visualize the permutation distribution
p5 = histogram(r_random, bins=50,
               xlabel="Correlation Coefficient",
               ylabel="Frequency",
               title="Permutation Distribution",
               legend=false,
               color=:skyblue,
               linecolor=:black)
vline!([r_obt], linewidth=2, linestyle=:dash, color=:red,
       label="Observed = $(round(r_obt, digits=3))")
savefig(p5, "plots/correlation_permutation_distribution.png")

# ============================================================================
# Summary
# ============================================================================

println("\n=== Summary ===")
println("Observed correlation: ", round(r_obt, digits=4))
println("Permutation distribution:")
println("  Mean: ", round(mean(r_random), digits=4))
println("  SD: ", round(std(r_random), digits=4))
println("  Min: ", round(minimum(r_random), digits=4))
println("  Max: ", round(maximum(r_random), digits=4))
println("P-value: ", round(prob, digits=4))

# Comparison with parametric test
# Julia's correlation test using CorrelationTest
cor_test = CorrelationTest(Score, SAT)
println("\n=== Comparison with Parametric Test ===")
println("Parametric correlation test p-value: ", pvalue(cor_test))
println("Permutation test p-value: ", round(prob, digits=4))

# ============================================================================
# Bootstrap vs Randomization (Permutation)
# ============================================================================

# Example: Simple data with 4 observations
x = [45, 53, 73, 80]
y = [22, 30, 29, 38]

println("\n=== Bootstrap vs Randomization ===")
println("Original data:")
println("x: ", x)
println("y: ", y)

println("\nRandomization resample 1 (permute y, keep x fixed):")
println("x: ", x)
println("y: ", shuffle(y))

println("\nRandomization resample 2 (permute y, keep x fixed):")
println("x: ", x)
println("y: ", shuffle(y))

# Key observations:
# 1. X values stay in the same order: 45, 53, 73, 80
# 2. Y values are randomly permuted: different orderings of 22, 30, 29, 38
# 3. No replacement: each y value appears exactly once in each resample
# 4. This tests if the pairing between x and y is meaningful
#
# In contrast, bootstrap resampling would:
# - Sample BOTH x and y WITH replacement
# - Preserve the original pairings (resample cases, not variables)
# - Be used to estimate standard errors or confidence intervals, not test hypotheses

println("\nScript completed successfully!")
