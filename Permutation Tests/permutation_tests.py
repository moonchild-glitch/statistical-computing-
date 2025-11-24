# ============================================================================
# Permutation Tests - Python Implementation
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Setup: Create plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

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
T = np.array([24, 43, 58, 71, 61, 44, 67, 49, 59, 52, 62, 54, 46, 43, 57, 
              43, 57, 56, 53, 49, 33])

# Control group DRP scores (n2 = 23 students)
C = np.array([42, 43, 55, 26, 33, 41, 19, 54, 46, 10, 17, 60, 37, 42, 55, 
              28, 62, 53, 37, 42, 20, 48, 85])

# Sample sizes
n1 = len(T)  # 21 treatment students
n2 = len(C)  # 23 control students

# Combine all observations into a single vector
Z = np.concatenate([T, C])
N = len(Z)   # Total: 44 students

# Step 3: Compute the observed test statistic
# The observed difference in means (Treatment - Control)
obs_stat = np.mean(T) - np.mean(C)

print(f"Observed difference in means (T - C): {obs_stat}")
print(f"Treatment mean: {np.mean(T)}")
print(f"Control mean: {np.mean(C)}\n")

# Step 2: Generate permutation distribution
# Number of permutation resamples
B = 1000

# Vector to store the test statistic for each permutation
new_stats = np.zeros(B)

# Set random seed for reproducibility
np.random.seed(42)

# Perform B permutation resamples
for i in range(B):
    # Step 1: Randomly select n1 observations for the "treatment" group
    # Sample without replacement from the combined data
    idx = np.random.choice(N, size=n1, replace=False)
    
    # Create new treatment and control groups
    newT = Z[idx]      # New treatment group (n1 observations)
    newC = np.delete(Z, idx)  # New control group (remaining n2 observations)
    
    # Calculate the test statistic for this permutation
    new_stats[i] = np.mean(newT) - np.mean(newC)

# Step 4: Calculate the p-value
# For a one-sided test (H₁: Treatment > Control)
# p-value = proportion of permutations with difference ≥ observed difference
#
# Note: We include obs_stat in the comparison to account for the observed data
# This gives us (# successes + 1) / (B + 1), which is the standard formula
pvalue = np.mean(np.append(new_stats, obs_stat) >= obs_stat)

print(f"P-value (one-sided test): {pvalue}")

# Interpretation:
# p-value ≈ 0.01 means that only about 1% of the random permutations produced
# a difference in means as large as or larger than what we observed.
# This provides strong evidence against H₀, suggesting that the treatment
# does have a positive effect on reading scores.

# ============================================================================
# Visualization: Permutation Distribution
# ============================================================================

# Plot the permutation distribution with observed statistic
plt.figure(figsize=(10, 6))
plt.hist(new_stats, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(obs_stat, color='red', linestyle='--', linewidth=2, label=f'Observed = {obs_stat:.2f}')
plt.xlabel('Difference in Means (T - C)')
plt.ylabel('Frequency')
plt.title('Permutation Distribution')
plt.legend()
plt.savefig('plots/reading_permutation_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================================
# Additional Analysis
# ============================================================================

# Summary statistics
print("\n=== Summary Statistics ===")
print("Treatment group:")
print(f"  n = {n1}")
print(f"  Mean = {np.mean(T):.5f}")
print(f"  SD = {np.std(T, ddof=1):.5f}")
print(f"  Min = {np.min(T)}")
print(f"  Max = {np.max(T)}\n")

print("Control group:")
print(f"  n = {n2}")
print(f"  Mean = {np.mean(C):.5f}")
print(f"  SD = {np.std(C, ddof=1):.5f}")
print(f"  Min = {np.min(C)}")
print(f"  Max = {np.max(C)}\n")

print(f"Observed difference: {obs_stat:.5f}")
print("Permutation distribution of differences:")
print(f"  Mean = {np.mean(new_stats):.5f}")
print(f"  SD = {np.std(new_stats, ddof=1):.5f}")
print(f"  Min = {np.min(new_stats):.5f}")
print(f"  Max = {np.max(new_stats):.5f}")

# Two-sided p-value (if we wanted to test for any difference)
pvalue_two_sided = np.mean(np.abs(np.append(new_stats, obs_stat)) >= np.abs(obs_stat))
print(f"\nTwo-sided p-value: {pvalue_two_sided}")

# ============================================================================
# Example: Chickwts (Chicken Weights by Feed Type)
# ============================================================================

# For this example, we'll simulate the chickwts dataset structure
# In practice, you would load from a dataset or CSV file

print("\n=== Chickwts Dataset ===")

# Simulate chickwts data (weights for different feed types)
chickwts_data = {
    'soybean': np.array([246, 309, 238, 229, 329, 266, 218, 237, 215, 245, 243, 230, 268, 289]),
    'linseed': np.array([141, 148, 169, 213, 257, 244, 271, 243, 230, 248, 327, 329])
}

# Extract weights for soybean and linseed feed groups
X = chickwts_data['soybean']
Y = chickwts_data['linseed']

print(f"Soybean group: n = {len(X)}, mean = {np.mean(X):.2f}")
print(f"Linseed group: n = {len(Y)}, mean = {np.mean(Y):.2f}")

# Visualize the data with a boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([X, Y], labels=['Soybean', 'Linseed'])
plt.ylabel('Weight')
plt.xlabel('Feed Type')
plt.title('Chicken Weights by Feed Type')
plt.savefig('plots/chickwts_boxplot.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================================
# Permutation Test: Soybean vs Linseed Feed
# ============================================================================

# Number of permutation resamples
B = 999

# Combine both groups into a single vector
Z = np.concatenate([X, Y])

# Vector to store the test statistic for each permutation
reps = np.zeros(B)

# Total observations (14 soybean + 12 linseed = 26)
K = len(Z)

# Compute the observed test statistic
# Using t-statistic from t-test (two-sample t-test)
t0, _ = stats.ttest_ind(X, Y)

print(f"\nObserved t-statistic: {t0:.6f}")

# Perform B permutation resamples
for i in range(B):
    # Randomly select 14 observations for the first group (same size as X)
    idx = np.random.choice(K, size=14, replace=False)
    
    # Create new groups
    x1 = Z[idx]      # New "soybean" group (14 observations)
    y1 = np.delete(Z, idx)  # New "linseed" group (remaining 12 observations)
    
    # Calculate the t-statistic for this permutation
    reps[i], _ = stats.ttest_ind(x1, y1)

# Calculate the p-value
# For a one-sided test (testing if soybean > linseed based on observed t0)
p = np.mean(np.append(reps, t0) >= t0)

print(f"P-value (one-sided test): {p:.3f}")

# Visualize the permutation distribution
plt.figure(figsize=(10, 6))
plt.hist(reps, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(t0, color='red', linestyle='--', linewidth=2, label=f'Observed = {t0:.2f}')
plt.xlabel('t-statistic')
plt.ylabel('Frequency')
plt.title('Permutation Distribution')
plt.legend()
plt.savefig('plots/chickwts_t_permutation_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# Comparison with parametric t-test
parametric_test = stats.ttest_ind(X, Y)
print("\n=== Comparison with Parametric t-test ===")
print(f"Parametric t-test p-value: {parametric_test.pvalue:.6f}")
print(f"Permutation test p-value: {p:.3f}")
print("\nNote: These should be similar if normality assumptions are met,")
print("but permutation test is valid even if assumptions are violated.")

# ============================================================================
# Example: Kolmogorov-Smirnov (K-S) Statistic
# ============================================================================

# Compute the observed K-S statistic
DO, _ = stats.ks_2samp(X, Y)

print(f"\n=== K-S Statistic Permutation Test ===")
print(f"Observed K-S statistic: {DO:.6f}")

# Vector to store K-S statistics from permutations
D = np.zeros(B)

# Perform B permutation resamples
for i in range(B):
    # Randomly select 14 observations for the first group
    idx = np.random.choice(K, size=14, replace=False)
    
    # Create new groups
    x1 = Z[idx]      # New group 1 (14 observations)
    y1 = np.delete(Z, idx)  # New group 2 (remaining 12 observations)
    
    # Calculate the K-S statistic for this permutation
    D[i], _ = stats.ks_2samp(x1, y1)

# Calculate the p-value
# K-S statistic is always positive, so we use one-sided test
p = np.mean(np.append(D, DO) >= DO)

print(f"P-value (K-S permutation test): {p:.3f}")

# Visualize the permutation distribution
plt.figure(figsize=(10, 6))
plt.hist(D, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(DO, color='red', linestyle='--', linewidth=2, label=f'Observed = {DO:.2f}')
plt.xlabel('K-S Statistic')
plt.ylabel('Frequency')
plt.title('Permutation Distribution')
plt.legend()
plt.savefig('plots/chickwts_ks_permutation_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================================
# Comparison: t-statistic vs K-S statistic
# ============================================================================

print("\n=== Comparison of Test Statistics ===")
print("t-statistic is sensitive to differences in:")
print("  - Location (mean difference)")
print("  - Best for detecting shifts in central tendency\n")

print("K-S statistic is sensitive to differences in:")
print("  - Location (shifts)")
print("  - Scale (spread/variance)")
print("  - Shape (skewness, modality)")
print("  - Any difference in the overall distributions\n")

print("When to use each:")
print("  - Use t-statistic when primarily interested in mean differences")
print("  - Use K-S statistic for more general distributional differences")
print("  - K-S is more conservative but detects broader alternatives")

# Compare with parametric K-S test
parametric_ks = stats.ks_2samp(X, Y)
print("\n=== Comparison with Parametric K-S Test ===")
print(f"Parametric K-S test p-value: {parametric_ks.pvalue:.6f}")
print(f"Permutation K-S test p-value: {p:.3f}")

# ============================================================================
# Example: Correlation Coefficients
# ============================================================================

# Score data (e.g., course scores)
Score = np.array([58, 48, 48, 41, 34, 43, 38, 53, 41, 60, 55, 44, 
                  43, 49, 47, 33, 47, 40, 46, 53, 40, 45, 39, 47, 
                  50, 53, 46, 53])

# SAT scores
SAT = np.array([590, 590, 580, 490, 550, 580, 550, 700, 560, 690, 800, 600, 
                650, 580, 660, 590, 600, 540, 610, 580, 620, 600, 560, 560, 
                570, 630, 510, 620])

# Compute the observed correlation coefficient
r_obt = np.corrcoef(Score, SAT)[0, 1]

print("\n=== Correlation Test ===")
print(f"The obtained correlation is {r_obt:.6f}")

# Permutation test for correlation
# Under H₀, the pairing between Score and SAT is arbitrary
# We permute one variable while keeping the other fixed

# Number of permutation resamples
nreps = 5000

# Vector to store correlation coefficients from permutations
r_random = np.zeros(nreps)

# Perform permutation test
for i in range(nreps):
    Y = Score  # Keep Score fixed
    X = np.random.permutation(SAT)  # Randomly permute SAT
    r_random[i] = np.corrcoef(X, Y)[0, 1]  # Compute correlation for this permutation

# Calculate p-value
# Proportion of permutations with correlation >= observed correlation
prob = np.sum(r_random >= r_obt) / nreps
print(f"Probability randomized r >= r.obt: {prob:.4f}")

# Visualize the permutation distribution
plt.figure(figsize=(10, 6))
plt.hist(r_random, bins=50, edgecolor='black', alpha=0.7)
plt.axvline(r_obt, color='red', linestyle='--', linewidth=2, label=f'Observed = {r_obt:.3f}')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Frequency')
plt.title('Permutation Distribution')
plt.legend()
plt.savefig('plots/correlation_permutation_distribution.png', dpi=100, bbox_inches='tight')
plt.close()

# ============================================================================
# Summary
# ============================================================================

print("\n=== Summary ===")
print(f"Observed correlation: {r_obt:.4f}")
print("Permutation distribution:")
print(f"  Mean: {np.mean(r_random):.4f}")
print(f"  SD: {np.std(r_random, ddof=1):.4f}")
print(f"  Min: {np.min(r_random):.4f}")
print(f"  Max: {np.max(r_random):.4f}")
print(f"P-value: {prob:.4f}")

# Comparison with parametric test
parametric_cor = stats.pearsonr(Score, SAT)
print("\n=== Comparison with Parametric Test ===")
print(f"Parametric correlation test p-value: {parametric_cor.pvalue:.8f}")
print(f"Permutation test p-value: {prob:.4f}")

# ============================================================================
# Bootstrap vs Randomization (Permutation)
# ============================================================================

# Example: Simple data with 4 observations
x = np.array([45, 53, 73, 80])
y = np.array([22, 30, 29, 38])

print("\n=== Bootstrap vs Randomization ===")
print("Original data:")
print("x:", x)
print("y:", y)

print("\nRandomization resample 1 (permute y, keep x fixed):")
print("x:", x)
print("y:", np.random.permutation(y))

print("\nRandomization resample 2 (permute y, keep x fixed):")
print("x:", x)
print("y:", np.random.permutation(y))

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

print("\nScript completed successfully!")
