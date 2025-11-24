# ============================================================================
# Permutation Tests
# ============================================================================

# Setup: Create plots directory if it doesn't exist
if (!dir.exists("plots")) {
    dir.create("plots")
}

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
T <- c(24, 43, 58, 71, 61, 44, 67, 49, 59, 52, 62, 54, 46, 43, 57, 
       43, 57, 56, 53, 49, 33)

# Control group DRP scores (n2 = 23 students)
C <- c(42, 43, 55, 26, 33, 41, 19, 54, 46, 10, 17, 60, 37, 42, 55, 
       28, 62, 53, 37, 42, 20, 48, 85)

# Sample sizes
n1 <- length(T)  # 21 treatment students
n2 <- length(C)  # 23 control students

# Combine all observations into a single vector
Z <- c(T, C)
N <- length(Z)   # Total: 44 students

# Step 3: Compute the observed test statistic
# The observed difference in means (Treatment - Control)
obs_stat <- mean(T) - mean(C)

cat("Observed difference in means (T - C):", obs_stat, "\n")
cat("Treatment mean:", mean(T), "\n")
cat("Control mean:", mean(C), "\n\n")

# Step 2: Generate permutation distribution
# Number of permutation resamples
B <- 1000

# Vector to store the test statistic for each permutation
new_stats <- numeric(B)

# Perform B permutation resamples
for(i in 1:B) {
    # Step 1: Randomly select n1 observations for the "treatment" group
    # Sample without replacement from the combined data
    idx <- sample(1:N, size = n1, replace = FALSE)
    
    # Create new treatment and control groups
    newT <- Z[idx]      # New treatment group (n1 observations)
    newC <- Z[-idx]     # New control group (remaining n2 observations)
    
    # Calculate the test statistic for this permutation
    new_stats[i] <- mean(newT) - mean(newC)
}

# Step 4: Calculate the p-value
# For a one-sided test (H₁: Treatment > Control)
# p-value = proportion of permutations with difference ≥ observed difference
#
# Note: We include obs_stat in the comparison to account for the observed data
# This gives us (# successes + 1) / (B + 1), which is the standard formula
pvalue <- mean(c(obs_stat, new_stats) >= obs_stat)

cat("P-value (one-sided test):", pvalue, "\n")

# Interpretation:
# p-value ≈ 0.01 means that only about 1% of the random permutations produced
# a difference in means as large as or larger than what we observed.
# This provides strong evidence against H₀, suggesting that the treatment
# does have a positive effect on reading scores.

# ============================================================================
# Visualization: Permutation Distribution
# ============================================================================

# Plot the permutation distribution with observed statistic
png("plots/reading_permutation_distribution.png", width = 800, height = 600)
hist(new_stats, main = "Permutation Distribution")
points(obs_stat, 0, cex = 1, pch = 16)
dev.off()

# ============================================================================
# Additional Analysis
# ============================================================================

# Summary statistics
cat("\n=== Summary Statistics ===\n")
cat("Treatment group:\n")
cat("  n =", n1, "\n")
cat("  Mean =", mean(T), "\n")
cat("  SD =", sd(T), "\n")
cat("  Min =", min(T), "\n")
cat("  Max =", max(T), "\n\n")

cat("Control group:\n")
cat("  n =", n2, "\n")
cat("  Mean =", mean(C), "\n")
cat("  SD =", sd(C), "\n")
cat("  Min =", min(C), "\n")
cat("  Max =", max(C), "\n\n")

cat("Observed difference:", obs_stat, "\n")
cat("Permutation distribution of differences:\n")
cat("  Mean =", mean(new_stats), "\n")
cat("  SD =", sd(new_stats), "\n")
cat("  Min =", min(new_stats), "\n")
cat("  Max =", max(new_stats), "\n")

# Two-sided p-value (if we wanted to test for any difference)
pvalue_two_sided <- mean(abs(c(obs_stat, new_stats)) >= abs(obs_stat))
cat("\nTwo-sided p-value:", pvalue_two_sided, "\n")

# ============================================================================
# Notes on Implementation
# ============================================================================

# 1. Why include obs_stat in the p-value calculation?
#    mean(c(obs_stat, new_stats) >= obs_stat)
#    
#    This gives us the formula: (# of permutations ≥ observed + 1) / (B + 1)
#    - The "+1" in numerator accounts for the observed data itself
#    - The "+1" in denominator is the total number of datasets (B permutations + 1 observed)
#    - This is the standard unbiased estimate of the p-value
#    - Ensures p-value is never exactly 0 (minimum is 1/(B+1))

# 2. Why sample without replacement?
#    - Under H₀, we're assuming all observations come from the same distribution
#    - We want to preserve the actual data values, just reassign group labels
#    - Sampling without replacement ensures each permutation uses all N observations
#    - This maintains the conditioning on the observed data

# 3. How many permutations should we use?
#    - Total possible permutations: choose(44, 21) = 5.36 × 10^12
#    - Too many to enumerate all possibilities
#    - B = 1000 gives p-value precision of about ±0.01
#    - B = 10000 gives precision of about ±0.003
#    - For publication, B = 10000 or more is recommended

# 4. Interpretation of p-value ≈ 0.01:
#    - If there were truly no treatment effect (H₀ true)
#    - Only about 1% of random group assignments would produce
#      a difference as large as what we observed
#    - This is strong evidence that the treatment has a real effect
#    - At α = 0.05, we would reject H₀

# ============================================================================
# Example: Chickwts (Chicken Weights by Feed Type)
# ============================================================================

# The chickwts dataset contains weights of chicks fed different feed supplements
# We can use a permutation test to determine if feed type affects weight

# Load the built-in chickwts dataset
data(chickwts)

# Display the structure of the data
cat("\n=== Chickwts Dataset ===\n")
cat("Structure:\n")
str(chickwts)

cat("\nFirst few rows:\n")
print(head(chickwts))

cat("\nSummary by feed type:\n")
print(summary(chickwts))

# Visualize the data with a boxplot
png("plots/chickwts_boxplot.png", width = 800, height = 600)
boxplot(weight ~ feed, data = chickwts)
dev.off()

# ============================================================================
# Permutation Test: Soybean vs Linseed Feed
# ============================================================================

# Compare two specific feed types: soybean and linseed
# Research Question: Does feed type (soybean vs. linseed) affect chick weight?
#
# Null Hypothesis (H₀): Feed type has no effect on chick weight.
#                       The feed labels are arbitrary.
#
# Alternative Hypothesis (Hₐ): The two feed types produce different weights.

# Extract weights for soybean and linseed feed groups
X <- as.vector(chickwts$weight[chickwts$feed == "soybean"])
Y <- as.vector(chickwts$weight[chickwts$feed == "linseed"])

cat("\n=== Comparing Soybean vs Linseed ===\n")
cat("Soybean group: n =", length(X), ", mean =", mean(X), "\n")
cat("Linseed group: n =", length(Y), ", mean =", mean(Y), "\n")

# Number of permutation resamples
B <- 999

# Combine both groups into a single vector
Z <- c(X, Y)

# Vector to store the test statistic for each permutation
reps <- numeric(B)

# Indices for all observations
K <- 1:26  # Total observations (14 soybean + 12 linseed = 26)

# Compute the observed test statistic
# Using t-statistic from t.test (two-sample t-test)
t0 <- t.test(X, Y)$statistic

cat("Observed t-statistic:", t0, "\n")

# Perform B permutation resamples
for(i in 1:B) {
    # Randomly select 14 observations for the first group (same size as X)
    k <- sample(K, size = 14, replace = FALSE)
    
    # Create new groups
    x1 <- Z[k]      # New "soybean" group (14 observations)
    y1 <- Z[-k]     # New "linseed" group (remaining 12 observations)
    
    # Calculate the t-statistic for this permutation
    reps[i] <- t.test(x1, y1)$statistic
}

# Calculate the p-value
# For a one-sided test (testing if soybean > linseed based on observed t0)
p <- mean(c(t0, reps) >= t0)

cat("P-value (one-sided test):", p, "\n")

# Note: For a two-sided test, we would use:
# p_two_sided <- mean(abs(c(t0, reps)) >= abs(t0))

# Visualize the permutation distribution
png("plots/chickwts_t_permutation_distribution.png", width = 800, height = 600)
hist(reps, main = "Permutation Distribution")
points(t0, 0, cex = 1, pch = 16)
dev.off()

# ============================================================================
# Key Observations: Chickwts Permutation Test
# ============================================================================

# Using t-statistic as the test statistic:
# - The t-statistic is a standardized measure of the difference between means
# - It accounts for both the difference in means and the variability within groups
# - Formula: t = (mean(X) - mean(Y)) / sqrt(s²/n₁ + s²/n₂)
# - More powerful than just using difference in means when variances are unequal

# Why permutation test with t-statistic?
# - Traditional t-test assumes normality and equal variances
# - Permutation test makes no such assumptions
# - We use the t-statistic as our test statistic, but get the p-value from
#   the permutation distribution rather than the t-distribution
# - This combines the power of the t-statistic with the flexibility of
#   permutation testing

# Interpretation:
# - If p-value is small (e.g., < 0.05), we have evidence that feed type matters
# - If p-value is large, we don't have enough evidence to conclude that
#   the feed types produce different weights

# Comparison with parametric t-test:
cat("\n=== Comparison with Parametric t-test ===\n")
parametric_test <- t.test(X, Y)
cat("Parametric t-test p-value:", parametric_test$p.value, "\n")
cat("Permutation test p-value:", p, "\n")
cat("\nNote: These should be similar if normality assumptions are met,\n")
cat("but permutation test is valid even if assumptions are violated.\n")

# ============================================================================
# Example: Kolmogorov-Smirnov (K-S) Statistic
# ============================================================================

# The K-S test is used to compare two distributions
# Unlike the t-test which focuses on means, the K-S test detects differences
# in any aspect of the distributions (location, spread, shape)
#
# K-S statistic: D = max|F₁(x) - F₂(x)|
# where F₁ and F₂ are the empirical CDFs of the two samples
#
# The K-S statistic measures the maximum vertical distance between the
# two empirical cumulative distribution functions

# Compute the observed K-S statistic
# Note: Setting exact=FALSE to use asymptotic approximation (faster)
DO <- ks.test(X, Y, exact = FALSE)$statistic
## Warning in ks.test(X, Y, exact = F): p-value will be approximate in the
## presence of ties

cat("\n=== K-S Statistic Permutation Test ===\n")
cat("Observed K-S statistic:", DO, "\n")

# Suppress warnings about ties (common in discrete data)
options(warn = -1)

# Vector to store K-S statistics from permutations
D <- numeric(B)

# Perform B permutation resamples
for(i in 1:B) {
    # Randomly select 14 observations for the first group
    k <- sample(K, size = 14, replace = FALSE)
    
    # Create new groups
    x1 <- Z[k]      # New group 1 (14 observations)
    y1 <- Z[-k]     # New group 2 (remaining 12 observations)
    
    # Calculate the K-S statistic for this permutation
    D[i] <- ks.test(x1, y1, exact = FALSE)$statistic
}

# Reset warning options to default
options(warn = 0)

# Calculate the p-value
# K-S statistic is always positive, so we use one-sided test
p <- mean(c(DO, D) >= DO)

cat("P-value (K-S permutation test):", p, "\n")

# Visualize the permutation distribution
png("plots/chickwts_ks_permutation_distribution.png", width = 800, height = 600)
hist(D, main = "Permutation Distribution")
points(DO, 0, cex = 1, pch = 16)
dev.off()

# ============================================================================
# Comparison: t-statistic vs K-S statistic
# ============================================================================

cat("\n=== Comparison of Test Statistics ===\n")
cat("t-statistic is sensitive to differences in:\n")
cat("  - Location (mean difference)\n")
cat("  - Best for detecting shifts in central tendency\n\n")

cat("K-S statistic is sensitive to differences in:\n")
cat("  - Location (shifts)\n")
cat("  - Scale (spread/variance)\n")
cat("  - Shape (skewness, modality)\n")
cat("  - Any difference in the overall distributions\n\n")

cat("When to use each:\n")
cat("  - Use t-statistic when primarily interested in mean differences\n")
cat("  - Use K-S statistic for more general distributional differences\n")
cat("  - K-S is more conservative but detects broader alternatives\n")

# Compare with parametric K-S test
parametric_ks <- ks.test(X, Y, exact = FALSE)
cat("\n=== Comparison with Parametric K-S Test ===\n")
cat("Parametric K-S test p-value:", parametric_ks$p.value, "\n")
cat("Permutation K-S test p-value:", p, "\n")

# ============================================================================
# Key Insights: Choice of Test Statistic
# ============================================================================

# The permutation test framework is flexible:
# - We can use ANY test statistic
# - Common choices:
#   * Difference in means (simple, intuitive)
#   * t-statistic (powerful for location differences)
#   * K-S statistic (detects any distributional difference)
#   * Difference in medians (robust to outliers)
#   * Difference in variances (tests for spread)
#   * Custom statistics designed for specific alternatives
#
# Choice depends on:
# - What aspect of the distributions we want to compare
# - What alternative hypothesis we have in mind
# - Power considerations for our specific problem
#
# Permutation test advantages:
# - Works with any test statistic
# - No parametric assumptions needed
# - P-value is exact (for complete enumeration) or approximate (Monte Carlo)
# - Maintains correct Type I error rate under the null hypothesis

# ============================================================================
# Example: Correlation Coefficients
# ============================================================================

# Testing whether two variables are correlated using permutation test
# Research Question: Is there a relationship between Score and SAT?
#
# Null Hypothesis (H₀): There is no correlation between Score and SAT.
#                       The pairing of observations is arbitrary.
#
# Alternative Hypothesis (Hₐ): Score and SAT are positively correlated.

# Score data (e.g., course scores)
Score <- c(58,  48,  48,  41,  34,  43,  38,  53,  41,  60,  55,  44,  
           43, 49,  47,  33,  47,  40,  46,  53,  40,  45,  39,  47,  
           50,  53,  46,  53)

# SAT scores
SAT <- c(590, 590, 580, 490, 550, 580, 550, 700, 560, 690, 800, 600, 
         650, 580, 660, 590, 600, 540, 610, 580, 620, 600, 560, 560, 
         570, 630, 510, 620)

# Compute the observed correlation coefficient
r.obt <- cor(Score, SAT)
cat("\n=== Correlation Test ===\n")
cat("The obtained correlation is ", r.obt, '\n')

# Permutation test for correlation
# Under H₀, the pairing between Score and SAT is arbitrary
# We permute one variable while keeping the other fixed

# Number of permutation resamples
nreps <- 5000

# Vector to store correlation coefficients from permutations
r.random <- numeric(nreps)

# Perform permutation test
for (i in 1:nreps) {
    Y <- Score  # Keep Score fixed
    X <- sample(SAT, 28, replace = FALSE)  # Randomly permute SAT
    r.random[i] <- cor(X, Y)  # Compute correlation for this permutation
}

# Calculate p-value
# Proportion of permutations with correlation >= observed correlation
prob <- length(r.random[r.random >= r.obt]) / nreps
cat("Probability randomized r >= r.obt", prob, "\n")

# Visualize the permutation distribution
png("plots/correlation_permutation_distribution.png", width = 800, height = 600)
hist(r.random, main = "Permutation Distribution")
points(r.obt, 0, cex = 1, pch = 16)
dev.off()

# ============================================================================
# Key Insights: Permutation Test for Correlation
# ============================================================================

# What we're testing:
# - H₀: No relationship between Score and SAT (correlation = 0)
# - Under H₀, any pairing of Score and SAT values is equally likely
# - We randomly permute one variable to break any association
#
# Permutation strategy:
# - Keep one variable (Score) in its original order
# - Randomly shuffle the other variable (SAT)
# - This breaks the original pairing while preserving the marginal distributions
# - Compute correlation for each random pairing
#
# Interpretation:
# - If observed correlation is in the tail of the permutation distribution,
#   we have evidence against H₀ (evidence of real correlation)
# - The p-value tells us how rare the observed correlation is under H₀

cat("\n=== Summary ===\n")
cat("Observed correlation:", round(r.obt, 4), "\n")
cat("Permutation distribution:\n")
cat("  Mean:", round(mean(r.random), 4), "\n")
cat("  SD:", round(sd(r.random), 4), "\n")
cat("  Min:", round(min(r.random), 4), "\n")
cat("  Max:", round(max(r.random), 4), "\n")
cat("P-value:", prob, "\n")

# Comparison with parametric test
parametric_cor <- cor.test(Score, SAT)
cat("\n=== Comparison with Parametric Test ===\n")
cat("Parametric correlation test p-value:", parametric_cor$p.value, "\n")
cat("Permutation test p-value:", prob, "\n")

# Notes on correlation permutation tests:
# 1. We permute ONE variable, not both
#    - Permuting both would preserve the original pairing
#    - Permuting one breaks the association we want to test
#
# 2. Alternative: permute the indices
#    - Equivalent to permuting one variable
#    - r.random[i] <- cor(Score, SAT[sample(1:28)])
#
# 3. This tests for ANY correlation (linear relationship)
#    - For testing specific alternatives (positive vs negative),
#      adjust the p-value calculation
#
# 4. Advantages over parametric cor.test:
#    - No assumption of bivariate normality
#    - Works with any sample size
#    - Can be adapted to test other association measures
#      (e.g., Spearman's rho, Kendall's tau)

# ============================================================================
# Bootstrap vs Randomization (Permutation)
# ============================================================================

# Key Conceptual Difference:
# - RANDOMIZATION (Permutation): Tests hypotheses about population parameters
#   by permuting observed data, holding some values constant
# - BOOTSTRAP: Estimates sampling distributions by resampling WITH replacement
#   from the observed data

# When we use a randomization approach, we permute the Y values
# while holding the X values constant.

# Example: Simple data with 4 observations
x <- c(45, 53, 73, 80)
y <- c(22, 30, 29, 38)

cat("\n=== Bootstrap vs Randomization ===\n")
cat("Original data:\n")
print(rbind(x, y))

cat("\nRandomization resample 1 (permute y, keep x fixed):\n")
print(rbind(x, sample(y, size = 4, replace = FALSE)))

cat("\nRandomization resample 2 (permute y, keep x fixed):\n")
print(rbind(x, sample(y, size = 4, replace = FALSE)))

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