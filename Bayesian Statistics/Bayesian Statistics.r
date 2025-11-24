# ============================================================================
# Bayesian Statistics
# ============================================================================

# Agenda:
# - Bayesian Inference
# - Priors
# - Bayesian Point Estimates
# - Bayesian Hypothesis Testing
# - Bayes Factors
# - Applications

cat(strrep("=", 80), "\n")
cat("Bayesian Statistics\n")
cat(strrep("=", 80), "\n\n")

# ============================================================================
# Bayes Factors
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Bayes Factors\n")
cat(strrep("=", 80), "\n\n")

cat("Bayes factors provide a way to quantify evidence for one hypothesis\n")
cat("relative to another hypothesis.\n\n")

cat("Definition:\n")
cat("  Bayes Factor (BF) = P(Data | H1) / P(Data | H0)\n")
cat("  where H1 is the alternative hypothesis and H0 is the null hypothesis\n\n")

cat("Interpretation:\n")
cat("  BF > 1:  Evidence favors H1 (alternative)\n")
cat("  BF < 1:  Evidence favors H0 (null)\n")
cat("  BF = 1:  No preference between hypotheses\n\n")

cat("Common interpretation scale (Kass & Raftery, 1995):\n")
cat("  BF < 1/10:    Strong evidence for H0\n")
cat("  1/10 < BF < 1/3:  Moderate evidence for H0\n")
cat("  1/3 < BF < 1:     Weak evidence for H0\n")
cat("  BF = 1:           No evidence either way\n")
cat("  1 < BF < 3:       Weak evidence for H1\n")
cat("  3 < BF < 10:      Moderate evidence for H1\n")
cat("  BF > 10:          Strong evidence for H1\n\n")

# ============================================================================
# Example: Bayes Factor Calculation
# ============================================================================

cat(strrep("=", 80), "\n")
cat("Example: Comparing Bayes Factor and p-value\n")
cat(strrep("=", 80), "\n\n")

# Simulate data for demonstration
set.seed(123)
n <- 50
mu_null <- 0
mu_alt <- 0.5
sigma <- 1
data <- rnorm(n, mean = mu_alt, sd = sigma)

cat("Simulated data:\n")
cat("  Sample size: n =", n, "\n")
cat("  True mean:", mu_alt, "\n")
cat("  True SD:", sigma, "\n")
cat("  Sample mean:", round(mean(data), 4), "\n")
cat("  Sample SD:", round(sd(data), 4), "\n\n")

# Frequentist p-value
t_test <- t.test(data, mu = mu_null)
pvalue <- t_test$p.value

cat("Frequentist t-test:\n")
cat("  H0: μ = 0\n")
cat("  Ha: μ ≠ 0\n")
cat("  t-statistic:", round(t_test$statistic, 4), "\n")
cat("  p-value:", round(pvalue, 4), "\n\n")

# Bayesian analysis with Bayes Factor
# Using BayesFactor package for proper Bayes factor calculation
if (!require("BayesFactor", quietly = TRUE)) {
  cat("Installing BayesFactor package...\n")
  install.packages("BayesFactor", repos = "http://cran.r-project.org")
  library(BayesFactor)
}

# Calculate Bayes Factor using ttestBF
bf_result <- ttestBF(data, mu = mu_null)
bayes_factor <- exp(bf_result@bayesFactor$bf)

cat("Bayesian t-test (Bayes Factor):\n")
cat("  H0: μ = 0\n")
cat("  H1: μ ≠ 0\n")
cat("  Bayes Factor (BF10):", round(bayes_factor, 4), "\n\n")

cat("Interpretation:\n")
if (bayes_factor > 10) {
  cat("  Strong evidence for H1 (μ ≠ 0)\n")
} else if (bayes_factor > 3) {
  cat("  Moderate evidence for H1 (μ ≠ 0)\n")
} else if (bayes_factor > 1) {
  cat("  Weak evidence for H1 (μ ≠ 0)\n")
} else if (bayes_factor > 1/3) {
  cat("  Weak evidence for H0 (μ = 0)\n")
} else if (bayes_factor > 1/10) {
  cat("  Moderate evidence for H0 (μ = 0)\n")
} else {
  cat("  Strong evidence for H0 (μ = 0)\n")
}
cat("\n")

# ============================================================================
# Comparison: Bayes Factor vs p-value
# ============================================================================

cat(strrep("=", 80), "\n")
cat("Comparison: Bayes Factor vs p-value\n")
cat(strrep("=", 80), "\n\n")

cat("For comparison, we calculated:\n")
cat("  - Bayes factor:", round(bayes_factor, 3), "\n")
cat("  - Frequentist p-value:", round(pvalue, 3), "\n\n")

cat("Key observations:\n\n")

cat("1. Different interpretations:\n")
cat("   - p-value: Probability of observing data at least as extreme as what\n")
cat("              we observed, assuming H0 is true\n")
cat("   - Bayes factor: Ratio of evidence for H1 relative to H0\n\n")

cat("2. Bayes factors and p-values are sort of comparable, but are not identical\n\n")

cat("3. Theoretical relationship:\n")
cat("   In fact, it is a theorem that in situations like this the Bayes factor\n")
cat("   is always larger than the p-value, at least asymptotically.\n\n")

cat("4. Implications:\n")
cat("   This makes Bayesian tests more conservative, less likely to reject\n")
cat("   the null hypothesis, than frequentists.\n\n")

cat("5. Philosophical question:\n")
cat("   Either the frequentists are too optimistic or the Bayesians are too\n")
cat("   conservative, or perhaps both.\n\n")

# ============================================================================
# Visualization: BF vs p-value relationship
# ============================================================================

cat("Creating visualization of Bayes Factor vs p-value relationship...\n\n")

# Simulate multiple datasets to show relationship
n_sims <- 100
sample_sizes <- c(20, 50, 100, 200)
results <- data.frame()

for (n_size in sample_sizes) {
  for (i in 1:n_sims) {
    sim_data <- rnorm(n_size, mean = 0.3, sd = 1)
    
    # p-value
    t_result <- t.test(sim_data, mu = 0)
    p <- t_result$p.value
    
    # Bayes factor
    bf <- tryCatch({
      bf_obj <- ttestBF(sim_data, mu = 0)
      exp(bf_obj@bayesFactor$bf)
    }, error = function(e) NA)
    
    if (!is.na(bf) && is.finite(bf)) {
      results <- rbind(results, data.frame(
        n = n_size,
        pvalue = p,
        bayes_factor = bf,
        log_bf = log10(bf)
      ))
    }
  }
}

# Create plots
par(mfrow = c(2, 2))

# Plot 1: Scatter plot of BF vs p-value
plot(results$pvalue, results$bayes_factor,
     xlab = "p-value",
     ylab = "Bayes Factor",
     main = "Bayes Factor vs p-value",
     pch = 19,
     col = rgb(0, 0, 1, 0.3),
     log = "y")
abline(h = 1, col = "red", lty = 2, lwd = 2)
abline(v = 0.05, col = "darkgreen", lty = 2, lwd = 2)
legend("topright", 
       legend = c("BF = 1 (no evidence)", "p = 0.05"),
       col = c("red", "darkgreen"),
       lty = 2,
       lwd = 2)

# Plot 2: By sample size
colors <- c("blue", "green", "orange", "red")
plot(NULL, NULL,
     xlim = c(0, 1),
     ylim = range(results$log_bf),
     xlab = "p-value",
     ylab = "log10(Bayes Factor)",
     main = "BF vs p-value by Sample Size")
for (i in seq_along(sample_sizes)) {
  subset_data <- results[results$n == sample_sizes[i], ]
  points(subset_data$pvalue, subset_data$log_bf,
         pch = 19,
         col = rgb(col2rgb(colors[i])[1]/255, 
                   col2rgb(colors[i])[2]/255, 
                   col2rgb(colors[i])[3]/255, 0.5))
}
abline(h = 0, col = "red", lty = 2, lwd = 2)
abline(v = 0.05, col = "darkgreen", lty = 2, lwd = 2)
legend("topright",
       legend = paste("n =", sample_sizes),
       col = colors,
       pch = 19)

# Plot 3: Histogram of p-values
hist(results$pvalue,
     breaks = 20,
     main = "Distribution of p-values",
     xlab = "p-value",
     col = "lightblue",
     border = "white")
abline(v = 0.05, col = "red", lty = 2, lwd = 2)

# Plot 4: Histogram of log(BF)
hist(results$log_bf,
     breaks = 20,
     main = "Distribution of log10(Bayes Factor)",
     xlab = "log10(BF)",
     col = "lightgreen",
     border = "white")
abline(v = 0, col = "red", lty = 2, lwd = 2)
abline(v = log10(3), col = "orange", lty = 2, lwd = 2)
abline(v = log10(10), col = "darkred", lty = 2, lwd = 2)

par(mfrow = c(1, 1))

cat("Visualization complete!\n\n")

# ============================================================================
# Decision boundaries comparison
# ============================================================================

cat(strrep("=", 80), "\n")
cat("Decision Boundaries: Frequentist vs Bayesian\n")
cat(strrep("=", 80), "\n\n")

# Count decisions at α = 0.05
freq_reject <- sum(results$pvalue < 0.05)
bayes_reject_weak <- sum(results$bayes_factor > 1)
bayes_reject_moderate <- sum(results$bayes_factor > 3)
bayes_reject_strong <- sum(results$bayes_factor > 10)

cat("Decision summary (out of", nrow(results), "simulations):\n\n")
cat("Frequentist (α = 0.05):\n")
cat("  Reject H0:", freq_reject, "(", 
    round(100 * freq_reject / nrow(results), 1), "%)\n\n")

cat("Bayesian:\n")
cat("  Any evidence for H1 (BF > 1):", bayes_reject_weak, "(",
    round(100 * bayes_reject_weak / nrow(results), 1), "%)\n")
cat("  Moderate evidence (BF > 3):", bayes_reject_moderate, "(",
    round(100 * bayes_reject_moderate / nrow(results), 1), "%)\n")
cat("  Strong evidence (BF > 10):", bayes_reject_strong, "(",
    round(100 * bayes_reject_strong / nrow(results), 1), "%)\n\n")

cat("This demonstrates that Bayesian tests are generally more conservative.\n\n")

# ============================================================================
# Summary
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Summary: Bayesian Statistics\n")
cat(strrep("=", 80), "\n\n")

cat("Key concepts covered:\n\n")

cat("1. Bayesian Inference\n")
cat("   - Updating beliefs based on data using Bayes' theorem\n")
cat("   - Prior × Likelihood = Posterior (up to normalization)\n\n")

cat("2. Priors\n")
cat("   - Represent initial beliefs before seeing data\n")
cat("   - Can be informative (based on previous knowledge) or non-informative\n")
cat("   - Choice of prior affects posterior, especially with small samples\n\n")

cat("3. Bayesian Point Estimates\n")
cat("   - Posterior mean: E[θ|data]\n")
cat("   - Posterior median: 50th percentile of posterior\n")
cat("   - Maximum a posteriori (MAP): mode of posterior\n\n")

cat("4. Bayesian Hypothesis Testing\n")
cat("   - Compare models/hypotheses rather than just reject/fail to reject\n")
cat("   - Posterior probability of hypotheses\n")
cat("   - More intuitive interpretation than p-values\n\n")

cat("5. Bayes Factors\n")
cat("   - Ratio of evidence for one hypothesis vs another\n")
cat("   - BF = P(Data|H1) / P(Data|H0)\n")
cat("   - Continuous measure of evidence (not just binary decision)\n")
cat("   - Generally more conservative than frequentist tests\n\n")

cat("Applications:\n")
cat("  - Pattern recognition\n")
cat("  - Spam detection\n")
cat("  - Search for lost objects\n")
cat("  - Medical diagnosis\n")
cat("  - Machine learning (Bayesian networks, naive Bayes)\n")
cat("  - A/B testing\n")
cat("  - Parameter estimation with uncertainty quantification\n\n")

cat("Important notes:\n")
cat("  - Calculations are trivial in our examples so far, not usually the case\n")
cat("  - Real-world Bayesian inference often requires:\n")
cat("    * Markov Chain Monte Carlo (MCMC) methods\n")
cat("    * Variational inference\n")
cat("    * Approximate Bayesian computation (ABC)\n")
cat("  - Modern computational tools make Bayesian methods practical\n")
cat("    (e.g., Stan, PyMC, JAGS, BayesFactor in R)\n\n")

cat("Philosophical perspective:\n")
cat("  Frequentist: Probability = long-run frequency\n")
cat("  Bayesian: Probability = degree of belief\n\n")
cat("  Neither approach is universally 'better' - they answer different questions\n")
cat("  and make different assumptions. Choose based on:\n")
cat("    - Nature of the problem\n")
cat("    - Available prior information\n")
cat("    - Desired interpretation of results\n")
cat("    - Computational resources\n\n")

cat(strrep("=", 80), "\n")
cat("Bayesian Statistics Complete!\n")
cat(strrep("=", 80), "\n")
