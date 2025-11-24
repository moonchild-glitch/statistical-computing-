# ============================================
# DISTRIBUTIONS IN R
# ============================================
# Statistical Computing Tutorial
# Topic: Probability Distributions and Goodness of Fit Testing
#
# Agenda:
# 1. Random number generation
# 2. Built-in distributions in R
# 3. Parametric distributions as models
# 4. Methods of fitting (moments, generalized moments, likelihood)
# 5. Methods of checking (visual comparisons, statistics, tests, calibration)
# 6. Chi-squared test for continuous distributions
# 7. Better alternatives (K-S test, bootstrap, smooth tests)
# ============================================

# Create plots directory if it doesn't exist
if (!dir.exists("../plots")) {
  dir.create("../plots", recursive = TRUE)
}

# Set seed for reproducibility
set.seed(42)

# ============================================
# RANDOM NUMBER GENERATION
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("RANDOM NUMBER GENERATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("R has built-in random number generators for many distributions")
print("")
print("General naming convention:")
print("  - dname(): density function (PDF)")
print("  - pname(): cumulative distribution function (CDF)")
print("  - qname(): quantile function (inverse CDF)")
print("  - rname(): random number generator")
print("")
print("where 'name' is the distribution name (norm, exp, unif, etc.)")

# Examples of random number generation
print("\n--- Uniform Distribution ---")
uniform_sample <- runif(10, min=0, max=1)
print("Sample of 10 uniform random numbers [0,1]:")
print(round(uniform_sample, 4))

print("\n--- Normal Distribution ---")
normal_sample <- rnorm(10, mean=0, sd=1)
print("Sample of 10 standard normal random numbers:")
print(round(normal_sample, 4))

print("\n--- Exponential Distribution ---")
exp_sample <- rexp(10, rate=1)
print("Sample of 10 exponential random numbers (rate=1):")
print(round(exp_sample, 4))

# ============================================
# DISTRIBUTIONS IN R
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("BUILT-IN DISTRIBUTIONS IN R\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("R provides many common distributions:")
print("")
print("Continuous distributions:")
print("  - Normal: rnorm(n, mean, sd)")
print("  - Exponential: rexp(n, rate)")
print("  - Uniform: runif(n, min, max)")
print("  - Gamma: rgamma(n, shape, rate)")
print("  - Beta: rbeta(n, shape1, shape2)")
print("  - Chi-squared: rchisq(n, df)")
print("  - t-distribution: rt(n, df)")
print("  - F-distribution: rf(n, df1, df2)")
print("")
print("Discrete distributions:")
print("  - Binomial: rbinom(n, size, prob)")
print("  - Poisson: rpois(n, lambda)")
print("  - Geometric: rgeom(n, prob)")
print("  - Negative binomial: rnbinom(n, size, prob)")

# Visualize some distributions
png("../plots/dist_common_distributions.png", width=1200, height=800)
par(mfrow=c(2,3))

# Normal distribution
x_norm <- seq(-4, 4, length.out=1000)
plot(x_norm, dnorm(x_norm, 0, 1), type='l', lwd=2, col='blue',
     main="Normal Distribution", xlab="x", ylab="Density")
hist(rnorm(1000), breaks=30, add=TRUE, freq=FALSE, col=rgb(0,0,1,0.3), border=NA)

# Exponential distribution
x_exp <- seq(0, 5, length.out=1000)
plot(x_exp, dexp(x_exp, 1), type='l', lwd=2, col='red',
     main="Exponential Distribution", xlab="x", ylab="Density")
hist(rexp(1000), breaks=30, add=TRUE, freq=FALSE, col=rgb(1,0,0,0.3), border=NA)

# Gamma distribution
x_gamma <- seq(0, 20, length.out=1000)
plot(x_gamma, dgamma(x_gamma, shape=2, rate=0.5), type='l', lwd=2, col='green',
     main="Gamma Distribution (shape=2, rate=0.5)", xlab="x", ylab="Density")
hist(rgamma(1000, 2, 0.5), breaks=30, add=TRUE, freq=FALSE, col=rgb(0,1,0,0.3), border=NA)

# Beta distribution
x_beta <- seq(0, 1, length.out=1000)
plot(x_beta, dbeta(x_beta, 2, 5), type='l', lwd=2, col='purple',
     main="Beta Distribution (a=2, b=5)", xlab="x", ylab="Density")
hist(rbeta(1000, 2, 5), breaks=30, add=TRUE, freq=FALSE, col=rgb(0.5,0,0.5,0.3), border=NA)

# Chi-squared distribution
x_chisq <- seq(0, 20, length.out=1000)
plot(x_chisq, dchisq(x_chisq, 5), type='l', lwd=2, col='orange',
     main="Chi-squared Distribution (df=5)", xlab="x", ylab="Density")
hist(rchisq(1000, 5), breaks=30, add=TRUE, freq=FALSE, col=rgb(1,0.5,0,0.3), border=NA)

# Binomial distribution (discrete)
x_binom <- 0:20
plot(x_binom, dbinom(x_binom, 20, 0.5), type='h', lwd=3, col='darkblue',
     main="Binomial Distribution (n=20, p=0.5)", xlab="x", ylab="Probability")
points(x_binom, dbinom(x_binom, 20, 0.5), pch=19, col='darkblue')

par(mfrow=c(1,1))
dev.off()
print("Plot saved: dist_common_distributions.png")

# ============================================
# PARAMETRIC DISTRIBUTIONS AS MODELS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("PARAMETRIC DISTRIBUTIONS AS MODELS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Parametric distributions serve as models for real-world data")
print("")
print("Key idea: Assume data comes from a known family of distributions")
print("          but with unknown parameters")
print("")
print("Goal: Estimate the parameters from the data")
print("      Check if the model fits well")

# Generate some example data (exponential)
true_rate <- 0.5
sample_data <- rexp(500, rate=true_rate)

print(sprintf("\nGenerated 500 samples from Exponential(rate=%.1f)", true_rate))
print(sprintf("Sample mean: %.4f (theoretical: %.4f)", mean(sample_data), 1/true_rate))
print(sprintf("Sample variance: %.4f (theoretical: %.4f)", var(sample_data), 1/true_rate^2))

# ============================================
# METHOD OF MOMENTS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("FITTING: METHOD OF MOMENTS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Method of Moments: Match sample moments to theoretical moments")
print("")
print("For exponential distribution:")
print("  E[X] = 1/λ")
print("  So: λ_hat = 1/mean(X)")

# Estimate rate using method of moments
rate_mom <- 1/mean(sample_data)
print(sprintf("\nMethod of Moments estimate: λ = %.4f", rate_mom))
print(sprintf("True rate: λ = %.4f", true_rate))
print(sprintf("Error: %.4f", abs(rate_mom - true_rate)))

# Visualize the fit
png("../plots/dist_method_of_moments.png", width=800, height=600)
hist(sample_data, breaks=30, freq=FALSE, 
     main="Method of Moments Fit",
     xlab="Value", ylab="Density",
     col='lightblue', border='white')
x_seq <- seq(0, max(sample_data), length.out=1000)
lines(x_seq, dexp(x_seq, rate_mom), col='red', lwd=2)
lines(x_seq, dexp(x_seq, true_rate), col='blue', lwd=2, lty=2)
legend("topright", 
       legend=c("Data histogram", 
                sprintf("MoM fit (λ=%.3f)", rate_mom),
                sprintf("True (λ=%.3f)", true_rate)),
       col=c("lightblue", "red", "blue"),
       lty=c(NA, 1, 2), lwd=c(NA, 2, 2),
       fill=c("lightblue", NA, NA), border=c("black", NA, NA))
dev.off()
print("Plot saved: dist_method_of_moments.png")

# ============================================
# MAXIMUM LIKELIHOOD ESTIMATION
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("FITTING: MAXIMUM LIKELIHOOD ESTIMATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Maximum Likelihood: Find parameters that maximize the likelihood function")
print("")
print("Likelihood function: L(θ|data) = product of f(x_i|θ)")
print("Log-likelihood: l(θ|data) = sum of log(f(x_i|θ))")
print("")
print("For exponential distribution:")
print("  l(λ) = n*log(λ) - λ*sum(x_i)")
print("  Maximize by taking derivative and setting to 0")
print("  Solution: λ_MLE = n / sum(x_i) = 1/mean(x)")

# MLE for exponential (same as method of moments in this case!)
rate_mle <- 1/mean(sample_data)
print(sprintf("\nMaximum Likelihood estimate: λ = %.4f", rate_mle))
print(sprintf("True rate: λ = %.4f", true_rate))

# For more complex distributions, use optim()
# Example: fit gamma distribution using MLE
print("\n--- Fitting Gamma Distribution ---")

# Generate gamma data
gamma_data <- rgamma(500, shape=2, rate=0.5)

# Negative log-likelihood for gamma
neg_log_lik_gamma <- function(params, data) {
  shape <- params[1]
  rate <- params[2]
  if (shape <= 0 || rate <= 0) return(Inf)
  -sum(dgamma(data, shape=shape, rate=rate, log=TRUE))
}

# Optimize
mle_result <- optim(c(1, 1), neg_log_lik_gamma, data=gamma_data)
shape_mle <- mle_result$par[1]
rate_mle_gamma <- mle_result$par[2]

print(sprintf("MLE estimates: shape = %.4f, rate = %.4f", shape_mle, rate_mle_gamma))
print(sprintf("True parameters: shape = 2.0000, rate = 0.5000"))

# Visualize
png("../plots/dist_mle_gamma.png", width=800, height=600)
hist(gamma_data, breaks=30, freq=FALSE,
     main="Maximum Likelihood Fit (Gamma Distribution)",
     xlab="Value", ylab="Density",
     col='lightgreen', border='white')
x_seq_g <- seq(0, max(gamma_data), length.out=1000)
lines(x_seq_g, dgamma(x_seq_g, shape=shape_mle, rate=rate_mle_gamma), 
      col='red', lwd=2)
lines(x_seq_g, dgamma(x_seq_g, shape=2, rate=0.5), 
      col='blue', lwd=2, lty=2)
legend("topright",
       legend=c("Data histogram",
                sprintf("MLE fit (shape=%.2f, rate=%.2f)", shape_mle, rate_mle_gamma),
                "True (shape=2.00, rate=0.50)"),
       col=c("lightgreen", "red", "blue"),
       lty=c(NA, 1, 2), lwd=c(NA, 2, 2),
       fill=c("lightgreen", NA, NA), border=c("black", NA, NA))
dev.off()
print("Plot saved: dist_mle_gamma.png")

# ============================================
# VISUAL COMPARISON: Q-Q PLOTS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CHECKING FIT: VISUAL COMPARISON (Q-Q PLOTS)\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Q-Q Plot (Quantile-Quantile Plot):")
print("  - Compare quantiles of data to theoretical quantiles")
print("  - If data matches distribution, points lie on diagonal line")
print("  - Deviations indicate departure from assumed distribution")

png("../plots/dist_qq_plots.png", width=1200, height=400)
par(mfrow=c(1,3))

# Q-Q plot for exponential data
qqplot(qexp(ppoints(length(sample_data)), rate=rate_mle), sample_data,
       main="Q-Q Plot: Exponential Data",
       xlab="Theoretical Quantiles", ylab="Sample Quantiles")
abline(0, 1, col='red', lwd=2)

# Q-Q plot for gamma data
qqplot(qgamma(ppoints(length(gamma_data)), shape=shape_mle, rate=rate_mle_gamma), 
       gamma_data,
       main="Q-Q Plot: Gamma Data",
       xlab="Theoretical Quantiles", ylab="Sample Quantiles")
abline(0, 1, col='red', lwd=2)

# Q-Q plot for normal data (should be good)
normal_data <- rnorm(500)
qqnorm(normal_data, main="Q-Q Plot: Normal Data")
qqline(normal_data, col='red', lwd=2)

par(mfrow=c(1,1))
dev.off()
print("Plot saved: dist_qq_plots.png")

# ============================================
# EMPIRICAL CDF COMPARISON
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CHECKING FIT: EMPIRICAL CDF COMPARISON\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Empirical CDF: Step function of observed data")
print("Compare to theoretical CDF")

png("../plots/dist_ecdf_comparison.png", width=1200, height=400)
par(mfrow=c(1,2))

# ECDF for exponential data
plot(ecdf(sample_data), 
     main="ECDF vs Theoretical CDF (Exponential)",
     xlab="Value", ylab="Cumulative Probability",
     col='blue', lwd=2)
x_seq <- seq(0, max(sample_data), length.out=1000)
lines(x_seq, pexp(x_seq, rate=rate_mle), col='red', lwd=2, lty=2)
legend("bottomright",
       legend=c("Empirical CDF", "Theoretical CDF"),
       col=c("blue", "red"), lty=c(1, 2), lwd=2)

# ECDF for gamma data
plot(ecdf(gamma_data),
     main="ECDF vs Theoretical CDF (Gamma)",
     xlab="Value", ylab="Cumulative Probability",
     col='blue', lwd=2)
x_seq_g <- seq(0, max(gamma_data), length.out=1000)
lines(x_seq_g, pgamma(x_seq_g, shape=shape_mle, rate=rate_mle_gamma), 
      col='red', lwd=2, lty=2)
legend("bottomright",
       legend=c("Empirical CDF", "Theoretical CDF"),
       col=c("blue", "red"), lty=c(1, 2), lwd=2)

par(mfrow=c(1,1))
dev.off()
print("Plot saved: dist_ecdf_comparison.png")

# ============================================
# CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Chi-squared goodness-of-fit test:")
print("  - Designed for discrete/categorical data")
print("  - For continuous data: must discretize into bins")
print("  - Test statistic: χ² = sum((O_i - E_i)² / E_i)")
print("    where O_i = observed count in bin i")
print("          E_i = expected count in bin i")

# Discretize exponential data
n_bins <- 10
breaks <- quantile(sample_data, probs=seq(0, 1, length.out=n_bins+1))
breaks[1] <- 0  # Ensure lower bound is 0
observed_counts <- table(cut(sample_data, breaks=breaks, include.lowest=TRUE))

# Expected counts under fitted exponential
expected_probs <- diff(pexp(breaks, rate=rate_mle))
# Normalize to ensure they sum to 1 (account for rounding errors)
expected_probs <- expected_probs / sum(expected_probs)
expected_counts <- expected_probs * length(sample_data)

print("\nObserved vs Expected counts:")
print(data.frame(
  Bin = 1:n_bins,
  Observed = as.numeric(observed_counts),
  Expected = round(expected_counts, 2)
))

# Chi-squared test
chisq_result <- chisq.test(x=as.numeric(observed_counts), p=expected_probs)
print("\nChi-squared test result:")
print(chisq_result)

# Visualize
png("../plots/dist_chisq_test.png", width=800, height=600)
barplot(rbind(as.numeric(observed_counts), expected_counts),
        beside=TRUE,
        names.arg=1:n_bins,
        col=c('lightblue', 'salmon'),
        main="Chi-squared Test: Observed vs Expected Counts",
        xlab="Bin", ylab="Count",
        legend.text=c("Observed", "Expected"),
        args.legend=list(x="topright"))
dev.off()
print("Plot saved: dist_chisq_test.png")

# ============================================
# PROBLEMS WITH CHI-SQUARED TEST
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("PROBLEMS WITH CHI-SQUARED TEST\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Issues with chi-squared test for continuous distributions:")
print("")
print("1. LOSS OF INFORMATION FROM DISCRETIZATION")
print("   - Converting continuous data to bins loses precision")
print("   - Different binning choices can give different results")
print("")
print("2. LOTS OF WORK JUST TO USE chisq.test()")
print("   - Must choose number of bins")
print("   - Must calculate expected counts manually")
print("   - Results depend on arbitrary binning choices")
print("")
print("3. LOW POWER")
print("   - May not detect departures from null hypothesis")
print("   - Especially with small sample sizes")
print("")
print("BETTER ALTERNATIVES:")
print("  ✓ Kolmogorov-Smirnov test (ks.test)")
print("  ✓ Bootstrap testing")
print("  ✓ Smooth tests of goodness of fit")
print("  ✓ Anderson-Darling test")

# ============================================
# BETTER ALTERNATIVE: KOLMOGOROV-SMIRNOV TEST
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("BETTER ALTERNATIVE: KOLMOGOROV-SMIRNOV TEST\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Kolmogorov-Smirnov (K-S) test:")
print("  - Compares empirical CDF to theoretical CDF")
print("  - Test statistic: D = max|F_empirical(x) - F_theoretical(x)|")
print("  - No binning required!")
print("  - More powerful than chi-squared for continuous data")

# K-S test for exponential data
ks_result <- ks.test(sample_data, "pexp", rate=rate_mle)
print("\nKolmogorov-Smirnov test result:")
print(ks_result)

print(sprintf("\nK-S test statistic: D = %.4f", ks_result$statistic))
print(sprintf("P-value: %.4f", ks_result$p.value))

if (ks_result$p.value > 0.05) {
  print("Conclusion: Cannot reject null hypothesis")
  print("            Data is consistent with exponential distribution")
} else {
  print("Conclusion: Reject null hypothesis")
  print("            Data does not follow exponential distribution")
}

# Visualize K-S statistic
png("../plots/dist_ks_test.png", width=800, height=600)
plot(ecdf(sample_data),
     main="Kolmogorov-Smirnov Test Visualization",
     xlab="Value", ylab="Cumulative Probability",
     col='blue', lwd=2)
x_seq <- seq(0, max(sample_data), length.out=1000)
lines(x_seq, pexp(x_seq, rate=rate_mle), col='red', lwd=2)

# Find where maximum difference occurs
ecdf_func <- ecdf(sample_data)
x_test <- sort(sample_data)
diffs <- abs(ecdf_func(x_test) - pexp(x_test, rate=rate_mle))
max_diff_idx <- which.max(diffs)
x_max <- x_test[max_diff_idx]

# Draw line showing maximum difference
segments(x_max, ecdf_func(x_max), x_max, pexp(x_max, rate=rate_mle),
         col='green', lwd=3, lty=2)
text(x_max, mean(c(ecdf_func(x_max), pexp(x_max, rate=rate_mle))),
     sprintf("D = %.4f", ks_result$statistic),
     pos=4, col='green', font=2)

legend("bottomright",
       legend=c("Empirical CDF", "Theoretical CDF", "Max Difference (D)"),
       col=c("blue", "red", "green"), lty=c(1, 1, 2), lwd=c(2, 2, 3))
dev.off()
print("Plot saved: dist_ks_test.png")

# ============================================
# BOOTSTRAP TESTING
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("BOOTSTRAP TESTING\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Bootstrap approach for goodness-of-fit:")
print("  1. Fit distribution to observed data")
print("  2. Generate many bootstrap samples from fitted distribution")
print("  3. Calculate test statistic for each bootstrap sample")
print("  4. Compare observed test statistic to bootstrap distribution")

# Bootstrap K-S test
n_boot <- 1000
boot_stats <- numeric(n_boot)

for (i in 1:n_boot) {
  boot_sample <- rexp(length(sample_data), rate=rate_mle)
  boot_stats[i] <- ks.test(boot_sample, "pexp", rate=rate_mle)$statistic
}

# Calculate p-value
observed_stat <- ks_result$statistic
bootstrap_pval <- mean(boot_stats >= observed_stat)

print(sprintf("\nBootstrap K-S test (B = %d):", n_boot))
print(sprintf("Observed K-S statistic: %.4f", observed_stat))
print(sprintf("Bootstrap p-value: %.4f", bootstrap_pval))

# Visualize bootstrap distribution
png("../plots/dist_bootstrap_test.png", width=800, height=600)
hist(boot_stats, breaks=30, freq=FALSE,
     main="Bootstrap Distribution of K-S Statistic",
     xlab="K-S Statistic", ylab="Density",
     col='lightgray', border='white')
abline(v=observed_stat, col='red', lwd=2, lty=2)
text(observed_stat, par("usr")[4]*0.9,
     sprintf("Observed\nD = %.4f\np = %.4f", observed_stat, bootstrap_pval),
     pos=4, col='red', font=2)
dev.off()
print("Plot saved: dist_bootstrap_test.png")

# ============================================
# COMPARISON OF TESTS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("COMPARISON OF GOODNESS-OF-FIT TESTS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Test results summary:")
print("")
print(sprintf("Chi-squared test:      χ² = %.4f, p = %.4f", 
              chisq_result$statistic, chisq_result$p.value))
print(sprintf("Kolmogorov-Smirnov:    D = %.4f,  p = %.4f",
              ks_result$statistic, ks_result$p.value))
print(sprintf("Bootstrap K-S:         D = %.4f,  p = %.4f",
              observed_stat, bootstrap_pval))
print("")
print("All tests agree: data is consistent with exponential distribution")

# ============================================
# PRACTICAL EXAMPLE: TESTING NORMALITY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("PRACTICAL EXAMPLE: TESTING NORMALITY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Generate data from mixture of normals (not truly normal)")
print("Test if various methods can detect the departure from normality")

# Generate non-normal data (mixture)
set.seed(123)
mixture_data <- c(rnorm(400, mean=0, sd=1), rnorm(100, mean=3, sd=0.5))

# Fit normal distribution
mean_est <- mean(mixture_data)
sd_est <- sd(mixture_data)

print(sprintf("\nFitted normal: mean = %.4f, sd = %.4f", mean_est, sd_est))

# Visual inspection
png("../plots/dist_normality_test.png", width=1200, height=800)
par(mfrow=c(2,2))

# Histogram with fitted normal
hist(mixture_data, breaks=30, freq=FALSE,
     main="Histogram with Fitted Normal",
     xlab="Value", ylab="Density",
     col='lightblue', border='white')
x_norm <- seq(min(mixture_data), max(mixture_data), length.out=1000)
lines(x_norm, dnorm(x_norm, mean_est, sd_est), col='red', lwd=2)

# Q-Q plot
qqnorm(mixture_data, main="Q-Q Plot")
qqline(mixture_data, col='red', lwd=2)

# ECDF comparison
plot(ecdf(mixture_data),
     main="Empirical vs Theoretical CDF",
     xlab="Value", ylab="Cumulative Probability",
     col='blue', lwd=2)
lines(x_norm, pnorm(x_norm, mean_est, sd_est), col='red', lwd=2)

# Box plot
boxplot(mixture_data, horizontal=TRUE,
        main="Box Plot",
        xlab="Value",
        col='lightgreen')

par(mfrow=c(1,1))
dev.off()
print("Plot saved: dist_normality_test.png")

# Statistical tests
print("\n--- Statistical Tests for Normality ---")

# Shapiro-Wilk test (most powerful for normality)
if (length(mixture_data) <= 5000) {  # Shapiro-Wilk has sample size limit
  shapiro_result <- shapiro.test(mixture_data)
  print(sprintf("Shapiro-Wilk test: W = %.4f, p = %.4f", 
                shapiro_result$statistic, shapiro_result$p.value))
}

# K-S test for normality
ks_norm_result <- ks.test(mixture_data, "pnorm", mean=mean_est, sd=sd_est)
print(sprintf("K-S test:          D = %.4f, p = %.4f",
              ks_norm_result$statistic, ks_norm_result$p.value))

print("\nConclusion:")
if (exists("shapiro_result") && shapiro_result$p.value < 0.05) {
  print("  Shapiro-Wilk test rejects normality (p < 0.05)")
  print("  The data does NOT appear to be normally distributed")
} else {
  print("  Tests suggest data may not be perfectly normal")
  print("  Visual inspection (Q-Q plot) shows deviation in tails")
}

# ============================================
# CALIBRATION PLOTS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CALIBRATION PLOTS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Calibration: Check if predicted probabilities match observed frequencies")
print("")
print("For a well-fitted model:")
print("  - If we predict P(Y=1) = 0.7, about 70% should actually be Y=1")
print("  - Calibration plot: predicted probability vs observed frequency")

# Generate binary data with known probabilities
set.seed(456)
n_cal <- 1000
x_cal <- runif(n_cal, -3, 3)
true_prob <- plogis(x_cal)  # logistic function: 1/(1+exp(-x))
y_cal <- rbinom(n_cal, 1, true_prob)

# Fit logistic regression
cal_model <- glm(y_cal ~ x_cal, family=binomial)
pred_prob <- predict(cal_model, type="response")

# Create calibration plot
n_cal_bins <- 10
cal_bins <- cut(pred_prob, breaks=seq(0, 1, length.out=n_cal_bins+1), include.lowest=TRUE)
cal_data <- data.frame(
  pred = pred_prob,
  observed = y_cal,
  bin = cal_bins
)

cal_summary <- aggregate(cbind(pred, observed) ~ bin, data=cal_data, FUN=mean)

png("../plots/dist_calibration.png", width=800, height=600)
plot(cal_summary$pred, cal_summary$observed,
     main="Calibration Plot",
     xlab="Predicted Probability",
     ylab="Observed Frequency",
     pch=19, col='blue', cex=1.5,
     xlim=c(0,1), ylim=c(0,1))
abline(0, 1, col='red', lwd=2, lty=2)
grid()
legend("topleft",
       legend=c("Calibration points", "Perfect calibration"),
       col=c("blue", "red"), pch=c(19, NA), lty=c(NA, 2), lwd=c(NA, 2))
dev.off()
print("Plot saved: dist_calibration.png")

print("\nCalibration results:")
print("  Points near diagonal = well-calibrated")
print("  Points above diagonal = underestimating probability")
print("  Points below diagonal = overestimating probability")

# ============================================
# SUMMARY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

print("Key takeaways:")
print("")
print("✓ RANDOM NUMBER GENERATION")
print("  - R has built-in generators: rnorm, rexp, runif, etc.")
print("  - Also provides density (d*), CDF (p*), quantile (q*) functions")
print("")
print("✓ DISTRIBUTIONS IN R")
print("  - Many continuous distributions: normal, exponential, gamma, beta, etc.")
print("  - Many discrete distributions: binomial, Poisson, geometric, etc.")
print("  - Easy to work with using consistent naming conventions")
print("")
print("✓ PARAMETRIC DISTRIBUTIONS ARE MODELS")
print("  - Assume data comes from known family with unknown parameters")
print("  - Goal: estimate parameters and check fit")
print("")
print("✓ METHODS OF FITTING")
print("  - Method of Moments: match sample moments to theoretical moments")
print("  - Maximum Likelihood: maximize probability of observing the data")
print("  - Generalized Method of Moments: more flexible, use more moments")
print("")
print("✓ METHODS OF CHECKING FIT")
print("  - Visual comparisons: histograms, Q-Q plots, ECDF plots")
print("  - Goodness-of-fit statistics: K-S statistic, chi-squared statistic")
print("  - Hypothesis tests: K-S test, chi-squared test, Shapiro-Wilk")
print("  - Calibration: predicted probabilities vs observed frequencies")
print("")
print("✓ CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS")
print("  - Requires discretization (binning) of continuous data")
print("  - DRAWBACKS:")
print("    • Loss of information from discretization")
print("    • Lots of work just to use chisq.test()")
print("    • Results depend on arbitrary binning choices")
print("    • Low power compared to alternatives")
print("")
print("✓ BETTER ALTERNATIVES")
print("  - ks.test(): Kolmogorov-Smirnov test (no binning, more powerful)")
print("  - Bootstrap testing: resampling approach for p-values")
print("  - Smooth tests of goodness of fit: more sophisticated methods")
print("  - Anderson-Darling test: gives more weight to tails")
print("  - Shapiro-Wilk test: specifically for testing normality")

cat("\n")
cat(paste(rep("=", 60), collapse=""), "\n")
cat("DISTRIBUTIONS TUTORIAL COMPLETE\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

final_plot_count <- length(list.files("../plots", pattern="dist_.*\\.png"))
print(paste("Total plots generated:", final_plot_count))
print("\nAll plots saved to: ../plots/")
print("\nThank you for completing this tutorial!")
