# ==============================================================================
# MONTE CARLO SIMULATIONS
# Statistical Computing Tutorial
#
# Topics covered:
# 1. Ordinary Monte Carlo (OMC) theory
# 2. Monte Carlo integration examples
# 3. Approximating distributions
# 4. Toy collector exercise (Coupon Collector Problem)
# ==============================================================================

cat(strrep("=", 70), "\n")
cat("MONTE CARLO SIMULATIONS TUTORIAL\n")
cat(strrep("=", 70), "\n\n")

# Set seed for reproducibility
set.seed(42)

# Create plots directory if it doesn't exist
if (!dir.exists("../plots")) {
  dir.create("../plots", recursive = TRUE)
}

# ==============================================================================
# PART 1: ORDINARY MONTE CARLO (OMC) - THEORY
# ==============================================================================

cat("\n", strrep("=", 70), "\n")
cat("PART 1: ORDINARY MONTE CARLO - THEORY\n")
cat(strrep("=", 70), "\n\n")

cat("The 'Monte Carlo method' refers to the theory and practice of learning\n")
cat("about probability distributions by simulation rather than calculus.\n\n")

cat("In Ordinary Monte Carlo (OMC) we use IID simulations from the\n")
cat("distribution of interest.\n\n")

cat("Setup:\n")
cat("------\n")
cat("Suppose X₁, X₂, ... are IID simulations from some distribution,\n")
cat("and we want to know an expectation:\n\n")
cat("  θ = E[Y₁] = E[g(X₁)]\n\n")

cat("Law of Large Numbers (LLN):\n")
cat("---------------------------\n")
cat("  ȳₙ = (1/n) Σᵢ Yᵢ = (1/n) Σᵢ g(Xᵢ)\n\n")
cat("converges in probability to θ.\n\n")

cat("Central Limit Theorem (CLT):\n")
cat("----------------------------\n")
cat("  √n(ȳₙ - θ)/σ →ᵈ N(0,1)\n\n")
cat("That is, for sufficiently large n:\n")
cat("  ȳₙ ~ N(θ, σ²/n)\n\n")

cat("Standard Error Estimation:\n")
cat("-------------------------\n")
cat("We can estimate the standard error σ/√n with sₙ/√n\n")
cat("where sₙ is the sample standard deviation.\n\n")

cat("KEY INSIGHT:\n")
cat("-----------\n")
cat("The theory of OMC is just the theory of frequentist statistical inference.\n")
cat("The only differences are that:\n\n")
cat("1. The 'data' X₁,...,Xₙ are computer simulations rather than\n")
cat("   measurements on objects in the real world\n\n")
cat("2. The 'sample size' n is the number of computer simulations\n")
cat("   rather than the size of some real world data\n\n")
cat("3. The unknown parameter θ is in principle completely known,\n")
cat("   given by some integral, which we are unable to do.\n\n")

cat("VECTOR CASE:\n")
cat("-----------\n")
cat("Everything works just the same when the data X₁, X₂, ...\n")
cat("(which are computer simulations) are vectors.\n")
cat("But the functions of interest g(X₁), g(X₂), ... are scalars.\n\n")

cat("LIMITATION:\n")
cat("----------\n")
cat("OMC works great, but it can be very difficult to simulate IID\n")
cat("simulations of random variables or random vectors whose\n")
cat("distribution is not brand name distributions.\n\n")

# ==============================================================================
# PART 2: APPROXIMATING THE BINOMIAL DISTRIBUTION
# ==============================================================================

cat("\n", strrep("=", 70), "\n")
cat("PART 2: APPROXIMATING THE BINOMIAL DISTRIBUTION\n")
cat(strrep("=", 70), "\n\n")

cat("Problem: Flip a coin 10 times. What is P(more than 3 heads)?\n")
cat(strrep("-", 60), "\n\n")

cat("This is trivial for the Binomial distribution, but we'll use\n")
cat("Monte Carlo simulation to demonstrate the method.\n\n")

# Monte Carlo simulation
runs <- 10000

one.trial <- function() {
  sum(sample(c(0, 1), 10, replace = TRUE)) > 3
}

cat(sprintf("Running %d Monte Carlo simulations...\n\n", runs))

set.seed(123)
mc.results <- replicate(runs, one.trial())
mc.binom <- sum(mc.results) / runs

# Exact probability
exact.prob <- pbinom(3, 10, 0.5, lower.tail = FALSE)

# Calculate Monte Carlo standard error
# For a proportion p, variance is p(1-p)
# Standard error is sqrt(p(1-p)/n)
mc.se <- sqrt(mc.binom * (1 - mc.binom) / runs)

cat("RESULTS:\n")
cat(sprintf("Monte Carlo estimate: %.6f\n", mc.binom))
cat(sprintf("Exact probability:    %.6f\n", exact.prob))
cat(sprintf("Absolute error:       %.6f\n", abs(mc.binom - exact.prob)))
cat(sprintf("\nMonte Carlo standard error: %.6f\n", mc.se))
cat(sprintf("95%% Confidence Interval: [%.6f, %.6f]\n", 
            mc.binom - 1.96*mc.se, mc.binom + 1.96*mc.se))

# Check if exact value is in CI
in_ci <- (exact.prob >= mc.binom - 1.96*mc.se) && 
         (exact.prob <= mc.binom + 1.96*mc.se)
cat(sprintf("Exact value in CI: %s\n", ifelse(in_ci, "YES ✓", "NO ✗")))

# ==============================================================================
# EXERCISE SOLUTION: MONTE CARLO STANDARD ERROR
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("EXERCISE: ESTIMATING MONTE CARLO STANDARD ERROR\n")
cat(strrep("=", 70), "\n\n")

cat("For a binary outcome (success/failure), the standard error is:\n")
cat("  SE = √[p(1-p)/n]\n\n")

cat("where:\n")
cat("  p = estimated probability (proportion of successes)\n")
cat("  n = number of Monte Carlo simulations\n\n")

cat("In our case:\n")
cat(sprintf("  p = %.6f\n", mc.binom))
cat(sprintf("  n = %d\n", runs))
cat(sprintf("  SE = √[%.6f × %.6f / %d] = %.6f\n", 
            mc.binom, 1-mc.binom, runs, mc.se))

# Demonstrate convergence with different sample sizes
cat("\n\nDemonstrating convergence with different sample sizes:\n")
cat(strrep("-", 60), "\n\n")

sample_sizes <- c(100, 1000, 10000, 100000)
results_table <- data.frame(
  n = integer(),
  estimate = numeric(),
  se = numeric(),
  ci_lower = numeric(),
  ci_upper = numeric(),
  in_ci = logical()
)

set.seed(456)
for (n in sample_sizes) {
  mc_trials <- replicate(n, one.trial())
  p_hat <- mean(mc_trials)
  se <- sqrt(p_hat * (1 - p_hat) / n)
  ci_lower <- p_hat - 1.96 * se
  ci_upper <- p_hat + 1.96 * se
  in_ci <- (exact.prob >= ci_lower) && (exact.prob <= ci_upper)
  
  results_table <- rbind(results_table, 
                         data.frame(n = n, estimate = p_hat, se = se,
                                   ci_lower = ci_lower, ci_upper = ci_upper,
                                   in_ci = in_ci))
}

print(results_table)

cat(sprintf("\nExact probability: %.6f\n", exact.prob))
cat("\nNote: Standard error decreases as O(1/√n)\n")

# ==============================================================================
# PART 3: APPROXIMATING π
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("PART 3: APPROXIMATING π USING MONTE CARLO\n")
cat(strrep("=", 70), "\n\n")

cat("Geometric Approach to Estimating π\n")
cat(strrep("-", 60), "\n\n")

cat("Key insight:\n")
cat("  Area of a circle = πr²\n")
cat("  Area of square containing the circle = (2r)² = 4r²\n\n")

cat("Therefore, the ratio of areas is:\n")
cat("  πr² / 4r² = π/4\n\n")

cat("If we can empirically determine the ratio of the area of the\n")
cat("circle to the area of the square, we can multiply by 4 to get π.\n\n")

cat("Method:\n")
cat("-------\n")
cat("1. Randomly sample (x, y) points on the unit square centered at 0\n")
cat("   (i.e., x, y ∈ [-0.5, 0.5])\n")
cat("2. Check if x² + y² ≤ 0.5² (point is inside the circle)\n")
cat("3. Ratio of points in circle × 4 = estimate of π\n\n")

# Monte Carlo estimation of π
runs <- 100000
set.seed(2024)

cat(sprintf("Running %d Monte Carlo simulations...\n\n", runs))

xs <- runif(runs, min = -0.5, max = 0.5)
ys <- runif(runs, min = -0.5, max = 0.5)
in.circle <- xs^2 + ys^2 <= 0.5^2
mc.pi <- (sum(in.circle) / runs) * 4

# Calculate standard error
# This is a proportion problem: p = proportion in circle
p <- sum(in.circle) / runs
se.p <- sqrt(p * (1 - p) / runs)
se.pi <- 4 * se.p  # SE for π estimate

cat("RESULTS:\n")
cat(sprintf("Monte Carlo estimate of π: %.6f\n", mc.pi))
cat(sprintf("True value of π:           %.6f\n", pi))
cat(sprintf("Absolute error:            %.6f\n", abs(mc.pi - pi)))
cat(sprintf("Relative error:            %.4f%%\n", 100 * abs(mc.pi - pi) / pi))
cat(sprintf("\nProportion in circle:      %.6f\n", p))
cat(sprintf("Standard error of π:       %.6f\n", se.pi))
cat(sprintf("95%% CI for π:              [%.6f, %.6f]\n", 
            mc.pi - 1.96*se.pi, mc.pi + 1.96*se.pi))

# Check if true π is in CI
in_ci_pi <- (pi >= mc.pi - 1.96*se.pi) && (pi <= mc.pi + 1.96*se.pi)
cat(sprintf("True π in CI: %s\n", ifelse(in_ci_pi, "YES ✓", "NO ✗")))

# Convergence analysis
cat("\n\nConvergence analysis with different sample sizes:\n")
cat(strrep("-", 60), "\n\n")

sample_sizes_pi <- c(100, 1000, 10000, 100000, 1000000)
pi_results <- data.frame(
  n = integer(),
  estimate = numeric(),
  error = numeric(),
  rel_error_pct = numeric(),
  se = numeric()
)

set.seed(12345)
for (n in sample_sizes_pi) {
  xs_temp <- runif(n, min = -0.5, max = 0.5)
  ys_temp <- runif(n, min = -0.5, max = 0.5)
  in_circle_temp <- xs_temp^2 + ys_temp^2 <= 0.5^2
  pi_est <- (sum(in_circle_temp) / n) * 4
  
  p_temp <- sum(in_circle_temp) / n
  se_temp <- 4 * sqrt(p_temp * (1 - p_temp) / n)
  
  pi_results <- rbind(pi_results, 
                      data.frame(n = n, 
                                estimate = pi_est,
                                error = abs(pi_est - pi),
                                rel_error_pct = 100 * abs(pi_est - pi) / pi,
                                se = se_temp))
}

print(pi_results)

cat(sprintf("\nTrue π = %.10f\n", pi))

# ==============================================================================
# VISUALIZATION: APPROXIMATING π
# ==============================================================================

cat("\n\nCreating π approximation visualization...\n")

png("../plots/monte_carlo_pi_approximation.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 3))

# Plot 1: Visualization of the method (small sample for clarity)
set.seed(999)
n_vis <- 2000
xs_vis <- runif(n_vis, min = -0.5, max = 0.5)
ys_vis <- runif(n_vis, min = -0.5, max = 0.5)
in_circle_vis <- xs_vis^2 + ys_vis^2 <= 0.5^2

plot(xs_vis[in_circle_vis], ys_vis[in_circle_vis], 
     col = "blue", pch = ".", cex = 2,
     xlim = c(-0.5, 0.5), ylim = c(-0.5, 0.5), asp = 1,
     xlab = "x", ylab = "y", 
     main = sprintf("Monte Carlo π Estimation (n=%d)", n_vis))
points(xs_vis[!in_circle_vis], ys_vis[!in_circle_vis], 
       col = "red", pch = ".", cex = 2)

# Draw the circle
theta <- seq(0, 2*pi, length.out = 200)
lines(0.5 * cos(theta), 0.5 * sin(theta), lwd = 2, col = "black")

# Draw the square
rect(-0.5, -0.5, 0.5, 0.5, border = "black", lwd = 2)

pi_est_vis <- (sum(in_circle_vis) / n_vis) * 4
legend("topright", 
       legend = c(sprintf("Inside (%.0f%%)", 100*sum(in_circle_vis)/n_vis),
                  sprintf("Outside (%.0f%%)", 100*sum(!in_circle_vis)/n_vis),
                  sprintf("π ≈ %.4f", pi_est_vis)),
       col = c("blue", "red", NA), pch = c(16, 16, NA), cex = 0.8)

# Plot 2: Convergence of π estimate
set.seed(111)
n_conv <- 50000
xs_conv <- runif(n_conv, min = -0.5, max = 0.5)
ys_conv <- runif(n_conv, min = -0.5, max = 0.5)
in_circle_conv <- xs_conv^2 + ys_conv^2 <= 0.5^2

cumulative_pi <- (cumsum(in_circle_conv) / (1:n_conv)) * 4

plot(1:n_conv, cumulative_pi, type = "l", col = "blue", lwd = 2,
     xlab = "Number of simulations", ylab = "Estimated π",
     main = "Convergence of π Estimate",
     ylim = c(2.8, 3.5))
abline(h = pi, col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("MC estimate", "True π"),
       col = c("blue", "red"), lwd = 2, lty = c(1, 2))

# Plot 3: Error vs sample size (log-log plot)
plot(pi_results$n, pi_results$error, 
     type = "b", col = "darkred", lwd = 2, pch = 16,
     log = "xy",
     xlab = "Sample size (n)", ylab = "Absolute error",
     main = "Error vs Sample Size")
# Add theoretical O(1/√n) reference line
abline(lm(log10(pi_results$error) ~ log10(pi_results$n)), 
       col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Actual error", "O(1/√n)"),
       col = c("darkred", "blue"), lwd = 2, lty = c(1, 2), pch = c(16, NA))

# Plot 4: Distribution of π estimates
set.seed(222)
n_reps <- 1000
n_per_rep <- 10000
pi_estimates <- replicate(n_reps, {
  xs_temp <- runif(n_per_rep, min = -0.5, max = 0.5)
  ys_temp <- runif(n_per_rep, min = -0.5, max = 0.5)
  (sum(xs_temp^2 + ys_temp^2 <= 0.5^2) / n_per_rep) * 4
})

hist(pi_estimates, breaks = 30, col = "lightgreen", border = "black",
     main = sprintf("Distribution of π Estimates\n(1000 reps, n=%d each)", n_per_rep),
     xlab = "Estimated π", freq = FALSE)
abline(v = pi, col = "red", lwd = 3, lty = 2)
abline(v = mean(pi_estimates), col = "blue", lwd = 2, lty = 2)

# Add theoretical normal distribution
p_theory <- pi / 4
se_theory <- 4 * sqrt(p_theory * (1 - p_theory) / n_per_rep)
x_seq <- seq(min(pi_estimates), max(pi_estimates), length.out = 100)
lines(x_seq, dnorm(x_seq, pi, se_theory), col = "darkgreen", lwd = 2)
legend("topright", 
       legend = c("True π", "Mean estimate", "Theoretical N"),
       col = c("red", "blue", "darkgreen"), 
       lwd = 2, lty = c(2, 2, 1), cex = 0.8)

# Plot 5: Relative error over sample sizes
plot(pi_results$n, pi_results$rel_error_pct,
     type = "b", col = "purple", lwd = 2, pch = 16,
     log = "x",
     xlab = "Sample size (n)", ylab = "Relative error (%)",
     main = "Relative Error vs Sample Size")
grid()

# Plot 6: Standard error vs sample size
plot(pi_results$n, pi_results$se,
     type = "b", col = "orange", lwd = 2, pch = 16,
     log = "xy",
     xlab = "Sample size (n)", ylab = "Standard error",
     main = "Standard Error vs Sample Size")
abline(lm(log10(pi_results$se) ~ log10(pi_results$n)), 
       col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Actual SE", "O(1/√n)"),
       col = c("orange", "blue"), lwd = 2, lty = c(1, 2), pch = c(16, NA))

par(mfrow = c(1, 1))
dev.off()
cat("Saved: ../plots/monte_carlo_pi_approximation.png\n")

# Alternative visualization: detailed π plot
cat("\nCreating detailed π visualization...\n")

png("../plots/monte_carlo_pi_detailed.png", width = 800, height = 800, res = 150)
plot(xs, ys, pch = '.', col = ifelse(in.circle, "blue", "grey"),
     xlab = '', ylab = '', asp = 1,
     main = paste("MC Approximation of π =", round(mc.pi, 4)))
# Draw the circle
theta_circle <- seq(0, 2*pi, length.out = 200)
lines(0.5 * cos(theta_circle), 0.5 * sin(theta_circle), lwd = 2, col = "black")
# Draw the square
rect(-0.5, -0.5, 0.5, 0.5, border = "black", lwd = 2)
dev.off()
cat("Saved: ../plots/monte_carlo_pi_detailed.png\n")

# ==============================================================================
# PART 4: MONTE CARLO INTEGRATION WITH SEQUENTIAL STOPPING
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("PART 4: MONTE CARLO INTEGRATION\n")
cat(strrep("=", 70), "\n\n")

cat("Example: Intractable Expectation\n")
cat(strrep("-", 60), "\n\n")

cat("Let X ~ Gamma(3/2, 1), i.e.\n")
cat("  f(x) = (2/√π) √x e^(-x) I(x > 0)\n\n")

cat("Suppose we want to find:\n")
cat("  θ = E[1/((X+1)log(X+3))]\n")
cat("    = ∫₀^∞ 1/((x+1)log(x+3)) * (2/√π) √x e^(-x) dx\n\n")

cat("The expectation (or integral) θ is intractable - we don't know\n")
cat("how to compute it analytically.\n\n")

cat("GOAL: Estimate θ such that the 95% CI length is less than 0.002\n\n")

# Initial Monte Carlo estimation
n <- 1000
set.seed(4040)

cat(sprintf("Initial estimation with n = %d:\n", n))
cat(strrep("-", 40), "\n")

x <- rgamma(n, 3/2, scale = 1)
cat(sprintf("Mean of X (theoretical = 3/2 = 1.5): %.6f\n", mean(x)))

y <- 1 / ((x + 1) * log(x + 3))
est <- mean(y)
cat(sprintf("\nInitial estimate of θ: %.7f\n", est))

mcse <- sd(y) / sqrt(length(y))
interval <- est + c(-1, 1) * 1.96 * mcse
cat(sprintf("Monte Carlo SE: %.7f\n", mcse))
cat(sprintf("95%% CI: [%.7f, %.7f]\n", interval[1], interval[2]))
cat(sprintf("CI length: %.7f\n", diff(interval)))

# ==============================================================================
# SEQUENTIAL STOPPING RULE
# ==============================================================================

cat("\n\nApplying Sequential Stopping Rule:\n")
cat(strrep("-", 60), "\n")
cat("Target CI length: 0.002\n")
cat("Adding samples in batches of 1000 until target is reached...\n\n")

eps <- 0.002
len <- diff(interval)
plotting.var <- c(est, interval)
iteration <- 1

while (len > eps) {
  new.x <- rgamma(n, 3/2, scale = 1)
  new.y <- 1 / ((new.x + 1) * log(new.x + 3))
  y <- c(y, new.y)
  est <- mean(y)
  mcse <- sd(y) / sqrt(length(y))
  interval <- est + c(-1, 1) * 1.96 * mcse
  len <- diff(interval)
  plotting.var <- rbind(plotting.var, c(est, interval))
  iteration <- iteration + 1
  
  # Print progress every 20 iterations
  if (iteration %% 20 == 0) {
    cat(sprintf("  Iteration %3d: n = %6d, CI length = %.6f\n", 
                iteration, length(y), len))
  }
}

cat("\nSequential stopping complete!\n")
cat(sprintf("Final sample size: %d\n", length(y)))
cat(sprintf("Final estimate: %.7f\n", est))
cat(sprintf("Final 95%% CI: [%.7f, %.7f]\n", interval[1], interval[2]))
cat(sprintf("Final CI length: %.7f (target: %.3f)\n", len, eps))
cat(sprintf("Final SE: %.7f\n", mcse))

# ==============================================================================
# VISUALIZATION: SEQUENTIAL STOPPING CONVERGENCE
# ==============================================================================

cat("\n\nCreating sequential stopping visualization...\n")

png("../plots/monte_carlo_integration_sequential.png", width = 1400, height = 600, res = 150)
par(mfrow = c(1, 2))

# Plot 1: Estimates with confidence bands
temp <- seq(1000, length(y), 1000)
plot(temp, plotting.var[, 1], type = "l", lwd = 2,
     ylim = c(min(plotting.var), max(plotting.var)),
     main = "Sequential Estimation with 95% CI",
     xlab = "Sample size (n)", ylab = "Estimate of θ")
lines(temp, plotting.var[, 2], col = "red", lwd = 2)
lines(temp, plotting.var[, 3], col = "red", lwd = 2)
abline(h = est, col = "blue", lwd = 1, lty = 2)
legend("topright", 
       legend = c("Estimate", "95% CI", "Final estimate"),
       col = c("black", "red", "blue"), 
       lwd = 2, lty = c(1, 1, 2))
grid()

# Plot 2: CI length convergence
ci_lengths <- plotting.var[, 3] - plotting.var[, 2]
plot(temp, ci_lengths, type = "l", col = "purple", lwd = 2,
     main = "Convergence of CI Length",
     xlab = "Sample size (n)", ylab = "CI length")
abline(h = eps, col = "red", lwd = 2, lty = 2)
text(max(temp) * 0.7, eps * 1.5, 
     sprintf("Target = %.3f", eps), col = "red")
grid()

par(mfrow = c(1, 1))
dev.off()
cat("Saved: ../plots/monte_carlo_integration_sequential.png\n")

# Additional analysis
cat("\n\nAdditional Analysis:\n")
cat(strrep("-", 60), "\n")

# Compare initial vs final
cat(sprintf("Sample size increase: %d → %d (%.1fx)\n", 
            1000, length(y), length(y) / 1000))
cat(sprintf("CI length reduction: %.6f → %.6f (%.1fx)\n",
            diff(plotting.var[1, 2:3]), len, 
            diff(plotting.var[1, 2:3]) / len))
cat(sprintf("SE reduction: √(1000/%d) = %.4f (theoretical)\n", 
            length(y), sqrt(1000 / length(y))))
cat(sprintf("Actual SE ratio: %.4f\n", 
            (plotting.var[1, 3] - plotting.var[1, 1]) / 1.96 / mcse))

# Distribution of Y values
cat("\n\nDistribution of transformed values Y = 1/((X+1)log(X+3)):\n")
cat(sprintf("  Mean: %.7f\n", mean(y)))
cat(sprintf("  SD:   %.7f\n", sd(y)))
cat(sprintf("  Min:  %.7f\n", min(y)))
cat(sprintf("  Max:  %.7f\n", max(y)))
cat(sprintf("  Median: %.7f\n", median(y)))

# Plot distribution of Y
png("../plots/monte_carlo_integration_distribution.png", width = 1200, height = 400, res = 150)
par(mfrow = c(1, 3))

# Plot 1: Distribution of X
hist(x[1:10000], breaks = 50, col = "lightblue", border = "black",
     main = "Distribution of X ~ Gamma(3/2, 1)",
     xlab = "X", freq = FALSE)
curve(dgamma(x, 3/2, scale = 1), add = TRUE, col = "red", lwd = 2)

# Plot 2: Distribution of Y
hist(y, breaks = 100, col = "lightgreen", border = "black",
     main = "Distribution of Y = 1/((X+1)log(X+3))",
     xlab = "Y", freq = FALSE, xlim = quantile(y, c(0, 0.99)))
abline(v = mean(y), col = "red", lwd = 2, lty = 2)

# Plot 3: Scatter plot X vs Y
plot(x[1:5000], y[1:5000], pch = 16, cex = 0.3, col = rgb(0, 0, 1, 0.3),
     main = "Relationship between X and Y",
     xlab = "X ~ Gamma(3/2, 1)", ylab = "Y = 1/((X+1)log(X+3))")
abline(h = mean(y), col = "red", lwd = 2, lty = 2)

par(mfrow = c(1, 1))
dev.off()
cat("\nSaved: ../plots/monte_carlo_integration_distribution.png\n")

# ==============================================================================
# VISUALIZATION: BINOMIAL APPROXIMATION
# ==============================================================================

cat("\n\nCreating binomial approximation visualization...\n")

png("../plots/monte_carlo_binomial.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 3))

# Plot 1: Distribution of successes
set.seed(789)
n_sims <- 10000
successes <- replicate(n_sims, sum(sample(c(0, 1), 10, replace = TRUE)))
hist(successes, breaks = seq(-0.5, 10.5, by = 1), col = "lightblue", 
     border = "black", main = "Distribution of Heads in 10 Flips",
     xlab = "Number of heads", ylab = "Frequency", freq = TRUE)
abline(v = 3, col = "red", lwd = 3, lty = 2)
text(3, max(table(successes)) * 0.9, "Threshold = 3", pos = 4, col = "red")

# Plot 2: Proportion > 3 heads
prop_gt3 <- mean(successes > 3)
barplot(c(prop_gt3, 1 - prop_gt3), 
        names.arg = c("> 3 heads", "≤ 3 heads"),
        col = c("lightcoral", "lightblue"),
        main = "Proportion of Outcomes",
        ylab = "Proportion",
        ylim = c(0, 1))
abline(h = exact.prob, col = "red", lwd = 2, lty = 2)
text(0.7, exact.prob + 0.05, sprintf("Exact: %.3f", exact.prob), col = "red")

# Plot 3: Convergence of estimate
set.seed(321)
n_max <- 10000
mc_sequence <- replicate(n_max, one.trial())
cumulative_prop <- cumsum(mc_sequence) / (1:n_max)

plot(1:n_max, cumulative_prop, type = "l", col = "blue", lwd = 2,
     xlab = "Number of simulations", ylab = "Estimated probability",
     main = "Convergence of Monte Carlo Estimate",
     ylim = c(0.75, 0.90))
abline(h = exact.prob, col = "red", lwd = 2, lty = 2)
legend("topright", legend = c("MC estimate", "Exact value"),
       col = c("blue", "red"), lwd = 2, lty = c(1, 2))

# Plot 4: Standard error vs sample size
n_seq <- seq(100, 10000, by = 100)
se_theoretical <- sqrt(exact.prob * (1 - exact.prob) / n_seq)
plot(n_seq, se_theoretical, type = "l", col = "darkred", lwd = 2,
     xlab = "Sample size (n)", ylab = "Standard error",
     main = "Standard Error vs Sample Size",
     log = "xy")
# Add O(1/√n) reference line
abline(lm(log(se_theoretical) ~ log(n_seq)), col = "blue", lwd = 2, lty = 2)
legend("topright", legend = c("Theoretical SE", "O(1/√n)"),
       col = c("darkred", "blue"), lwd = 2, lty = c(1, 2))

# Plot 5: Multiple MC runs distribution
set.seed(999)
n_experiments <- 1000
n_per_exp <- 1000
mc_estimates <- replicate(n_experiments, {
  mean(replicate(n_per_exp, one.trial()))
})

hist(mc_estimates, breaks = 30, col = "lightgreen", border = "black",
     main = sprintf("Distribution of MC Estimates\n(1000 experiments, n=%d each)", n_per_exp),
     xlab = "Estimated probability", ylab = "Frequency", freq = FALSE)
abline(v = exact.prob, col = "red", lwd = 3, lty = 2)
abline(v = mean(mc_estimates), col = "blue", lwd = 2, lty = 2)

# Add theoretical normal distribution
x_seq <- seq(min(mc_estimates), max(mc_estimates), length.out = 100)
theoretical_sd <- sqrt(exact.prob * (1 - exact.prob) / n_per_exp)
lines(x_seq, dnorm(x_seq, exact.prob, theoretical_sd), 
      col = "darkgreen", lwd = 2)
legend("topright", 
       legend = c("Exact", "MC mean", "Theoretical"),
       col = c("red", "blue", "darkgreen"), 
       lwd = 2, lty = c(2, 2, 1), cex = 0.8)

# Plot 6: Confidence interval coverage
set.seed(1111)
n_ci_experiments <- 100
n_per_ci <- 1000
ci_coverage <- data.frame(
  experiment = 1:n_ci_experiments,
  estimate = numeric(n_ci_experiments),
  lower = numeric(n_ci_experiments),
  upper = numeric(n_ci_experiments),
  covers = logical(n_ci_experiments)
)

for (i in 1:n_ci_experiments) {
  p_hat <- mean(replicate(n_per_ci, one.trial()))
  se <- sqrt(p_hat * (1 - p_hat) / n_per_ci)
  ci_coverage$estimate[i] <- p_hat
  ci_coverage$lower[i] <- p_hat - 1.96 * se
  ci_coverage$upper[i] <- p_hat + 1.96 * se
  ci_coverage$covers[i] <- (exact.prob >= ci_coverage$lower[i]) && 
                           (exact.prob <= ci_coverage$upper[i])
}

coverage_rate <- mean(ci_coverage$covers)

plot(1:n_ci_experiments, ci_coverage$estimate, pch = 16, cex = 0.5,
     ylim = c(min(ci_coverage$lower), max(ci_coverage$upper)),
     xlab = "Experiment number", ylab = "Estimate and 95% CI",
     main = sprintf("95%% CI Coverage (n=%d per experiment)\nCoverage rate: %.1f%%", 
                    n_per_ci, 100*coverage_rate),
     col = ifelse(ci_coverage$covers, "blue", "red"))
segments(1:n_ci_experiments, ci_coverage$lower, 
         1:n_ci_experiments, ci_coverage$upper,
         col = ifelse(ci_coverage$covers, "blue", "red"))
abline(h = exact.prob, col = "darkgreen", lwd = 2, lty = 2)
legend("topright", 
       legend = c("Covers", "Doesn't cover", "Exact"),
       col = c("blue", "red", "darkgreen"), 
       pch = c(16, 16, NA), lty = c(NA, NA, 2), lwd = c(NA, NA, 2),
       cex = 0.8)

par(mfrow = c(1, 1))
dev.off()
cat("Saved: ../plots/monte_carlo_binomial.png\n")

# ==============================================================================
# PART 3: GENERAL EXPECTATION ESTIMATION
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("PART 3: GENERAL EXPECTATION ESTIMATION\n")
cat(strrep("=", 70), "\n\n")

cat("Example: Estimate E[X²] where X ~ N(5, 2)\n")
cat(strrep("-", 60), "\n\n")

# Function to estimate
g <- function(x) x^2

# True value: E[X²] = Var(X) + E[X]² = 2² + 5² = 4 + 25 = 29
true_value <- 4 + 25

# Monte Carlo estimation
set.seed(2024)
n_mc <- 10000
x_samples <- rnorm(n_mc, mean = 5, sd = 2)
y_samples <- g(x_samples)

# Estimate and standard error
theta_hat <- mean(y_samples)
se_hat <- sd(y_samples) / sqrt(n_mc)

cat("RESULTS:\n")
cat(sprintf("True E[X²]:          %.6f\n", true_value))
cat(sprintf("MC estimate:         %.6f\n", theta_hat))
cat(sprintf("Absolute error:      %.6f\n", abs(theta_hat - true_value)))
cat(sprintf("Standard error (SE): %.6f\n", se_hat))
cat(sprintf("95%% CI: [%.6f, %.6f]\n", 
            theta_hat - 1.96*se_hat, theta_hat + 1.96*se_hat))

# Demonstrate with different functions
cat("\n\nEstimating expectations for various functions:\n")
cat(strrep("-", 60), "\n\n")

functions_list <- list(
  list(name = "E[X]", func = function(x) x, 
       true = 5),
  list(name = "E[X²]", func = function(x) x^2, 
       true = 29),
  list(name = "E[X³]", func = function(x) x^3, 
       true = 5^3 + 3*5*4),  # μ³ + 3μσ²
  list(name = "E[exp(X)]", func = function(x) exp(x), 
       true = exp(5 + 4/2)),  # For X~N(μ,σ²): E[exp(X)] = exp(μ + σ²/2)
  list(name = "E[sin(X)]", func = function(x) sin(x), 
       true = NA)  # No closed form
)

results_df <- data.frame(
  Expectation = character(),
  True_Value = character(),
  MC_Estimate = numeric(),
  SE = numeric(),
  Rel_Error = character(),
  stringsAsFactors = FALSE
)

set.seed(3030)
x_samples <- rnorm(10000, mean = 5, sd = 2)

for (func_info in functions_list) {
  y <- func_info$func(x_samples)
  estimate <- mean(y)
  se <- sd(y) / sqrt(length(y))
  
  if (is.na(func_info$true)) {
    true_val_str <- "N/A"
    rel_error_str <- "N/A"
  } else {
    true_val_str <- sprintf("%.6f", func_info$true)
    rel_error <- abs(estimate - func_info$true) / abs(func_info$true)
    rel_error_str <- sprintf("%.4f%%", 100 * rel_error)
  }
  
  results_df <- rbind(results_df, data.frame(
    Expectation = func_info$name,
    True_Value = true_val_str,
    MC_Estimate = estimate,
    SE = se,
    Rel_Error = rel_error_str,
    stringsAsFactors = FALSE
  ))
}

print(results_df, row.names = FALSE)

# ==============================================================================
# PART 5: BOOTSTRAP AND PERMUTATION METHODS
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("PART 5: BOOTSTRAP AND PERMUTATION METHODS\n")
cat(strrep("=", 70), "\n\n")

cat("HIGH-DIMENSIONAL EXAMPLES:\n")
cat("Monte Carlo methods are essential for complex, high-dimensional problems:\n")
cat("  - FiveThirtyEight's Election Forecast\n")
cat("  - FiveThirtyEight's NBA Predictions\n")
cat("  - Vanguard's Retirement Nest Egg Calculator\n")
cat("  - Fisher's Exact Test in R\n\n")

# ==============================================================================
# PERMUTATIONS WITH sample()
# ==============================================================================

cat("PERMUTATIONS WITH sample():\n")
cat(strrep("-", 60), "\n")
cat("sample() is powerful - it works on any object with a defined length()\n\n")

cat("Example 1: Simple permutations\n")
set.seed(5050)
cat("sample(5):\n")
print(sample(5))

cat("\nsample(1:6):\n")
print(sample(1:6))

cat("\nExample 2: Permuting character vectors\n")
cat("replicate(3, sample(c('Curly', 'Larry', 'Moe', 'Shemp'))):\n")
set.seed(6060)
stooges_perms <- replicate(3, sample(c("Curly", "Larry", "Moe", "Shemp")))
print(stooges_perms)

# ==============================================================================
# RESAMPLING WITH sample() - BOOTSTRAP
# ==============================================================================

cat("\n\nRESAMPLING WITH sample() - BOOTSTRAP:\n")
cat(strrep("-", 60), "\n")
cat("Resampling from any existing distribution gives bootstrap estimators\n\n")

cat("Key difference from jackknife:\n")
cat("  - Jackknife: removes one point and recalculates\n")
cat("  - Bootstrap: resamples same length WITH REPLACEMENT\n\n")

# Bootstrap resample function
bootstrap.resample <- function(object) {
  sample(object, length(object), replace = TRUE)
}

cat("Example: Bootstrap resampling\n")
set.seed(7070)
cat("replicate(5, bootstrap.resample(6:10)):\n")
bootstrap_example <- replicate(5, bootstrap.resample(6:10))
print(bootstrap_example)

cat("\nNote: Values can (and do) repeat due to replacement\n")

# ==============================================================================
# BOOTSTRAP TEST: TWO-SAMPLE DIFFERENCE
# ==============================================================================

cat("\n\nBOOTSTRAP TEST: TWO-SAMPLE DIFFERENCE IN MEANS\n")
cat(strrep("-", 60), "\n")

cat("The 2-sample t-test checks for differences in means according to\n")
cat("a known null distribution. Let's use bootstrap to generate the\n")
cat("sampling distribution under the bootstrap assumption.\n\n")

cat("Example: Cat heart weights by sex (MASS::cats dataset)\n\n")

# Load MASS library for cats data
library(MASS)

# Function to calculate difference in means
diff.in.means <- function(df) {
  mean(df[df$Sex == "M", "Hwt"]) - mean(df[df$Sex == "F", "Hwt"])
}

# Observed difference
obs.diff <- diff.in.means(cats)
cat(sprintf("Observed difference in means (M - F): %.4f g\n", obs.diff))

# Summary statistics
cat("\nSummary by sex:\n")
cat(sprintf("Males:   n=%d, mean=%.2f g, sd=%.2f g\n",
            sum(cats$Sex == "M"),
            mean(cats[cats$Sex == "M", "Hwt"]),
            sd(cats[cats$Sex == "M", "Hwt"])))
cat(sprintf("Females: n=%d, mean=%.2f g, sd=%.2f g\n",
            sum(cats$Sex == "F"),
            mean(cats[cats$Sex == "F", "Hwt"]),
            sd(cats[cats$Sex == "F", "Hwt"])))

# Bootstrap resampling
cat("\nGenerating bootstrap distribution (1000 replicates)...\n")
set.seed(8080)
resample.diffs <- replicate(1000, {
  boot.indices <- bootstrap.resample(1:nrow(cats))
  diff.in.means(cats[boot.indices, ])
})

# Bootstrap statistics
cat(sprintf("\nBootstrap results:\n"))
cat(sprintf("Mean of bootstrap diffs: %.4f g\n", mean(resample.diffs)))
cat(sprintf("SD of bootstrap diffs:   %.4f g\n", sd(resample.diffs)))
cat(sprintf("95%% CI (percentile):     [%.4f, %.4f] g\n", 
            quantile(resample.diffs, 0.025), 
            quantile(resample.diffs, 0.975)))

# Compare with t-test
t_test_result <- t.test(Hwt ~ Sex, data = cats)
cat(sprintf("\nComparison with t-test:\n"))
cat(sprintf("t-test 95%% CI: [%.4f, %.4f] g\n", 
            t_test_result$conf.int[1], t_test_result$conf.int[2]))
cat(sprintf("t-test p-value: %.6f\n", t_test_result$p.value))

# Visualization
png("../plots/monte_carlo_bootstrap_test.png", width = 1200, height = 800, res = 150)
par(mfrow = c(2, 2))

# Plot 1: Bootstrap distribution
hist(resample.diffs, breaks = 40, col = "lightblue", border = "black",
     main = "Bootstrap Distribution of Difference in Means",
     xlab = "Difference in heart weight (M - F, grams)",
     freq = FALSE)
abline(v = obs.diff, col = "red", lwd = 3, lty = 2)
abline(v = mean(resample.diffs), col = "blue", lwd = 2, lty = 2)
ci.boot <- quantile(resample.diffs, c(0.025, 0.975))
abline(v = ci.boot, col = "darkgreen", lwd = 2, lty = 3)
legend("topright", 
       legend = c("Observed", "Bootstrap mean", "95% CI"),
       col = c("red", "blue", "darkgreen"), 
       lwd = 2, lty = c(2, 2, 3), cex = 0.8)

# Plot 2: Original data
boxplot(Hwt ~ Sex, data = cats, col = c("pink", "lightblue"),
        main = "Cat Heart Weights by Sex",
        xlab = "Sex", ylab = "Heart weight (grams)")
points(1:2, c(mean(cats[cats$Sex == "F", "Hwt"]),
              mean(cats[cats$Sex == "M", "Hwt"])),
       pch = 18, col = "red", cex = 2)

# Plot 3: Q-Q plot of bootstrap distribution
qqnorm(resample.diffs, main = "Q-Q Plot: Bootstrap Distribution")
qqline(resample.diffs, col = "red", lwd = 2)

# Plot 4: ECDF
plot(ecdf(resample.diffs), 
     main = "ECDF of Bootstrap Differences",
     xlab = "Difference in heart weight (grams)",
     ylab = "Cumulative probability")
abline(v = obs.diff, col = "red", lwd = 2, lty = 2)
abline(v = 0, col = "gray", lwd = 1, lty = 3)

par(mfrow = c(1, 1))
dev.off()
cat("\nSaved: ../plots/monte_carlo_bootstrap_test.png\n")

# ==============================================================================
# PART 6: TOY COLLECTOR EXERCISE (COUPON COLLECTOR PROBLEM)
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("PART 6: TOY COLLECTOR EXERCISE (COUPON COLLECTOR PROBLEM)\n")
cat(strrep("=", 70), "\n\n")

cat("Problem: Children are enticed to buy cereal to collect action figures.\n")
cat("Assume there are 15 action figures and each box contains exactly one,\n")
cat("with each figure being equally likely initially.\n\n")

# Simulation function
simulate_collection <- function(n_toys, probs = NULL, max_boxes = 10000) {
  # Simulate collecting all n_toys with given probabilities.
  # Parameters:
  # - n_toys: number of unique toys
  # - probs: probability of each toy (NULL = equal probability)
  # - max_boxes: maximum boxes to try
  # Returns: number of boxes needed to collect all toys
  
  if (is.null(probs)) {
    probs <- rep(1 / n_toys, n_toys)
  }
  
  collected <- rep(FALSE, n_toys)
  n_boxes <- 0
  
  while (sum(collected) < n_toys && n_boxes < max_boxes) {
    n_boxes <- n_boxes + 1
    toy <- sample(1:n_toys, 1, prob = probs)
    collected[toy] <- TRUE
  }
  
  return(n_boxes)
}

# ==============================================================================
# Question 1 & 2: Equal Probabilities
# ==============================================================================

cat("QUESTIONS 1 & 2: EQUAL PROBABILITIES (1/15 each)\n")
cat(strrep("-", 70), "\n\n")

n_toys <- 15
n_simulations <- 10000

cat(sprintf("Running %d simulations...\n", n_simulations))
set.seed(9090)
boxes_needed_equal <- replicate(n_simulations, simulate_collection(n_toys))

# Calculate statistics
mean_boxes_equal <- mean(boxes_needed_equal)
sd_boxes_equal <- sd(boxes_needed_equal)
se_boxes_equal <- sd_boxes_equal / sqrt(n_simulations)

# Theoretical expectation: n * H_n (harmonic number)
harmonic_number <- sum(1 / (1:n_toys))
theoretical_mean <- n_toys * harmonic_number

cat("\nRESULTS FOR EQUAL PROBABILITIES:\n")
cat(sprintf("Q1. Expected number of boxes (simulated):   %.2f\n", mean_boxes_equal))
cat(sprintf("    Expected number of boxes (theoretical): %.2f\n", theoretical_mean))
cat(sprintf("Q2. Standard deviation:                     %.2f\n", sd_boxes_equal))
cat(sprintf("    Standard error of estimate:             %.4f\n", se_boxes_equal))
cat(sprintf("    Median: %.0f, Range: [%d, %d]\n", 
            median(boxes_needed_equal), 
            min(boxes_needed_equal), 
            max(boxes_needed_equal)))

# Quantiles
quantiles_equal <- quantile(boxes_needed_equal, c(0.25, 0.5, 0.75, 0.9, 0.95))
cat("\nQuantiles:\n")
for (i in seq_along(quantiles_equal)) {
  cat(sprintf("  %s: %d boxes\n", names(quantiles_equal)[i], quantiles_equal[i]))
}

# ==============================================================================
# Questions 3, 4, 5: Unequal Probabilities
# ==============================================================================

cat("\n\nQUESTIONS 3, 4, 5: UNEQUAL PROBABILITIES\n")
cat(strrep("-", 70), "\n\n")

# Define probabilities
toy_names <- LETTERS[1:15]
toy_probs <- c(.2, .1, .1, .1, .1, .1, .05, .05, .05, .05, .02, .02, .02, .02, .02)

cat("Figure probabilities:\n")
prob_table <- data.frame(Figure = toy_names, Probability = toy_probs)
print(prob_table, row.names = FALSE)
cat(sprintf("\nSum of probabilities: %.2f (must equal 1.0)\n", sum(toy_probs)))

cat(sprintf("\nRunning %d simulations...\n", n_simulations))
set.seed(10101)
boxes_needed_unequal <- replicate(n_simulations, 
                                   simulate_collection(n_toys, toy_probs))

# Calculate statistics
mean_boxes_unequal <- mean(boxes_needed_unequal)
sd_boxes_unequal <- sd(boxes_needed_unequal)
se_boxes_unequal <- sd_boxes_unequal / sqrt(n_simulations)

cat("\nRESULTS FOR UNEQUAL PROBABILITIES:\n")
cat(sprintf("Q3. Expected number of boxes: %.2f\n", mean_boxes_unequal))
cat(sprintf("Q4. Uncertainty of estimate:\n"))
cat(sprintf("    Standard deviation: %.2f\n", sd_boxes_unequal))
cat(sprintf("    Standard error:     %.4f\n", se_boxes_unequal))
cat(sprintf("    95%% CI: [%.2f, %.2f]\n", 
            mean_boxes_unequal - 1.96 * se_boxes_unequal,
            mean_boxes_unequal + 1.96 * se_boxes_unequal))
cat(sprintf("    Relative error: %.2f%%\n", 
            100 * se_boxes_unequal / mean_boxes_unequal))

# Thresholds
cat("\nQ5. Probability of buying more than X boxes:\n")
thresholds <- c(50, 100, 200)
for (threshold in thresholds) {
  prob <- mean(boxes_needed_unequal > threshold)
  count <- sum(boxes_needed_unequal > threshold)
  cat(sprintf("    P(boxes > %3d) = %.4f (%.2f%%) - %d/%d simulations\n", 
              threshold, prob, 100 * prob, count, n_simulations))
}

# Additional statistics
cat("\nAdditional statistics:\n")
cat(sprintf("Median: %.0f, Range: [%d, %d]\n", 
            median(boxes_needed_unequal),
            min(boxes_needed_unequal), 
            max(boxes_needed_unequal)))

quantiles_unequal <- quantile(boxes_needed_unequal, c(0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
cat("\nQuantiles:\n")
for (i in seq_along(quantiles_unequal)) {
  cat(sprintf("  %s: %d boxes\n", names(quantiles_unequal)[i], quantiles_unequal[i]))
}

# ==============================================================================
# VISUALIZATION: TOY COLLECTOR
# ==============================================================================

cat("\n\nCreating toy collector visualizations...\n")

png("../plots/monte_carlo_toy_collector.png", width = 1400, height = 1000, res = 150)
par(mfrow = c(2, 3))

# Plot 1: Equal probabilities distribution
hist(boxes_needed_equal, breaks = 50, col = "lightblue", border = "black",
     main = "Equal Probabilities\n(each figure: 1/15)",
     xlab = "Number of boxes needed", ylab = "Frequency")
abline(v = mean_boxes_equal, col = "red", lwd = 3, lty = 2)
abline(v = theoretical_mean, col = "darkgreen", lwd = 3, lty = 3)
legend("topright", 
       legend = c(sprintf("Simulated: %.1f", mean_boxes_equal),
                  sprintf("Theoretical: %.1f", theoretical_mean)),
       col = c("red", "darkgreen"), lwd = 2, lty = c(2, 3), cex = 0.8)

# Plot 2: Unequal probabilities distribution
hist(boxes_needed_unequal, breaks = 50, col = "lightcoral", border = "black",
     main = "Unequal Probabilities\n(rare figures: 0.02)",
     xlab = "Number of boxes needed", ylab = "Frequency")
abline(v = mean_boxes_unequal, col = "red", lwd = 3, lty = 2)
legend("topright", 
       legend = sprintf("Mean: %.1f", mean_boxes_unequal),
       col = "red", lwd = 2, lty = 2, cex = 0.8)

# Plot 3: Comparison CDFs
plot(ecdf(boxes_needed_equal), col = "blue", lwd = 2,
     main = "Cumulative Distribution Comparison",
     xlab = "Number of boxes", ylab = "Cumulative Probability",
     xlim = c(0, max(boxes_needed_unequal)))
plot(ecdf(boxes_needed_unequal), col = "red", lwd = 2, add = TRUE)
abline(v = c(50, 100, 200), lty = 3, col = "gray")
legend("bottomright", 
       legend = c("Equal probs", "Unequal probs", "Thresholds"),
       col = c("blue", "red", "gray"), lwd = 2, lty = c(1, 1, 3), cex = 0.8)

# Plot 4: Box plots comparison
boxplot(list(Equal = boxes_needed_equal, Unequal = boxes_needed_unequal),
        col = c("lightblue", "lightcoral"),
        main = "Distribution Comparison",
        ylab = "Number of boxes needed")
points(c(1, 2), c(mean_boxes_equal, mean_boxes_unequal), 
       col = "red", pch = 18, cex = 2)
legend("topright", legend = "Mean", col = "red", pch = 18, cex = 0.8)

# Plot 5: Probability of exceeding thresholds
threshold_seq <- seq(0, 400, by = 10)
prob_exceed_equal <- sapply(threshold_seq, function(x) mean(boxes_needed_equal > x))
prob_exceed_unequal <- sapply(threshold_seq, function(x) mean(boxes_needed_unequal > x))

plot(threshold_seq, prob_exceed_equal, type = "l", col = "blue", lwd = 2,
     xlab = "Number of boxes", ylab = "P(boxes > x)",
     main = "Probability of Exceeding Threshold",
     ylim = c(0, 1))
lines(threshold_seq, prob_exceed_unequal, col = "red", lwd = 2)
abline(h = c(0.5, 0.9, 0.95), lty = 3, col = "gray")
abline(v = c(50, 100, 200), lty = 3, col = "gray")
legend("topright", 
       legend = c("Equal probs", "Unequal probs"),
       col = c("blue", "red"), lwd = 2, cex = 0.8)

# Plot 6: Impact of rare items
barplot(toy_probs, names.arg = toy_names, col = "lightcoral",
        main = "Figure Probabilities\n(Unequal Case)",
        xlab = "Figure", ylab = "Probability",
        ylim = c(0, max(toy_probs) * 1.1))
abline(h = 1/15, col = "blue", lty = 2, lwd = 2)
text(5, 1/15 + 0.01, "Equal prob = 1/15", col = "blue")

par(mfrow = c(1, 1))
dev.off()
cat("Saved: ../plots/monte_carlo_toy_collector.png\n")

# ==============================================================================
# KEY INSIGHTS
# ==============================================================================

cat("\n\nKEY INSIGHTS:\n")
cat(strrep("-", 70), "\n")
cat(sprintf("1. Impact of unequal probabilities:\n"))
cat(sprintf("   Equal case:   %.1f boxes expected\n", mean_boxes_equal))
cat(sprintf("   Unequal case: %.1f boxes expected\n", mean_boxes_unequal))
cat(sprintf("   Increase: %.1f boxes (%.0f%%)\n", 
            mean_boxes_unequal - mean_boxes_equal,
            100 * (mean_boxes_unequal - mean_boxes_equal) / mean_boxes_equal))

cat(sprintf("\n2. Rare items dominate collection time:\n"))
cat(sprintf("   Rarest figures have probability 0.02 (vs 1/15=0.067)\n"))
cat(sprintf("   Expected wait for a specific rare item: 1/0.02 = 50 boxes\n"))

cat(sprintf("\n3. High variability in unequal case:\n"))
cat(sprintf("   95th percentile: %d boxes (%.1f%% above mean)\n",
            quantiles_unequal["95%"],
            100 * (quantiles_unequal["95%"] - mean_boxes_unequal) / mean_boxes_unequal))

cat(sprintf("\n4. Practical implications:\n"))
cat(sprintf("   P(> 100 boxes) = %.1f%% - significant risk of extreme cases\n",
            100 * mean(boxes_needed_unequal > 100)))
cat(sprintf("   P(> 200 boxes) = %.1f%% - rare but possible\n",
            100 * mean(boxes_needed_unequal > 200)))

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("SUMMARY: ORDINARY MONTE CARLO\n")
cat(strrep("=", 70), "\n\n")

cat("KEY PRINCIPLES:\n")
cat("1. OMC uses IID simulations to estimate expectations: θ = E[g(X)]\n")
cat("2. Law of Large Numbers: ȳₙ converges to θ\n")
cat("3. Central Limit Theorem: ȳₙ ~ N(θ, σ²/n) for large n\n")
cat("4. Standard error decreases as O(1/√n)\n")
cat("5. We can construct confidence intervals using SE = s/√n\n\n")

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("\n\n", strrep("=", 70), "\n")
cat("SUMMARY: MONTE CARLO SIMULATIONS\n")
cat(strrep("=", 70), "\n\n")

cat("KEY PRINCIPLES:\n")
cat("1. OMC uses IID simulations to estimate expectations: θ = E[g(X)]\n")
cat("2. Law of Large Numbers: ȳₙ converges to θ\n")
cat("3. Central Limit Theorem: ȳₙ ~ N(θ, σ²/n) for large n\n")
cat("4. Standard error decreases as O(1/√n)\n")
cat("5. We can construct confidence intervals using SE = s/√n\n\n")

cat("METHODS COVERED:\n")
cat("1. Ordinary Monte Carlo - basic estimation with IID samples\n")
cat("2. Approximating π - geometric probability method\n")
cat("3. Monte Carlo integration - sequential stopping rule\n")
cat("4. Bootstrap methods - resampling with replacement\n")
cat("5. Permutation tests - resampling without replacement\n")
cat("6. Coupon collector problem - complex probability estimation\n\n")

cat("PRACTICAL APPLICATIONS:\n")
cat("- High-dimensional problems (election forecasts, sports predictions)\n")
cat("- Intractable integrals and expectations\n")
cat("- Hypothesis testing without parametric assumptions\n")
cat("- Uncertainty quantification in complex systems\n")
cat("- Sequential decision making with stopping rules\n\n")

cat("PRACTICAL CONSIDERATIONS:\n")
cat("1. More simulations → better accuracy (but diminishing returns)\n")
cat("2. Doubling accuracy requires 4× more simulations\n")
cat("3. Always report standard error or confidence intervals\n")
cat("4. Check convergence by monitoring running estimates\n")
cat("5. For proportions: SE = √[p(1-p)/n]\n")
cat("6. For general expectations: SE = s/√n where s is sample SD\n")
cat("7. Sequential stopping can achieve target precision efficiently\n")
cat("8. Bootstrap provides distribution-free inference\n\n")

cat(strrep("=", 70), "\n")
cat("Generated plots:\n")
cat("  1. monte_carlo_binomial.png - Binomial approximation analysis\n")
cat("  2. monte_carlo_pi_approximation.png - π estimation convergence\n")
cat("  3. monte_carlo_pi_detailed.png - π scatter plot visualization\n")
cat("  4. monte_carlo_integration_sequential.png - Sequential stopping\n")
cat("  5. monte_carlo_integration_distribution.png - Integration distributions\n")
cat("  6. monte_carlo_bootstrap_test.png - Bootstrap hypothesis test\n")
cat("  7. monte_carlo_toy_collector.png - Coupon collector analysis\n")
cat(strrep("=", 70), "\n")

cat("\nMONTE CARLO SIMULATIONS TUTORIAL COMPLETE!\n")
cat("All methods demonstrated with practical examples.\n")
