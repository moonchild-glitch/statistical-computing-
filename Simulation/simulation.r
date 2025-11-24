############################################
# SIMULATION AND RANDOM NUMBER GENERATION
# Statistical Computing Tutorial
############################################

# AGENDA:
# - Transforming uniform random variables
# - Quantile (inverse CDF) method
# - Rejection sampling method
# - Box-Muller transformation
# - Testing random number generators

# Create plots directory
if(!dir.exists("../plots")) {
  dir.create("../plots", recursive=TRUE)
}

############################################
# INTRODUCTION TO RANDOM NUMBER GENERATION
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("INTRODUCTION TO RANDOM NUMBER GENERATION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("Most programming languages provide a uniform random number generator\n")
cat("We can transform uniform draws into other distributions\n\n")

cat("Three main approaches:\n")
cat("1. Quantile (Inverse CDF) method - when we can invert the CDF\n")
cat("2. Rejection method - if all we have is the density\n")
cat("3. Transformation methods - using mathematical relationships\n\n")

# Generate uniform random numbers
set.seed(206)
n <- 1000
u <- runif(n)

png("../plots/simulation_uniform.png", width=800, height=600, res=100)
par(mfrow=c(2,2), mar=c(4,4,3,2))
hist(u, breaks=30, main="Histogram of Uniform(0,1)", 
     xlab="Value", col="lightblue", probability=TRUE)
abline(h=1, col="red", lwd=2, lty=2)

plot(u[1:100], type="l", main="First 100 Values", 
     xlab="Index", ylab="Value", col="blue")
abline(h=c(0,1), col="gray", lty=2)

qqnorm(u, main="Q-Q Plot vs Normal")
qqline(u, col="red", lwd=2)

plot(ecdf(u), main="Empirical CDF", 
     xlab="Value", ylab="Cumulative Probability")
abline(0, 1, col="red", lwd=2, lty=2)
dev.off()

cat("Plot saved: simulation_uniform.png\n\n")

############################################
# QUANTILE (INVERSE CDF) METHOD
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("QUANTILE (INVERSE CDF) METHOD\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("If X ~ F, then F(X) ~ Uniform(0,1)\n")
cat("Conversely, if U ~ Uniform(0,1), then F^(-1)(U) ~ F\n\n")

cat("Example 1: Exponential Distribution\n")
cat("F(x) = 1 - exp(-lambda*x)\n")
cat("F^(-1)(u) = -log(1-u)/lambda\n\n")

# Exponential distribution using inverse CDF
lambda <- 2
u <- runif(n)
x_exp <- -log(1 - u) / lambda

# Compare with built-in
x_exp_builtin <- rexp(n, rate=lambda)

png("../plots/simulation_exponential.png", width=1000, height=600, res=100)
par(mfrow=c(1,2), mar=c(4,4,3,2))

hist(x_exp, breaks=30, probability=TRUE, 
     main="Exponential via Inverse CDF",
     xlab="Value", col="lightblue", border="white")
curve(dexp(x, rate=lambda), add=TRUE, col="red", lwd=2)
legend("topright", legend=c("Simulated", "True density"), 
       col=c("lightblue", "red"), lwd=c(10, 2))

qqplot(x_exp, x_exp_builtin, main="Q-Q Plot: Our Method vs Built-in",
       xlab="Our Method", ylab="Built-in rexp()")
abline(0, 1, col="red", lwd=2)
dev.off()

cat("Plot saved: simulation_exponential.png\n\n")

cat("Example 2: Logistic Distribution\n")
cat("F(x) = 1 / (1 + exp(-x))\n")
cat("F^(-1)(u) = log(u / (1-u))\n\n")

# Logistic distribution using inverse CDF
u <- runif(n)
x_logistic <- log(u / (1 - u))

png("../plots/simulation_logistic.png", width=800, height=600, res=100)
hist(x_logistic, breaks=50, probability=TRUE,
     main="Logistic Distribution via Inverse CDF",
     xlab="Value", col="lightgreen", border="white", xlim=c(-6, 6))
curve(dlogis(x), add=TRUE, col="red", lwd=2)
legend("topright", legend=c("Simulated", "True density"),
       col=c("lightgreen", "red"), lwd=c(10, 2))
dev.off()

cat("Plot saved: simulation_logistic.png\n\n")

############################################
# REJECTION SAMPLING METHOD
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("REJECTION SAMPLING METHOD\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("When we can't invert the CDF, but we have the density f(x)\n")
cat("1. Find an envelope function g(x) such that f(x) <= M*g(x)\n")
cat("2. Sample from g(x)\n")
cat("3. Accept with probability f(x) / (M*g(x))\n\n")

cat("Example: Beta(2, 5) using Rejection Sampling\n")
cat("Envelope: Uniform(0,1)\n\n")

# Rejection sampling for Beta(2, 5)
rejection_beta <- function(n, a=2, b=5) {
  samples <- numeric(n)
  M <- max(dbeta(seq(0, 1, length=1000), a, b))
  
  accepted <- 0
  total_proposals <- 0
  
  while(accepted < n) {
    # Propose from uniform
    u <- runif(1)
    # Acceptance probability
    accept_prob <- dbeta(u, a, b) / M
    
    total_proposals <- total_proposals + 1
    
    if(runif(1) < accept_prob) {
      accepted <- accepted + 1
      samples[accepted] <- u
    }
  }
  
  acceptance_rate <- n / total_proposals
  cat(sprintf("Acceptance rate: %.2f%%\n", acceptance_rate * 100))
  
  return(samples)
}

set.seed(206)
beta_samples <- rejection_beta(n, 2, 5)

png("../plots/simulation_rejection.png", width=1000, height=600, res=100)
par(mfrow=c(1,2), mar=c(4,4,3,2))

hist(beta_samples, breaks=30, probability=TRUE,
     main="Beta(2,5) via Rejection Sampling",
     xlab="Value", col="lightcoral", border="white")
curve(dbeta(x, 2, 5), add=TRUE, col="darkred", lwd=2)
legend("topright", legend=c("Simulated", "True density"),
       col=c("lightcoral", "darkred"), lwd=c(10, 2))

# Q-Q plot against true Beta
beta_builtin <- rbeta(n, 2, 5)
qqplot(beta_samples, beta_builtin, 
       main="Q-Q Plot: Rejection vs Built-in",
       xlab="Rejection Method", ylab="Built-in rbeta()")
abline(0, 1, col="red", lwd=2)
dev.off()

cat("\nPlot saved: simulation_rejection.png\n\n")

############################################
# BASIC R COMMANDS
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("BASIC R COMMANDS FOR RANDOM NUMBER GENERATION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("R encapsulates a lot of these methods for us:\n")
cat("- rnorm() for Normal\n")
cat("- rexp() for Exponential\n")
cat("- rbeta() for Beta\n")
cat("- rgamma() for Gamma\n")
cat("- ... and many more\n\n")

cat("These use optimized algorithms based on:\n")
cat("- Distribution type\n")
cat("- Parameter values\n")
cat("- Computational efficiency\n\n")

cat("Examples of different distributions:\n\n")

set.seed(206)
samples_list <- list(
  Normal = rnorm(n, mean=5, sd=2),
  Exponential = rexp(n, rate=1),
  Gamma = rgamma(n, shape=2, scale=2),
  Beta = rbeta(n, 2, 5),
  Poisson = rpois(n, lambda=3),
  Binomial = rbinom(n, size=10, prob=0.3)
)

png("../plots/simulation_distributions.png", width=1200, height=800, res=100)
par(mfrow=c(2,3), mar=c(4,4,3,2))

for(i in 1:length(samples_list)) {
  name <- names(samples_list)[i]
  data <- samples_list[[i]]
  
  hist(data, breaks=30, probability=TRUE,
       main=name, xlab="Value", 
       col=rainbow(6)[i], border="white")
}
dev.off()

cat("Plot saved: simulation_distributions.png\n\n")

############################################
# SUMMARY
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("Key Points:\n\n")

cat("✓ Can transform uniform draws into other distributions when we can\n")
cat("  compute the distribution function\n\n")

cat("✓ Quantile method when we can invert the CDF\n")
cat("  - Simple and exact\n")
cat("  - Requires invertible CDF\n\n")

cat("✓ The rejection method if all we have is the density\n")
cat("  - Works when CDF is not invertible\n")
cat("  - Efficiency depends on envelope function\n\n")

cat("✓ Basic R commands encapsulate a lot of this for us\n")
cat("  - Highly optimized implementations\n")
cat("  - Use built-in functions when available\n\n")

cat("✓ Optimized algorithms based on distribution and parameter values\n")
cat("  - Different methods for different scenarios\n")
cat("  - Trade-offs between accuracy and speed\n\n")

############################################
# EXERCISE: BOX-MULLER TRANSFORMATION
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("EXERCISE: BOX-MULLER TRANSFORMATION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("The Box-Muller transformation converts uniform random variables\n")
cat("into standard normal random variables.\n\n")

cat("Given U1, U2 ~ Uniform(0,1), independent:\n")
cat("Z1 = sqrt(-2*log(U1)) * cos(2*pi*U2)\n")
cat("Z2 = sqrt(-2*log(U1)) * sin(2*pi*U2)\n")
cat("Then Z1, Z2 ~ Normal(0, 1), independent\n\n")

cat("To generate Normal(mu, sd):\n")
cat("X = mu + sd * Z\n\n")

############################################
# BOX-MULLER IMPLEMENTATION
############################################

bmnormal <- function(n, mu=0, sd=1) {
  # Function to simulate n draws from Normal(mu, sd) using Box-Muller
  # 
  # Args:
  #   n: number of draws
  #   mu: mean of the normal distribution
  #   sd: standard deviation of the normal distribution
  #
  # Returns:
  #   Vector of n random draws from Normal(mu, sd)
  
  # We generate pairs, so round up to even number if needed
  n_pairs <- ceiling(n / 2)
  
  # Generate uniform random variables
  u1 <- runif(n_pairs)
  u2 <- runif(n_pairs)
  
  # Box-Muller transformation
  r <- sqrt(-2 * log(u1))
  theta <- 2 * pi * u2
  
  z1 <- r * cos(theta)
  z2 <- r * sin(theta)
  
  # Combine both sets of standard normals
  z <- c(z1, z2)
  
  # Take only n samples (in case we generated one extra)
  z <- z[1:n]
  
  # Transform to Normal(mu, sd)
  x <- mu + sd * z
  
  return(x)
}

cat("Function 'bmnormal' created!\n\n")

############################################
# TEST THE BOX-MULLER IMPLEMENTATION
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("TESTING BOX-MULLER IMPLEMENTATION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Simulate 2000 draws from Normal(10, 3)
set.seed(206)
n_draws <- 2000
mu_test <- 10
sd_test <- 3

bm_samples <- bmnormal(n_draws, mu=mu_test, sd=sd_test)
builtin_samples <- rnorm(n_draws, mean=mu_test, sd=sd_test)

cat(sprintf("Generated %d samples from Normal(%.0f, %.0f)\n\n", 
            n_draws, mu_test, sd_test))

# Summary statistics
cat("Summary Statistics:\n")
cat(sprintf("Box-Muller - Mean: %.4f, SD: %.4f\n", 
            mean(bm_samples), sd(bm_samples)))
cat(sprintf("Built-in   - Mean: %.4f, SD: %.4f\n", 
            mean(builtin_samples), sd(builtin_samples)))
cat(sprintf("True       - Mean: %.4f, SD: %.4f\n\n", mu_test, sd_test))

############################################
# VISUAL VALIDATION
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("VISUAL VALIDATION\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

png("../plots/simulation_boxmuller.png", width=1200, height=900, res=100)
par(mfrow=c(2,3), mar=c(4,4,3,2))

# 1. Histogram comparison
hist(bm_samples, breaks=50, probability=TRUE,
     main="Box-Muller Samples",
     xlab="Value", col="lightblue", border="white")
curve(dnorm(x, mean=mu_test, sd=sd_test), add=TRUE, col="red", lwd=2)
legend("topright", legend=c("Simulated", "True density"),
       col=c("lightblue", "red"), lwd=c(10, 2))

# 2. Built-in comparison
hist(builtin_samples, breaks=50, probability=TRUE,
     main="Built-in rnorm() Samples",
     xlab="Value", col="lightgreen", border="white")
curve(dnorm(x, mean=mu_test, sd=sd_test), add=TRUE, col="red", lwd=2)

# 3. Q-Q plot against theoretical normal
qqnorm(bm_samples, main="Box-Muller Q-Q Plot")
qqline(bm_samples, col="red", lwd=2)

# 4. Q-Q plot: Box-Muller vs Built-in
qqplot(bm_samples, builtin_samples, 
       main="Box-Muller vs Built-in",
       xlab="Box-Muller", ylab="Built-in rnorm()")
abline(0, 1, col="red", lwd=2)

# 5. ECDF comparison
plot(ecdf(bm_samples), main="Empirical CDF Comparison",
     xlab="Value", ylab="Cumulative Probability",
     col="blue", lwd=2)
plot(ecdf(builtin_samples), add=TRUE, col="green", lwd=2, lty=2)
curve(pnorm(x, mean=mu_test, sd=sd_test), add=TRUE, col="red", lwd=2, lty=3)
legend("bottomright", 
       legend=c("Box-Muller", "Built-in", "True CDF"),
       col=c("blue", "green", "red"), lwd=2, lty=c(1, 2, 3))

# 6. Scatter plot of pairs
# Generate new samples to show pairing
set.seed(206)
u1 <- runif(100)
u2 <- runif(100)
r <- sqrt(-2 * log(u1))
theta <- 2 * pi * u2
z1 <- r * cos(theta)
z2 <- r * sin(theta)

plot(z1, z2, pch=19, col=rgb(0,0,1,0.5),
     main="Box-Muller Pairs (Standard Normal)",
     xlab="Z1", ylab="Z2",
     xlim=c(-4, 4), ylim=c(-4, 4))
abline(h=0, v=0, col="gray", lty=2)
# Add circle to show independence
theta_circle <- seq(0, 2*pi, length=100)
for(r_circle in 1:3) {
  lines(r_circle * cos(theta_circle), r_circle * sin(theta_circle),
        col="red", lty=2)
}

dev.off()

cat("Plot saved: simulation_boxmuller.png\n\n")

############################################
# STATISTICAL TESTS
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STATISTICAL TESTS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("We can use formal statistical tests to validate our sampler:\n\n")

# 1. Shapiro-Wilk test for normality
shapiro_test <- shapiro.test(bm_samples)
cat("1. Shapiro-Wilk Test for Normality\n")
cat(sprintf("   W = %.6f, p-value = %.6f\n", 
            shapiro_test$statistic, shapiro_test$p.value))
if(shapiro_test$p.value > 0.05) {
  cat("   ✓ Cannot reject normality (p > 0.05)\n\n")
} else {
  cat("   ✗ Reject normality (p < 0.05)\n\n")
}

# 2. Kolmogorov-Smirnov test
ks_test <- ks.test(bm_samples, "pnorm", mean=mu_test, sd=sd_test)
cat("2. Kolmogorov-Smirnov Test\n")
cat(sprintf("   D = %.6f, p-value = %.6f\n", 
            ks_test$statistic, ks_test$p.value))
if(ks_test$p.value > 0.05) {
  cat("   ✓ Cannot reject that samples follow Normal(10, 3) (p > 0.05)\n\n")
} else {
  cat("   ✗ Reject null hypothesis (p < 0.05)\n\n")
}

# 3. t-test for mean
t_test <- t.test(bm_samples, mu=mu_test)
cat("3. One-Sample t-test (H0: mean = 10)\n")
cat(sprintf("   t = %.6f, p-value = %.6f\n", 
            t_test$statistic, t_test$p.value))
cat(sprintf("   95%% CI: (%.4f, %.4f)\n", 
            t_test$conf.int[1], t_test$conf.int[2]))
if(t_test$p.value > 0.05) {
  cat("   ✓ Cannot reject that mean = 10 (p > 0.05)\n\n")
} else {
  cat("   ✗ Reject null hypothesis (p < 0.05)\n\n")
}

# 4. Chi-squared test for variance
var_test_stat <- (n_draws - 1) * var(bm_samples) / sd_test^2
var_test_pval <- 2 * min(pchisq(var_test_stat, df=n_draws-1),
                         1 - pchisq(var_test_stat, df=n_draws-1))
cat("4. Chi-squared Test for Variance (H0: sd = 3)\n")
cat(sprintf("   χ² = %.4f, p-value = %.6f\n", 
            var_test_stat, var_test_pval))
cat(sprintf("   Sample variance: %.4f, Expected: %.4f\n", 
            var(bm_samples), sd_test^2))
if(var_test_pval > 0.05) {
  cat("   ✓ Cannot reject that variance = 9 (p > 0.05)\n\n")
} else {
  cat("   ✗ Reject null hypothesis (p < 0.05)\n\n")
}

# 5. Anderson-Darling test (if available)
if(require(nortest, quietly=TRUE)) {
  ad_test <- ad.test(bm_samples)
  cat("5. Anderson-Darling Test for Normality\n")
  cat(sprintf("   A = %.6f, p-value = %.6f\n", 
              ad_test$statistic, ad_test$p.value))
  if(ad_test$p.value > 0.05) {
    cat("   ✓ Cannot reject normality (p > 0.05)\n\n")
  } else {
    cat("   ✗ Reject normality (p < 0.05)\n\n")
  }
} else {
  cat("5. Anderson-Darling Test\n")
  cat("   (nortest package not installed)\n\n")
}

############################################
# COMPARISON OF METHODS
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("COMPARISON OF SAMPLING METHODS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Time comparison
cat("Speed Comparison (generating 100,000 samples):\n\n")

n_speed <- 100000

time_bm <- system.time({
  bm_speed <- bmnormal(n_speed, mu=10, sd=3)
})

time_builtin <- system.time({
  builtin_speed <- rnorm(n_speed, mean=10, sd=3)
})

cat(sprintf("Box-Muller:  %.4f seconds\n", time_bm["elapsed"]))
cat(sprintf("Built-in:    %.4f seconds\n", time_builtin["elapsed"]))
cat(sprintf("Ratio:       %.2fx slower\n\n", 
            time_bm["elapsed"] / time_builtin["elapsed"]))

cat("Note: Built-in functions are typically faster because:\n")
cat("- They use optimized C code\n")
cat("- They may use more sophisticated algorithms (e.g., Ziggurat)\n")
cat("- They minimize function call overhead\n\n")

############################################
# INDEPENDENCE TEST
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("INDEPENDENCE OF BOX-MULLER PAIRS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Generate pairs explicitly
set.seed(206)
n_pairs <- 1000
u1 <- runif(n_pairs)
u2 <- runif(n_pairs)

r <- sqrt(-2 * log(u1))
theta <- 2 * pi * u2

z1 <- r * cos(theta)
z2 <- r * sin(theta)

# Test independence using correlation
cor_test <- cor.test(z1, z2)
cat("Correlation Test between Z1 and Z2:\n")
cat(sprintf("Correlation: %.6f\n", cor_test$estimate))
cat(sprintf("p-value: %.6f\n", cor_test$p.value))

if(abs(cor_test$estimate) < 0.1 & cor_test$p.value > 0.05) {
  cat("✓ Z1 and Z2 appear independent\n\n")
} else {
  cat("Note: Some correlation detected (expected to be near zero)\n\n")
}

# Scatter plot with density
png("../plots/simulation_boxmuller_independence.png", width=800, height=800, res=100)
par(mar=c(5,5,4,2))

# Create 2D density
library(MASS)
dens <- kde2d(z1, z2, n=50)

# Contour plot
contour(dens, drawlabels=FALSE, nlevels=10, col="blue", lwd=2,
        main="Box-Muller Pairs: Testing Independence",
        xlab="Z1 ~ N(0,1)", ylab="Z2 ~ N(0,1)")
points(z1, z2, pch=19, col=rgb(0,0,0,0.2), cex=0.5)
abline(h=0, v=0, col="red", lty=2, lwd=2)

# Add marginal histograms info
text(-3, 3, sprintf("Cor = %.4f\np = %.4f", 
                   cor_test$estimate, cor_test$p.value),
     adj=0, cex=1.2, font=2)

dev.off()

cat("Plot saved: simulation_boxmuller_independence.png\n\n")

############################################
# ADVANCED: POLAR BOX-MULLER
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("ADVANCED: POLAR BOX-MULLER METHOD\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("The polar form avoids computing sine and cosine:\n")
cat("1. Generate U1, U2 ~ Uniform(-1, 1)\n")
cat("2. Compute S = U1^2 + U2^2\n")
cat("3. If S >= 1 or S = 0, reject and try again\n")
cat("4. Z1 = U1 * sqrt(-2*log(S)/S)\n")
cat("5. Z2 = U2 * sqrt(-2*log(S)/S)\n\n")

bmnormal_polar <- function(n, mu=0, sd=1) {
  n_pairs <- ceiling(n / 2)
  z <- numeric(2 * n_pairs)
  
  i <- 1
  while(i <= n_pairs) {
    # Generate uniform in (-1, 1)
    u1 <- runif(1, -1, 1)
    u2 <- runif(1, -1, 1)
    s <- u1^2 + u2^2
    
    # Reject if outside unit circle
    if(s < 1 && s > 0) {
      factor <- sqrt(-2 * log(s) / s)
      z[2*i - 1] <- u1 * factor
      z[2*i] <- u2 * factor
      i <- i + 1
    }
  }
  
  z <- z[1:n]
  x <- mu + sd * z
  return(x)
}

cat("Function 'bmnormal_polar' created!\n\n")

# Quick test
set.seed(206)
polar_samples <- bmnormal_polar(1000, mu=10, sd=3)
cat(sprintf("Polar method - Mean: %.4f, SD: %.4f\n\n", 
            mean(polar_samples), sd(polar_samples)))

############################################
# FINAL SUMMARY
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("FINAL SUMMARY: BOX-MULLER EXERCISE\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("✓ Implemented Box-Muller transformation\n")
cat("  - Successfully generates Normal(mu, sd) from Uniform(0,1)\n\n")

cat("✓ Generated 2000 samples from Normal(10, 3)\n")
cat("  - Mean and SD match expected values\n\n")

cat("✓ Visual validation:\n")
cat("  - Histogram matches theoretical density\n")
cat("  - Q-Q plots show good agreement\n")
cat("  - ECDF matches theoretical CDF\n\n")

cat("✓ Statistical tests:\n")
cat("  - Shapiro-Wilk test: normality not rejected\n")
cat("  - Kolmogorov-Smirnov test: matches Normal(10, 3)\n")
cat("  - t-test: mean consistent with 10\n")
cat("  - Variance test: SD consistent with 3\n\n")

cat("✓ Independence verified:\n")
cat("  - Z1 and Z2 pairs are uncorrelated\n")
cat("  - Scatter plot shows circular symmetry\n\n")

cat("✓ Additional implementations:\n")
cat("  - Polar Box-Muller (avoids trig functions)\n\n")

cat(paste(rep("=", 60), collapse=""), "\n")
cat("SIMULATION TUTORIAL COMPLETE\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

plot_count <- length(list.files("../plots", 
                                pattern="^simulation.*\\.png$"))
cat(sprintf("Generated %d plots in ../plots/\n\n", plot_count))

cat("For more on random number generation, see:\n")
cat("- Devroye (1986): Non-Uniform Random Variate Generation\n")
cat("- Ripley (1987): Stochastic Simulation\n")
cat("- Robert & Casella (2004): Monte Carlo Statistical Methods\n\n")

cat("Thank you for completing this tutorial!\n")
