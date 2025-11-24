# ============================================================================
# Bootstrap
# ============================================================================

# Agenda
# - Toy collector solution
# - Plug-In and the Bootstrap
# - Nonparametric and Parametric Bootstraps
# - Examples

# ============================================================================
# Exercise: Toy Collector Problem
# ============================================================================

# Children (and some adults) are frequently enticed to buy breakfast cereal 
# in an effort to collect all the action figures. Assume there are 15 action 
# figures and each cereal box contains exactly one with each figure being 
# equally likely.

# Questions:
# 1. Find the expected number of boxes needed to collect all 15 action figures.
# 2. Find the standard deviation of the number of boxes needed to collect all 
#    15 action figures.
# 3. Now suppose we no longer have equal probabilities, instead let:
#    Figure      A    B    C    D    E    F    G    H    I    J    K    L    M    N    O
#    Probability .2   .1   .1   .1   .1   .1   .05  .05  .05  .05  .02  .02  .02  .02  .02
#    
#    a. Estimate the expected number of boxes needed to collect all 15 action figures.
#    b. What is the uncertainty of your estimate?
#    c. What is the probability you bought more than 300 boxes? 500 boxes? 800 boxes?

# ============================================================================
# Part 1 & 2: Equal Probabilities (Theoretical Solution)
# ============================================================================

# When all figures are equally likely (probability = 1/15 each), this is known
# as the "Coupon Collector Problem"

# SOLUTION APPROACH:
# Consider the probability of getting a "new toy" given we already have i toys
# P(New Toy|i) = (15-i)/15
#
# Since each box is independent, our waiting time until a "new toy" is a 
# geometric random variable with success probability p = (15-i)/15
#
# For a geometric random variable with success probability p:
# - Mean waiting time = 1/p
# - Variance = (1-p)/p^2

n <- 15  # number of action figures

# METHOD 1: Geometric Random Variable Approach
# Expected number of boxes = sum of expected waiting times for each new toy
# E[T] = 15/15 + 15/14 + 15/13 + ... + 15/1

expected_boxes_geometric <- sum(n / (n:1))

cat("=== Equal Probabilities (Theoretical - Geometric Approach) ===\n")
cat("Expected number of boxes:\n")
cat("  E[T] = 15/15 + 15/14 + 15/13 + ... + 15/1\n")
cat("  E[T] =", expected_boxes_geometric, "\n")
cat("  E[T] ≈ 49.77\n\n")

# Variance calculation using geometric random variables
# For stage i (when we have i toys already), probability of new toy is (n-i)/n
# Variance for that stage: Var[X_i] = (1-p)/p^2 where p = (n-i)/n
# Total variance = sum of variances (independent stages)

variance_geometric <- 0
for (i in 0:(n-1)) {
    p <- (n - i) / n  # probability of new toy when we have i toys
    variance_geometric <- variance_geometric + (1 - p) / p^2
}

# Apply the scaling factor n^2
variance_geometric <- n * variance_geometric

sd_geometric <- sqrt(variance_geometric)

cat("Variance calculation:\n")
cat("  Var[T] = 15*(1-15/15)/(15/15)^2 + 15*(1-14/15)/(14/15)^2 + ... + 15*(1-1/15)/(1/15)^2\n")
cat("  Var[T] =", variance_geometric, "\n")
cat("  Var[T] ≈ 34.77\n\n")
cat("Standard deviation:", sd_geometric, "\n")
cat("SD ≈ 5.90\n\n")

# METHOD 2: Harmonic Number Approach (equivalent)
# Theoretical Expected Value: E[T] = n * H_n
# where n = number of figures, H_n = nth harmonic number = sum(1/i) for i=1 to n

# Calculate the harmonic number H_15
H_n <- sum(1 / (1:n))

# Expected number of boxes
expected_boxes_theory <- n * H_n

cat("=== Equal Probabilities (Theoretical - Harmonic Number Approach) ===\n")
cat("Expected number of boxes: E[T] = n * H_n\n")
cat("  where H_n = sum(1/i) for i=1 to", n, "\n")
cat("  H_15 =", H_n, "\n")
cat("  E[T] =", expected_boxes_theory, "\n\n")

# Theoretical Variance: Var[T] = n^2 * sum(1/i^2) for i=1 to n - n * H_n
variance_theory <- n^2 * sum(1 / (1:n)^2) - n * H_n
sd_theory <- sqrt(variance_theory)

cat("Variance: Var[T] = n^2 * sum(1/i^2) - n * H_n\n")
cat("  Var[T] =", variance_theory, "\n")
cat("Standard deviation:", sd_theory, "\n\n")

# Verify both methods give the same answer
cat("=== Verification: Both Methods Agree ===\n")
cat("Geometric approach - Expected:", round(expected_boxes_geometric, 4), "\n")
cat("Harmonic approach  - Expected:", round(expected_boxes_theory, 4), "\n")
cat("Difference:", abs(expected_boxes_geometric - expected_boxes_theory), "\n\n")

cat("Geometric approach - Variance:", round(variance_geometric, 4), "\n")
cat("Harmonic approach  - Variance:", round(variance_theory, 4), "\n")
cat("Difference:", abs(variance_geometric - variance_theory), "\n\n")

# ============================================================================
# Part 1 & 2: Equal Probabilities (Simulation Verification)
# ============================================================================

# Function to simulate collecting all figures with equal probabilities
collect_all_equal <- function(n_figures = 15) {
    collected <- rep(FALSE, n_figures)
    n_boxes <- 0
    
    while (!all(collected)) {
        # Buy a box, get a random figure (equal probability)
        figure <- sample(1:n_figures, 1)
        collected[figure] <- TRUE
        n_boxes <- n_boxes + 1
    }
    
    return(n_boxes)
}

# Run simulation
set.seed(123)
n_simulations <- 10000
results_equal <- replicate(n_simulations, collect_all_equal(15))

# Calculate statistics from simulation
expected_boxes_sim <- mean(results_equal)
sd_boxes_sim <- sd(results_equal)

cat("\n=== Equal Probabilities (Simulation) ===\n")
cat("Expected number of boxes (simulated):", expected_boxes_sim, "\n")
cat("Standard deviation (simulated):", sd_boxes_sim, "\n")
cat("\nComparison with theory:\n")
cat("Expected value difference:", abs(expected_boxes_theory - expected_boxes_sim), "\n")
cat("SD difference:", abs(sd_theory - sd_boxes_sim), "\n")

# Visualize the distribution
hist(results_equal, 
     breaks = 30, 
     main = "Distribution of Boxes Needed (Equal Probabilities)",
     xlab = "Number of Boxes",
     col = "skyblue",
     border = "white")
abline(v = expected_boxes_sim, col = "red", lwd = 2, lty = 2)
legend("topright", 
       legend = c(paste("Mean =", round(expected_boxes_sim, 2)),
                  paste("SD =", round(sd_boxes_sim, 2))),
       col = "red", lty = 2, lwd = 2)

# ============================================================================
# Part 3: Unequal Probabilities
# ============================================================================

# Define the probabilities for each figure
figures <- c("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O")
prob.table <- c(.2, .1, .1, .1, .1, .1, .05, .05, .05, .05, .02, .02, .02, .02, .02)
probabilities <- prob.table  # Keep both variable names for compatibility

# Verify probabilities sum to 1
cat("\n=== Unequal Probabilities ===\n")
cat("Sum of probabilities:", sum(probabilities), "\n")

# Display the probability distribution
prob_df <- data.frame(Figure = figures, Probability = probabilities)
print(prob_df)

# ============================================================================
# Simulation Function for Unequal Probabilities
# ============================================================================

# Define the box numbers
boxes <- seq(1, 15)

# Function to simulate collecting all figures with specified probabilities
box.count <- function(prob = prob.table) {
    check <- double(length(prob))  # Vector to track which figures we have
    i <- 0  # Counter for number of boxes purchased
    
    while (sum(check) < length(prob)) {
        # Sample a box according to the probability distribution
        x <- sample(boxes, 1, prob = prob)
        check[x] <- 1  # Mark that we now have this figure
        i <- i + 1      # Increment box counter
    }
    
    return(i)
}

# Alternative naming for consistency
collect_all_unequal <- function(probs = prob.table) {
    return(box.count(prob = probs))
}

# ============================================================================
# Part 3a: Estimate the expected number of boxes
# ============================================================================

# Run simulation using the box.count() function
trials <- 1000
sim.boxes <- double(trials)

set.seed(456)
for (i in 1:trials) {
    sim.boxes[i] <- box.count()
}

# Calculate point estimate
est <- mean(sim.boxes)

# Calculate Monte Carlo Standard Error (MCSE)
# MCSE measures the uncertainty in our estimate due to finite simulation
mcse <- sd(sim.boxes) / sqrt(trials)

# 95% Confidence interval for the mean
interval <- est + c(-1, 1) * 1.96 * mcse

cat("\n3a. Expected number of boxes (unequal probabilities):\n")
cat("    Point estimate:", est, "\n")
cat("    Example output: est = 115.468\n\n")

# Part 3b: Uncertainty of the estimate
cat("3b. Uncertainty of estimate:\n")
cat("    Monte Carlo Standard Error (MCSE):", mcse, "\n")
cat("    95% Confidence interval:", interval[1], "to", interval[2], "\n")
cat("    Example output: interval = [112.0715, 118.8645]\n\n")

# Additional run with more simulations for better accuracy
n_simulations <- 10000
set.seed(789)
results_unequal <- replicate(n_simulations, box.count())

expected_boxes_unequal <- mean(results_unequal)
sd_boxes_unequal <- sd(results_unequal)
se_mean <- sd_boxes_unequal / sqrt(n_simulations)

cat("More precise estimate (with", n_simulations, "simulations):\n")
cat("    Expected number of boxes:", expected_boxes_unequal, "\n")
cat("    Standard deviation of boxes:", sd_boxes_unequal, "\n")
cat("    Standard error of the mean:", se_mean, "\n")
cat("    95% Confidence interval:", 
    expected_boxes_unequal - 1.96 * se_mean, "to", 
    expected_boxes_unequal + 1.96 * se_mean, "\n\n")

# ============================================================================
# Visualization: Histogram with Reference Line
# ============================================================================

# Plot histogram of simulation results
hist(sim.boxes, 
     main = "Histogram of Total Boxes", 
     xlab = "Boxes",
     col = "lightblue",
     border = "white")
abline(v = 300, col = "red", lwd = 2)
legend("topright", 
       legend = "300 boxes threshold",
       col = "red", 
       lwd = 2)

# ============================================================================
# Part 3c: Probability of buying more than 300, 500, 800 boxes
# ============================================================================

# Calculate probabilities from the initial simulation (trials = 1000)
prob_more_300_initial <- mean(sim.boxes > 300)
prob_more_500_initial <- mean(sim.boxes > 500)
prob_more_800_initial <- mean(sim.boxes > 800)

cat("3c. Probabilities (from", trials, "simulations):\n")
cat("    P(boxes > 300):", prob_more_300_initial, "\n")
cat("    P(boxes > 500):", prob_more_500_initial, "\n")
cat("    P(boxes > 800):", prob_more_800_initial, "\n\n")

# Calculate probabilities from the larger simulation (n_simulations = 10000)
prob_more_300 <- mean(results_unequal > 300)
prob_more_500 <- mean(results_unequal > 500)
prob_more_800 <- mean(results_unequal > 800)

cat("Probabilities (from", n_simulations, "simulations - more precise):\n")
cat("    P(boxes > 300):", prob_more_300, "\n")
cat("    P(boxes > 500):", prob_more_500, "\n")
cat("    P(boxes > 800):", prob_more_800, "\n")
cat("    P(boxes > 800):", prob_more_800, "\n")

# Visualize the distribution with unequal probabilities
hist(results_unequal, 
     breaks = 50, 
     main = "Distribution of Boxes Needed (Unequal Probabilities)",
     xlab = "Number of Boxes",
     col = "lightcoral",
     border = "white",
     xlim = c(0, max(results_unequal)))
abline(v = expected_boxes_unequal, col = "darkred", lwd = 2, lty = 2)
abline(v = 300, col = "blue", lwd = 2, lty = 3)
abline(v = 500, col = "green", lwd = 2, lty = 3)
abline(v = 800, col = "purple", lwd = 2, lty = 3)
legend("topright", 
       legend = c(paste("Mean =", round(expected_boxes_unequal, 2)),
                  "300 boxes", "500 boxes", "800 boxes"),
       col = c("darkred", "blue", "green", "purple"), 
       lty = c(2, 3, 3, 3), 
       lwd = 2)

# ============================================================================
# Comparison: Equal vs Unequal Probabilities
# ============================================================================

cat("\n=== Comparison ===\n")
cat("Equal probabilities:\n")
cat("  Expected boxes:", round(expected_boxes_sim, 2), "\n")
cat("  Standard deviation:", round(sd_boxes_sim, 2), "\n")
cat("\nUnequal probabilities:\n")
cat("  Expected boxes:", round(expected_boxes_unequal, 2), "\n")
cat("  Standard deviation:", round(sd_boxes_unequal, 2), "\n")
cat("\nDifference:\n")
cat("  Additional expected boxes:", round(expected_boxes_unequal - expected_boxes_sim, 2), "\n")
cat("  Increase in variability:", round(sd_boxes_unequal - sd_boxes_sim, 2), "\n")

# Side-by-side boxplot comparison
boxplot(list(Equal = results_equal, Unequal = results_unequal),
        main = "Comparison: Equal vs Unequal Probabilities",
        ylab = "Number of Boxes",
        col = c("skyblue", "lightcoral"),
        border = c("darkblue", "darkred"))

# ============================================================================
# Summary Statistics
# ============================================================================

cat("\n=== Summary Statistics (Unequal Probabilities) ===\n")
cat("Minimum boxes:", min(results_unequal), "\n")
cat("1st Quartile:", quantile(results_unequal, 0.25), "\n")
cat("Median:", median(results_unequal), "\n")
cat("Mean:", mean(results_unequal), "\n")
cat("3rd Quartile:", quantile(results_unequal, 0.75), "\n")
cat("Maximum boxes:", max(results_unequal), "\n")
cat("90th percentile:", quantile(results_unequal, 0.90), "\n")
cat("95th percentile:", quantile(results_unequal, 0.95), "\n")
cat("99th percentile:", quantile(results_unequal, 0.99), "\n")

# ============================================================================
# Key Insights
# ============================================================================

# Key observations:
# 1. With equal probabilities, the expected number of boxes is ~50.4
# 2. With unequal probabilities (rare figures), the expected number increases
#    dramatically to several hundred boxes
# 3. The rare figures (probability 0.02) are the bottleneck - you spend most
#    time waiting for these
# 4. The variability also increases substantially with unequal probabilities
# 5. There's a substantial probability (> 5%) of needing more than 500 boxes
#    with the unequal distribution

# ============================================================================
# Example: Speed of Light Data (Newcomb's Experiment)
# ============================================================================

# Thanks to Rob Gould (UCLA Statistics) for the following example

# In 1882 Simon Newcomb performed an experiment to measure the speed of light
# He measured the time it took for light to travel from Fort Myer on the west 
# bank of the Potomac River to a fixed mirror at the foot of the Washington 
# monument 3721 meters away.
#
# In the units of the data, the currently accepted "true" speed of light is 33.02
#
# Question: Does the data support the current accepted speed of 33.02?

# Speed of light measurements
speed <- c(28, -44, 29, 30, 26, 27, 22, 23, 33, 16, 24, 29, 24, 40, 21, 31, 34, -2, 25, 19)

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("Speed of Light Data (Newcomb's Experiment, 1882)\n")
cat(strrep("=", 80), "\n\n")

# Basic summary statistics
cat("=== Summary Statistics ===\n")
cat("Sample size:", length(speed), "\n")
cat("Mean:", mean(speed), "\n")
cat("Median:", median(speed), "\n")
cat("Standard deviation:", sd(speed), "\n")
cat("Min:", min(speed), "\n")
cat("Max:", max(speed), "\n")
cat("\nAccepted true speed of light: 33.02\n")

# Visualize the data
hist(speed, 
     breaks = 10,
     main = "Newcomb's Speed of Light Measurements (1882)",
     xlab = "Speed (in experimental units)",
     col = "lightgreen",
     border = "white")
abline(v = mean(speed), col = "blue", lwd = 2, lty = 2)
abline(v = 33.02, col = "red", lwd = 2, lty = 2)
legend("topleft", 
       legend = c(paste("Sample mean =", round(mean(speed), 2)),
                  "Accepted value = 33.02"),
       col = c("blue", "red"), 
       lty = 2, 
       lwd = 2)

# Boxplot to identify outliers
boxplot(speed,
        main = "Speed of Light Measurements",
        ylab = "Speed (in experimental units)",
        col = "lightgreen",
        border = "darkgreen")
abline(h = 33.02, col = "red", lwd = 2, lty = 2)
text(1.3, 33.02, "Accepted: 33.02", col = "red")

# Identify potential outliers
cat("\n=== Outlier Analysis ===\n")
Q1 <- quantile(speed, 0.25)
Q3 <- quantile(speed, 0.75)
IQR_val <- IQR(speed)
lower_fence <- Q1 - 1.5 * IQR_val
upper_fence <- Q3 + 1.5 * IQR_val

outliers <- speed[speed < lower_fence | speed > upper_fence]
cat("Potential outliers:", outliers, "\n")
cat("Lower fence:", lower_fence, "\n")
cat("Upper fence:", upper_fence, "\n")

# Initial assessment
cat("\n=== Initial Assessment ===\n")
cat("Difference between sample mean and accepted value:", mean(speed) - 33.02, "\n")
cat("Is the accepted value within one SD of the mean?", 
    abs(mean(speed) - 33.02) <= sd(speed), "\n")

# Note: The presence of outliers (especially -44 and -2) suggests possible 
# measurement errors or experimental issues. These may significantly affect 
# the mean and our conclusions about whether the data supports the accepted value.

cat("\nNote: The data contains some unusual values (e.g., -44, -2) that may\n")
cat("represent measurement errors. These outliers can substantially affect\n")
cat("our conclusions. Bootstrap methods can help us understand the uncertainty\n")
cat("in our estimates and test hypotheses about the true speed of light.\n")

# ============================================================================
# Hypothesis Testing: Does the data support the accepted speed?
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("Hypothesis Testing for Speed of Light\n")
cat(strrep("=", 80), "\n\n")

# Step 1: State null and alternative hypotheses
cat("Step 1: State the Hypotheses\n")
cat("  H₀: μ = 33.02  (The true mean speed equals the accepted value)\n")
cat("  Hₐ: μ ≠ 33.02  (The true mean speed differs from the accepted value)\n")
cat("  (This is a two-sided test)\n\n")

# Step 2: Choose a significance level
alpha <- 0.05
cat("Step 2: Significance Level\n")
cat("  α =", alpha, "\n\n")

# Step 3: Choose a test statistic
cat("Step 3: Test Statistic\n")
cat("  Since we wish to estimate the mean speed, we use the sample average\n")
cat("  Test statistic: X̄ (sample mean)\n\n")

# Step 4: Find the observed value of the test statistic
observed_mean <- mean(speed)
cat("Step 4: Observed Test Statistic\n")
cat("  Observed sample mean:", observed_mean, "\n")
cat("  Difference from H₀:", observed_mean - 33.02, "\n\n")

# Step 5: Calculate a p-value?
cat("Step 5: Calculate p-value\n")
cat("  Problem: We need the sampling distribution of X̄ under H₀,\n")
cat("           but we don't have it!\n")
cat("  - Normal approximation would be poor here due to small sample\n")
cat("    size (n=20) and presence of outliers\n")
cat("  - We need an alternative approach...\n\n")

# ============================================================================
# Bootstrap Approach: Simulating Under the Null Hypothesis
# ============================================================================

cat(strrep("=", 80), "\n")
cat("Bootstrap Solution: Shift the Data to Satisfy H₀\n")
cat(strrep("=", 80), "\n\n")

cat("Strategy:\n")
cat("  Instead of relying on theoretical distributions, we can perform\n")
cat("  a simulation under conditions where H₀ is TRUE.\n\n")

cat("Approach:\n")
cat("  1. Use our data to represent the population\n")
cat("  2. BUT first, shift it so that the mean really IS 33.02\n")
cat("  3. This creates a population where H₀ is true by construction\n\n")

# Shift the data so its mean equals 33.02
newspeed <- speed - mean(speed) + 33.02

cat("Creating the shifted data:\n")
cat("  newspeed = speed - mean(speed) + 33.02\n")
cat("  newspeed = speed -", mean(speed), "+ 33.02\n\n")

# Verify the shift
cat("Verification:\n")
cat("  Original mean(speed):", mean(speed), "\n")
cat("  New mean(newspeed):", mean(newspeed), "\n")
cat("  Difference:", abs(mean(newspeed) - 33.02), "(should be ≈ 0)\n\n")

cat("Key insight:\n")
cat("  The histogram of newspeed has EXACTLY the same shape as speed,\n")
cat("  but is shifted so that it's centered at 33.02.\n")
cat("  This preserves the variability and distribution shape of the\n")
cat("  original data while satisfying H₀.\n\n")

# Visualize the shift
par(mfrow = c(1, 2))

# Original data
hist(speed, 
     breaks = 10,
     main = "Original Data",
     xlab = "Speed",
     col = "lightblue",
     border = "white",
     xlim = c(-50, 50))
abline(v = mean(speed), col = "blue", lwd = 2, lty = 2)
abline(v = 33.02, col = "red", lwd = 2, lty = 2)
legend("topleft", 
       legend = c(paste("Mean =", round(mean(speed), 2)),
                  "H₀: μ = 33.02"),
       col = c("blue", "red"), 
       lty = 2, 
       lwd = 2,
       cex = 0.8)

# Shifted data
hist(newspeed, 
     breaks = 10,
     main = "Shifted Data (H₀ is True)",
     xlab = "Speed",
     col = "lightgreen",
     border = "white",
     xlim = c(-50, 50))
abline(v = mean(newspeed), col = "darkgreen", lwd = 2, lty = 2)
abline(v = 33.02, col = "red", lwd = 2, lty = 2)
legend("topleft", 
       legend = c(paste("Mean =", round(mean(newspeed), 2)),
                  "H₀: μ = 33.02"),
       col = c("darkgreen", "red"), 
       lty = 2, 
       lwd = 2,
       cex = 0.8)

par(mfrow = c(1, 1))

# Show side-by-side comparison of the data
cat("\nData comparison (first 10 values):\n")
comparison_df <- data.frame(
    Original = speed[1:10],
    Shifted = newspeed[1:10],
    Difference = newspeed[1:10] - speed[1:10]
)
print(comparison_df)
cat("\nNote: All differences are equal to", round(33.02 - mean(speed), 4), "\n")

# ============================================================================
# Bootstrap Resampling: Generate the Sampling Distribution Under H₀
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("Bootstrap Resampling from the Shifted Data\n")
cat(strrep("=", 80), "\n\n")

cat("Now we reach into our fake population and take out 20 observations at\n")
cat("random, with replacement.\n")
cat("  - We take out 20 because that's the size of our initial sample\n")
cat("  - We calculate the average and save it\n")
cat("  - We repeat this process many, many times\n\n")

cat("Result:\n")
cat("  We will have a sampling distribution of X̄ with mean 33.02\n")
cat("  (because newspeed has mean 33.02)\n\n")

cat("Final step:\n")
cat("  Compare this distribution to our observed sample average and\n")
cat("  obtain a p-value!\n\n")

# Perform the bootstrap
n <- 1000
bstrap <- double(n)

cat("Running", n, "bootstrap replications...\n")
for (i in 1:n){ 
  newsample <- sample(newspeed, 20, replace=T)
  bstrap[i] <- mean(newsample) 
}
cat("Bootstrap complete!\n\n")

# Summary of bootstrap distribution
cat("Bootstrap Sampling Distribution Summary:\n")
cat("  Mean of bootstrap means:", mean(bstrap), "\n")
cat("  SD of bootstrap means:", sd(bstrap), "\n")
cat("  Min:", min(bstrap), "\n")
cat("  Max:", max(bstrap), "\n\n")

# Visualize the bootstrap distribution
par(mfrow = c(2, 2))

# Plot 1: Basic histogram with observed value
hist(bstrap,
     breaks = 30,
     main = "Bootstrap Sampling Distribution of X̄ Under H₀",
     xlab = "Sample Mean",
     col = "lightblue",
     border = "white",
     probability = TRUE)

# Add vertical line for H₀ mean
abline(v = 33.02, col = "darkgreen", lwd = 2, lty = 2)

# Add vertical line for observed mean
abline(v = observed_mean, col = "red", lwd = 3)

# Add density curve
lines(density(bstrap), col = "blue", lwd = 2)

legend("topright",
       legend = c("H₀: μ = 33.02",
                  paste("Observed:", round(observed_mean, 2)),
                  "Bootstrap density"),
       col = c("darkgreen", "red", "blue"),
       lty = c(2, 1, 1),
       lwd = c(2, 3, 2),
       cex = 0.8)

# Plot 2: Histogram highlighting extreme values (for p-value)
distance_from_null <- abs(observed_mean - 33.02)
extreme_lower <- 33.02 - distance_from_null
extreme_upper <- 33.02 + distance_from_null

hist(bstrap,
     breaks = 30,
     main = "P-value Region (Two-Sided Test)",
     xlab = "Sample Mean",
     col = "lightgray",
     border = "white",
     probability = TRUE)

# Color the extreme regions
hist_data <- hist(bstrap, breaks = 30, plot = FALSE)
extreme_bins <- which(hist_data$mids <= extreme_lower | hist_data$mids >= extreme_upper)
for (i in extreme_bins) {
    rect(hist_data$breaks[i], 0, hist_data$breaks[i+1], hist_data$density[i],
         col = "coral", border = "white")
}

abline(v = c(extreme_lower, extreme_upper), col = "red", lwd = 2, lty = 2)
abline(v = 33.02, col = "darkgreen", lwd = 2, lty = 2)
abline(v = observed_mean, col = "red", lwd = 3)

legend("topright",
       legend = c("H₀: μ = 33.02",
                  paste("Observed:", round(observed_mean, 2)),
                  "Extreme regions"),
       col = c("darkgreen", "red", "coral"),
       lty = c(2, 1, NA),
       lwd = c(2, 3, NA),
       pch = c(NA, NA, 15),
       cex = 0.7)

# Plot 3: QQ plot to check normality of bootstrap distribution
qqnorm(bstrap, 
       main = "Q-Q Plot: Bootstrap Distribution",
       pch = 20,
       col = rgb(0, 0, 1, 0.5))
qqline(bstrap, col = "red", lwd = 2)

# Plot 4: Cumulative distribution
plot(ecdf(bstrap),
     main = "Empirical CDF of Bootstrap Means",
     xlab = "Sample Mean",
     ylab = "Cumulative Probability",
     col = "blue",
     lwd = 2)
abline(v = 33.02, col = "darkgreen", lwd = 2, lty = 2)
abline(v = observed_mean, col = "red", lwd = 2)
abline(h = c(0.025, 0.975), col = "gray", lty = 3)

legend("bottomright",
       legend = c("H₀: μ = 33.02",
                  paste("Observed:", round(observed_mean, 2)),
                  "2.5% / 97.5%"),
       col = c("darkgreen", "red", "gray"),
       lty = c(2, 1, 3),
       lwd = c(2, 2, 1),
       cex = 0.7)

par(mfrow = c(1, 1))

# ============================================================================
# Calculate p-value
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Step 6: Calculate the p-value\n")
cat(strrep("=", 80), "\n\n")

cat("Observed test statistic: X̄ =", observed_mean, "\n")
cat("Null hypothesis value: μ₀ = 33.02\n")
cat("Difference:", observed_mean - 33.02, "\n\n")

cat("The p-value is the probability of getting something more extreme than\n")
cat("what we observed.\n\n")

cat("Notice that", observed_mean, "is", 33.02 - observed_mean, "units away from\n")
cat("the null hypothesis (33.02 -", observed_mean, "=", 33.02 - observed_mean, ")\n\n")

cat("So the p-value is the probability of being more than", 33.02 - observed_mean, "\n")
cat("units away from 33.02 in EITHER direction (two-sided test).\n\n")

cat("For a two-sided test, we need to find:\n")
cat("  P(X̄ <", observed_mean, "OR X̄ >", 33.02 + (33.02 - observed_mean), "| H₀ is true)\n\n")

# Calculate how far our observed mean is from the null hypothesis
distance_from_null <- abs(observed_mean - 33.02)
extreme_lower <- observed_mean
extreme_upper <- 33.02 + distance_from_null

cat("This means we count bootstrap samples where:\n")
cat("  - X̄ <", extreme_lower, "(as low or lower than observed)\n")
cat("  - X̄ >", round(extreme_upper, 2), "(equally far above 33.02)\n\n")

# Count how many bootstrap means are at least as extreme
lower_tail <- sum(bstrap < extreme_lower)
upper_tail <- sum(bstrap > extreme_upper)
extreme_count <- lower_tail + upper_tail

cat("Calculation:\n")
cat("  (sum(bstrap <", extreme_lower, ") + sum(bstrap >", round(extreme_upper, 2), ")) /", n, "\n")
cat("  = (", lower_tail, "+", upper_tail, ") /", n, "\n")

# Calculate p-value
pvalue <- extreme_count / n

cat("  =", extreme_count, "/", n, "\n")
cat("  =", pvalue, "\n\n")

cat("p-value =", pvalue, "\n\n")

# ============================================================================
# Conclusion
# ============================================================================

cat(strrep("=", 80), "\n")
cat("Step 7: Make a Conclusion\n")
cat(strrep("=", 80), "\n\n")

cat("Decision rule: Reject H₀ if p-value < α =", alpha, "\n\n")

if (pvalue < alpha) {
    cat("Result: p-value =", pvalue, "<", alpha, "\n")
    cat("Decision: REJECT H₀\n\n")
    cat("Conclusion:\n")
    cat("  Since our significance level is", alpha, "(5%), we REJECT H₀\n")
    cat("  and conclude that Newcomb's measurements were NOT consistent with\n")
    cat("  the currently accepted figure.\n\n")
    cat("  Interpretation:\n")
    cat("  At the α =", alpha, "significance level, we have sufficient evidence\n")
    cat("  to conclude that the true mean speed differs from the accepted value\n")
    cat("  of 33.02. Newcomb's measurements appear to be systematically lower\n")
    cat("  than the accepted speed of light.\n\n")
    cat("  Possible explanations:\n")
    cat("  - Systematic measurement error in the experimental setup\n")
    cat("  - Equipment calibration issues\n")
    cat("  - The presence of outliers affecting the sample mean\n")
} else {
    cat("Result: p-value =", pvalue, ">=", alpha, "\n")
    cat("Decision: FAIL TO REJECT H₀\n\n")
    cat("Conclusion:\n")
    cat("  Since our significance level is", alpha, "(5%), we FAIL TO REJECT H₀.\n")
    cat("  At the α =", alpha, "significance level, we do not have sufficient\n")
    cat("  evidence to conclude that the true mean speed differs from 33.02.\n")
    cat("  The observed difference could plausibly be due to random sampling\n")
    cat("  variation.\n")
}

cat("\n")
cat(strrep("=", 80), "\n")
cat("What Did We Just Do? The Bootstrap Principle\n")
cat(strrep("=", 80), "\n\n")

cat("This is an example of the BOOTSTRAP HYPOTHESIS TEST:\n\n")

cat("1. We wanted to test if μ = 33.02\n\n")

cat("2. We created a 'fake population' (newspeed) where μ = 33.02 is TRUE\n")
cat("   by shifting our sample\n\n")

cat("3. We repeatedly sampled from this fake population (with replacement)\n")
cat("   to simulate what sample means would look like IF H₀ were true\n\n")

cat("4. We compared our actual observed mean to this distribution to see\n")
cat("   if it's unusually far from 33.02\n\n")

cat("Key insight:\n")
cat("  The BOOTSTRAP uses the sample to represent the population!\n")
cat("  - When we shift the sample, we're creating a population where\n")
cat("    H₀ is exactly true\n")
cat("  - Resampling from this shifted data gives us the sampling\n")
cat("    distribution we need\n")
cat("  - No normal approximation needed!\n")
cat("  - No assumptions about the population distribution!\n\n")

# ============================================================================
# Example: Sleep Study
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("Example: Sleep Study - Bootstrap for Two-Sample Comparison\n")
cat(strrep("=", 80), "\n\n")

cat("The two-sample t-test checks for differences in means according to a\n")
cat("known null distribution.\n")
cat("This is similar to permutation tests, but we'll use bootstrap resampling.\n\n")

cat("Let's resample and generate the sampling distribution under the\n")
cat("bootstrap assumption.\n\n")

# Load the sleep data
cat("Dataset: Student's Sleep Data\n")
cat("Description: Effect of two soporific drugs on sleep increase\n")
cat("  - 10 patients given drug 1\n")
cat("  - 10 patients given drug 2\n")
cat("  - Measured: increase in hours of sleep\n\n")

# Display the data
cat("Sleep data:\n")
print(sleep)
cat("\n")

# Summary by group
cat("Summary statistics by group:\n")
cat("Group 1 (Drug 1):\n")
group1_data <- sleep[sleep$group == 1, "extra"]
cat("  Mean:", mean(group1_data), "\n")
cat("  SD:", sd(group1_data), "\n")
cat("  n:", length(group1_data), "\n\n")

cat("Group 2 (Drug 2):\n")
group2_data <- sleep[sleep$group == 2, "extra"]
cat("  Mean:", mean(group2_data), "\n")
cat("  SD:", sd(group2_data), "\n")
cat("  n:", length(group2_data), "\n\n")

observed_diff <- mean(group1_data) - mean(group2_data)
cat("Observed difference in means (Group 1 - Group 2):", observed_diff, "\n\n")

# Visualize the data
par(mfrow = c(1, 2))

# Boxplot comparison
boxplot(extra ~ group, data = sleep,
        main = "Sleep Increase by Drug",
        xlab = "Group",
        ylab = "Increase in Hours of Sleep",
        col = c("lightblue", "lightcoral"),
        names = c("Drug 1", "Drug 2"))
abline(h = 0, col = "gray", lty = 2)

# Dot plot showing individual observations
stripchart(extra ~ group, data = sleep,
           main = "Individual Sleep Increases",
           xlab = "Group",
           ylab = "Increase in Hours of Sleep",
           vertical = TRUE,
           pch = 19,
           col = c("blue", "red"),
           method = "jitter",
           jitter = 0.1)
abline(h = 0, col = "gray", lty = 2)
points(1, mean(group1_data), pch = 18, cex = 2, col = "darkblue")
points(2, mean(group2_data), pch = 18, cex = 2, col = "darkred")

par(mfrow = c(1, 1))

# ============================================================================
# Bootstrap Resampling for Two-Sample Problem
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Bootstrap Approach\n")
cat(strrep("=", 80), "\n\n")

cat("Strategy:\n")
cat("  1. Resample rows from the sleep dataset with replacement\n")
cat("  2. Calculate the difference in means for each bootstrap sample\n")
cat("  3. Build up the sampling distribution of the difference\n\n")

# Define helper functions
bootstrap.resample <- function(object) {
  sample(object, length(object), replace = TRUE)
}

diff.in.means <- function(df) {
  mean(df[df$group == 1, "extra"]) - mean(df[df$group == 2, "extra"])
}

cat("Functions defined:\n")
cat("  bootstrap.resample <- function(object)\n")
cat("      sample(object, length(object), replace = TRUE)\n\n")
cat("  diff.in.means <- function(df)\n")
cat("      mean(df[df$group == 1, 'extra']) - mean(df[df$group == 2, 'extra'])\n\n")

# Perform bootstrap resampling
cat("Performing 2000 bootstrap replications...\n")
set.seed(123)  # For reproducibility
resample.diffs <- replicate(2000, 
                             diff.in.means(sleep[bootstrap.resample(1:nrow(sleep)), ]))
cat("Bootstrap complete!\n\n")

# Summary of bootstrap distribution
cat("Bootstrap Distribution Summary:\n")
cat("  Mean of bootstrap differences:", mean(resample.diffs), "\n")
cat("  SD of bootstrap differences:", sd(resample.diffs), "\n")
cat("  Min:", min(resample.diffs), "\n")
cat("  Max:", max(resample.diffs), "\n")
cat("  Observed difference:", observed_diff, "\n\n")

# Visualize the bootstrap distribution
par(mfrow = c(1, 1))

# Simple histogram showing the bootstrap sampling distribution
hist(resample.diffs, 
     main = "Bootstrap Sampling Distribution",
     xlab = "Difference in Means (Drug 1 - Drug 2)",
     col = "lightblue",
     border = "white")
abline(v = diff.in.means(sleep), col = 2, lwd = 3)

cat("\nBasic visualization created: Bootstrap sampling distribution with\n")
cat("observed difference marked in red.\n\n")

# Add more detailed visualizations
par(mfrow = c(2, 2))

# Plot 1: Histogram of bootstrap differences
hist(resample.diffs,
     breaks = 40,
     main = "Bootstrap Distribution of\nDifference in Means",
     xlab = "Difference in Means (Drug 1 - Drug 2)",
     col = "lightblue",
     border = "white",
     probability = TRUE)
abline(v = observed_diff, col = "red", lwd = 2)
abline(v = 0, col = "darkgreen", lwd = 2, lty = 2)
lines(density(resample.diffs), col = "blue", lwd = 2)
legend("topright",
       legend = c(paste("Observed:", round(observed_diff, 2)),
                  "No difference (0)",
                  "Bootstrap density"),
       col = c("red", "darkgreen", "blue"),
       lty = c(1, 2, 1),
       lwd = 2,
       cex = 0.7)

# Plot 2: Bootstrap differences with confidence interval
boot_ci_lower <- quantile(resample.diffs, 0.025)
boot_ci_upper <- quantile(resample.diffs, 0.975)

hist(resample.diffs,
     breaks = 40,
     main = "95% Bootstrap Confidence Interval",
     xlab = "Difference in Means (Drug 1 - Drug 2)",
     col = "lightgray",
     border = "white",
     probability = TRUE)

# Highlight the confidence interval region
hist_data <- hist(resample.diffs, breaks = 40, plot = FALSE)
ci_bins <- which(hist_data$mids >= boot_ci_lower & hist_data$mids <= boot_ci_upper)
for (i in ci_bins) {
    rect(hist_data$breaks[i], 0, hist_data$breaks[i+1], hist_data$density[i],
         col = "lightblue", border = "white")
}

abline(v = c(boot_ci_lower, boot_ci_upper), col = "blue", lwd = 2, lty = 2)
abline(v = observed_diff, col = "red", lwd = 2)
abline(v = 0, col = "darkgreen", lwd = 2, lty = 2)

legend("topright",
       legend = c(paste("Observed:", round(observed_diff, 2)),
                  "95% CI bounds",
                  "No difference (0)"),
       col = c("red", "blue", "darkgreen"),
       lty = c(1, 2, 2),
       lwd = 2,
       cex = 0.7)

# Plot 3: QQ plot
qqnorm(resample.diffs,
       main = "Q-Q Plot: Bootstrap Differences",
       pch = 20,
       col = rgb(0, 0, 1, 0.5))
qqline(resample.diffs, col = "red", lwd = 2)

# Plot 4: Cumulative distribution
plot(ecdf(resample.diffs),
     main = "Empirical CDF",
     xlab = "Difference in Means",
     ylab = "Cumulative Probability",
     col = "blue",
     lwd = 2)
abline(v = observed_diff, col = "red", lwd = 2)
abline(v = 0, col = "darkgreen", lwd = 2, lty = 2)
abline(h = c(0.025, 0.975), col = "gray", lty = 3)

par(mfrow = c(1, 1))

# ============================================================================
# Bootstrap Confidence Interval
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Bootstrap 95% Confidence Interval\n")
cat(strrep("=", 80), "\n\n")

cat("Using the percentile method:\n")
cat("  Lower bound (2.5th percentile):", boot_ci_lower, "\n")
cat("  Upper bound (97.5th percentile):", boot_ci_upper, "\n\n")

cat("Interpretation:\n")
cat("  We are 95% confident that the true difference in mean sleep increase\n")
cat("  between Drug 1 and Drug 2 is between", round(boot_ci_lower, 3), "and",
    round(boot_ci_upper, 3), "hours.\n\n")

if (boot_ci_lower > 0) {
    cat("  Since the entire confidence interval is ABOVE 0, we can conclude\n")
    cat("  that Drug 1 appears to increase sleep more than Drug 2.\n")
} else if (boot_ci_upper < 0) {
    cat("  Since the entire confidence interval is BELOW 0, we can conclude\n")
    cat("  that Drug 2 appears to increase sleep more than Drug 1.\n")
} else {
    cat("  Since the confidence interval INCLUDES 0, we cannot conclude\n")
    cat("  that there is a significant difference between the two drugs.\n")
}

cat("\n")

# Compare with traditional t-test
cat(strrep("=", 80), "\n")
cat("Comparison with Traditional Two-Sample t-test\n")
cat(strrep("=", 80), "\n\n")

t_test_result <- t.test(extra ~ group, data = sleep, var.equal = TRUE)
cat("Traditional t-test results:\n")
print(t_test_result)

cat("\nComparison:\n")
cat("  Bootstrap 95% CI: [", round(boot_ci_lower, 3), ",", round(boot_ci_upper, 3), "]\n")
cat("  t-test 95% CI:    [", round(t_test_result$conf.int[1], 3), ",", 
    round(t_test_result$conf.int[2], 3), "]\n\n")

cat("Note: The bootstrap approach:\n")
cat("  - Does NOT assume normality\n")
cat("  - Does NOT assume equal variances\n")
cat("  - Works better with small samples or skewed data\n")
cat("  - Provides empirical sampling distribution\n\n")

# ============================================================================
# Bootstrapping Functions in R
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("Bootstrapping Functions in R\n")
cat(strrep("=", 80), "\n\n")

cat("R has numerous built-in bootstrapping functions, too many to mention.\n")
cat("See the 'boot' library for comprehensive bootstrapping tools.\n\n")

cat("The boot library provides:\n")
cat("  - boot(): Main function for bootstrap resampling\n")
cat("  - boot.ci(): Confidence interval calculations (multiple methods)\n")
cat("  - Various helper functions for specialized bootstrapping\n\n")

# ============================================================================
# Example: Using the boot() Function
# ============================================================================

cat(strrep("=", 80), "\n")
cat("Example: Bootstrap of the Ratio of Means\n")
cat(strrep("=", 80), "\n\n")

cat("Dataset: City data from the boot package\n")
cat("Goal: Bootstrap the ratio of means using the city data\n\n")

# Load the boot library
library(boot)

# Load the city data
data(city)

cat("City data structure:\n")
str(city)
cat("\n")

cat("First few rows:\n")
print(head(city))
cat("\n")

# Define the statistic function
# This calculates the ratio of weighted sums
ratio <- function(d, w) sum(d$x * w)/sum(d$u * w)

cat("Statistic function defined:\n")
cat("  ratio <- function(d, w) sum(d$x * w)/sum(d$u * w)\n\n")

cat("This function:\n")
cat("  - Takes data 'd' and weights 'w' as inputs\n")
cat("  - Calculates the ratio of weighted sums\n")
cat("  - d$x and d$u are variables from the city dataset\n")
cat("  - w contains bootstrap weights (frequency of each observation)\n\n")

# Perform bootstrap
cat("Performing bootstrap with 1000 replications...\n")
cat("  boot(city, ratio, R=1000, stype='w')\n\n")
results <- boot(city, ratio, R = 1000, stype = "w")

cat("Bootstrap complete!\n\n")

# Display results
cat(strrep("=", 80), "\n")
cat("Bootstrap Results\n")
cat(strrep("=", 80), "\n\n")

print(results)

cat("\n\n")
cat("Interpretation:\n")
cat("  original:    The observed statistic from the original data\n")
cat("               (", results$t0, ")\n")
cat("  bias:        Estimated bias of the bootstrap estimator\n")
cat("               (", mean(results$t) - results$t0, ")\n")
cat("  std. error:  Standard error estimated from bootstrap distribution\n")
cat("               (", sd(results$t), ")\n\n")

# Visualize bootstrap distribution
par(mfrow = c(2, 2))

# Plot 1: Histogram of bootstrap replicates
hist(results$t,
     breaks = 30,
     main = "Bootstrap Distribution of Ratio",
     xlab = "Ratio of Means",
     col = "lightblue",
     border = "white",
     probability = TRUE)
abline(v = results$t0, col = "red", lwd = 3)
lines(density(results$t), col = "blue", lwd = 2)
legend("topright",
       legend = c(paste("Original:", round(results$t0, 3)),
                  "Bootstrap density"),
       col = c("red", "blue"),
       lty = 1,
       lwd = c(3, 2),
       cex = 0.8)

# Plot 2: Q-Q plot
qqnorm(results$t,
       main = "Q-Q Plot: Bootstrap Ratios",
       pch = 20,
       col = rgb(0, 0, 1, 0.5))
qqline(results$t, col = "red", lwd = 2)

# Plot 3: Bootstrap replicates over index
plot(results$t,
     type = "l",
     main = "Bootstrap Replicates Sequence",
     xlab = "Bootstrap Replicate Number",
     ylab = "Ratio",
     col = "blue")
abline(h = results$t0, col = "red", lwd = 2, lty = 2)

# Plot 4: Empirical CDF
plot(ecdf(results$t),
     main = "Empirical CDF",
     xlab = "Ratio of Means",
     ylab = "Cumulative Probability",
     col = "blue",
     lwd = 2)
abline(v = results$t0, col = "red", lwd = 2)

par(mfrow = c(1, 1))

# ============================================================================
# Bootstrap Confidence Intervals
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Bootstrap Confidence Intervals using boot.ci()\n")
cat(strrep("=", 80), "\n\n")

cat("The boot.ci() function provides several types of confidence intervals:\n")
cat("  - norm:     Normal approximation interval\n")
cat("  - basic:    Basic bootstrap interval\n")
cat("  - perc:     Percentile interval\n")
cat("  - bca:      Bias-corrected and accelerated (BCa) interval\n")
cat("  - stud:     Studentized interval\n\n")

cat("We'll use the BCa (bias-corrected and accelerated) method:\n")
cat("  boot.ci(results, type='bca')\n\n")

# Calculate BCa confidence interval
ci_results <- boot.ci(results, type = "bca")

cat(strrep("=", 80), "\n\n")
print(ci_results)
cat("\n")
cat(strrep("=", 80), "\n\n")

cat("Interpretation:\n")
cat("  Level:   Confidence level (95%)\n")
cat("  BCa:     Bias-corrected and accelerated interval\n")
cat("           95% CI: [", ci_results$bca[4], ",", ci_results$bca[5], "]\n\n")

cat("Conclusion:\n")
cat("  We are 95% confident that the true ratio of means is between\n")
cat("  ", round(ci_results$bca[4], 3), "and", round(ci_results$bca[5], 3), ".\n\n")

cat("  Calculations and Intervals on Original Scale\n\n")

cat("Advantages of BCa intervals:\n")
cat("  - Corrects for bias in the bootstrap distribution\n")
cat("  - Adjusts for skewness (acceleration)\n")
cat("  - Generally more accurate than percentile intervals\n")
cat("  - Better coverage properties for skewed distributions\n\n")

# Compare different CI methods
cat(strrep("=", 80), "\n")
cat("Comparison of Different Bootstrap CI Methods\n")
cat(strrep("=", 80), "\n\n")

ci_all <- boot.ci(results, type = c("norm", "basic", "perc", "bca"))
print(ci_all)

cat("\n\nNote: Different methods may give different intervals,\n")
cat("especially for skewed distributions or small sample sizes.\n")
cat("BCa is often preferred for its superior theoretical properties.\n\n")

# ============================================================================
# Bootstrapping a Single Statistic
# ============================================================================

cat("\n\n")
cat(strrep("=", 80), "\n")
cat("Bootstrapping a Single Statistic: R-squared Example\n")
cat(strrep("=", 80), "\n\n")

cat("Goal: Use the bootstrap to generate a 95% confidence interval for R²\n\n")

cat("Context:\n")
cat("  - Linear regression of miles per gallon (mpg) on car weight (wt)\n")
cat("    and displacement (disp)\n")
cat("  - Data source: mtcars\n")
cat("  - The bootstrapped confidence interval is based on 1000 replications\n\n")

# Explore the mtcars data
cat("Dataset: mtcars\n")
cat("Structure:\n")
str(mtcars[, c("mpg", "wt", "disp")])
cat("\n")

cat("Summary statistics:\n")
summary(mtcars[, c("mpg", "wt", "disp")])
cat("\n")

# Fit the original model to see the R-squared
original_model <- lm(mpg ~ wt + disp, data = mtcars)
cat("Original linear regression model:\n")
cat("  Formula: mpg ~ wt + disp\n\n")
print(summary(original_model))
cat("\n")

cat("Original R²:", summary(original_model)$r.squared, "\n")
cat("This means", round(summary(original_model)$r.squared * 100, 2), 
    "% of the variance in mpg is explained by wt and disp.\n\n")

# Visualize the data
par(mfrow = c(1, 2))

# Scatterplot: mpg vs wt
plot(mtcars$wt, mtcars$mpg,
     main = "MPG vs Weight",
     xlab = "Weight (1000 lbs)",
     ylab = "Miles per Gallon",
     pch = 19,
     col = "blue")
abline(lm(mpg ~ wt, data = mtcars), col = "red", lwd = 2)

# Scatterplot: mpg vs disp
plot(mtcars$disp, mtcars$mpg,
     main = "MPG vs Displacement",
     xlab = "Displacement (cu.in.)",
     ylab = "Miles per Gallon",
     pch = 19,
     col = "blue")
abline(lm(mpg ~ disp, data = mtcars), col = "red", lwd = 2)

par(mfrow = c(1, 1))

# ============================================================================
# Bootstrap the R-squared Statistic
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Bootstrap Procedure\n")
cat(strrep("=", 80), "\n\n")

cat("Step 1: Define the statistic function\n\n")

# Define the R-squared function for bootstrapping
rsq <- function(formula, data, indices) {
  d <- data[indices, ]  
  fit <- lm(formula, data = d)
  return(summary(fit)$r.square)
} 

cat("Function defined:\n")
cat("  rsq <- function(formula, data, indices) {\n")
cat("    d <- data[indices, ]  \n")
cat("    fit <- lm(formula, data = d)\n")
cat("    return(summary(fit)$r.square)\n")
cat("  }\n\n")

cat("This function:\n")
cat("  - Takes a formula, dataset, and bootstrap indices\n")
cat("  - Subsets the data according to the bootstrap indices\n")
cat("  - Fits a linear model to the bootstrap sample\n")
cat("  - Returns the R² value from the model\n\n")

cat("Step 2: Perform bootstrap resampling\n\n")
cat("Running bootstrap with 1000 replications...\n")
cat("  boot(data=mtcars, statistic=rsq, R=1000, formula=mpg~wt+disp)\n\n")

set.seed(456)  # For reproducibility
results <- boot(data = mtcars, statistic = rsq, 
                R = 1000, formula = mpg ~ wt + disp)

cat("Bootstrap complete!\n\n")

# Display results
cat(strrep("=", 80), "\n")
cat("Bootstrap Results\n")
cat(strrep("=", 80), "\n\n")

print(results)

cat("\n\n")
cat("Interpretation:\n")
cat("  original:    R² from the original model =", results$t0, "\n")
cat("  bias:        Bootstrap estimate of bias =", mean(results$t) - results$t0, "\n")
cat("               (Small bias suggests R² is nearly unbiased)\n")
cat("  std. error:  Standard error of R² =", sd(results$t), "\n")
cat("               (Measure of variability in R² estimates)\n\n")

# Built-in plot method for boot objects
cat(strrep("=", 80), "\n")
cat("Built-in Diagnostic Plots\n")
cat(strrep("=", 80), "\n\n")

cat("The boot package provides a plot() method that creates diagnostic plots:\n")
cat("  plot(results)\n\n")

# Create the built-in boot plots
plot(results)

cat("\nThese plots show:\n")
cat("  1. Histogram of bootstrap replicates (left)\n")
cat("     - Shows the distribution of R² values across bootstrap samples\n")
cat("     - Red vertical line marks the original statistic\n\n")
cat("  2. Q-Q plot (right)\n")
cat("     - Compares bootstrap distribution to normal distribution\n")
cat("     - Points on the line indicate approximate normality\n")
cat("     - Deviations suggest non-normal distribution\n\n")

# Additional custom visualizations
cat(strrep("=", 80), "\n")
cat("Custom Visualizations\n")
cat(strrep("=", 80), "\n\n")

# Visualize the bootstrap distribution of R-squared
par(mfrow = c(2, 2))

# Plot 1: Histogram of bootstrap R² values
hist(results$t,
     breaks = 30,
     main = "Bootstrap Distribution of R²",
     xlab = "R-squared",
     col = "lightgreen",
     border = "white",
     probability = TRUE,
     xlim = c(min(results$t) - 0.05, max(results$t) + 0.05))
abline(v = results$t0, col = "red", lwd = 3)
lines(density(results$t), col = "darkgreen", lwd = 2)
legend("topleft",
       legend = c(paste("Original R² =", round(results$t0, 4)),
                  "Bootstrap density"),
       col = c("red", "darkgreen"),
       lty = 1,
       lwd = c(3, 2),
       cex = 0.8)

# Plot 2: Histogram with confidence interval region
boot_ci_lower <- quantile(results$t, 0.025)
boot_ci_upper <- quantile(results$t, 0.975)

hist(results$t,
     breaks = 30,
     main = "95% Percentile CI for R²",
     xlab = "R-squared",
     col = "lightgray",
     border = "white",
     probability = TRUE)

# Highlight CI region
hist_data <- hist(results$t, breaks = 30, plot = FALSE)
ci_bins <- which(hist_data$mids >= boot_ci_lower & hist_data$mids <= boot_ci_upper)
for (i in ci_bins) {
    rect(hist_data$breaks[i], 0, hist_data$breaks[i+1], hist_data$density[i],
         col = "lightgreen", border = "white")
}

abline(v = c(boot_ci_lower, boot_ci_upper), col = "blue", lwd = 2, lty = 2)
abline(v = results$t0, col = "red", lwd = 2)

legend("topleft",
       legend = c(paste("Original:", round(results$t0, 3)),
                  "95% CI bounds"),
       col = c("red", "blue"),
       lty = c(1, 2),
       lwd = 2,
       cex = 0.8)

# Plot 3: Q-Q plot
qqnorm(results$t,
       main = "Q-Q Plot: Bootstrap R²",
       pch = 20,
       col = rgb(0, 0.5, 0, 0.5))
qqline(results$t, col = "red", lwd = 2)

# Plot 4: Bootstrap replicates sequence
plot(results$t,
     type = "l",
     main = "Bootstrap R² Sequence",
     xlab = "Bootstrap Replicate",
     ylab = "R-squared",
     col = "darkgreen")
abline(h = results$t0, col = "red", lwd = 2, lty = 2)
abline(h = mean(results$t), col = "blue", lwd = 1, lty = 3)

par(mfrow = c(1, 1))

# ============================================================================
# Bootstrap Confidence Interval for R-squared
# ============================================================================

cat("\n")
cat(strrep("=", 80), "\n")
cat("Bootstrap Confidence Interval for R²\n")
cat(strrep("=", 80), "\n\n")

cat("Calculating BCa confidence interval...\n")
cat("  boot.ci(results, type='bca')\n\n")

# Calculate confidence interval
ci_rsq <- boot.ci(results, type = "bca")

cat(strrep("=", 80), "\n\n")
print(ci_rsq)
cat("\n")
cat(strrep("=", 80), "\n\n")

cat("Results:\n")
cat("  Confidence Level: 95%\n")
cat("  Method: BCa (Bias-corrected and accelerated)\n")
cat("  Interval: [", round(ci_rsq$bca[4], 4), ",", round(ci_rsq$bca[5], 4), "]\n\n")

cat("Interpretation:\n")
cat("  We are 95% confident that the true R² (proportion of variance\n")
cat("  explained by weight and displacement) is between",
    round(ci_rsq$bca[4], 4), "and", round(ci_rsq$bca[5], 4), ".\n\n")

cat("  This means we're 95% confident that between",
    round(ci_rsq$bca[4] * 100, 2), "% and",
    round(ci_rsq$bca[5] * 100, 2), "% of the\n")
cat("  variance in fuel efficiency is explained by these predictors.\n\n")

cat("  Calculations and Intervals on Original Scale\n")

if (grepl("unstable", paste(capture.output(print(ci_rsq)), collapse = " "), ignore.case = TRUE)) {
    cat("\n  Note: 'Some BCa intervals may be unstable' warning suggests:\n")
    cat("    - The bootstrap distribution may be skewed\n")
    cat("    - Consider using more bootstrap replications (R > 1000)\n")
    cat("    - Alternative: use 'perc' or 'basic' interval types\n")
}

cat("\n")
cat("Why bootstrap for R²?\n")
cat("  - R² distribution is bounded [0, 1] and often skewed\n")
cat("  - Traditional confidence intervals assume normality\n")
cat("  - Bootstrap provides more accurate intervals for bounded statistics\n")
cat("  - No assumptions about the sampling distribution needed\n\n")
