# Agenda
# - Like Ordinary Monte Carlo (OMC), but better?
# - SLLN and Markov chain CLT
# - Variance estimation
# - AR(1) example
# - Metropolis-Hastings algorithm (with an exercise)

# Markov chain Monte Carlo
# The central limit theorem (CLT) for Markov chains says
# sqrt(n) * (μ_hat_n - E_π g(X_i)) → N(0, σ²)
#
# where
# σ² = Var g(X_i) + 2 * Σ_{k=1}^∞ Cov[g(X_i), g(X_{i+k})]
#
# CLT holds if E_π |g(X_i)|^{2+ε} < ∞
# and the Markov chain is geometrically ergodic
#
# Can estimate σ² in various ways
#
# Verifying such a mixing condition is generally very challenging
#
# Nevertheless, we expect the CLT to hold in practice when using a smart sampler

# Batch means
# In order to make MCMC practical, need a method to estimate the variance σ²
# in the CLT, then can proceed just like in OMC
#
# If σ_hat² is a consistent estimate of σ², then an asymptotic 95% confidence interval for μ_g is
# μ_hat_n ± 1.96 * σ_hat / sqrt(n)
#
# The method of batch means estimates the asymptotic variance for a stationary time series

# Example: AR(1)
# Consider the Markov chain such that
# X_i = ρ * X_{i-1} + ε_i
# where ε_i ~ iid N(0, 1)
#
# Consider X_1 = 0, ρ = 0.95, and estimating E_π X = 0
#
# Run until
# w_n = 2 * z_{0.975} * σ_hat / sqrt(n) ≤ 0.2
# where σ_hat is calculated using batch means

# Example: AR(1)
# The following will provide an observation from the MC 1 step ahead

ar1 <- function(m, rho, tau) {
  rho*m + rnorm(1, 0, tau)
}

# Next, we add to this function so that we can give it a Markov chain 
# and the result will be p observations from the Markov chain

ar1.gen <- function(mc, p, rho, tau, q=1) {
  loc <- length(mc)
  junk <- double(p)
  mc <- append(mc, junk)
  
  for(i in 1:p){
    j <- i+loc-1
    mc[(j+1)] <- ar1(mc[j], rho, tau)
  }
  return(mc)
}

# Example: AR(1)
set.seed(20)
library(mcmcse)
## mcmcse: Monte Carlo Standard Errors for MCMC
## Version 1.2-1 created on 2016-03-24.
## copyright (c) 2012, James M. Flegal, University of California,Riverside
##                     John Hughes, University of Minnesota
##                     Dootika Vats, University of Minnesota
##  For citation information, type citation("mcmcse").
##  Type help("mcmcse-package") to get started.
tau <- 1
rho <- .95
out <- 0
eps <- 0.1
start <- 1000
r <- 1000

# Example: AR(1)
out <- ar1.gen(out, start, rho, tau)
MCSE <- mcse(out)$se
N <- length(out)
t <- qt(.975, (floor(sqrt(N) - 1)))
muhat <- mean(out)
check <- MCSE * t

while(eps < check) {
  out <- ar1.gen(out, r, rho, tau)
  MCSE <- append(MCSE, mcse(out)$se)
  N <- length(out)
  t <- qt(.975, (floor(sqrt(N) - 1)))
  muhat <- append(muhat, mean(out))
  check <- MCSE[length(MCSE)] * t
}

N <- seq(start, length(out), r) 
t <- qt(.975, (floor(sqrt(N) - 1)))
half <- MCSE * t
sigmahat <- MCSE*sqrt(N)
N <- seq(start, length(out), r) / 1000

# Example: AR(1)
plot(N, muhat, main="Estimates of the Mean", xlab="Iterations (in 1000's)")
points(N, muhat, type="l", col="red") ; abline(h=0, lwd=3)
legend("bottomright", legend=c("Observed", "Actual"), lty=c(1,1), col=c(2,1), lwd=c(1,3))

# Example: AR(1)
plot(N, sigmahat, main="Estimates of Sigma", xlab="Iterations (in 1000's)")
points(N, sigmahat, type="l", col="red"); abline(h=20, lwd=3)
legend("bottomright", legend=c("Observed", "Actual"), lty=c(1,1), col=c(2,1), lwd=c(1,3))

# Example: AR(1)
plot(N, 2*half, main="Calculated Interval Widths", xlab="Iterations (in 1000's)", ylab="Width", ylim=c(0, 1.8))
points(N, 2*half, type="l", col="red"); abline(h=0.2, lwd=3)
legend("topright", legend=c("Observed", "Cut-off"), lty=c(1,1), col=c(2,1), lwd=c(1,3))

# Markov chain Monte Carlo
# MCMC methods are used most often in Bayesian inference where the equilibrium 
# (invariant, stationary) distribution is a posterior distribution
#
# Challenge lies in construction of a suitable Markov chain with f
# as its stationary distribution
#
# A key problem is we only get to observe t observations from {X_t},
# which are serially dependent

# Other questions to consider
# - How good are my MCMC estimators?
# - How long to run my Markov chain simulation?
# - How to compare MCMC samplers?
# - What to do in high-dimensional settings?

# Metropolis-Hastings algorithm
# Setting X_0 = x_0 (somehow), the Metropolis-Hastings algorithm generates X_{t+1}
# given X_t = x_t as follows:
#
# 1. Sample a candidate value X* ~ g(·|x_t) where g is the proposal distribution
#
# 2. Compute the MH ratio R(x_t, X*), where
#    R(x_t, X*) = [f(x*) * g(x_t|x*)] / [f(x_t) * g(x*|x_t)]
#
# 3. Set
#    X_{t+1} = { x*  with probability min{R(x_t, X*), 1}
#              { x_t otherwise

# Metropolis-Hastings algorithm
# Irreducibility and aperiodicity depend on the choice of g, these must be checked
# Performance (finite sample) depends on the choice of g also, be careful

# Independence chains
# Suppose g(x*|x_t) = g(x*), this yields an independence chain since the 
# proposal does not depend on the current state
#
# In this case, the MH ratio is
# R(x_t, X*) = [f(x*) * g(x_t)] / [f(x_t) * g(x*)]
#
# and the resulting Markov chain will be irreducible and aperiodic if g > 0 where f > 0
#
# A good envelope function g should resemble f, but should cover f in the tails

# Random walk chains
# Generate X* such that ε ~ h(·) and set X* = X_t + ε, then g(x*|x_t) = h(x* - x_t)
#
# Common choices of h(·) are symmetric zero mean random variables with a scale parameter,
# e.g. a Uniform(-a, a), Normal(0, σ²), c * T_ν, ...
#
# For symmetric zero mean random variables, the MH ratio is
# R(x_t, X*) = f(x*) / f(x_t)
#
# If the support of f is connected and h is positive in a neighborhood of 0,
# then the chain is irreducible and aperiodic.

# Example: Markov chain basics
# Exercise: Suppose f ~ Exp(1)
#
# 1. Write an independence MH sampler with g ~ Exp(θ)
#    Show R(x_t, X*) = exp{(x_t - x*)(1 - θ)}
#    Generate 1000 draws from f with θ ∈ {1/2, 1, 2}
#
# 2. Write a random walk MH sampler with h ~ N(0, σ²)
#    Show R(x_t, X*) = exp{x_t - x*} * I(x* > 0)
#    Generate 1000 draws from f with σ ∈ {0.2, 1, 5}
#
# 3. In general, do you prefer an independence chain or a random walk MH sampler? Why?
#
# 4. Implement the fixed-width stopping rule for your preferred chain

# Example: Markov chain basics
# Independence Metropolis sampler with Exp(θ) proposal

ind.chain <- function(x, n, theta = 1) {
  ## if theta = 1, then this is an iid sampler
  m <- length(x)
  x <- append(x, double(n))
  for(i in (m+1):length(x)){
    x.prime <- rexp(1, rate=theta)
    u <- exp((x[(i-1)]-x.prime)*(1-theta))
    if(runif(1) < u)
      x[i] <- x.prime
    else
      x[i] <- x[(i-1)]
  }
  return(x)
}

# Example: Markov chain basics
# Random Walk Metropolis sampler with N(0, σ) proposal

rw.chain <- function(x, n, sigma = 1) {
  m <- length(x)
  x <- append(x, double(n))
  for(i in (m+1):length(x)){
    x.prime <- x[(i-1)] + rnorm(1, sd = sigma)
    u <- exp((x[(i-1)]-x.prime))
    u
    if(runif(1) < u && x.prime > 0)
      x[i] <- x.prime
    else
      x[i] <- x[(i-1)]
  }
  return(x)
}

# Example: Markov chain basics
trial0 <- ind.chain(1, 500, 1)
trial1 <- ind.chain(1, 500, 2)
trial2 <- ind.chain(1, 500, 1/2)
rw1 <- rw.chain(1, 500, .2)
rw2 <- rw.chain(1, 500, 1)
rw3 <- rw.chain(1, 500, 5)

# ============================================================================
# SOLUTION TO EXERCISE

# ============================================================================


# Part 1: Independence MH sampler with g ~ Exp(θ)
# ------------------------------------------------
# Target: f(x) = exp(-x) for x > 0 (Exp(1))
# Proposal: g(x|θ) = θ * exp(-θ*x) for x > 0 (Exp(θ))
#
# Derivation of MH ratio:
# R(x_t, x*) = [f(x*) * g(x_t|x*)] / [f(x_t) * g(x*|x_t)]
#            = [exp(-x*) * θ*exp(-θ*x_t)] / [exp(-x_t) * θ*exp(-θ*x*)]
#            = exp(-x* + θ*x_t) / exp(-x_t + θ*x*)
#            = exp(-x* + θ*x_t + x_t - θ*x*)
#            = exp(x_t(1 + θ) - x*(1 + θ))
#            = exp((x_t - x*)(1 - θ))  ✓

# Independence MH sampler function
independence_mh <- function(n_iter, theta, x0 = 1) {
  x <- numeric(n_iter)
  x[1] <- x0
  accept_count <- 0
  
  for (i in 2:n_iter) {
    # Propose from Exp(theta)
    x_star <- rexp(1, rate = theta)
    
    # Compute MH ratio
    R <- exp((x[i-1] - x_star) * (1 - theta))
    
    # Accept/reject
    if (runif(1) < R) {
      x[i] <- x_star
      accept_count <- accept_count + 1
    } else {
      x[i] <- x[i-1]
    }
  }
  
  list(chain = x, acceptance_rate = accept_count / (n_iter - 1))
}

# Generate 1000 draws for θ ∈ {1/2, 1, 2}
set.seed(123)
n <- 1000

indep_theta_0.5 <- independence_mh(n, theta = 0.5)
indep_theta_1.0 <- independence_mh(n, theta = 1.0)
indep_theta_2.0 <- independence_mh(n, theta = 2.0)

cat("Independence MH Acceptance Rates:\n")
cat("θ = 0.5:", indep_theta_0.5$acceptance_rate, "\n")
cat("θ = 1.0:", indep_theta_1.0$acceptance_rate, "\n")
cat("θ = 2.0:", indep_theta_2.0$acceptance_rate, "\n\n")

# Part 2: Random Walk MH sampler with h ~ N(0, σ²)
# -------------------------------------------------
# Target: f(x) = exp(-x) for x > 0
# Proposal: X* = X_t + ε where ε ~ N(0, σ²)
#
# Derivation of MH ratio:
# For symmetric proposals, g(x*|x_t) = g(x_t|x*)
# R(x_t, x*) = f(x*) / f(x_t)
#            = exp(-x*) / exp(-x_t)  if x* > 0
#            = exp(x_t - x*) * I(x* > 0)  ✓

# Random Walk MH sampler function
random_walk_mh <- function(n_iter, sigma, x0 = 1) {
  x <- numeric(n_iter)
  x[1] <- x0
  accept_count <- 0
  
  for (i in 2:n_iter) {
    # Propose from random walk
    x_star <- x[i-1] + rnorm(1, mean = 0, sd = sigma)
    
    # Compute MH ratio (only accept if x_star > 0)
    if (x_star > 0) {
      R <- exp(x[i-1] - x_star)
      
      # Accept/reject
      if (runif(1) < R) {
        x[i] <- x_star
        accept_count <- accept_count + 1
      } else {
        x[i] <- x[i-1]
      }
    } else {
      x[i] <- x[i-1]  # Reject if x_star <= 0
    }
  }
  
  list(chain = x, acceptance_rate = accept_count / (n_iter - 1))
}

# Generate 1000 draws for σ ∈ {0.2, 1, 5}
set.seed(123)

rw_sigma_0.2 <- random_walk_mh(n, sigma = 0.2)
rw_sigma_1.0 <- random_walk_mh(n, sigma = 1.0)
rw_sigma_5.0 <- random_walk_mh(n, sigma = 5.0)

cat("Random Walk MH Acceptance Rates:\n")
cat("σ = 0.2:", rw_sigma_0.2$acceptance_rate, "\n")
cat("σ = 1.0:", rw_sigma_1.0$acceptance_rate, "\n")
cat("σ = 5.0:", rw_sigma_5.0$acceptance_rate, "\n\n")

# Visualize the chains
par(mfrow = c(2, 3))

# Independence chains
plot(indep_theta_0.5$chain, type = "l", main = "Independence: θ = 0.5", 
     ylab = "X", xlab = "Iteration")
plot(indep_theta_1.0$chain, type = "l", main = "Independence: θ = 1.0", 
     ylab = "X", xlab = "Iteration")
plot(indep_theta_2.0$chain, type = "l", main = "Independence: θ = 2.0", 
     ylab = "X", xlab = "Iteration")

# Random walk chains
plot(rw_sigma_0.2$chain, type = "l", main = "Random Walk: σ = 0.2", 
     ylab = "X", xlab = "Iteration")
plot(rw_sigma_1.0$chain, type = "l", main = "Random Walk: σ = 1.0", 
     ylab = "X", xlab = "Iteration")
plot(rw_sigma_5.0$chain, type = "l", main = "Random Walk: σ = 5.0", 
     ylab = "X", xlab = "Iteration")

# Compare histograms with true Exp(1) distribution
par(mfrow = c(2, 3))
hist(indep_theta_0.5$chain, probability = TRUE, main = "Independence: θ = 0.5", 
     xlab = "X", breaks = 30)
curve(dexp(x, 1), add = TRUE, col = "red", lwd = 2)

hist(indep_theta_1.0$chain, probability = TRUE, main = "Independence: θ = 1.0", 
     xlab = "X", breaks = 30)
curve(dexp(x, 1), add = TRUE, col = "red", lwd = 2)

hist(indep_theta_2.0$chain, probability = TRUE, main = "Independence: θ = 2.0", 
     xlab = "X", breaks = 30)
curve(dexp(x, 1), add = TRUE, col = "red", lwd = 2)

hist(rw_sigma_0.2$chain, probability = TRUE, main = "Random Walk: σ = 0.2", 
     xlab = "X", breaks = 30)
curve(dexp(x, 1), add = TRUE, col = "red", lwd = 2)

hist(rw_sigma_1.0$chain, probability = TRUE, main = "Random Walk: σ = 1.0", 
     xlab = "X", breaks = 30)
curve(dexp(x, 1), add = TRUE, col = "red", lwd = 2)

hist(rw_sigma_5.0$chain, probability = TRUE, main = "Random Walk: σ = 5.0", 
     xlab = "X", breaks = 30)
curve(dexp(x, 1), add = TRUE, col = "red", lwd = 2)

par(mfrow = c(1, 1))

# Part 3: Preference Discussion
# ------------------------------
cat("\n=== Part 3: Independence vs Random Walk ===\n")
cat("For this Exp(1) example:\n\n")
cat("Independence Chain (θ = 1):\n")
cat("- Acceptance rate:", indep_theta_1.0$acceptance_rate, "\n")
cat("- Proposal matches target exactly, optimal performance\n")
cat("- When θ = 1, every proposal is accepted (R = 1 always)\n\n")

cat("Random Walk Chain:\n")
cat("- σ = 0.2: High acceptance (", rw_sigma_0.2$acceptance_rate, 
    ") but slow mixing (small steps)\n")
cat("- σ = 1.0: Moderate acceptance (", rw_sigma_1.0$acceptance_rate, 
    ") with good mixing\n")
cat("- σ = 5.0: Low acceptance (", rw_sigma_5.0$acceptance_rate, 
    ") due to large steps, many rejected\n\n")

cat("PREFERENCE:\n")
cat("- If we know the target well: Independence chain is better (when g ≈ f)\n")
cat("- In general: Random walk is more robust and easier to tune\n")
cat("- For this problem: Random walk with σ ≈ 1 is preferred as it doesn't\n")
cat("  require knowing the exact form of the target distribution\n\n")

# Part 4: Fixed-width stopping rule (using random walk with σ = 1)
# -----------------------------------------------------------------
cat("\n=== Part 4: Fixed-Width Stopping Rule ===\n")

# Using random walk MH with σ = 1 and mcmcse package
library(mcmcse)

random_walk_mh_stopping <- function(sigma, target_width = 0.2, 
                                     start_n = 1000, batch_n = 500, 
                                     max_iter = 50000, x0 = 1) {
  x <- x0
  chain <- numeric(start_n)
  chain[1] <- x0
  
  # Initial run
  for (i in 2:start_n) {
    x_star <- x + rnorm(1, mean = 0, sd = sigma)
    if (x_star > 0) {
      R <- exp(x - x_star)
      if (runif(1) < R) {
        x <- x_star
      }
    }
    chain[i] <- x
  }
  
  # Check stopping criterion
  mcse_result <- mcse(chain)
  se <- mcse_result$se
  n <- length(chain)
  half_width <- qnorm(0.975) * se
  
  cat("Initial n =", n, ", Mean =", mean(chain), 
      ", Half-width =", half_width, "\n")
  
  # Continue until criterion met
  while (half_width > target_width / 2 && n < max_iter) {
    # Generate more samples
    new_samples <- numeric(batch_n)
    for (i in 1:batch_n) {
      x_star <- x + rnorm(1, mean = 0, sd = sigma)
      if (x_star > 0) {
        R <- exp(x - x_star)
        if (runif(1) < R) {
          x <- x_star
        }
      }
      new_samples[i] <- x
    }
    
    chain <- c(chain, new_samples)
    mcse_result <- mcse(chain)
    se <- mcse_result$se
    n <- length(chain)
    half_width <- qnorm(0.975) * se
    
    cat("n =", n, ", Mean =", mean(chain), 
        ", Half-width =", half_width, "\n")
  }
  
  list(chain = chain, 
       final_mean = mean(chain),
       final_se = se,
       final_half_width = half_width,
       final_n = n,
       converged = half_width <= target_width / 2)
}

set.seed(456)
result <- random_walk_mh_stopping(sigma = 1.0, target_width = 0.2)

cat("\n=== Final Results ===\n")
cat("Target half-width:", 0.1, "\n")
cat("Achieved half-width:", result$final_half_width, "\n")
cat("Final sample size:", result$final_n, "\n")
cat("Final mean estimate:", result$final_mean, "\n")
cat("True mean (Exp(1)):", 1.0, "\n")
cat("95% CI: [", result$final_mean - result$final_half_width, ",", 
    result$final_mean + result$final_half_width, "]\n")
cat("Converged:", result$converged, "\n")






