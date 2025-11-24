# Agenda
# - Markov chain Monte Carlo, again
# - Gibbs sampling
# - Output analysis for MCMC
# - Convergence diagnostics
# - Examples: Capture-recapture and toy example

# Gibbs Sampling
# 1. Select starting values x_0 and set t = 0
# 2. Generate in turn (deterministic scan Gibbs sampler)
#    x^(1)_{t+1} ~ f(x^(1) | x^(-1)_t)
#    x^(2)_{t+1} ~ f(x^(2) | x^(1)_{t+1}, x^(3)_t, ..., x^(p)_t)
#    x^(3)_{t+1} ~ f(x^(3) | x^(1)_{t+1}, x^(2)_{t+1}, x^(4)_t, ..., x^(p)_t)
#    ...
#    x^(p)_{t+1} ~ f(x^(p) | x^(-p)_{t+1})
# 3. Increment t and go to Step 2

# Gibbs Sampling
# Common to have one or more components not available in closed form
# Then one can just use a MH sampler for those components known as a 
# Metropolis within Gibbs or Hybrid Gibbs sampling
# Common to "block" groups of random variables

# Example: Capture-recapture
# First, we can write the data into R

captured <- c(30, 22, 29, 26, 31, 32, 35)
new.captures <- c(30, 8, 17, 7, 9, 8, 5)
total.r <- sum(new.captures)

# Example: Capture-recapture
# The following R code implements the Gibbs sampler

gibbs.chain <- function(n, N.start = 94, alpha.start = rep(.5,7)) {
    output <- matrix(0, nrow=n, ncol=8)
    for(i in 1:n){
        neg.binom.prob <- 1 - prod(1-alpha.start)
        N.new <- rnbinom(1, 85, neg.binom.prob) + total.r
        beta1 <- captured + .5
        beta2 <- N.new - captured + .5
        alpha.new <- rbeta(7, beta1, beta2)
        output[i,] <- c(N.new, alpha.new)
        N.start <- N.new
        alpha.start <- alpha.new    
    }
    return(output)
}

# MCMC output analysis
# How can we tell if the chain is mixing well?
#
# - Trace plots or time-series plots
# - Autocorrelation plots
# - Plot of estimate versus Markov chain sample size
# - Effective sample size (ESS)
#   ESS(n) = n / (1 + 2 * Σ_{k=1}^∞ ρ_k(g))
#   where ρ_k(g) is the autocorrelation of lag k for g
#
# Alternative, ESS can be written as
#   ESS(n) = n * σ² / Var g
#   where σ² is the asymptotic variance from a Markov chain CLT

# Example: Capture-recapture
# Then we consider some preliminary simulations to ensure the chain is mixing well

trial <- gibbs.chain(1000)
plot.ts(trial[,1], main = "Trace Plot for N")
for(i in 1:7){
    plot.ts(trial[,(i+1)], main = paste("Alpha", i))
    }

acf(trial[,1], main = "Lag Plot for N")
for(i in 1:7){
    acf(trial[,(i+1)], main = paste("Lag Alpha", i))
    }

# Example: Capture-recapture
# Now for a more complete simulation to estimate posterior means and a 90% Bayesian credible region

sim <- gibbs.chain(10000)
N <- sim[,1]
alpha1 <- sim[,2]

# Example: Capture-recapture
par(mfrow=c(1,2))
hist(N, freq=F, main="Estimated Marginal Posterior for N")
hist(alpha1, freq=F, main ="Estimating Marginal Posterior for Alpha 1")

# Example: Capture-recapture
library(mcmcse)
## mcmcse: Monte Carlo Standard Errors for MCMC
## Version 1.2-1 created on 2016-03-24.
## copyright (c) 2012, James M. Flegal, University of California,Riverside
##                     John Hughes, University of Minnesota
##                     Dootika Vats, University of Minnesota
##  For citation information, type citation("mcmcse").
##  Type help("mcmcse-package") to get started.
ess(N)
##       se 
## 8421.642
ess(alpha1)
##       se 
## 8519.459

# Example: Capture-recapture
par(mfrow=c(1,2))
estvssamp(N)
estvssamp(alpha1)

# Example: Capture-recapture
mcse(N)
## $est
## [1] 89.5495
## 
## $se
## [1] 0.0307835
mcse.q(N, .05)
## $est
## [1] 86
## 
## $se
## [1] 0.02930293
mcse.q(N, .95)
## $est
## [1] 95
## 
## $se

# Example: Capture-recapture
mcse(alpha1)
## $est
## [1] 0.3374951
## 
## $se
## [1] 0.0005427247
mcse.q(alpha1, .05)
## $est
## [1] 0.2556669
## 
## $se
## [1] 0.0009743541
mcse.q(alpha1, .95)
## $est
## [1] 0.4221397
## 
## $se

# Example: Capture-recapture
current <- sim[10000,] # start from here is you need more simulations
sim <- rbind(sim, gibbs.chain(10000, N.start = current[1], alpha.start = current[2:8]))
N.big <- sim[,1]

# Example: Capture-recapture
hist(N.big, freq=F, main="Estimated Marginal Posterior for N")

# Example: Capture-recapture
ess(N)
##       se 
## 8421.642
ess(N.big)
##       se 
## 14446.78

# Example: Capture-recapture
estvssamp(N.big)

# Example: Capture-recapture
mcse(N)
## $est
## [1] 89.5495
## 
## $se
## [1] 0.0307835
mcse(N.big)
## $est
## [1] 89.50495
## 
## $se
## [1] 0.02318482

# Example: Capture-recapture
mcse.q(N, .05)
## $est
## [1] 86
## 
## $se
## [1] 0.02930293
mcse.q(N.big, .05)
## $est
## [1] 86
## 
## $se
## [1] 0.01798403

# Example: Capture-recapture
mcse.q(N, .95)
## $est
## [1] 95
## 
## $se
## [1] 0.06565672
mcse.q(N.big, .95)
## $est
## [1] 95
## 
## $se
## [1] 0.04391722

# Toy example
# Histograms of μ̄_n for both stopping methods.

# Summary
# - Bayesian inference usually requires a MCMC simulation
# - Metropolis-Hastings algorithm and Gibbs samplers
# - Basic idea is similar to OMC, but sampling from a Markov chain yields dependent draws
# - MCMC output analysis is often ignored or poorly understood
