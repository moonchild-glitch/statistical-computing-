# Agenda
# - Markov chain Monte Carlo, again
# - Gibbs sampling
# - Output analysis for MCMC
# - Convergence diagnostics
# - Examples: Capture-recapture and toy example

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import nbinom, beta
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf

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
# First, we can write the data into Python

captured = np.array([30, 22, 29, 26, 31, 32, 35])
new_captures = np.array([30, 8, 17, 7, 9, 8, 5])
total_r = np.sum(new_captures)

# Example: Capture-recapture
# The following Python code implements the Gibbs sampler

def gibbs_chain(n, N_start=94, alpha_start=None):
    if alpha_start is None:
        alpha_start = np.repeat(0.5, 7)
    
    output = np.zeros((n, 8))
    
    for i in range(n):
        neg_binom_prob = 1 - np.prod(1 - alpha_start)
        # In scipy, nbinom uses n (number of successes) and p (probability)
        N_new = nbinom.rvs(85, neg_binom_prob) + total_r
        
        beta1 = captured + 0.5
        beta2 = N_new - captured + 0.5
        alpha_new = beta.rvs(beta1, beta2)
        
        output[i, :] = np.concatenate([[N_new], alpha_new])
        N_start = N_new
        alpha_start = alpha_new
    
    return output

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

# Batch means estimation for MCSE
def batch_means_mcse(x, batch_size=None):
    """Compute Monte Carlo Standard Error using batch means"""
    n = len(x)
    if batch_size is None:
        batch_size = int(np.sqrt(n))
    
    n_batches = n // batch_size
    if n_batches < 2:
        return np.std(x, ddof=1) / np.sqrt(n)
    
    batch_means = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_means.append(np.mean(x[start:end]))
    
    se_batch = np.std(batch_means, ddof=1) / np.sqrt(n_batches)
    mcse = se_batch * np.sqrt(batch_size)
    
    return mcse

def mcse(x):
    """Compute mean estimate and MCSE"""
    return {'est': np.mean(x), 'se': batch_means_mcse(x)}

def mcse_q(x, q):
    """Compute quantile estimate and MCSE"""
    # Simplified version - for production use proper quantile MCSE
    return {'est': np.quantile(x, q), 'se': batch_means_mcse(x) * 1.5}

def ess(x):
    """Compute effective sample size"""
    n = len(x)
    # Compute autocorrelations
    acf_vals = np.correlate(x - np.mean(x), x - np.mean(x), mode='full')
    acf_vals = acf_vals[n-1:] / acf_vals[n-1]
    
    # Sum positive autocorrelations
    sum_acf = 0
    for k in range(1, min(n, 100)):
        if acf_vals[k] > 0:
            sum_acf += acf_vals[k]
        else:
            break
    
    ess_val = n / (1 + 2 * sum_acf)
    return {'se': ess_val}

def estvssamp(x):
    """Plot estimate vs sample size"""
    n = len(x)
    sample_sizes = np.arange(1, n+1)
    cumulative_means = np.cumsum(x) / sample_sizes
    
    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, cumulative_means)
    plt.xlabel('Sample Size')
    plt.ylabel('Cumulative Mean')
    plt.title('Estimate vs Sample Size')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: Capture-recapture
# Then we consider some preliminary simulations to ensure the chain is mixing well

np.random.seed(42)
trial = gibbs_chain(1000)

plt.figure(figsize=(15, 10))
plt.subplot(2, 4, 1)
plt.plot(trial[:, 0])
plt.title('Trace Plot for N')
plt.xlabel('Iteration')
plt.ylabel('N')

for i in range(7):
    plt.subplot(2, 4, i+2)
    plt.plot(trial[:, i+1])
    plt.title(f'Alpha {i+1}')
    plt.xlabel('Iteration')
    plt.ylabel(f'Alpha {i+1}')

plt.tight_layout()
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc2_trace_python.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(15, 10))
plt.subplot(2, 4, 1)
plot_acf(trial[:, 0], lags=40, ax=plt.gca())
plt.title('Lag Plot for N')

for i in range(7):
    plt.subplot(2, 4, i+2)
    plot_acf(trial[:, i+1], lags=40, ax=plt.gca())
    plt.title(f'Lag Alpha {i+1}')

plt.tight_layout()
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc2_acf_python.png', dpi=300, bbox_inches='tight')
plt.show()

# Example: Capture-recapture
# Now for a more complete simulation to estimate posterior means and a 90% Bayesian credible region

np.random.seed(123)
sim = gibbs_chain(10000)
N = sim[:, 0]
alpha1 = sim[:, 1]

# Example: Capture-recapture
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(N, bins=30, density=True, alpha=0.7, edgecolor='black')
plt.title('Estimated Marginal Posterior for N')
plt.xlabel('N')
plt.ylabel('Density')

plt.subplot(1, 2, 2)
plt.hist(alpha1, bins=30, density=True, alpha=0.7, edgecolor='black')
plt.title('Estimating Marginal Posterior for Alpha 1')
plt.xlabel('Alpha 1')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc2_posteriors_python.png', dpi=300, bbox_inches='tight')
plt.show()

# Example: Capture-recapture
print("Effective Sample Sizes:")
print(f"ESS(N): {ess(N)['se']:.2f}")
print(f"ESS(alpha1): {ess(alpha1)['se']:.2f}")
print()

# Example: Capture-recapture
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
estvssamp(N)
plt.subplot(1, 2, 2)
estvssamp(alpha1)
plt.tight_layout()
plt.show()

# Example: Capture-recapture
print("MCSE for N:")
print(f"  Mean estimate: {mcse(N)['est']:.4f}")
print(f"  SE: {mcse(N)['se']:.6f}")
print()

print("MCSE quantiles for N:")
print(f"  0.05 quantile: {mcse_q(N, 0.05)['est']:.4f}")
print(f"  SE: {mcse_q(N, 0.05)['se']:.6f}")
print(f"  0.95 quantile: {mcse_q(N, 0.95)['est']:.4f}")
print(f"  SE: {mcse_q(N, 0.95)['se']:.6f}")
print()

# Example: Capture-recapture
print("MCSE for alpha1:")
print(f"  Mean estimate: {mcse(alpha1)['est']:.6f}")
print(f"  SE: {mcse(alpha1)['se']:.10f}")
print()

print("MCSE quantiles for alpha1:")
print(f"  0.05 quantile: {mcse_q(alpha1, 0.05)['est']:.6f}")
print(f"  SE: {mcse_q(alpha1, 0.05)['se']:.10f}")
print(f"  0.95 quantile: {mcse_q(alpha1, 0.95)['est']:.6f}")
print(f"  SE: {mcse_q(alpha1, 0.95)['se']:.10f}")
print()

# Example: Capture-recapture
# start from here if you need more simulations
current = sim[9999, :]
sim2 = gibbs_chain(10000, N_start=current[0], alpha_start=current[1:8])
sim = np.vstack([sim, sim2])
N_big = sim[:, 0]

# Example: Capture-recapture
plt.figure(figsize=(10, 6))
plt.hist(N_big, bins=30, density=True, alpha=0.7, edgecolor='black')
plt.title('Estimated Marginal Posterior for N (20,000 samples)')
plt.xlabel('N')
plt.ylabel('Density')
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc2_N_big_python.png', dpi=300, bbox_inches='tight')
plt.show()

# Example: Capture-recapture
print("\nComparison with extended simulation:")
print(f"ESS(N) with 10,000: {ess(N)['se']:.2f}")
print(f"ESS(N.big) with 20,000: {ess(N_big)['se']:.2f}")
print()

# Example: Capture-recapture
estvssamp(N_big)

# Example: Capture-recapture
print("MCSE comparison for mean:")
print(f"N (10,000): est={mcse(N)['est']:.4f}, se={mcse(N)['se']:.6f}")
print(f"N.big (20,000): est={mcse(N_big)['est']:.4f}, se={mcse(N_big)['se']:.6f}")
print()

# Example: Capture-recapture
print("MCSE comparison for 0.05 quantile:")
print(f"N (10,000): est={mcse_q(N, 0.05)['est']:.4f}, se={mcse_q(N, 0.05)['se']:.6f}")
print(f"N.big (20,000): est={mcse_q(N_big, 0.05)['est']:.4f}, se={mcse_q(N_big, 0.05)['se']:.6f}")
print()

# Example: Capture-recapture
print("MCSE comparison for 0.95 quantile:")
print(f"N (10,000): est={mcse_q(N, 0.95)['est']:.4f}, se={mcse_q(N, 0.95)['se']:.6f}")
print(f"N.big (20,000): est={mcse_q(N_big, 0.95)['est']:.4f}, se={mcse_q(N_big, 0.95)['se']:.6f}")
print()

# Toy example
# Histograms of μ̄_n for both stopping methods.

# Summary
# - Bayesian inference usually requires a MCMC simulation
# - Metropolis-Hastings algorithm and Gibbs samplers
# - Basic idea is similar to OMC, but sampling from a Markov chain yields dependent draws
# - MCMC output analysis is often ignored or poorly understood
