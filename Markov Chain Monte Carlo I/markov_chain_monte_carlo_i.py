# Agenda
# - Like Ordinary Monte Carlo (OMC), but better?
# - SLLN and Markov chain CLT
# - Variance estimation
# - AR(1) example
# - Metropolis-Hastings algorithm (with an exercise)

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t as t_dist

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

def ar1(m, rho, tau):
    return rho * m + np.random.normal(0, tau)

# Next, we add to this function so that we can give it a Markov chain 
# and the result will be p observations from the Markov chain

def ar1_gen(mc, p, rho, tau, q=1):
    mc = list(mc) if not isinstance(mc, list) else mc
    loc = len(mc)
    
    for i in range(p):
        j = i + loc - 1
        mc.append(ar1(mc[j], rho, tau))
    
    return np.array(mc)

# Batch means estimation function
def batch_means_mcse(x, batch_size=None):
    """Compute Monte Carlo Standard Error using batch means"""
    n = len(x)
    if batch_size is None:
        batch_size = int(np.sqrt(n))
    
    n_batches = n // batch_size
    if n_batches < 2:
        return np.std(x) / np.sqrt(n)
    
    batch_means = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        batch_means.append(np.mean(x[start:end]))
    
    se_batch = np.std(batch_means, ddof=1) / np.sqrt(n_batches)
    mcse = se_batch * np.sqrt(batch_size)
    
    return mcse

# Example: AR(1)
np.random.seed(20)

tau = 1
rho = 0.95
out = np.array([0])
eps = 0.1
start = 1000
r = 1000

# Example: AR(1)
out = ar1_gen(out, start, rho, tau)
MCSE = [batch_means_mcse(out)]
N = len(out)
t_val = t_dist.ppf(0.975, int(np.floor(np.sqrt(N) - 1)))
muhat = [np.mean(out)]
check = MCSE[0] * t_val

while eps < check:
    out = ar1_gen(out, r, rho, tau)
    MCSE.append(batch_means_mcse(out))
    N = len(out)
    t_val = t_dist.ppf(0.975, int(np.floor(np.sqrt(N) - 1)))
    muhat.append(np.mean(out))
    check = MCSE[-1] * t_val

N_vals = np.arange(start, len(out) + 1, r)
t_vals = t_dist.ppf(0.975, np.floor(np.sqrt(N_vals) - 1).astype(int))
half = np.array(MCSE) * t_vals
sigmahat = np.array(MCSE) * np.sqrt(N_vals)
N_vals = N_vals / 1000

# Example: AR(1)
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(N_vals, muhat, 'r-', label='Observed')
plt.axhline(y=0, color='black', linewidth=3, label='Actual')
plt.title("Estimates of the Mean")
plt.xlabel("Iterations (in 1000's)")
plt.ylabel("Mean")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(N_vals, sigmahat, 'r-', label='Observed')
plt.axhline(y=20, color='black', linewidth=3, label='Actual')
plt.title("Estimates of Sigma")
plt.xlabel("Iterations (in 1000's)")
plt.ylabel("Sigma")
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(N_vals, 2*half, 'r-', label='Observed')
plt.axhline(y=0.2, color='black', linewidth=3, label='Cut-off')
plt.title("Calculated Interval Widths")
plt.xlabel("Iterations (in 1000's)")
plt.ylabel("Width")
plt.ylim(0, 1.8)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc_ar1_python.png', dpi=300, bbox_inches='tight')
plt.show()

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

def ind_chain(x, n, theta=1):
    """if theta = 1, then this is an iid sampler"""
    x = list(x) if not isinstance(x, list) else x
    m = len(x)
    
    for i in range(m, m + n):
        x_prime = np.random.exponential(1/theta)
        u = np.exp((x[i-1] - x_prime) * (1 - theta))
        
        if np.random.uniform() < u:
            x.append(x_prime)
        else:
            x.append(x[i-1])
    
    return np.array(x)

# Example: Markov chain basics
# Random Walk Metropolis sampler with N(0, σ) proposal

def rw_chain(x, n, sigma=1):
    x = list(x) if not isinstance(x, list) else x
    m = len(x)
    
    for i in range(m, m + n):
        x_prime = x[i-1] + np.random.normal(0, sigma)
        u = np.exp(x[i-1] - x_prime)
        
        if np.random.uniform() < u and x_prime > 0:
            x.append(x_prime)
        else:
            x.append(x[i-1])
    
    return np.array(x)

# Example: Markov chain basics
np.random.seed(42)
trial0 = ind_chain([1], 500, 1)
trial1 = ind_chain([1], 500, 2)
trial2 = ind_chain([1], 500, 1/2)
rw1 = rw_chain([1], 500, 0.2)
rw2 = rw_chain([1], 500, 1)
rw3 = rw_chain([1], 500, 5)

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
def independence_mh(n_iter, theta, x0=1):
    x = np.zeros(n_iter)
    x[0] = x0
    accept_count = 0
    
    for i in range(1, n_iter):
        # Propose from Exp(theta)
        x_star = np.random.exponential(1/theta)
        
        # Compute MH ratio
        R = np.exp((x[i-1] - x_star) * (1 - theta))
        
        # Accept/reject
        if np.random.uniform() < R:
            x[i] = x_star
            accept_count += 1
        else:
            x[i] = x[i-1]
    
    return {'chain': x, 'acceptance_rate': accept_count / (n_iter - 1)}

# Generate 1000 draws for θ ∈ {1/2, 1, 2}
np.random.seed(123)
n = 1000

indep_theta_0_5 = independence_mh(n, theta=0.5)
indep_theta_1_0 = independence_mh(n, theta=1.0)
indep_theta_2_0 = independence_mh(n, theta=2.0)

print("Independence MH Acceptance Rates:")
print(f"θ = 0.5: {indep_theta_0_5['acceptance_rate']:.4f}")
print(f"θ = 1.0: {indep_theta_1_0['acceptance_rate']:.4f}")
print(f"θ = 2.0: {indep_theta_2_0['acceptance_rate']:.4f}\n")

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
def random_walk_mh(n_iter, sigma, x0=1):
    x = np.zeros(n_iter)
    x[0] = x0
    accept_count = 0
    
    for i in range(1, n_iter):
        # Propose from random walk
        x_star = x[i-1] + np.random.normal(0, sigma)
        
        # Compute MH ratio (only accept if x_star > 0)
        if x_star > 0:
            R = np.exp(x[i-1] - x_star)
            
            # Accept/reject
            if np.random.uniform() < R:
                x[i] = x_star
                accept_count += 1
            else:
                x[i] = x[i-1]
        else:
            x[i] = x[i-1]  # Reject if x_star <= 0
    
    return {'chain': x, 'acceptance_rate': accept_count / (n_iter - 1)}

# Generate 1000 draws for σ ∈ {0.2, 1, 5}
np.random.seed(123)

rw_sigma_0_2 = random_walk_mh(n, sigma=0.2)
rw_sigma_1_0 = random_walk_mh(n, sigma=1.0)
rw_sigma_5_0 = random_walk_mh(n, sigma=5.0)

print("Random Walk MH Acceptance Rates:")
print(f"σ = 0.2: {rw_sigma_0_2['acceptance_rate']:.4f}")
print(f"σ = 1.0: {rw_sigma_1_0['acceptance_rate']:.4f}")
print(f"σ = 5.0: {rw_sigma_5_0['acceptance_rate']:.4f}\n")

# Visualize the chains
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Independence chains
axes[0, 0].plot(indep_theta_0_5['chain'])
axes[0, 0].set_title('Independence: θ = 0.5')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel('X')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(indep_theta_1_0['chain'])
axes[0, 1].set_title('Independence: θ = 1.0')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel('X')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(indep_theta_2_0['chain'])
axes[0, 2].set_title('Independence: θ = 2.0')
axes[0, 2].set_xlabel('Iteration')
axes[0, 2].set_ylabel('X')
axes[0, 2].grid(True, alpha=0.3)

# Random walk chains
axes[1, 0].plot(rw_sigma_0_2['chain'])
axes[1, 0].set_title('Random Walk: σ = 0.2')
axes[1, 0].set_xlabel('Iteration')
axes[1, 0].set_ylabel('X')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(rw_sigma_1_0['chain'])
axes[1, 1].set_title('Random Walk: σ = 1.0')
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('X')
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(rw_sigma_5_0['chain'])
axes[1, 2].set_title('Random Walk: σ = 5.0')
axes[1, 2].set_xlabel('Iteration')
axes[1, 2].set_ylabel('X')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc_chains_python.png', dpi=300, bbox_inches='tight')
plt.show()

# Compare histograms with true Exp(1) distribution
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
x_range = np.linspace(0, 8, 1000)
exp_pdf = stats.expon.pdf(x_range, scale=1)

axes[0, 0].hist(indep_theta_0_5['chain'], bins=30, density=True, alpha=0.7, edgecolor='black')
axes[0, 0].plot(x_range, exp_pdf, 'r-', linewidth=2, label='Exp(1)')
axes[0, 0].set_title('Independence: θ = 0.5')
axes[0, 0].set_xlabel('X')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(indep_theta_1_0['chain'], bins=30, density=True, alpha=0.7, edgecolor='black')
axes[0, 1].plot(x_range, exp_pdf, 'r-', linewidth=2, label='Exp(1)')
axes[0, 1].set_title('Independence: θ = 1.0')
axes[0, 1].set_xlabel('X')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].hist(indep_theta_2_0['chain'], bins=30, density=True, alpha=0.7, edgecolor='black')
axes[0, 2].plot(x_range, exp_pdf, 'r-', linewidth=2, label='Exp(1)')
axes[0, 2].set_title('Independence: θ = 2.0')
axes[0, 2].set_xlabel('X')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].hist(rw_sigma_0_2['chain'], bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 0].plot(x_range, exp_pdf, 'r-', linewidth=2, label='Exp(1)')
axes[1, 0].set_title('Random Walk: σ = 0.2')
axes[1, 0].set_xlabel('X')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(rw_sigma_1_0['chain'], bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 1].plot(x_range, exp_pdf, 'r-', linewidth=2, label='Exp(1)')
axes[1, 1].set_title('Random Walk: σ = 1.0')
axes[1, 1].set_xlabel('X')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].hist(rw_sigma_5_0['chain'], bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1, 2].plot(x_range, exp_pdf, 'r-', linewidth=2, label='Exp(1)')
axes[1, 2].set_title('Random Walk: σ = 5.0')
axes[1, 2].set_xlabel('X')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/kevinbanker4/statistical-computing-/plots/mcmc_histograms_python.png', dpi=300, bbox_inches='tight')
plt.show()

# Part 3: Preference Discussion
# ------------------------------
print("\n=== Part 3: Independence vs Random Walk ===")
print("For this Exp(1) example:\n")
print("Independence Chain (θ = 1):")
print(f"- Acceptance rate: {indep_theta_1_0['acceptance_rate']:.4f}")
print("- Proposal matches target exactly, optimal performance")
print("- When θ = 1, every proposal is accepted (R = 1 always)\n")

print("Random Walk Chain:")
print(f"- σ = 0.2: High acceptance ({rw_sigma_0_2['acceptance_rate']:.4f}) but slow mixing (small steps)")
print(f"- σ = 1.0: Moderate acceptance ({rw_sigma_1_0['acceptance_rate']:.4f}) with good mixing")
print(f"- σ = 5.0: Low acceptance ({rw_sigma_5_0['acceptance_rate']:.4f}) due to large steps, many rejected\n")

print("PREFERENCE:")
print("- If we know the target well: Independence chain is better (when g ≈ f)")
print("- In general: Random walk is more robust and easier to tune")
print("- For this problem: Random walk with σ ≈ 1 is preferred as it doesn't")
print("  require knowing the exact form of the target distribution\n")

# Part 4: Fixed-width stopping rule (using random walk with σ = 1)
# -----------------------------------------------------------------
print("\n=== Part 4: Fixed-Width Stopping Rule ===\n")

def random_walk_mh_stopping(sigma, target_width=0.2, start_n=1000, batch_n=500, 
                             max_iter=50000, x0=1):
    x = x0
    chain = np.zeros(start_n)
    chain[0] = x0
    
    # Initial run
    for i in range(1, start_n):
        x_star = x + np.random.normal(0, sigma)
        if x_star > 0:
            R = np.exp(x - x_star)
            if np.random.uniform() < R:
                x = x_star
        chain[i] = x
    
    # Check stopping criterion
    se = batch_means_mcse(chain)
    n = len(chain)
    half_width = stats.norm.ppf(0.975) * se
    
    print(f"Initial n = {n}, Mean = {np.mean(chain):.4f}, Half-width = {half_width:.4f}")
    
    # Continue until criterion met
    while half_width > target_width / 2 and n < max_iter:
        # Generate more samples
        new_samples = np.zeros(batch_n)
        for i in range(batch_n):
            x_star = x + np.random.normal(0, sigma)
            if x_star > 0:
                R = np.exp(x - x_star)
                if np.random.uniform() < R:
                    x = x_star
            new_samples[i] = x
        
        chain = np.concatenate([chain, new_samples])
        se = batch_means_mcse(chain)
        n = len(chain)
        half_width = stats.norm.ppf(0.975) * se
        
        print(f"n = {n}, Mean = {np.mean(chain):.4f}, Half-width = {half_width:.4f}")
    
    return {
        'chain': chain,
        'final_mean': np.mean(chain),
        'final_se': se,
        'final_half_width': half_width,
        'final_n': n,
        'converged': half_width <= target_width / 2
    }

np.random.seed(456)
result = random_walk_mh_stopping(sigma=1.0, target_width=0.2)

print("\n=== Final Results ===")
print(f"Target half-width: 0.1")
print(f"Achieved half-width: {result['final_half_width']:.4f}")
print(f"Final sample size: {result['final_n']}")
print(f"Final mean estimate: {result['final_mean']:.4f}")
print(f"True mean (Exp(1)): 1.0")
print(f"95% CI: [{result['final_mean'] - result['final_half_width']:.4f}, {result['final_mean'] + result['final_half_width']:.4f}]")
print(f"Converged: {result['converged']}")
