"""
SIMULATION AND RANDOM NUMBER GENERATION
Statistical Computing Tutorial - Python Version

Topics covered:
1. Random number generation theory
2. Box-Muller transformation
3. Inverse CDF method
4. Rejection sampling
5. Statistical validation techniques
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, beta, uniform, expon, gamma, chi2
from scipy.optimize import minimize_scalar
import time
import os

# Create plots directory if it doesn't exist
os.makedirs('../plots', exist_ok=True)

print("=" * 70)
print("SIMULATION AND RANDOM NUMBER GENERATION TUTORIAL")
print("=" * 70)

# =============================================================================
# PART 1: RANDOM NUMBER GENERATION BASICS
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: RANDOM NUMBER GENERATION BASICS")
print("=" * 70)

# Set seed for reproducibility
np.random.seed(42)

# Basic uniform random numbers
print("\nBasic uniform random numbers U(0,1):")
u = np.random.uniform(0, 1, 10)
print(u[:5])

# Transform to other distributions using inverse CDF
print("\nInverse CDF method - Exponential(rate=2):")
rate = 2
x_exp = -np.log(1 - u) / rate
print(f"First 5 values: {x_exp[:5]}")
print(f"Compare with built-in: {np.random.exponential(1/rate, 5)}")

# =============================================================================
# PART 2: BOX-MULLER TRANSFORMATION
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: BOX-MULLER TRANSFORMATION")
print("=" * 70)

print("\nThe Box-Muller transformation converts uniform random variables")
print("to normal random variables using the transformation:")
print("Z1 = sqrt(-2*ln(U1)) * cos(2*pi*U2)")
print("Z2 = sqrt(-2*ln(U1)) * sin(2*pi*U2)")
print("where U1, U2 ~ Uniform(0,1) and Z1, Z2 ~ Normal(0,1)")

def bmnormal(n, mu=0, sd=1):
    """
    Generate n draws from Normal(mu, sd) using Box-Muller transformation.
    
    Parameters:
    -----------
    n : int
        Number of samples to generate
    mu : float
        Mean of the normal distribution (default: 0)
    sd : float
        Standard deviation (default: 1)
    
    Returns:
    --------
    numpy.ndarray
        Array of n samples from Normal(mu, sd)
    
    Notes:
    ------
    The Box-Muller transformation converts pairs of uniform random variables
    into pairs of independent standard normal variables. We generate ceiling(n/2)
    pairs and return the first n values.
    """
    # Number of pairs needed
    n_pairs = int(np.ceil(n / 2))
    
    # Generate uniform random variables
    u1 = np.random.uniform(0, 1, n_pairs)
    u2 = np.random.uniform(0, 1, n_pairs)
    
    # Box-Muller transformation
    r = np.sqrt(-2 * np.log(u1))
    theta = 2 * np.pi * u2
    
    z1 = r * np.cos(theta)
    z2 = r * np.sin(theta)
    
    # Combine and take first n values
    z = np.concatenate([z1, z2])[:n]
    
    # Transform to desired mean and sd
    x = mu + sd * z
    
    return x

# Test the function
print("\nTesting bmnormal function:")
samples = bmnormal(5, mu=0, sd=1)
print(f"5 samples from N(0,1): {samples}")

samples = bmnormal(5, mu=10, sd=3)
print(f"5 samples from N(10,3): {samples}")

# =============================================================================
# EXERCISE: BOX-MULLER VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("EXERCISE: VALIDATE BOX-MULLER IMPLEMENTATION")
print("=" * 70)

# Generate 2000 samples from Normal(10, 3)
n = 2000
mu = 10
sd = 3

print(f"\nGenerating {n} samples from Normal({mu}, {sd}) using Box-Muller...")
np.random.seed(123)
samples_bm = bmnormal(n, mu, sd)

# Calculate sample statistics
sample_mean = np.mean(samples_bm)
sample_sd = np.std(samples_bm, ddof=1)

print(f"\nSample Statistics:")
print(f"Sample mean: {sample_mean:.4f} (expected: {mu})")
print(f"Sample SD:   {sample_sd:.4f} (expected: {sd})")

# =============================================================================
# VISUAL VALIDATION
# =============================================================================

print("\n" + "=" * 70)
print("VISUAL VALIDATION")
print("=" * 70)

# Create comprehensive validation plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Box-Muller Transformation Validation: N(10, 3) with n=2000', 
             fontsize=14, fontweight='bold')

# 1. Histogram with theoretical density
ax = axes[0, 0]
ax.hist(samples_bm, bins=50, density=True, alpha=0.7, color='skyblue', 
        edgecolor='black', label='Box-Muller samples')
x_range = np.linspace(samples_bm.min(), samples_bm.max(), 200)
ax.plot(x_range, norm.pdf(x_range, mu, sd), 'r-', linewidth=2, 
        label=f'N({mu}, {sd}) density')
ax.axvline(sample_mean, color='blue', linestyle='--', linewidth=2, 
          label=f'Sample mean: {sample_mean:.2f}')
ax.axvline(mu, color='red', linestyle='--', linewidth=2, 
          label=f'True mean: {mu}')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Histogram vs Theoretical Density')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 2. Q-Q plot
ax = axes[0, 1]
stats.probplot(samples_bm, dist=stats.norm(mu, sd), plot=ax)
ax.set_title('Q-Q Plot')
ax.grid(True, alpha=0.3)

# 3. ECDF comparison
ax = axes[0, 2]
sorted_samples = np.sort(samples_bm)
ecdf_y = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)
ax.plot(sorted_samples, ecdf_y, 'b-', linewidth=2, label='Empirical CDF')
ax.plot(sorted_samples, norm.cdf(sorted_samples, mu, sd), 'r--', 
        linewidth=2, label='Theoretical CDF')
ax.set_xlabel('Value')
ax.set_ylabel('Cumulative Probability')
ax.set_title('ECDF vs Theoretical CDF')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Box plot with mean and median
ax = axes[1, 0]
bp = ax.boxplot(samples_bm, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
ax.axhline(mu, color='red', linestyle='--', linewidth=2, label=f'True mean: {mu}')
ax.axhline(sample_mean, color='blue', linestyle='--', linewidth=2, 
          label=f'Sample mean: {sample_mean:.2f}')
ax.set_ylabel('Value')
ax.set_title('Box Plot')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5. Running mean convergence
ax = axes[1, 1]
running_mean = np.cumsum(samples_bm) / np.arange(1, len(samples_bm) + 1)
ax.plot(running_mean, 'b-', linewidth=1, label='Running mean')
ax.axhline(mu, color='red', linestyle='--', linewidth=2, label=f'True mean: {mu}')
ax.fill_between(range(len(running_mean)), 
                mu - 1.96*sd/np.sqrt(np.arange(1, len(samples_bm) + 1)),
                mu + 1.96*sd/np.sqrt(np.arange(1, len(samples_bm) + 1)),
                alpha=0.3, color='red', label='95% CI')
ax.set_xlabel('Sample size')
ax.set_ylabel('Mean')
ax.set_title('Convergence of Sample Mean')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Variance convergence
ax = axes[1, 2]
running_var = np.array([np.var(samples_bm[:i], ddof=1) for i in range(2, len(samples_bm) + 1)])
ax.plot(range(2, len(samples_bm) + 1), running_var, 'b-', linewidth=1, 
        label='Running variance')
ax.axhline(sd**2, color='red', linestyle='--', linewidth=2, 
          label=f'True variance: {sd**2}')
ax.set_xlabel('Sample size')
ax.set_ylabel('Variance')
ax.set_title('Convergence of Sample Variance')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/simulation_boxmuller_validation.png', dpi=300, bbox_inches='tight')
print("\nSaved: ../plots/simulation_boxmuller_validation.png")
plt.close()

# =============================================================================
# STATISTICAL TESTS
# =============================================================================

print("\n" + "=" * 70)
print("STATISTICAL TESTS")
print("=" * 70)

# 1. Shapiro-Wilk test for normality
print("\n1. Shapiro-Wilk test for normality:")
print("   H0: Data comes from a normal distribution")
stat_sw, p_sw = stats.shapiro(samples_bm)
print(f"   Test statistic W = {stat_sw:.6f}")
print(f"   P-value = {p_sw:.4f}")
if p_sw > 0.05:
    print("   ✓ Cannot reject H0: Data appears normally distributed (p > 0.05)")
else:
    print("   ✗ Reject H0: Data does not appear normally distributed (p ≤ 0.05)")

# 2. Kolmogorov-Smirnov test
print("\n2. Kolmogorov-Smirnov test:")
print(f"   H0: Data comes from Normal({mu}, {sd})")
stat_ks, p_ks = stats.kstest(samples_bm, lambda x: norm.cdf(x, mu, sd))
print(f"   Test statistic D = {stat_ks:.6f}")
print(f"   P-value = {p_ks:.4f}")
if p_ks > 0.05:
    print(f"   ✓ Cannot reject H0: Data consistent with N({mu},{sd}) (p > 0.05)")
else:
    print(f"   ✗ Reject H0: Data not consistent with N({mu},{sd}) (p ≤ 0.05)")

# 3. t-test for mean
print("\n3. One-sample t-test for mean:")
print(f"   H0: Population mean = {mu}")
stat_t, p_t = stats.ttest_1samp(samples_bm, mu)
print(f"   Test statistic t = {stat_t:.4f}")
print(f"   P-value = {p_t:.4f}")
if p_t > 0.05:
    print(f"   ✓ Cannot reject H0: Mean consistent with {mu} (p > 0.05)")
else:
    print(f"   ✗ Reject H0: Mean different from {mu} (p ≤ 0.05)")

# 4. Chi-squared test for variance
print("\n4. Chi-squared test for variance:")
print(f"   H0: Population variance = {sd**2}")
chi_stat = (n - 1) * sample_sd**2 / sd**2
p_chi = 2 * min(stats.chi2.cdf(chi_stat, n-1), 1 - stats.chi2.cdf(chi_stat, n-1))
print(f"   Test statistic χ² = {chi_stat:.4f}")
print(f"   P-value = {p_chi:.4f}")
if p_chi > 0.05:
    print(f"   ✓ Cannot reject H0: Variance consistent with {sd**2} (p > 0.05)")
else:
    print(f"   ✗ Reject H0: Variance different from {sd**2} (p ≤ 0.05)")

# 5. Independence test for Z1 and Z2
print("\n5. Testing independence of Z1 and Z2 pairs:")
print("   H0: Z1 and Z2 are independent (correlation = 0)")
# Generate pairs
np.random.seed(123)
n_pairs = 1000
u1 = np.random.uniform(0, 1, n_pairs)
u2 = np.random.uniform(0, 1, n_pairs)
r = np.sqrt(-2 * np.log(u1))
theta = 2 * np.pi * u2
z1 = r * np.cos(theta)
z2 = r * np.sin(theta)

corr, p_corr = stats.pearsonr(z1, z2)
print(f"   Correlation coefficient r = {corr:.6f}")
print(f"   P-value = {p_corr:.4f}")
if p_corr > 0.05:
    print("   ✓ Cannot reject H0: Z1 and Z2 appear independent (p > 0.05)")
else:
    print("   ✗ Reject H0: Z1 and Z2 appear correlated (p ≤ 0.05)")

# Create scatter plot of Z1 vs Z2
plt.figure(figsize=(8, 8))
plt.scatter(z1, z2, alpha=0.5, s=10)
plt.xlabel('Z1')
plt.ylabel('Z2')
plt.title(f'Independence of Z1 and Z2\n(Correlation: r={corr:.4f}, p={p_corr:.4f})')
plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(0, color='gray', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.savefig('../plots/simulation_boxmuller_independence.png', dpi=300, bbox_inches='tight')
print("\nSaved: ../plots/simulation_boxmuller_independence.png")
plt.close()

# =============================================================================
# PERFORMANCE COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)

n_test = 10000
n_reps = 100

# Time Box-Muller implementation
start = time.time()
for _ in range(n_reps):
    _ = bmnormal(n_test, mu, sd)
time_bm = time.time() - start

# Time built-in normal generator
start = time.time()
for _ in range(n_reps):
    _ = np.random.normal(mu, sd, n_test)
time_builtin = time.time() - start

print(f"\nGenerating {n_test} samples, {n_reps} repetitions:")
print(f"Box-Muller implementation: {time_bm:.4f} seconds")
print(f"Built-in np.random.normal: {time_builtin:.4f} seconds")
print(f"Ratio (Box-Muller / Built-in): {time_bm/time_builtin:.2f}x")

# =============================================================================
# ADDITIONAL METHODS: REJECTION SAMPLING
# =============================================================================

print("\n" + "=" * 70)
print("ADDITIONAL METHOD: REJECTION SAMPLING")
print("=" * 70)

print("\nRejection sampling is useful when:")
print("1. The inverse CDF is difficult to compute")
print("2. We can bound the target density with a proposal density")
print("\nExample: Sampling from Beta(2, 5) using Uniform(0, 1) proposal")

def rejection_beta(n, a=2, b=5):
    """
    Sample from Beta(a, b) using rejection sampling with Uniform(0,1) proposal.
    
    The Beta(2,5) density is f(x) = 30*x*(1-x)^4 for x in [0,1].
    Maximum occurs at x = 1/5, where f(1/5) ≈ 2.4576.
    We use M = 2.5 to ensure M*g(x) ≥ f(x) for all x.
    """
    samples = []
    attempts = 0
    
    # Find maximum of Beta density for scaling
    M = 2.5  # Upper bound on Beta(2,5) density
    
    while len(samples) < n:
        # Propose from Uniform(0, 1)
        u = np.random.uniform(0, 1)
        
        # Accept/reject
        acceptance_prob = beta.pdf(u, a, b) / M
        if np.random.uniform(0, 1) < acceptance_prob:
            samples.append(u)
        
        attempts += 1
    
    acceptance_rate = n / attempts
    return np.array(samples), acceptance_rate

# Generate samples
np.random.seed(456)
samples_beta, acc_rate = rejection_beta(1000)

print(f"\nGenerated 1000 samples from Beta(2, 5)")
print(f"Acceptance rate: {acc_rate:.4f} ({acc_rate*100:.2f}%)")
print(f"Sample mean: {np.mean(samples_beta):.4f} (expected: {2/(2+5):.4f})")
print(f"Sample variance: {np.var(samples_beta, ddof=1):.4f} (expected: {2*5/((2+5)**2*(2+5+1)):.4f})")

# Visualize rejection sampling
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram vs density
ax = axes[0]
ax.hist(samples_beta, bins=40, density=True, alpha=0.7, color='skyblue', 
        edgecolor='black', label='Rejection samples')
x_range = np.linspace(0, 1, 200)
ax.plot(x_range, beta.pdf(x_range, 2, 5), 'r-', linewidth=2, 
        label='Beta(2, 5) density')
ax.axhline(2.5, color='green', linestyle='--', linewidth=2, 
          label='Proposal bound M*g(x)')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Rejection Sampling: Beta(2, 5)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Q-Q plot
ax = axes[1]
stats.probplot(samples_beta, dist=beta(2, 5), plot=ax)
ax.set_title('Q-Q Plot: Beta(2, 5)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/simulation_rejection_sampling.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/simulation_rejection_sampling.png")
plt.close()

# =============================================================================
# POLAR BOX-MULLER METHOD
# =============================================================================

print("\n" + "=" * 70)
print("POLAR BOX-MULLER METHOD")
print("=" * 70)

print("\nThe polar Box-Muller method avoids trigonometric functions:")
print("1. Generate U1, U2 ~ Uniform(-1, 1)")
print("2. Calculate S = U1² + U2²")
print("3. If S ≥ 1, reject and try again")
print("4. Z1 = U1 * sqrt(-2*ln(S)/S)")
print("5. Z2 = U2 * sqrt(-2*ln(S)/S)")

def bmnormal_polar(n, mu=0, sd=1):
    """
    Generate n draws from Normal(mu, sd) using polar Box-Muller method.
    This method avoids computing sine and cosine.
    """
    samples = []
    
    while len(samples) < n:
        # Generate uniform in (-1, 1) x (-1, 1)
        u1 = np.random.uniform(-1, 1)
        u2 = np.random.uniform(-1, 1)
        s = u1**2 + u2**2
        
        # Reject if outside unit circle
        if s >= 1 or s == 0:
            continue
        
        # Polar transformation
        factor = np.sqrt(-2 * np.log(s) / s)
        z1 = u1 * factor
        z2 = u2 * factor
        
        samples.extend([z1, z2])
    
    # Take first n samples and transform
    z = np.array(samples[:n])
    x = mu + sd * z
    
    return x

# Test polar method
print("\nTesting polar Box-Muller:")
np.random.seed(789)
samples_polar = bmnormal_polar(2000, mu=10, sd=3)
print(f"Sample mean: {np.mean(samples_polar):.4f} (expected: 10)")
print(f"Sample SD:   {np.std(samples_polar, ddof=1):.4f} (expected: 3)")

# Compare methods visually
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.hist(samples_bm, bins=50, density=True, alpha=0.6, color='skyblue', 
        label='Standard Box-Muller', edgecolor='black')
ax.hist(samples_polar, bins=50, density=True, alpha=0.6, color='lightcoral', 
        label='Polar Box-Muller', edgecolor='black')
x_range = np.linspace(min(samples_bm.min(), samples_polar.min()), 
                      max(samples_bm.max(), samples_polar.max()), 200)
ax.plot(x_range, norm.pdf(x_range, 10, 3), 'k-', linewidth=2, 
        label='N(10, 3) density')
ax.set_xlabel('Value')
ax.set_ylabel('Density')
ax.set_title('Comparison of Box-Muller Methods')
ax.legend()
ax.grid(True, alpha=0.3)

# Q-Q plot comparing the two methods
ax = axes[1]
stats.probplot(samples_polar, dist=stats.norm(10, 3), plot=ax)
ax.set_title('Q-Q Plot: Polar Box-Muller')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/simulation_polar_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/simulation_polar_comparison.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: RANDOM NUMBER GENERATION METHODS")
print("=" * 70)

print("\n1. INVERSE CDF METHOD")
print("   - Best when inverse CDF has closed form")
print("   - Example: Exponential, Uniform, Pareto")
print("   - X = F^(-1)(U) where U ~ Uniform(0,1)")

print("\n2. BOX-MULLER TRANSFORMATION")
print("   - Converts uniform to normal random variables")
print("   - Two variants: standard (uses trig) and polar (avoids trig)")
print("   - Generates pairs of independent normals")
print("   - Exact method (not approximate)")

print("\n3. REJECTION SAMPLING")
print("   - Useful when density is known up to a constant")
print("   - Requires proposal distribution and bound M")
print("   - Acceptance rate = 1/M (want M close to 1)")
print("   - Example: Beta, truncated distributions")

print("\n4. TRANSFORMATION METHOD")
print("   - Use properties of distributions")
print("   - Example: If Z ~ N(0,1), then X = μ + σZ ~ N(μ, σ²)")
print("   - Example: If Z ~ N(0,1), then Z² ~ χ²(1)")

print("\n" + "=" * 70)
print("VALIDATION CHECKLIST")
print("=" * 70)

print("\n✓ Visual checks:")
print("  - Histogram matches theoretical density")
print("  - Q-Q plot shows points near diagonal")
print("  - ECDF matches theoretical CDF")
print("  - Running mean converges to true mean")

print("\n✓ Statistical tests:")
print("  - Shapiro-Wilk for normality")
print("  - Kolmogorov-Smirnov for distribution fit")
print("  - t-test for mean")
print("  - Chi-squared test for variance")
print("  - Correlation test for independence")

print("\n" + "=" * 70)
print("Generated plots:")
print("  1. simulation_boxmuller_validation.png - 6-panel validation")
print("  2. simulation_boxmuller_independence.png - Z1 vs Z2 scatter")
print("  3. simulation_rejection_sampling.png - Beta distribution example")
print("  4. simulation_polar_comparison.png - Standard vs Polar methods")
print("=" * 70)

print("\nSIMULATION TUTORIAL COMPLETE!")
print("All methods validated and working correctly.")
