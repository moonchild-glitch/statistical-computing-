# ============================================================================
# Bayesian Statistics
# ============================================================================

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ============================================================================
# Bayes Factors
# ============================================================================

def bayes_factor(data, mu_null=0):
    """
    Calculate Bayes Factor for a one-sample t-test.
    """
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    t_stat = (sample_mean - mu_null) / (sample_std / np.sqrt(n))
    
    # Calculate p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))

    # Approximate Bayes Factor (using BIC approximation)
    bic_null = n * np.log(sample_std**2) + 2 * np.log(n)
    bic_alt = n * np.log(sample_std**2) + 2 * np.log(n) + t_stat**2
    bf = np.exp((bic_null - bic_alt) / 2)

    return bf, p_value

# ============================================================================
# Example: Bayes Factor Calculation
# ============================================================================

np.random.seed(123)
n = 50
mu_null = 0
mu_alt = 0.5
sigma = 1
data = np.random.normal(mu_alt, sigma, n)

print("Simulated data:")
print(f"  Sample size: n = {n}")
print(f"  True mean: {mu_alt}")
print(f"  True SD: {sigma}")
print(f"  Sample mean: {np.mean(data):.4f}")
print(f"  Sample SD: {np.std(data, ddof=1):.4f}\n")

# Calculate Bayes Factor and p-value
bf, p_value = bayes_factor(data, mu_null=mu_null)

print("Frequentist t-test:")
print(f"  H0: μ = {mu_null}")
print("  Ha: μ ≠ 0")
print(f"  p-value: {p_value:.4f}\n")

print("Bayesian t-test (Bayes Factor):")
print(f"  H0: μ = {mu_null}")
print("  H1: μ ≠ 0")
print(f"  Bayes Factor (BF10): {bf:.4f}\n")

# ============================================================================
# Visualization: BF vs p-value relationship
# ============================================================================

print("Creating visualization of Bayes Factor vs p-value relationship...\n")

# Simulate multiple datasets to show relationship
n_sims = 100
sample_sizes = [20, 50, 100, 200]
results = []

for n_size in sample_sizes:
    for _ in range(n_sims):
        sim_data = np.random.normal(0.3, 1, n_size)
        bf, p = bayes_factor(sim_data, mu_null=0)
        results.append((n_size, p, bf))

# Convert results to numpy array
results = np.array(results, dtype=[('n', int), ('pvalue', float), ('bayes_factor', float)])

# Plot 1: Scatter plot of BF vs p-value
plt.figure(figsize=(10, 6))
plt.scatter(results['pvalue'], results['bayes_factor'], alpha=0.5, c='blue')
plt.xscale('log')
plt.yscale('log')
plt.axhline(1, color='red', linestyle='--', label='BF = 1 (no evidence)')
plt.axvline(0.05, color='green', linestyle='--', label='p = 0.05')
plt.xlabel('p-value')
plt.ylabel('Bayes Factor')
plt.title('Bayes Factor vs p-value')
plt.legend()
plt.show()

# ============================================================================
# Summary
# ============================================================================

print("Summary: Bayesian Statistics\n")
print("Key concepts covered:\n")
print("1. Bayesian Inference\n")
print("2. Priors\n")
print("3. Bayesian Point Estimates\n")
print("4. Bayesian Hypothesis Testing\n")
print("5. Bayes Factors\n")
print("Applications: Pattern recognition, spam detection, etc.\n")