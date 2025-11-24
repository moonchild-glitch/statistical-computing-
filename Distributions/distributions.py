#!/usr/bin/env python3
"""
============================================
DISTRIBUTIONS IN PYTHON
============================================
Statistical Computing Tutorial
Topic: Probability Distributions and Goodness of Fit Testing

Agenda:
1. Random number generation
2. Built-in distributions in Python (scipy.stats)
3. Parametric distributions as models
4. Methods of fitting (moments, generalized moments, likelihood)
5. Methods of checking (visual comparisons, statistics, tests, calibration)
6. Chi-squared test for continuous distributions
7. Better alternatives (K-S test, bootstrap, smooth tests)
============================================
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Create plots directory if it doesn't exist
os.makedirs("../plots", exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

print("\n" + "="*50)
print("RANDOM NUMBER GENERATION")
print("="*50 + "\n")

print("Python has built-in random number generators via NumPy and SciPy")
print("")
print("General naming convention in scipy.stats:")
print("  - dist.pdf(x): probability density function")
print("  - dist.cdf(x): cumulative distribution function")
print("  - dist.ppf(q): percent point function (inverse CDF)")
print("  - dist.rvs(size): random number generator")
print("")
print("where 'dist' is the distribution object (norm, expon, uniform, etc.)")

# Examples of random number generation
print("\n--- Uniform Distribution ---")
uniform_sample = np.random.uniform(0, 1, 10)
print("Sample of 10 uniform random numbers [0,1]:")
print(np.round(uniform_sample, 4))

print("\n--- Normal Distribution ---")
normal_sample = np.random.normal(0, 1, 10)
print("Sample of 10 standard normal random numbers:")
print(np.round(normal_sample, 4))

print("\n--- Exponential Distribution ---")
exp_sample = np.random.exponential(1, 10)
print("Sample of 10 exponential random numbers (scale=1):")
print(np.round(exp_sample, 4))

# ============================================
# DISTRIBUTIONS IN PYTHON
# ============================================
print("\n" + "="*50)
print("BUILT-IN DISTRIBUTIONS IN PYTHON")
print("="*50 + "\n")

print("Python (scipy.stats) provides many common distributions:")
print("")
print("Continuous distributions:")
print("  - Normal: stats.norm(loc, scale)")
print("  - Exponential: stats.expon(scale)")
print("  - Uniform: stats.uniform(loc, scale)")
print("  - Gamma: stats.gamma(a, scale)")
print("  - Beta: stats.beta(a, b)")
print("  - Chi-squared: stats.chi2(df)")
print("  - t-distribution: stats.t(df)")
print("  - F-distribution: stats.f(dfn, dfd)")
print("")
print("Discrete distributions:")
print("  - Binomial: stats.binom(n, p)")
print("  - Poisson: stats.poisson(mu)")
print("  - Geometric: stats.geom(p)")
print("  - Negative binomial: stats.nbinom(n, p)")

# Visualize some distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Normal distribution
x_norm = np.linspace(-4, 4, 1000)
axes[0, 0].plot(x_norm, stats.norm.pdf(x_norm, 0, 1), 'b-', lw=2, label='PDF')
axes[0, 0].hist(np.random.normal(0, 1, 1000), bins=30, density=True, 
                alpha=0.3, color='blue', label='Sample')
axes[0, 0].set_title('Normal Distribution')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].legend()

# Exponential distribution
x_exp = np.linspace(0, 5, 1000)
axes[0, 1].plot(x_exp, stats.expon.pdf(x_exp, scale=1), 'r-', lw=2, label='PDF')
axes[0, 1].hist(np.random.exponential(1, 1000), bins=30, density=True,
                alpha=0.3, color='red', label='Sample')
axes[0, 1].set_title('Exponential Distribution')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()

# Gamma distribution
x_gamma = np.linspace(0, 20, 1000)
axes[0, 2].plot(x_gamma, stats.gamma.pdf(x_gamma, a=2, scale=2), 'g-', lw=2, label='PDF')
axes[0, 2].hist(stats.gamma.rvs(a=2, scale=2, size=1000), bins=30, density=True,
                alpha=0.3, color='green', label='Sample')
axes[0, 2].set_title('Gamma Distribution (shape=2, scale=2)')
axes[0, 2].set_xlabel('x')
axes[0, 2].set_ylabel('Density')
axes[0, 2].legend()

# Beta distribution
x_beta = np.linspace(0, 1, 1000)
axes[1, 0].plot(x_beta, stats.beta.pdf(x_beta, 2, 5), color='purple', lw=2, label='PDF')
axes[1, 0].hist(stats.beta.rvs(2, 5, size=1000), bins=30, density=True,
                alpha=0.3, color='purple', label='Sample')
axes[1, 0].set_title('Beta Distribution (a=2, b=5)')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Density')
axes[1, 0].legend()

# Chi-squared distribution
x_chisq = np.linspace(0, 20, 1000)
axes[1, 1].plot(x_chisq, stats.chi2.pdf(x_chisq, 5), color='orange', lw=2, label='PDF')
axes[1, 1].hist(stats.chi2.rvs(5, size=1000), bins=30, density=True,
                alpha=0.3, color='orange', label='Sample')
axes[1, 1].set_title('Chi-squared Distribution (df=5)')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()

# Binomial distribution (discrete)
x_binom = np.arange(0, 21)
axes[1, 2].stem(x_binom, stats.binom.pmf(x_binom, 20, 0.5), basefmt=' ')
axes[1, 2].set_title('Binomial Distribution (n=20, p=0.5)')
axes[1, 2].set_xlabel('x')
axes[1, 2].set_ylabel('Probability')

plt.tight_layout()
plt.savefig('../plots/dist_common_distributions.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_common_distributions.png")

# ============================================
# PARAMETRIC DISTRIBUTIONS AS MODELS
# ============================================
print("\n" + "="*50)
print("PARAMETRIC DISTRIBUTIONS AS MODELS")
print("="*50 + "\n")

print("Parametric distributions serve as models for real-world data")
print("")
print("Key idea: Assume data comes from a known family of distributions")
print("          but with unknown parameters")
print("")
print("Goal: Estimate the parameters from the data")
print("      Check if the model fits well")

# Generate some example data (exponential)
true_rate = 0.5
sample_data = np.random.exponential(scale=1/true_rate, size=500)

print(f"\nGenerated 500 samples from Exponential(rate={true_rate:.1f})")
print(f"Sample mean: {np.mean(sample_data):.4f} (theoretical: {1/true_rate:.4f})")
print(f"Sample variance: {np.var(sample_data, ddof=1):.4f} (theoretical: {(1/true_rate)**2:.4f})")

# ============================================
# METHOD OF MOMENTS
# ============================================
print("\n" + "="*50)
print("FITTING: METHOD OF MOMENTS")
print("="*50 + "\n")

print("Method of Moments: Match sample moments to theoretical moments")
print("")
print("For exponential distribution:")
print("  E[X] = 1/λ")
print("  So: λ_hat = 1/mean(X)")

# Estimate rate using method of moments
rate_mom = 1/np.mean(sample_data)
print(f"\nMethod of Moments estimate: λ = {rate_mom:.4f}")
print(f"True rate: λ = {true_rate:.4f}")
print(f"Error: {abs(rate_mom - true_rate):.4f}")

# Visualize the fit
plt.figure(figsize=(10, 6))
plt.hist(sample_data, bins=30, density=True, alpha=0.6, color='lightblue',
         edgecolor='white', label='Data histogram')
x_seq = np.linspace(0, np.max(sample_data), 1000)
plt.plot(x_seq, stats.expon.pdf(x_seq, scale=1/rate_mom), 'r-', lw=2,
         label=f'MoM fit (λ={rate_mom:.3f})')
plt.plot(x_seq, stats.expon.pdf(x_seq, scale=1/true_rate), 'b--', lw=2,
         label=f'True (λ={true_rate:.3f})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Method of Moments Fit')
plt.legend()
plt.savefig('../plots/dist_method_of_moments.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_method_of_moments.png")

# ============================================
# MAXIMUM LIKELIHOOD ESTIMATION
# ============================================
print("\n" + "="*50)
print("FITTING: MAXIMUM LIKELIHOOD ESTIMATION")
print("="*50 + "\n")

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
rate_mle = 1/np.mean(sample_data)
print(f"\nMaximum Likelihood estimate: λ = {rate_mle:.4f}")
print(f"True rate: λ = {true_rate:.4f}")

# For more complex distributions, use scipy.optimize
# Example: fit gamma distribution using MLE
print("\n--- Fitting Gamma Distribution ---")

# Generate gamma data (note: scipy uses shape and scale, not rate)
gamma_data = stats.gamma.rvs(a=2, scale=2, size=500)  # scale = 1/rate, so rate=0.5

# Negative log-likelihood for gamma
def neg_log_lik_gamma(params, data):
    shape, scale = params
    if shape <= 0 or scale <= 0:
        return np.inf
    return -np.sum(stats.gamma.logpdf(data, a=shape, scale=scale))

# Optimize
result = minimize(neg_log_lik_gamma, x0=[1, 1], args=(gamma_data,), method='L-BFGS-B',
                  bounds=[(0.01, None), (0.01, None)])
shape_mle, scale_mle = result.x
rate_mle_gamma = 1/scale_mle

print(f"MLE estimates: shape = {shape_mle:.4f}, scale = {scale_mle:.4f} (rate = {rate_mle_gamma:.4f})")
print(f"True parameters: shape = 2.0000, scale = 2.0000 (rate = 0.5000)")

# Visualize
plt.figure(figsize=(10, 6))
plt.hist(gamma_data, bins=30, density=True, alpha=0.6, color='lightgreen',
         edgecolor='white', label='Data histogram')
x_seq_g = np.linspace(0, np.max(gamma_data), 1000)
plt.plot(x_seq_g, stats.gamma.pdf(x_seq_g, a=shape_mle, scale=scale_mle), 'r-', lw=2,
         label=f'MLE fit (shape={shape_mle:.2f}, scale={scale_mle:.2f})')
plt.plot(x_seq_g, stats.gamma.pdf(x_seq_g, a=2, scale=2), 'b--', lw=2,
         label='True (shape=2.00, scale=2.00)')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title('Maximum Likelihood Fit (Gamma Distribution)')
plt.legend()
plt.savefig('../plots/dist_mle_gamma.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_mle_gamma.png")

# ============================================
# VISUAL COMPARISON: Q-Q PLOTS
# ============================================
print("\n" + "="*50)
print("CHECKING FIT: VISUAL COMPARISON (Q-Q PLOTS)")
print("="*50 + "\n")

print("Q-Q Plot (Quantile-Quantile Plot):")
print("  - Compare quantiles of data to theoretical quantiles")
print("  - If data matches distribution, points lie on diagonal line")
print("  - Deviations indicate departure from assumed distribution")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Q-Q plot for exponential data
stats.probplot(sample_data, dist=stats.expon(scale=1/rate_mle), plot=axes[0])
axes[0].set_title('Q-Q Plot: Exponential Data')
axes[0].get_lines()[0].set_markerfacecolor('blue')
axes[0].get_lines()[0].set_markeredgecolor('blue')

# Q-Q plot for gamma data
stats.probplot(gamma_data, dist=stats.gamma(a=shape_mle, scale=scale_mle), plot=axes[1])
axes[1].set_title('Q-Q Plot: Gamma Data')
axes[1].get_lines()[0].set_markerfacecolor('green')
axes[1].get_lines()[0].set_markeredgecolor('green')

# Q-Q plot for normal data (should be good)
normal_data = np.random.normal(0, 1, 500)
stats.probplot(normal_data, dist='norm', plot=axes[2])
axes[2].set_title('Q-Q Plot: Normal Data')

plt.tight_layout()
plt.savefig('../plots/dist_qq_plots.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_qq_plots.png")

# ============================================
# EMPIRICAL CDF COMPARISON
# ============================================
print("\n" + "="*50)
print("CHECKING FIT: EMPIRICAL CDF COMPARISON")
print("="*50 + "\n")

print("Empirical CDF: Step function of observed data")
print("Compare to theoretical CDF")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# ECDF for exponential data
sorted_data = np.sort(sample_data)
ecdf_exp = np.arange(1, len(sorted_data)+1) / len(sorted_data)
axes[0].step(sorted_data, ecdf_exp, 'b-', lw=2, label='Empirical CDF', where='post')
x_seq = np.linspace(0, np.max(sample_data), 1000)
axes[0].plot(x_seq, stats.expon.cdf(x_seq, scale=1/rate_mle), 'r--', lw=2,
             label='Theoretical CDF')
axes[0].set_title('ECDF vs Theoretical CDF (Exponential)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Cumulative Probability')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ECDF for gamma data
sorted_gamma = np.sort(gamma_data)
ecdf_gamma = np.arange(1, len(sorted_gamma)+1) / len(sorted_gamma)
axes[1].step(sorted_gamma, ecdf_gamma, 'b-', lw=2, label='Empirical CDF', where='post')
x_seq_g = np.linspace(0, np.max(gamma_data), 1000)
axes[1].plot(x_seq_g, stats.gamma.cdf(x_seq_g, a=shape_mle, scale=scale_mle), 'r--', lw=2,
             label='Theoretical CDF')
axes[1].set_title('ECDF vs Theoretical CDF (Gamma)')
axes[1].set_xlabel('Value')
axes[1].set_ylabel('Cumulative Probability')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/dist_ecdf_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_ecdf_comparison.png")

# ============================================
# CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS
# ============================================
print("\n" + "="*50)
print("CHI-SQUARED TEST FOR CONTINUOUS DISTRIBUTIONS")
print("="*50 + "\n")

print("Chi-squared goodness-of-fit test:")
print("  - Designed for discrete/categorical data")
print("  - For continuous data: must discretize into bins")
print("  - Test statistic: χ² = sum((O_i - E_i)² / E_i)")
print("    where O_i = observed count in bin i")
print("          E_i = expected count in bin i")

# Discretize exponential data
n_bins = 10
breaks = np.percentile(sample_data, np.linspace(0, 100, n_bins+1))
breaks[0] = 0  # Ensure lower bound is 0
observed_counts, _ = np.histogram(sample_data, bins=breaks)

# Expected counts under fitted exponential
expected_probs = np.diff(stats.expon.cdf(breaks, scale=1/rate_mle))
expected_probs = expected_probs / np.sum(expected_probs)  # Normalize
expected_counts = expected_probs * len(sample_data)

print("\nObserved vs Expected counts:")
import pandas as pd
df_counts = pd.DataFrame({
    'Bin': range(1, n_bins+1),
    'Observed': observed_counts,
    'Expected': np.round(expected_counts, 2)
})
print(df_counts)

# Chi-squared test
chisq_stat = np.sum((observed_counts - expected_counts)**2 / expected_counts)
df_chisq = n_bins - 1  # degrees of freedom
chisq_pval = 1 - stats.chi2.cdf(chisq_stat, df_chisq)

print("\nChi-squared test result:")
print(f"  χ² = {chisq_stat:.4f}")
print(f"  df = {df_chisq}")
print(f"  p-value = {chisq_pval:.4f}")

# Visualize
x_pos = np.arange(n_bins)
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x_pos - width/2, observed_counts, width, label='Observed', color='lightblue')
ax.bar(x_pos + width/2, expected_counts, width, label='Expected', color='salmon')
ax.set_xlabel('Bin')
ax.set_ylabel('Count')
ax.set_title('Chi-squared Test: Observed vs Expected Counts')
ax.set_xticks(x_pos)
ax.set_xticklabels(range(1, n_bins+1))
ax.legend()
plt.savefig('../plots/dist_chisq_test.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_chisq_test.png")

# ============================================
# PROBLEMS WITH CHI-SQUARED TEST
# ============================================
print("\n" + "="*50)
print("PROBLEMS WITH CHI-SQUARED TEST")
print("="*50 + "\n")

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
print("  ✓ Kolmogorov-Smirnov test (scipy.stats.kstest)")
print("  ✓ Bootstrap testing")
print("  ✓ Smooth tests of goodness of fit")
print("  ✓ Anderson-Darling test")

# ============================================
# BETTER ALTERNATIVE: KOLMOGOROV-SMIRNOV TEST
# ============================================
print("\n" + "="*50)
print("BETTER ALTERNATIVE: KOLMOGOROV-SMIRNOV TEST")
print("="*50 + "\n")

print("Kolmogorov-Smirnov (K-S) test:")
print("  - Compares empirical CDF to theoretical CDF")
print("  - Test statistic: D = max|F_empirical(x) - F_theoretical(x)|")
print("  - No binning required!")
print("  - More powerful than chi-squared for continuous data")

# K-S test for exponential data
ks_stat, ks_pval = stats.kstest(sample_data, lambda x: stats.expon.cdf(x, scale=1/rate_mle))
print("\nKolmogorov-Smirnov test result:")
print(f"  D = {ks_stat:.4f}")
print(f"  p-value = {ks_pval:.4f}")

if ks_pval > 0.05:
    print("Conclusion: Cannot reject null hypothesis")
    print("            Data is consistent with exponential distribution")
else:
    print("Conclusion: Reject null hypothesis")
    print("            Data does not follow exponential distribution")

# Visualize K-S statistic
plt.figure(figsize=(10, 6))
sorted_data = np.sort(sample_data)
ecdf_vals = np.arange(1, len(sorted_data)+1) / len(sorted_data)
plt.step(sorted_data, ecdf_vals, 'b-', lw=2, label='Empirical CDF', where='post')
x_seq = np.linspace(0, np.max(sample_data), 1000)
plt.plot(x_seq, stats.expon.cdf(x_seq, scale=1/rate_mle), 'r-', lw=2,
         label='Theoretical CDF')

# Find where maximum difference occurs
theo_cdf = stats.expon.cdf(sorted_data, scale=1/rate_mle)
diffs = np.abs(ecdf_vals - theo_cdf)
max_idx = np.argmax(diffs)
x_max = sorted_data[max_idx]

plt.plot([x_max, x_max], [ecdf_vals[max_idx], theo_cdf[max_idx]], 'g--', lw=3,
         label=f'Max Difference (D={ks_stat:.4f})')
plt.xlabel('Value')
plt.ylabel('Cumulative Probability')
plt.title('Kolmogorov-Smirnov Test Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../plots/dist_ks_test.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_ks_test.png")

# ============================================
# BOOTSTRAP TESTING
# ============================================
print("\n" + "="*50)
print("BOOTSTRAP TESTING")
print("="*50 + "\n")

print("Bootstrap approach for goodness-of-fit:")
print("  1. Fit distribution to observed data")
print("  2. Generate many bootstrap samples from fitted distribution")
print("  3. Calculate test statistic for each bootstrap sample")
print("  4. Compare observed test statistic to bootstrap distribution")

# Bootstrap K-S test
n_boot = 1000
boot_stats = np.zeros(n_boot)

for i in range(n_boot):
    boot_sample = np.random.exponential(scale=1/rate_mle, size=len(sample_data))
    boot_stats[i], _ = stats.kstest(boot_sample, 
                                      lambda x: stats.expon.cdf(x, scale=1/rate_mle))

# Calculate p-value
bootstrap_pval = np.mean(boot_stats >= ks_stat)

print(f"\nBootstrap K-S test (B = {n_boot}):")
print(f"Observed K-S statistic: {ks_stat:.4f}")
print(f"Bootstrap p-value: {bootstrap_pval:.4f}")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(boot_stats, bins=30, density=True, alpha=0.7, color='lightgray',
         edgecolor='white')
plt.axvline(ks_stat, color='red', linestyle='--', lw=2,
            label=f'Observed D = {ks_stat:.4f}\np = {bootstrap_pval:.4f}')
plt.xlabel('K-S Statistic')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of K-S Statistic')
plt.legend()
plt.savefig('../plots/dist_bootstrap_test.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_bootstrap_test.png")

# ============================================
# COMPARISON OF TESTS
# ============================================
print("\n" + "="*50)
print("COMPARISON OF GOODNESS-OF-FIT TESTS")
print("="*50 + "\n")

print("Test results summary:")
print("")
print(f"Chi-squared test:      χ² = {chisq_stat:.4f}, p = {chisq_pval:.4f}")
print(f"Kolmogorov-Smirnov:    D = {ks_stat:.4f},  p = {ks_pval:.4f}")
print(f"Bootstrap K-S:         D = {ks_stat:.4f},  p = {bootstrap_pval:.4f}")
print("")
print("All tests agree: data is consistent with exponential distribution")

# ============================================
# PRACTICAL EXAMPLE: TESTING NORMALITY
# ============================================
print("\n" + "="*50)
print("PRACTICAL EXAMPLE: TESTING NORMALITY")
print("="*50 + "\n")

print("Generate data from mixture of normals (not truly normal)")
print("Test if various methods can detect the departure from normality")

# Generate non-normal data (mixture)
np.random.seed(123)
mixture_data = np.concatenate([
    np.random.normal(0, 1, 400),
    np.random.normal(3, 0.5, 100)
])

# Fit normal distribution
mean_est = np.mean(mixture_data)
sd_est = np.std(mixture_data, ddof=1)

print(f"\nFitted normal: mean = {mean_est:.4f}, sd = {sd_est:.4f}")

# Visual inspection
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Histogram with fitted normal
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(mixture_data, bins=30, density=True, alpha=0.6, color='lightblue',
         edgecolor='white', label='Data histogram')
x_norm = np.linspace(np.min(mixture_data), np.max(mixture_data), 1000)
ax1.plot(x_norm, stats.norm.pdf(x_norm, mean_est, sd_est), 'r-', lw=2,
         label='Fitted Normal')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.set_title('Histogram with Fitted Normal')
ax1.legend()

# Q-Q plot
ax2 = fig.add_subplot(gs[0, 1])
stats.probplot(mixture_data, dist='norm', plot=ax2)
ax2.set_title('Q-Q Plot')

# ECDF comparison
ax3 = fig.add_subplot(gs[1, 0])
sorted_mix = np.sort(mixture_data)
ecdf_mix = np.arange(1, len(sorted_mix)+1) / len(sorted_mix)
ax3.step(sorted_mix, ecdf_mix, 'b-', lw=2, label='Empirical CDF', where='post')
ax3.plot(x_norm, stats.norm.cdf(x_norm, mean_est, sd_est), 'r-', lw=2,
         label='Theoretical CDF')
ax3.set_xlabel('Value')
ax3.set_ylabel('Cumulative Probability')
ax3.set_title('Empirical vs Theoretical CDF')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Box plot
ax4 = fig.add_subplot(gs[1, 1])
ax4.boxplot(mixture_data, vert=False)
ax4.set_xlabel('Value')
ax4.set_title('Box Plot')

plt.savefig('../plots/dist_normality_test.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_normality_test.png")

# Statistical tests
print("\n--- Statistical Tests for Normality ---")

# Shapiro-Wilk test
if len(mixture_data) <= 5000:
    shapiro_stat, shapiro_pval = stats.shapiro(mixture_data)
    print(f"Shapiro-Wilk test: W = {shapiro_stat:.4f}, p = {shapiro_pval:.4f}")

# K-S test for normality
ks_norm_stat, ks_norm_pval = stats.kstest(mixture_data, 
                                           lambda x: stats.norm.cdf(x, mean_est, sd_est))
print(f"K-S test:          D = {ks_norm_stat:.4f}, p = {ks_norm_pval:.4f}")

print("\nConclusion:")
if shapiro_pval < 0.05:
    print("  Shapiro-Wilk test rejects normality (p < 0.05)")
    print("  The data does NOT appear to be normally distributed")
else:
    print("  Tests suggest data may not be perfectly normal")
    print("  Visual inspection (Q-Q plot) shows deviation in tails")

# ============================================
# CALIBRATION PLOTS
# ============================================
print("\n" + "="*50)
print("CALIBRATION PLOTS")
print("="*50 + "\n")

print("Calibration: Check if predicted probabilities match observed frequencies")
print("")
print("For a well-fitted model:")
print("  - If we predict P(Y=1) = 0.7, about 70% should actually be Y=1")
print("  - Calibration plot: predicted probability vs observed frequency")

# Generate binary data with known probabilities
np.random.seed(456)
n_cal = 1000
x_cal = np.random.uniform(-3, 3, n_cal)
true_prob = 1 / (1 + np.exp(-x_cal))  # logistic function
y_cal = np.random.binomial(1, true_prob)

# Fit logistic regression
from sklearn.linear_model import LogisticRegression
cal_model = LogisticRegression()
cal_model.fit(x_cal.reshape(-1, 1), y_cal)
pred_prob = cal_model.predict_proba(x_cal.reshape(-1, 1))[:, 1]

# Create calibration plot
n_cal_bins = 10
cal_bins = pd.cut(pred_prob, bins=np.linspace(0, 1, n_cal_bins+1), include_lowest=True)
cal_df = pd.DataFrame({
    'pred': pred_prob,
    'observed': y_cal,
    'bin': cal_bins
})

cal_summary = cal_df.groupby('bin', observed=True).agg({'pred': 'mean', 'observed': 'mean'})

plt.figure(figsize=(10, 6))
plt.scatter(cal_summary['pred'], cal_summary['observed'], s=150, 
            color='blue', alpha=0.6, edgecolors='black', linewidth=1.5)
plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect calibration')
plt.xlabel('Predicted Probability')
plt.ylabel('Observed Frequency')
plt.title('Calibration Plot')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('../plots/dist_calibration.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: dist_calibration.png")

print("\nCalibration results:")
print("  Points near diagonal = well-calibrated")
print("  Points above diagonal = underestimating probability")
print("  Points below diagonal = overestimating probability")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*50)
print("SUMMARY")
print("="*50 + "\n")

print("Key takeaways:")
print("")
print("✓ RANDOM NUMBER GENERATION")
print("  - Python has built-in generators: numpy.random and scipy.stats")
print("  - Also provides density (pdf), CDF (cdf), quantile (ppf) functions")
print("")
print("✓ DISTRIBUTIONS IN PYTHON")
print("  - Many continuous distributions: normal, exponential, gamma, beta, etc.")
print("  - Many discrete distributions: binomial, Poisson, geometric, etc.")
print("  - Easy to work with using scipy.stats")
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
print("  - scipy.stats.kstest(): Kolmogorov-Smirnov test (no binning, more powerful)")
print("  - Bootstrap testing: resampling approach for p-values")
print("  - Smooth tests of goodness of fit: more sophisticated methods")
print("  - Anderson-Darling test: gives more weight to tails")
print("  - Shapiro-Wilk test: specifically for testing normality")

print("\n" + "="*60)
print("DISTRIBUTIONS TUTORIAL COMPLETE")
print("="*60 + "\n")

import glob
final_plot_count = len(glob.glob("../plots/dist_*.png"))
print(f"Total plots generated: {final_plot_count}")
print("\nAll plots saved to: ../plots/")
print("\nThank you for completing this tutorial!")
