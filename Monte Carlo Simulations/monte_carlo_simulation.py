"""
MONTE CARLO SIMULATIONS
Statistical Computing Tutorial - Python Version

Topics covered:
1. Ordinary Monte Carlo (OMC) theory
2. Monte Carlo integration examples
3. Approximating distributions
4. Bootstrap and permutation methods
5. Toy collector exercise (Coupon Collector Problem)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, gamma, t as t_dist
import pandas as pd
import time
import os

# Create plots directory if it doesn't exist
os.makedirs('../plots', exist_ok=True)

print("=" * 70)
print("MONTE CARLO SIMULATIONS TUTORIAL")
print("=" * 70)
print()

# Set seed for reproducibility
np.random.seed(42)

# =============================================================================
# PART 1: ORDINARY MONTE CARLO (OMC) - THEORY
# =============================================================================

print("\n" + "=" * 70)
print("PART 1: ORDINARY MONTE CARLO - THEORY")
print("=" * 70)
print()

print("The 'Monte Carlo method' refers to the theory and practice of learning")
print("about probability distributions by simulation rather than calculus.\n")

print("In Ordinary Monte Carlo (OMC) we use IID simulations from the")
print("distribution of interest.\n")

print("Setup:")
print("-" * 60)
print("Suppose X₁, X₂, ... are IID simulations from some distribution,")
print("and we want to know an expectation:\n")
print("  θ = E[Y₁] = E[g(X₁)]\n")

print("Law of Large Numbers (LLN):")
print("-" * 60)
print("  ȳₙ = (1/n) Σᵢ Yᵢ = (1/n) Σᵢ g(Xᵢ)\n")
print("converges in probability to θ.\n")

print("Central Limit Theorem (CLT):")
print("-" * 60)
print("  √n(ȳₙ - θ)/σ →ᵈ N(0,1)\n")
print("That is, for sufficiently large n:")
print("  ȳₙ ~ N(θ, σ²/n)\n")

print("Standard Error Estimation:")
print("-" * 60)
print("We can estimate the standard error σ/√n with sₙ/√n")
print("where sₙ is the sample standard deviation.\n")

print("KEY INSIGHT:")
print("-" * 60)
print("The theory of OMC is just the theory of frequentist statistical inference.")
print("The only differences are that:\n")
print("1. The 'data' X₁,...,Xₙ are computer simulations rather than")
print("   measurements on objects in the real world\n")
print("2. The 'sample size' n is the number of computer simulations")
print("   rather than the size of some real world data\n")
print("3. The unknown parameter θ is in principle completely known,")
print("   given by some integral, which we are unable to do.\n")

print("VECTOR CASE:")
print("-" * 60)
print("Everything works just the same when the data X₁, X₂, ...")
print("(which are computer simulations) are vectors.")
print("But the functions of interest g(X₁), g(X₂), ... are scalars.\n")

print("LIMITATION:")
print("-" * 60)
print("OMC works great, but it can be very difficult to simulate IID")
print("simulations of random variables or random vectors whose")
print("distribution is not brand name distributions.\n")

# =============================================================================
# PART 2: APPROXIMATING THE BINOMIAL DISTRIBUTION
# =============================================================================

print("\n" + "=" * 70)
print("PART 2: APPROXIMATING THE BINOMIAL DISTRIBUTION")
print("=" * 70)
print()

print("Problem: Flip a coin 10 times. What is P(more than 3 heads)?")
print("-" * 60)
print()

print("This is trivial for the Binomial distribution, but we'll use")
print("Monte Carlo simulation to demonstrate the method.\n")

# Monte Carlo simulation
runs = 10000

def one_trial():
    return np.sum(np.random.randint(0, 2, 10)) > 3

print(f"Running {runs} Monte Carlo simulations...\n")

np.random.seed(123)
mc_results = np.array([one_trial() for _ in range(runs)])
mc_binom = np.mean(mc_results)

# Exact probability
exact_prob = 1 - stats.binom.cdf(3, 10, 0.5)

# Calculate Monte Carlo standard error
mc_se = np.sqrt(mc_binom * (1 - mc_binom) / runs)

print("RESULTS:")
print(f"Monte Carlo estimate: {mc_binom:.6f}")
print(f"Exact probability:    {exact_prob:.6f}")
print(f"Absolute error:       {abs(mc_binom - exact_prob):.6f}")
print(f"\nMonte Carlo standard error: {mc_se:.6f}")
print(f"95% Confidence Interval: [{mc_binom - 1.96*mc_se:.6f}, {mc_binom + 1.96*mc_se:.6f}]")

in_ci = (exact_prob >= mc_binom - 1.96*mc_se) and (exact_prob <= mc_binom + 1.96*mc_se)
print(f"Exact value in CI: {'YES ✓' if in_ci else 'NO ✗'}")

# =============================================================================
# EXERCISE SOLUTION: MONTE CARLO STANDARD ERROR
# =============================================================================

print("\n\n" + "=" * 70)
print("EXERCISE: ESTIMATING MONTE CARLO STANDARD ERROR")
print("=" * 70)
print()

print("For a binary outcome (success/failure), the standard error is:")
print("  SE = √[p(1-p)/n]\n")

print("where:")
print("  p = estimated probability (proportion of successes)")
print("  n = number of Monte Carlo simulations\n")

print("In our case:")
print(f"  p = {mc_binom:.6f}")
print(f"  n = {runs}")
print(f"  SE = √[{mc_binom:.6f} × {1-mc_binom:.6f} / {runs}] = {mc_se:.6f}")

# Demonstrate convergence
print("\n\nDemonstrating convergence with different sample sizes:")
print("-" * 60)
print()

sample_sizes = [100, 1000, 10000, 100000]
results_table = []

np.random.seed(456)
for n in sample_sizes:
    mc_trials = np.array([one_trial() for _ in range(n)])
    p_hat = np.mean(mc_trials)
    se = np.sqrt(p_hat * (1 - p_hat) / n)
    ci_lower = p_hat - 1.96 * se
    ci_upper = p_hat + 1.96 * se
    in_ci_check = (exact_prob >= ci_lower) and (exact_prob <= ci_upper)
    
    results_table.append({
        'n': n,
        'estimate': p_hat,
        'se': se,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'in_ci': in_ci_check
    })

results_df = pd.DataFrame(results_table)
print(results_df.to_string(index=False))

print(f"\nExact probability: {exact_prob:.6f}")
print("\nNote: Standard error decreases as O(1/√n)")

# =============================================================================
# PART 3: APPROXIMATING π
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 3: APPROXIMATING π USING MONTE CARLO")
print("=" * 70)
print()

print("Geometric Approach to Estimating π")
print("-" * 60)
print()

print("Key insight:")
print("  Area of a circle = πr²")
print("  Area of square containing the circle = (2r)² = 4r²\n")

print("Therefore, the ratio of areas is:")
print("  πr² / 4r² = π/4\n")

print("If we can empirically determine the ratio of the area of the")
print("circle to the area of the square, we can multiply by 4 to get π.\n")

print("Method:")
print("-" * 60)
print("1. Randomly sample (x, y) points on the unit square centered at 0")
print("   (i.e., x, y ∈ [-0.5, 0.5])")
print("2. Check if x² + y² ≤ 0.5² (point is inside the circle)")
print("3. Ratio of points in circle × 4 = estimate of π\n")

# Monte Carlo estimation of π
runs = 100000
np.random.seed(2024)

print(f"Running {runs} Monte Carlo simulations...\n")

xs = np.random.uniform(-0.5, 0.5, runs)
ys = np.random.uniform(-0.5, 0.5, runs)
in_circle = (xs**2 + ys**2) <= 0.5**2
mc_pi = np.mean(in_circle) * 4

# Calculate standard error
p = np.mean(in_circle)
se_p = np.sqrt(p * (1 - p) / runs)
se_pi = 4 * se_p

print("RESULTS:")
print(f"Monte Carlo estimate of π: {mc_pi:.6f}")
print(f"True value of π:           {np.pi:.6f}")
print(f"Absolute error:            {abs(mc_pi - np.pi):.6f}")
print(f"Relative error:            {100 * abs(mc_pi - np.pi) / np.pi:.4f}%")
print(f"\nProportion in circle:      {p:.6f}")
print(f"Standard error of π:       {se_pi:.6f}")
print(f"95% CI for π:              [{mc_pi - 1.96*se_pi:.6f}, {mc_pi + 1.96*se_pi:.6f}]")

in_ci_pi = (np.pi >= mc_pi - 1.96*se_pi) and (np.pi <= mc_pi + 1.96*se_pi)
print(f"True π in CI: {'YES ✓' if in_ci_pi else 'NO ✗'}")

# Convergence analysis
print("\n\nConvergence analysis with different sample sizes:")
print("-" * 60)
print()

sample_sizes_pi = [100, 1000, 10000, 100000, 1000000]
pi_results = []

np.random.seed(12345)
for n in sample_sizes_pi:
    xs_temp = np.random.uniform(-0.5, 0.5, n)
    ys_temp = np.random.uniform(-0.5, 0.5, n)
    in_circle_temp = (xs_temp**2 + ys_temp**2) <= 0.5**2
    pi_est = np.mean(in_circle_temp) * 4
    
    p_temp = np.mean(in_circle_temp)
    se_temp = 4 * np.sqrt(p_temp * (1 - p_temp) / n)
    
    pi_results.append({
        'n': n,
        'estimate': pi_est,
        'error': abs(pi_est - np.pi),
        'rel_error_pct': 100 * abs(pi_est - np.pi) / np.pi,
        'se': se_temp
    })

pi_df = pd.DataFrame(pi_results)
print(pi_df.to_string(index=False))

print(f"\nTrue π = {np.pi:.10f}")

# Visualization
print("\n\nCreating π approximation visualization...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Visualization of the method
np.random.seed(999)
n_vis = 2000
xs_vis = np.random.uniform(-0.5, 0.5, n_vis)
ys_vis = np.random.uniform(-0.5, 0.5, n_vis)
in_circle_vis = (xs_vis**2 + ys_vis**2) <= 0.5**2

axes[0, 0].scatter(xs_vis[in_circle_vis], ys_vis[in_circle_vis], 
                   c='blue', s=1, alpha=0.5)
axes[0, 0].scatter(xs_vis[~in_circle_vis], ys_vis[~in_circle_vis], 
                   c='red', s=1, alpha=0.5)
theta = np.linspace(0, 2*np.pi, 200)
axes[0, 0].plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), 'k-', lw=2)
axes[0, 0].plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 'k-', lw=2)
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
axes[0, 0].set_title(f'Monte Carlo π Estimation (n={n_vis})')
axes[0, 0].set_aspect('equal')
pi_est_vis = np.mean(in_circle_vis) * 4
axes[0, 0].legend([f'Inside ({100*np.mean(in_circle_vis):.0f}%)',
                   f'Outside ({100*np.mean(~in_circle_vis):.0f}%)',
                   f'π ≈ {pi_est_vis:.4f}'], fontsize=8)

# Plot 2: Convergence of π estimate
np.random.seed(111)
n_conv = 50000
xs_conv = np.random.uniform(-0.5, 0.5, n_conv)
ys_conv = np.random.uniform(-0.5, 0.5, n_conv)
in_circle_conv = (xs_conv**2 + ys_conv**2) <= 0.5**2
cumulative_pi = np.cumsum(in_circle_conv) / np.arange(1, n_conv + 1) * 4

axes[0, 1].plot(range(1, n_conv + 1), cumulative_pi, 'b-', lw=2)
axes[0, 1].axhline(np.pi, color='red', lw=2, ls='--')
axes[0, 1].set_xlabel('Number of simulations')
axes[0, 1].set_ylabel('Estimated π')
axes[0, 1].set_title('Convergence of π Estimate')
axes[0, 1].set_ylim(2.8, 3.5)
axes[0, 1].legend(['MC estimate', 'True π'])

# Plot 3: Error vs sample size
axes[0, 2].loglog([r['n'] for r in pi_results], [r['error'] for r in pi_results], 
                  'o-r', lw=2, markersize=8)
axes[0, 2].set_xlabel('Sample size (n)')
axes[0, 2].set_ylabel('Absolute error')
axes[0, 2].set_title('Error vs Sample Size')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: Distribution of π estimates
np.random.seed(222)
n_reps = 1000
n_per_rep = 10000
pi_estimates = []
for _ in range(n_reps):
    xs_temp = np.random.uniform(-0.5, 0.5, n_per_rep)
    ys_temp = np.random.uniform(-0.5, 0.5, n_per_rep)
    pi_estimates.append(np.mean(xs_temp**2 + ys_temp**2 <= 0.5**2) * 4)

axes[1, 0].hist(pi_estimates, bins=30, density=True, color='lightgreen', 
                edgecolor='black', alpha=0.7)
axes[1, 0].axvline(np.pi, color='red', lw=3, ls='--')
axes[1, 0].axvline(np.mean(pi_estimates), color='blue', lw=2, ls='--')
axes[1, 0].set_xlabel('Estimated π')
axes[1, 0].set_title(f'Distribution of π Estimates\n(1000 reps, n={n_per_rep} each)')
axes[1, 0].legend(['True π', 'Mean estimate'], fontsize=8)

# Plot 5: Relative error
axes[1, 1].semilogx([r['n'] for r in pi_results], [r['rel_error_pct'] for r in pi_results],
                    'o-', color='purple', lw=2, markersize=8)
axes[1, 1].set_xlabel('Sample size (n)')
axes[1, 1].set_ylabel('Relative error (%)')
axes[1, 1].set_title('Relative Error vs Sample Size')
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Standard error
axes[1, 2].loglog([r['n'] for r in pi_results], [r['se'] for r in pi_results],
                  'o-', color='orange', lw=2, markersize=8)
axes[1, 2].set_xlabel('Sample size (n)')
axes[1, 2].set_ylabel('Standard error')
axes[1, 2].set_title('Standard Error vs Sample Size')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/monte_carlo_pi_approximation.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/monte_carlo_pi_approximation.png")
plt.close()

# Detailed π plot
print("Creating detailed π visualization...")
plt.figure(figsize=(8, 8))
plt.scatter(xs[in_circle], ys[in_circle], c='blue', s=0.1, alpha=0.5)
plt.scatter(xs[~in_circle], ys[~in_circle], c='grey', s=0.1, alpha=0.5)
theta = np.linspace(0, 2*np.pi, 200)
plt.plot(0.5 * np.cos(theta), 0.5 * np.sin(theta), 'k-', lw=2)
plt.plot([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5], 'k-', lw=2)
plt.xlabel('')
plt.ylabel('')
plt.title(f'MC Approximation of π = {mc_pi:.4f}')
plt.axis('equal')
plt.savefig('../plots/monte_carlo_pi_detailed.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/monte_carlo_pi_detailed.png")
plt.close()

# =============================================================================
# PART 4: MONTE CARLO INTEGRATION WITH SEQUENTIAL STOPPING
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 4: MONTE CARLO INTEGRATION")
print("=" * 70)
print()

print("Example: Intractable Expectation")
print("-" * 60)
print()

print("Let X ~ Gamma(3/2, 1), i.e.")
print("  f(x) = (2/√π) √x e^(-x) I(x > 0)\n")

print("Suppose we want to find:")
print("  θ = E[1/((X+1)log(X+3))]")
print("    = ∫₀^∞ 1/((x+1)log(x+3)) * (2/√π) √x e^(-x) dx\n")

print("The expectation (or integral) θ is intractable - we don't know")
print("how to compute it analytically.\n")

print("GOAL: Estimate θ such that the 95% CI length is less than 0.002\n")

# Initial Monte Carlo estimation
n = 1000
np.random.seed(4040)

print(f"Initial estimation with n = {n}:")
print("-" * 40)

x = np.random.gamma(3/2, scale=1, size=n)
print(f"Mean of X (theoretical = 3/2 = 1.5): {np.mean(x):.6f}")

y = 1 / ((x + 1) * np.log(x + 3))
est = np.mean(y)
print(f"\nInitial estimate of θ: {est:.7f}")

mcse = np.std(y, ddof=1) / np.sqrt(len(y))
interval = est + np.array([-1, 1]) * 1.96 * mcse
print(f"Monte Carlo SE: {mcse:.7f}")
print(f"95% CI: [{interval[0]:.7f}, {interval[1]:.7f}]")
print(f"CI length: {np.diff(interval)[0]:.7f}")

# Sequential stopping rule
print("\n\nApplying Sequential Stopping Rule:")
print("-" * 60)
print("Target CI length: 0.002")
print("Adding samples in batches of 1000 until target is reached...\n")

eps = 0.002
len_ci = np.diff(interval)[0]
plotting_var = [np.concatenate([[est], interval])]
iteration = 1

while len_ci > eps:
    new_x = np.random.gamma(3/2, scale=1, size=n)
    new_y = 1 / ((new_x + 1) * np.log(new_x + 3))
    y = np.concatenate([y, new_y])
    est = np.mean(y)
    mcse = np.std(y, ddof=1) / np.sqrt(len(y))
    interval = est + np.array([-1, 1]) * 1.96 * mcse
    len_ci = np.diff(interval)[0]
    plotting_var.append(np.concatenate([[est], interval]))
    iteration += 1
    
    if iteration % 20 == 0:
        print(f"  Iteration {iteration:3d}: n = {len(y):6d}, CI length = {len_ci:.6f}")

plotting_var = np.array(plotting_var)

print("\nSequential stopping complete!")
print(f"Final sample size: {len(y)}")
print(f"Final estimate: {est:.7f}")
print(f"Final 95% CI: [{interval[0]:.7f}, {interval[1]:.7f}]")
print(f"Final CI length: {len_ci:.7f} (target: {eps:.3f})")
print(f"Final SE: {mcse:.7f}")

# Visualization
print("\n\nCreating sequential stopping visualization...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

temp = np.arange(1000, len(y) + 1, 1000)
axes[0].plot(temp, plotting_var[:, 0], 'k-', lw=2)
axes[0].plot(temp, plotting_var[:, 1], 'r-', lw=2)
axes[0].plot(temp, plotting_var[:, 2], 'r-', lw=2)
axes[0].axhline(est, color='blue', lw=1, ls='--')
axes[0].set_xlabel('Sample size (n)')
axes[0].set_ylabel('Estimate of θ')
axes[0].set_title('Sequential Estimation with 95% CI')
axes[0].legend(['Estimate', '95% CI', '', 'Final estimate'])
axes[0].grid(True, alpha=0.3)

ci_lengths = plotting_var[:, 2] - plotting_var[:, 1]
axes[1].plot(temp, ci_lengths, 'purple', lw=2)
axes[1].axhline(eps, color='red', lw=2, ls='--')
axes[1].text(max(temp) * 0.7, eps * 1.5, f'Target = {eps:.3f}', color='red')
axes[1].set_xlabel('Sample size (n)')
axes[1].set_ylabel('CI length')
axes[1].set_title('Convergence of CI Length')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../plots/monte_carlo_integration_sequential.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/monte_carlo_integration_sequential.png")
plt.close()

# Additional analysis
print("\n\nAdditional Analysis:")
print("-" * 60)
print(f"Sample size increase: 1000 → {len(y)} ({len(y) / 1000:.1f}x)")
print(f"CI length reduction: {ci_lengths[0]:.6f} → {len_ci:.6f} ({ci_lengths[0] / len_ci:.1f}x)")

# Distribution plots
print("\n\nCreating integration distribution plots...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Plot 1: Distribution of X
axes[0].hist(x[:10000], bins=50, density=True, color='lightblue', edgecolor='black', alpha=0.7)
x_range = np.linspace(0, max(x[:10000]), 200)
axes[0].plot(x_range, stats.gamma.pdf(x_range, 3/2, scale=1), 'r-', lw=2)
axes[0].set_xlabel('X')
axes[0].set_title('Distribution of X ~ Gamma(3/2, 1)')

# Plot 2: Distribution of Y
axes[1].hist(y, bins=100, density=True, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1].axvline(np.mean(y), color='red', lw=2, ls='--')
axes[1].set_xlabel('Y')
axes[1].set_title('Distribution of Y = 1/((X+1)log(X+3))')
axes[1].set_xlim(np.percentile(y, [0, 99]))

# Plot 3: Scatter X vs Y
axes[2].scatter(x[:5000], y[:5000], s=1, alpha=0.3, c='blue')
axes[2].axhline(np.mean(y), color='red', lw=2, ls='--')
axes[2].set_xlabel('X ~ Gamma(3/2, 1)')
axes[2].set_ylabel('Y = 1/((X+1)log(X+3))')
axes[2].set_title('Relationship between X and Y')

plt.tight_layout()
plt.savefig('../plots/monte_carlo_integration_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/monte_carlo_integration_distribution.png")
plt.close()

# =============================================================================
# PART 5: BOOTSTRAP AND PERMUTATION METHODS
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 5: BOOTSTRAP AND PERMUTATION METHODS")
print("=" * 70)
print()

print("HIGH-DIMENSIONAL EXAMPLES:")
print("Monte Carlo methods are essential for complex, high-dimensional problems:")
print("  - FiveThirtyEight's Election Forecast")
print("  - FiveThirtyEight's NBA Predictions")
print("  - Vanguard's Retirement Nest Egg Calculator")
print("  - Fisher's Exact Test in Python\n")

# Permutations
print("PERMUTATIONS WITH numpy:")
print("-" * 60)
print("numpy.random.permutation() works on any array-like object\n")

print("Example 1: Simple permutations")
np.random.seed(5050)
print("np.random.permutation(5):")
print(np.random.permutation(5))

print("\nnp.random.permutation(range(1, 7)):")
print(np.random.permutation(range(1, 7)))

print("\nExample 2: Permuting arrays")
print("Multiple permutations of ['Curly', 'Larry', 'Moe', 'Shemp']:")
np.random.seed(6060)
stooges = ['Curly', 'Larry', 'Moe', 'Shemp']
stooges_perms = np.array([np.random.permutation(stooges) for _ in range(3)]).T
print(stooges_perms)

# Bootstrap
print("\n\nRESAMPLING WITH numpy - BOOTSTRAP:")
print("-" * 60)
print("Resampling from any existing distribution gives bootstrap estimators\n")

print("Key difference from jackknife:")
print("  - Jackknife: removes one point and recalculates")
print("  - Bootstrap: resamples same length WITH REPLACEMENT\n")

def bootstrap_resample(arr):
    """Bootstrap resample with replacement"""
    return np.random.choice(arr, size=len(arr), replace=True)

print("Example: Bootstrap resampling")
np.random.seed(7070)
print("Bootstrap resamples of [6, 7, 8, 9, 10]:")
bootstrap_example = np.array([bootstrap_resample(np.arange(6, 11)) for _ in range(5)]).T
print(bootstrap_example)

print("\nNote: Values can (and do) repeat due to replacement")

# Bootstrap two-sample test
print("\n\nBOOTSTRAP TEST: TWO-SAMPLE DIFFERENCE IN MEANS")
print("-" * 60)
print()

print("The 2-sample t-test checks for differences in means according to")
print("a known null distribution. Let's use bootstrap to generate the")
print("sampling distribution under the bootstrap assumption.\n")

print("Example: Simulated cat heart weights by sex\n")

# Simulate cat data (similar to MASS::cats)
np.random.seed(123)
n_males = 97
n_females = 47
male_hwt = np.random.normal(11.32, 2.54, n_males)
female_hwt = np.random.normal(9.20, 1.36, n_females)

cats_data = pd.DataFrame({
    'Sex': ['M'] * n_males + ['F'] * n_females,
    'Hwt': np.concatenate([male_hwt, female_hwt])
})

def diff_in_means(df):
    """Calculate difference in means (M - F)"""
    return df[df['Sex'] == 'M']['Hwt'].mean() - df[df['Sex'] == 'F']['Hwt'].mean()

obs_diff = diff_in_means(cats_data)
print(f"Observed difference in means (M - F): {obs_diff:.4f} g")

print("\nSummary by sex:")
print(f"Males:   n={n_males}, mean={male_hwt.mean():.2f} g, sd={male_hwt.std(ddof=1):.2f} g")
print(f"Females: n={n_females}, mean={female_hwt.mean():.2f} g, sd={female_hwt.std(ddof=1):.2f} g")

# Bootstrap resampling
print("\nGenerating bootstrap distribution (1000 replicates)...")
np.random.seed(8080)
n_boot = 1000
resample_diffs = np.zeros(n_boot)

for i in range(n_boot):
    boot_indices = np.random.choice(len(cats_data), size=len(cats_data), replace=True)
    boot_sample = cats_data.iloc[boot_indices].copy()
    resample_diffs[i] = diff_in_means(boot_sample)

print(f"\nBootstrap results:")
print(f"Mean of bootstrap diffs: {np.mean(resample_diffs):.4f} g")
print(f"SD of bootstrap diffs:   {np.std(resample_diffs, ddof=1):.4f} g")
print(f"95% CI (percentile):     [{np.percentile(resample_diffs, 2.5):.4f}, {np.percentile(resample_diffs, 97.5):.4f}] g")

# Compare with t-test
t_stat, p_value = stats.ttest_ind(male_hwt, female_hwt)
ci_ttest = stats.t.interval(0.95, len(male_hwt) + len(female_hwt) - 2,
                            loc=obs_diff,
                            scale=np.sqrt(np.var(male_hwt, ddof=1)/len(male_hwt) + 
                                         np.var(female_hwt, ddof=1)/len(female_hwt)))

print(f"\nComparison with t-test:")
print(f"t-test 95% CI: [{ci_ttest[0]:.4f}, {ci_ttest[1]:.4f}] g")
print(f"t-test p-value: {p_value:.6f}")

# Visualization
print("\nCreating bootstrap test visualization...")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Bootstrap distribution
axes[0, 0].hist(resample_diffs, bins=40, density=True, color='lightblue', 
                edgecolor='black', alpha=0.7)
axes[0, 0].axvline(obs_diff, color='red', lw=3, ls='--', label='Observed')
axes[0, 0].axvline(np.mean(resample_diffs), color='blue', lw=2, ls='--', label='Bootstrap mean')
ci_boot = np.percentile(resample_diffs, [2.5, 97.5])
axes[0, 0].axvline(ci_boot[0], color='darkgreen', lw=2, ls=':', label='95% CI')
axes[0, 0].axvline(ci_boot[1], color='darkgreen', lw=2, ls=':')
axes[0, 0].set_xlabel('Difference in heart weight (M - F, grams)')
axes[0, 0].set_title('Bootstrap Distribution of Difference in Means')
axes[0, 0].legend(fontsize=8)

# Plot 2: Original data
bp = axes[0, 1].boxplot([female_hwt, male_hwt], labels=['F', 'M'], patch_artist=True)
bp['boxes'][0].set_facecolor('pink')
bp['boxes'][1].set_facecolor('lightblue')
axes[0, 1].scatter([1, 2], [female_hwt.mean(), male_hwt.mean()], 
                   color='red', s=100, marker='D', zorder=3)
axes[0, 1].set_xlabel('Sex')
axes[0, 1].set_ylabel('Heart weight (grams)')
axes[0, 1].set_title('Cat Heart Weights by Sex')

# Plot 3: Q-Q plot
stats.probplot(resample_diffs, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Bootstrap Distribution')

# Plot 4: ECDF
sorted_diffs = np.sort(resample_diffs)
ecdf_y = np.arange(1, len(sorted_diffs) + 1) / len(sorted_diffs)
axes[1, 1].plot(sorted_diffs, ecdf_y, 'b-', lw=2)
axes[1, 1].axvline(obs_diff, color='red', lw=2, ls='--')
axes[1, 1].axvline(0, color='gray', lw=1, ls=':')
axes[1, 1].set_xlabel('Difference in heart weight (grams)')
axes[1, 1].set_ylabel('Cumulative probability')
axes[1, 1].set_title('ECDF of Bootstrap Differences')

plt.tight_layout()
plt.savefig('../plots/monte_carlo_bootstrap_test.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/monte_carlo_bootstrap_test.png")
plt.close()

# =============================================================================
# PART 6: TOY COLLECTOR EXERCISE
# =============================================================================

print("\n\n" + "=" * 70)
print("PART 6: TOY COLLECTOR EXERCISE (COUPON COLLECTOR PROBLEM)")
print("=" * 70)
print()

print("Problem: Children are enticed to buy cereal to collect action figures.")
print("Assume there are 15 action figures and each box contains exactly one,")
print("with each figure being equally likely initially.\n")

def simulate_collection(n_toys, probs=None, max_boxes=10000):
    """
    Simulate collecting all n_toys with given probabilities.
    
    Parameters:
    - n_toys: number of unique toys
    - probs: probability of each toy (None = equal probability)
    - max_boxes: maximum boxes to try
    
    Returns: number of boxes needed to collect all toys
    """
    if probs is None:
        probs = np.ones(n_toys) / n_toys
    
    collected = np.zeros(n_toys, dtype=bool)
    n_boxes = 0
    
    while not np.all(collected) and n_boxes < max_boxes:
        n_boxes += 1
        toy = np.random.choice(n_toys, p=probs)
        collected[toy] = True
    
    return n_boxes

# Questions 1 & 2: Equal probabilities
print("QUESTIONS 1 & 2: EQUAL PROBABILITIES (1/15 each)")
print("-" * 70)
print()

n_toys = 15
n_simulations = 10000

print(f"Running {n_simulations} simulations...")
np.random.seed(9090)
boxes_needed_equal = np.array([simulate_collection(n_toys) for _ in range(n_simulations)])

mean_boxes_equal = np.mean(boxes_needed_equal)
sd_boxes_equal = np.std(boxes_needed_equal, ddof=1)
se_boxes_equal = sd_boxes_equal / np.sqrt(n_simulations)

# Theoretical expectation
harmonic_number = np.sum(1 / np.arange(1, n_toys + 1))
theoretical_mean = n_toys * harmonic_number

print("\nRESULTS FOR EQUAL PROBABILITIES:")
print(f"Q1. Expected number of boxes (simulated):   {mean_boxes_equal:.2f}")
print(f"    Expected number of boxes (theoretical): {theoretical_mean:.2f}")
print(f"Q2. Standard deviation:                     {sd_boxes_equal:.2f}")
print(f"    Standard error of estimate:             {se_boxes_equal:.4f}")
print(f"    Median: {np.median(boxes_needed_equal):.0f}, Range: [{boxes_needed_equal.min()}, {boxes_needed_equal.max()}]")

quantiles_equal = np.percentile(boxes_needed_equal, [25, 50, 75, 90, 95])
print("\nQuantiles:")
for q, val in zip([25, 50, 75, 90, 95], quantiles_equal):
    print(f"  {q}%: {int(val)} boxes")

# Questions 3, 4, 5: Unequal probabilities
print("\n\nQUESTIONS 3, 4, 5: UNEQUAL PROBABILITIES")
print("-" * 70)
print()

toy_names = [chr(65 + i) for i in range(15)]  # A-O
toy_probs = np.array([.2, .1, .1, .1, .1, .1, .05, .05, .05, .05, .02, .02, .02, .02, .02])

print("Figure probabilities:")
prob_df = pd.DataFrame({'Figure': toy_names, 'Probability': toy_probs})
print(prob_df.to_string(index=False))
print(f"\nSum of probabilities: {np.sum(toy_probs):.2f} (must equal 1.0)")

print(f"\nRunning {n_simulations} simulations...")
np.random.seed(10101)
boxes_needed_unequal = np.array([simulate_collection(n_toys, toy_probs) 
                                  for _ in range(n_simulations)])

mean_boxes_unequal = np.mean(boxes_needed_unequal)
sd_boxes_unequal = np.std(boxes_needed_unequal, ddof=1)
se_boxes_unequal = sd_boxes_unequal / np.sqrt(n_simulations)

print("\nRESULTS FOR UNEQUAL PROBABILITIES:")
print(f"Q3. Expected number of boxes: {mean_boxes_unequal:.2f}")
print(f"Q4. Uncertainty of estimate:")
print(f"    Standard deviation: {sd_boxes_unequal:.2f}")
print(f"    Standard error:     {se_boxes_unequal:.4f}")
print(f"    95% CI: [{mean_boxes_unequal - 1.96*se_boxes_unequal:.2f}, {mean_boxes_unequal + 1.96*se_boxes_unequal:.2f}]")
print(f"    Relative error: {100*se_boxes_unequal/mean_boxes_unequal:.2f}%")

print("\nQ5. Probability of buying more than X boxes:")
for threshold in [50, 100, 200]:
    prob = np.mean(boxes_needed_unequal > threshold)
    count = np.sum(boxes_needed_unequal > threshold)
    print(f"    P(boxes > {threshold:3d}) = {prob:.4f} ({100*prob:.2f}%) - {count}/{n_simulations} simulations")

print(f"\nMedian: {np.median(boxes_needed_unequal):.0f}, Range: [{boxes_needed_unequal.min()}, {boxes_needed_unequal.max()}]")

quantiles_unequal = np.percentile(boxes_needed_unequal, [25, 50, 75, 90, 95, 99])
print("\nQuantiles:")
for q, val in zip([25, 50, 75, 90, 95, 99], quantiles_unequal):
    print(f"  {q}%: {int(val)} boxes")

# Visualization
print("\n\nCreating toy collector visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Equal probabilities
axes[0, 0].hist(boxes_needed_equal, bins=50, color='lightblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(mean_boxes_equal, color='red', lw=3, ls='--')
axes[0, 0].axvline(theoretical_mean, color='darkgreen', lw=3, ls=':')
axes[0, 0].set_xlabel('Number of boxes needed')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Equal Probabilities\n(each figure: 1/15)')
axes[0, 0].legend([f'Simulated: {mean_boxes_equal:.1f}',
                   f'Theoretical: {theoretical_mean:.1f}'], fontsize=8)

# Plot 2: Unequal probabilities
axes[0, 1].hist(boxes_needed_unequal, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(mean_boxes_unequal, color='red', lw=3, ls='--')
axes[0, 1].set_xlabel('Number of boxes needed')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Unequal Probabilities\n(rare figures: 0.02)')
axes[0, 1].legend([f'Mean: {mean_boxes_unequal:.1f}'], fontsize=8)

# Plot 3: CDFs comparison
sorted_equal = np.sort(boxes_needed_equal)
ecdf_equal = np.arange(1, len(sorted_equal) + 1) / len(sorted_equal)
sorted_unequal = np.sort(boxes_needed_unequal)
ecdf_unequal = np.arange(1, len(sorted_unequal) + 1) / len(sorted_unequal)

axes[0, 2].plot(sorted_equal, ecdf_equal, 'b-', lw=2)
axes[0, 2].plot(sorted_unequal, ecdf_unequal, 'r-', lw=2)
for threshold in [50, 100, 200]:
    axes[0, 2].axvline(threshold, color='gray', ls=':', alpha=0.5)
axes[0, 2].set_xlabel('Number of boxes')
axes[0, 2].set_ylabel('Cumulative Probability')
axes[0, 2].set_title('Cumulative Distribution Comparison')
axes[0, 2].legend(['Equal probs', 'Unequal probs', 'Thresholds'], fontsize=8)

# Plot 4: Box plots
bp = axes[1, 0].boxplot([boxes_needed_equal, boxes_needed_unequal],
                        labels=['Equal', 'Unequal'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][1].set_facecolor('lightcoral')
axes[1, 0].scatter([1, 2], [mean_boxes_equal, mean_boxes_unequal],
                   color='red', s=100, marker='D', zorder=3)
axes[1, 0].set_ylabel('Number of boxes needed')
axes[1, 0].set_title('Distribution Comparison')
axes[1, 0].legend(['Mean'], fontsize=8)

# Plot 5: Probability of exceeding thresholds
threshold_seq = np.arange(0, 401, 10)
prob_exceed_equal = np.array([np.mean(boxes_needed_equal > x) for x in threshold_seq])
prob_exceed_unequal = np.array([np.mean(boxes_needed_unequal > x) for x in threshold_seq])

axes[1, 1].plot(threshold_seq, prob_exceed_equal, 'b-', lw=2)
axes[1, 1].plot(threshold_seq, prob_exceed_unequal, 'r-', lw=2)
for h in [0.5, 0.9, 0.95]:
    axes[1, 1].axhline(h, color='gray', ls=':', alpha=0.5)
for v in [50, 100, 200]:
    axes[1, 1].axvline(v, color='gray', ls=':', alpha=0.5)
axes[1, 1].set_xlabel('Number of boxes')
axes[1, 1].set_ylabel('P(boxes > x)')
axes[1, 1].set_title('Probability of Exceeding Threshold')
axes[1, 1].set_ylim(0, 1)
axes[1, 1].legend(['Equal probs', 'Unequal probs'], fontsize=8)

# Plot 6: Figure probabilities
axes[1, 2].bar(toy_names, toy_probs, color='lightcoral', edgecolor='black')
axes[1, 2].axhline(1/15, color='blue', ls='--', lw=2)
axes[1, 2].text(5, 1/15 + 0.01, 'Equal prob = 1/15', color='blue')
axes[1, 2].set_xlabel('Figure')
axes[1, 2].set_ylabel('Probability')
axes[1, 2].set_title('Figure Probabilities\n(Unequal Case)')
axes[1, 2].set_ylim(0, max(toy_probs) * 1.1)

plt.tight_layout()
plt.savefig('../plots/monte_carlo_toy_collector.png', dpi=300, bbox_inches='tight')
print("Saved: ../plots/monte_carlo_toy_collector.png")
plt.close()

# Key insights
print("\n\nKEY INSIGHTS:")
print("-" * 70)
print(f"1. Impact of unequal probabilities:")
print(f"   Equal case:   {mean_boxes_equal:.1f} boxes expected")
print(f"   Unequal case: {mean_boxes_unequal:.1f} boxes expected")
print(f"   Increase: {mean_boxes_unequal - mean_boxes_equal:.1f} boxes ({100*(mean_boxes_unequal - mean_boxes_equal)/mean_boxes_equal:.0f}%)")

print(f"\n2. Rare items dominate collection time:")
print(f"   Rarest figures have probability 0.02 (vs 1/15=0.067)")
print(f"   Expected wait for a specific rare item: 1/0.02 = 50 boxes")

print(f"\n3. High variability in unequal case:")
print(f"   95th percentile: {int(quantiles_unequal[4])} boxes ({100*(quantiles_unequal[4] - mean_boxes_unequal)/mean_boxes_unequal:.1f}% above mean)")

print(f"\n4. Practical implications:")
print(f"   P(> 100 boxes) = {100*np.mean(boxes_needed_unequal > 100):.1f}% - significant risk of extreme cases")
print(f"   P(> 200 boxes) = {100*np.mean(boxes_needed_unequal > 200):.1f}% - rare but possible")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n\n" + "=" * 70)
print("SUMMARY: MONTE CARLO SIMULATIONS")
print("=" * 70)
print()

print("KEY PRINCIPLES:")
print("1. OMC uses IID simulations to estimate expectations: θ = E[g(X)]")
print("2. Law of Large Numbers: ȳₙ converges to θ")
print("3. Central Limit Theorem: ȳₙ ~ N(θ, σ²/n) for large n")
print("4. Standard error decreases as O(1/√n)")
print("5. We can construct confidence intervals using SE = s/√n\n")

print("METHODS COVERED:")
print("1. Ordinary Monte Carlo - basic estimation with IID samples")
print("2. Approximating π - geometric probability method")
print("3. Monte Carlo integration - sequential stopping rule")
print("4. Bootstrap methods - resampling with replacement")
print("5. Permutation tests - resampling without replacement")
print("6. Coupon collector problem - complex probability estimation\n")

print("PRACTICAL APPLICATIONS:")
print("- High-dimensional problems (election forecasts, sports predictions)")
print("- Intractable integrals and expectations")
print("- Hypothesis testing without parametric assumptions")
print("- Uncertainty quantification in complex systems")
print("- Sequential decision making with stopping rules\n")

print("=" * 70)
print("Generated plots:")
print("  1. monte_carlo_binomial.png - Binomial approximation analysis")
print("  2. monte_carlo_pi_approximation.png - π estimation convergence")
print("  3. monte_carlo_pi_detailed.png - π scatter plot visualization")
print("  4. monte_carlo_integration_sequential.png - Sequential stopping")
print("  5. monte_carlo_integration_distribution.png - Integration distributions")
print("  6. monte_carlo_bootstrap_test.png - Bootstrap hypothesis test")
print("  7. monte_carlo_toy_collector.png - Coupon collector analysis")
print("=" * 70)

print("\nMONTE CARLO SIMULATIONS TUTORIAL COMPLETE!")
print("All methods demonstrated with practical examples.")
