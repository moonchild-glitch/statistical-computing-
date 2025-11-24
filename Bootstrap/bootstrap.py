"""
============================================================================
Bootstrap Methods in Python
============================================================================

Agenda:
- Toy collector solution
- Plug-In and the Bootstrap
- Nonparametric and Parametric Bootstraps
- Examples
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import seaborn as sns
from typing import Callable, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("Bootstrap Methods - Python Implementation")
print("="*80)
print()

# ============================================================================
# Exercise: Toy Collector Problem
# ============================================================================

print("\n" + "="*80)
print("Exercise: Toy Collector Problem")
print("="*80)
print()

print("Children (and some adults) are frequently enticed to buy breakfast cereal")
print("in an effort to collect all the action figures. Assume there are 15 action")
print("figures and each cereal box contains exactly one with each figure being")
print("equally likely.\n")

print("Questions:")
print("1. Find the expected number of boxes needed to collect all 15 action figures.")
print("2. Find the standard deviation of the number of boxes needed to collect all")
print("   15 action figures.")
print("3. Now suppose we no longer have equal probabilities...\n")

# ============================================================================
# Part 1 & 2: Equal Probabilities (Theoretical Solution)
# ============================================================================

print("="*80)
print("Part 1 & 2: Equal Probabilities (Theoretical Solution)")
print("="*80)
print()

n = 15  # number of action figures

# Geometric Approach
print("=== Equal Probabilities (Theoretical - Geometric Approach) ===")
print("Expected number of boxes:")
print("  E[T] = 15/15 + 15/14 + 15/13 + ... + 15/1")

expected_boxes = sum(n / (n - i + 1) for i in range(1, n + 1))
print(f"  E[T] = {expected_boxes}")
print(f"  E[T] ≈ {expected_boxes:.2f}\n")

print("Variance calculation:")
print("  Var[T] = 15*(1-15/15)/(15/15)^2 + 15*(1-14/15)/(14/15)^2 + ... + 15*(1-1/15)/(1/15)^2")

variance_boxes = sum(n * (1 - (n - i + 1)/n) / ((n - i + 1)/n)**2 for i in range(1, n + 1))
sd_boxes_geometric = np.sqrt(variance_boxes)
print(f"  Var[T] = {variance_boxes}")
print(f"  Var[T] ≈ {variance_boxes:.2f}\n")
print(f"Standard deviation: {sd_boxes_geometric}")
print(f"SD ≈ {sd_boxes_geometric:.2f}\n")

# Harmonic Number Approach
print("=== Equal Probabilities (Theoretical - Harmonic Number Approach) ===")
harmonic_n = sum(1/i for i in range(1, n + 1))
expected_harmonic = n * harmonic_n
print(f"Expected number of boxes: E[T] = n * H_n")
print(f"  where H_n = sum(1/i) for i=1 to {n}")
print(f"  H_{n} = {harmonic_n}")
print(f"  E[T] = {expected_harmonic}\n")

variance_harmonic = n**2 * sum(1/i**2 for i in range(1, n + 1)) - n * harmonic_n
sd_boxes_harmonic = np.sqrt(variance_harmonic)
print(f"Variance: Var[T] = n^2 * sum(1/i^2) - n * H_n")
print(f"  Var[T] = {variance_harmonic}")
print(f"Standard deviation: {sd_boxes_harmonic}\n")

# Verification
print("=== Verification: Both Methods Agree ===")
print(f"Geometric approach - Expected: {expected_boxes:.4f}")
print(f"Harmonic approach  - Expected: {expected_harmonic:.4f}")
print(f"Difference: {abs(expected_boxes - expected_harmonic)}\n")

print(f"Geometric approach - Variance: {variance_boxes}")
print(f"Harmonic approach  - Variance: {variance_harmonic}")
print(f"Difference: {abs(variance_boxes - variance_harmonic)}\n")

# ============================================================================
# Simulation with Equal Probabilities
# ============================================================================

print("\n" + "="*80)
print("Simulation with Equal Probabilities")
print("="*80)
print()

def count_boxes_equal_prob(n_toys=15):
    """Simulate collecting all toys with equal probabilities."""
    collected = set()
    boxes = 0
    while len(collected) < n_toys:
        boxes += 1
        collected.add(np.random.randint(0, n_toys))
    return boxes

# Run simulation
trials_equal = 10000
sim_boxes_equal = [count_boxes_equal_prob(n) for _ in range(trials_equal)]

print("=== Equal Probabilities (Simulation) ===")
print(f"Expected number of boxes (simulated): {np.mean(sim_boxes_equal):.4f}")
print(f"Standard deviation (simulated): {np.std(sim_boxes_equal, ddof=1):.5f}\n")

print("Comparison with theory:")
print(f"Expected value difference: {abs(np.mean(sim_boxes_equal) - expected_boxes):.7f}")
print(f"SD difference: {abs(np.std(sim_boxes_equal, ddof=1) - sd_boxes_harmonic):.8f}\n")

# ============================================================================
# Part 3: Unequal Probabilities
# ============================================================================

print("="*80)
print("Part 3: Unequal Probabilities")
print("="*80)
print()

# Define probability table
prob_table = np.array([0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 
                       0.02, 0.02, 0.02, 0.02, 0.02])

print("=== Unequal Probabilities ===")
print(f"Sum of probabilities: {prob_table.sum()}")

# Create probability dataframe
figures = list("ABCDEFGHIJKLMNO")
prob_df = pd.DataFrame({'Figure': figures, 'Probability': prob_table})
print(prob_df.to_string(index=False))
print()

def box_count(prob_table):
    """Count boxes needed with unequal probabilities."""
    check = np.zeros(len(prob_table), dtype=bool)
    count = 0
    while not check.all():
        count += 1
        toy = np.random.choice(len(prob_table), p=prob_table)
        check[toy] = True
    return count

# Part 3a: Expected number of boxes
trials = 1000
sim_boxes = np.array([box_count(prob_table) for _ in range(trials)])

print(f"3a. Expected number of boxes (unequal probabilities):")
print(f"    Point estimate: {np.mean(sim_boxes):.3f}")
print(f"    Example output: est = 115.468\n")

# Part 3b: Uncertainty estimate
mcse = np.std(sim_boxes, ddof=1) / np.sqrt(trials)
ci_lower = np.mean(sim_boxes) - 1.96 * mcse
ci_upper = np.mean(sim_boxes) + 1.96 * mcse

print(f"3b. Uncertainty of estimate:")
print(f"    Monte Carlo Standard Error (MCSE): {mcse:.6f}")
print(f"    95% Confidence interval: {ci_lower:.4f} to {ci_upper:.4f}")
print(f"    Example output: interval = [112.0715, 118.8645]\n")

# More precise estimate with 10000 simulations
trials_precise = 10000
sim_boxes_precise = np.array([box_count(prob_table) for _ in range(trials_precise)])

print(f"More precise estimate (with {trials_precise} simulations):")
print(f"    Expected number of boxes: {np.mean(sim_boxes_precise):.4f}")
print(f"    Standard deviation of boxes: {np.std(sim_boxes_precise, ddof=1):.5f}")
print(f"    Standard error of the mean: {np.std(sim_boxes_precise, ddof=1) / np.sqrt(trials_precise):.7f}")
mcse_precise = np.std(sim_boxes_precise, ddof=1) / np.sqrt(trials_precise)
print(f"    95% Confidence interval: {np.mean(sim_boxes_precise) - 1.96*mcse_precise:.4f} to {np.mean(sim_boxes_precise) + 1.96*mcse_precise:.4f}\n")

# Part 3c: Probabilities
print(f"3c. Probabilities (from {trials} simulations):")
print(f"    P(boxes > 300): {np.mean(sim_boxes > 300):.3f}")
print(f"    P(boxes > 500): {np.mean(sim_boxes > 500):.3f}")
print(f"    P(boxes > 800): {np.mean(sim_boxes > 800):.3f}\n")

print(f"Probabilities (from {trials_precise} simulations - more precise):")
print(f"    P(boxes > 300): {np.mean(sim_boxes_precise > 300):.3f}")
print(f"    P(boxes > 500): {np.mean(sim_boxes_precise > 500):.4f}")
print(f"    P(boxes > 800): {np.mean(sim_boxes_precise > 800):.3f}")
print(f"    P(boxes > 800): {np.mean(sim_boxes_precise > 800):.3f}\n")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram for equal probabilities
axes[0, 0].hist(sim_boxes_equal, bins=30, color='lightblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(expected_boxes, color='red', linestyle='--', linewidth=2, label=f'Theoretical: {expected_boxes:.2f}')
axes[0, 0].set_xlabel('Number of Boxes')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Equal Probabilities Simulation')
axes[0, 0].legend()

# Histogram for unequal probabilities
axes[0, 1].hist(sim_boxes_precise, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(300, color='red', linestyle='--', linewidth=2, label='300 boxes')
axes[0, 1].set_xlabel('Number of Boxes')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Unequal Probabilities Simulation (10,000 trials)')
axes[0, 1].legend()

# Box plots comparison
comparison_data = pd.DataFrame({
    'Equal': sim_boxes_equal[:1000],
    'Unequal': sim_boxes_precise[:1000]
})
axes[1, 0].boxplot([comparison_data['Equal'], comparison_data['Unequal']], labels=['Equal', 'Unequal'])
axes[1, 0].set_ylabel('Number of Boxes')
axes[1, 0].set_title('Comparison: Equal vs Unequal Probabilities')

# Probability bar chart
probs_df = pd.DataFrame({
    'Threshold': ['> 300', '> 500', '> 800'],
    'Probability': [np.mean(sim_boxes_precise > 300), 
                    np.mean(sim_boxes_precise > 500),
                    np.mean(sim_boxes_precise > 800)]
})
axes[1, 1].bar(probs_df['Threshold'], probs_df['Probability'], color=['red', 'orange', 'yellow'])
axes[1, 1].set_ylabel('Probability')
axes[1, 1].set_title('Probability of Exceeding Thresholds')
axes[1, 1].set_ylim(0, max(probs_df['Probability']) * 1.2)

plt.tight_layout()
plt.savefig('toy_collector_analysis.png', dpi=300, bbox_inches='tight')
print("Saved plot: toy_collector_analysis.png\n")

# ============================================================================
# Speed of Light Example - Bootstrap Hypothesis Testing
# ============================================================================

print("\n" + "="*80)
print("Speed of Light Example - Bootstrap Hypothesis Testing")
print("="*80)
print()

# Newcomb's speed of light data (1882)
speed = np.array([28, -44, 29, 30, 26, 27, 22, 23, 33, 16, 24, 29, 24, 40, 21, 31, 34, -2, 25, 19])

print("Dataset: Newcomb's Speed of Light Measurements (1882)")
print(f"Number of observations: {len(speed)}")
print(f"Measurements (passage time in nanoseconds above 24,800):\n{speed}\n")

print(f"Summary statistics:")
print(f"  Mean: {np.mean(speed):.2f}")
print(f"  Median: {np.median(speed):.2f}")
print(f"  Standard deviation: {np.std(speed, ddof=1):.2f}")
print(f"  Min: {np.min(speed)}, Max: {np.max(speed)}\n")

# Hypothesis Testing
print("="*80)
print("Hypothesis Testing: Does the data support the accepted speed?")
print("="*80)
print()

print("Step 1: State the Hypotheses")
print("  H₀: μ = 33.02  (The true mean speed equals the accepted value)")
print("  Hₐ: μ ≠ 33.02  (The true mean speed differs from the accepted value)")
print("  (This is a two-sided test)\n")

alpha = 0.05
print(f"Step 2: Significance Level")
print(f"  α = {alpha}\n")

print("Step 3: Test Statistic")
print("  Test statistic: X̄ (sample mean)\n")

observed_mean = np.mean(speed)
print(f"Step 4: Observed Test Statistic")
print(f"  Observed sample mean: {observed_mean:.2f}")
print(f"  Difference from H₀: {observed_mean - 33.02:.2f}\n")

# Shift data to satisfy H0
newspeed = speed - np.mean(speed) + 33.02

print("Bootstrap Solution: Shift the Data to Satisfy H₀")
print(f"  newspeed = speed - mean(speed) + 33.02")
print(f"  Original mean: {np.mean(speed):.2f}")
print(f"  Shifted mean: {np.mean(newspeed):.2f}\n")

# Bootstrap resampling
n_bootstrap = 1000
bstrap = np.zeros(n_bootstrap)

print(f"Performing {n_bootstrap} bootstrap replications...")
for i in range(n_bootstrap):
    newsample = np.random.choice(newspeed, size=20, replace=True)
    bstrap[i] = np.mean(newsample)
print("Bootstrap complete!\n")

# Calculate p-value
distance_from_null = abs(observed_mean - 33.02)
extreme_lower = observed_mean
extreme_upper = 33.02 + distance_from_null

lower_tail = np.sum(bstrap < extreme_lower)
upper_tail = np.sum(bstrap > extreme_upper)
extreme_count = lower_tail + upper_tail
pvalue = extreme_count / n_bootstrap

print("Step 6: Calculate the p-value")
print(f"  Distance from null: {distance_from_null:.2f}")
print(f"  Count in lower tail (< {extreme_lower:.2f}): {lower_tail}")
print(f"  Count in upper tail (> {extreme_upper:.2f}): {upper_tail}")
print(f"  p-value = {pvalue:.3f}\n")

# Decision
print("Step 7: Make a Conclusion")
if pvalue < alpha:
    print(f"Result: p-value = {pvalue:.3f} < {alpha}")
    print("Decision: REJECT H₀")
    print("\nConclusion:")
    print("  We have sufficient evidence to conclude that Newcomb's measurements")
    print("  were NOT consistent with the currently accepted figure.\n")
else:
    print(f"Result: p-value = {pvalue:.3f} >= {alpha}")
    print("Decision: FAIL TO REJECT H₀\n")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original vs shifted data
axes[0, 0].hist(speed, bins=10, color='lightblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(speed), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(speed):.2f}')
axes[0, 0].axvline(33.02, color='red', linestyle='--', linewidth=2, label='H₀: μ = 33.02')
axes[0, 0].set_xlabel('Speed')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Original Data')
axes[0, 0].legend()

axes[0, 1].hist(newspeed, bins=10, color='lightgreen', edgecolor='black', alpha=0.7)
axes[0, 1].axvline(np.mean(newspeed), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {np.mean(newspeed):.2f}')
axes[0, 1].axvline(33.02, color='red', linestyle='--', linewidth=2, label='H₀: μ = 33.02')
axes[0, 1].set_xlabel('Speed')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Shifted Data (H₀ is True)')
axes[0, 1].legend()

# Bootstrap distribution
axes[1, 0].hist(bstrap, bins=30, color='lightblue', edgecolor='black', alpha=0.7, density=True)
axes[1, 0].axvline(33.02, color='darkgreen', linestyle='--', linewidth=2, label='H₀: μ = 33.02')
axes[1, 0].axvline(observed_mean, color='red', linewidth=3, label=f'Observed: {observed_mean:.2f}')
from scipy.stats import gaussian_kde
kde = gaussian_kde(bstrap)
x_range = np.linspace(bstrap.min(), bstrap.max(), 100)
axes[1, 0].plot(x_range, kde(x_range), 'b-', linewidth=2, label='Bootstrap density')
axes[1, 0].set_xlabel('Sample Mean')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Bootstrap Sampling Distribution of X̄ Under H₀')
axes[1, 0].legend()

# ECDF
from scipy.stats import ecdf
ecdf_result = ecdf(bstrap)
axes[1, 1].plot(ecdf_result.cdf.quantiles, ecdf_result.cdf.probabilities, 'b-', linewidth=2)
axes[1, 1].axvline(33.02, color='darkgreen', linestyle='--', linewidth=2, label='H₀: μ = 33.02')
axes[1, 1].axvline(observed_mean, color='red', linewidth=2, label=f'Observed: {observed_mean:.2f}')
axes[1, 1].axhline(0.025, color='gray', linestyle=':', linewidth=1)
axes[1, 1].axhline(0.975, color='gray', linestyle=':', linewidth=1)
axes[1, 1].set_xlabel('Sample Mean')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('Empirical CDF')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('speed_of_light_bootstrap.png', dpi=300, bbox_inches='tight')
print("Saved plot: speed_of_light_bootstrap.png\n")

# ============================================================================
# Sleep Study Example - Two-Sample Bootstrap
# ============================================================================

print("\n" + "="*80)
print("Sleep Study Example - Two-Sample Bootstrap")
print("="*80)
print()

# Load sleep data (simulated similar to R's sleep dataset)
sleep_data = pd.DataFrame({
    'extra': [0.7, -1.6, -0.2, -1.2, -0.1, 3.4, 3.7, 0.8, 0.0, 2.0,
              1.9, 0.8, 1.1, 0.1, -0.1, 4.4, 5.5, 1.6, 4.6, 3.4],
    'group': [1]*10 + [2]*10
})

print("Dataset: Student's Sleep Data")
print("Description: Effect of two soporific drugs on sleep increase")
print(f"\nSleep data:\n{sleep_data}\n")

group1_data = sleep_data[sleep_data['group'] == 1]['extra'].values
group2_data = sleep_data[sleep_data['group'] == 2]['extra'].values

print("Summary by group:")
print(f"Group 1: Mean = {np.mean(group1_data):.3f}, SD = {np.std(group1_data, ddof=1):.3f}")
print(f"Group 2: Mean = {np.mean(group2_data):.3f}, SD = {np.std(group2_data, ddof=1):.3f}")

observed_diff = np.mean(group1_data) - np.mean(group2_data)
print(f"\nObserved difference (Group 1 - Group 2): {observed_diff:.3f}\n")

# Bootstrap functions
def bootstrap_resample(data):
    """Resample data with replacement."""
    return np.random.choice(data, size=len(data), replace=True)

def diff_in_means(df):
    """Calculate difference in means."""
    group1 = df[df['group'] == 1]['extra'].values
    group2 = df[df['group'] == 2]['extra'].values
    return np.mean(group1) - np.mean(group2)

# Perform bootstrap
n_resamples = 2000
resample_diffs = np.zeros(n_resamples)

print(f"Performing {n_resamples} bootstrap replications...")
for i in range(n_resamples):
    indices = bootstrap_resample(np.arange(len(sleep_data)))
    boot_sample = sleep_data.iloc[indices]
    resample_diffs[i] = diff_in_means(boot_sample)
print("Bootstrap complete!\n")

print(f"Bootstrap Distribution Summary:")
print(f"  Mean: {np.mean(resample_diffs):.3f}")
print(f"  SD: {np.std(resample_diffs, ddof=1):.3f}\n")

# Bootstrap confidence interval
boot_ci_lower = np.percentile(resample_diffs, 2.5)
boot_ci_upper = np.percentile(resample_diffs, 97.5)

print("95% Bootstrap Confidence Interval (Percentile Method)")
print(f"  Lower bound: {boot_ci_lower:.3f}")
print(f"  Upper bound: {boot_ci_upper:.3f}\n")

if boot_ci_lower > 0:
    print("  Since the entire CI is above 0, Drug 1 increases sleep more than Drug 2.\n")
elif boot_ci_upper < 0:
    print("  Since the entire CI is below 0, Drug 2 increases sleep more than Drug 1.\n")
else:
    print("  Since the CI includes 0, we cannot conclude a significant difference.\n")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Boxplot
sleep_data.boxplot(column='extra', by='group', ax=axes[0, 0])
axes[0, 0].set_title('Sleep Increase by Drug')
axes[0, 0].set_xlabel('Group')
axes[0, 0].set_ylabel('Increase in Hours of Sleep')

# Bootstrap distribution
axes[0, 1].hist(resample_diffs, bins=40, color='lightblue', edgecolor='black', alpha=0.7, density=True)
axes[0, 1].axvline(observed_diff, color='red', linewidth=2, label=f'Observed: {observed_diff:.2f}')
axes[0, 1].axvline(0, color='darkgreen', linestyle='--', linewidth=2, label='No difference (0)')
axes[0, 1].set_xlabel('Difference in Means')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Bootstrap Sampling Distribution')
axes[0, 1].legend()

# Confidence interval visualization
axes[1, 0].hist(resample_diffs, bins=40, color='lightgray', edgecolor='black', alpha=0.7, density=True)
# Highlight CI region
in_ci = (resample_diffs >= boot_ci_lower) & (resample_diffs <= boot_ci_upper)
axes[1, 0].hist(resample_diffs[in_ci], bins=40, color='lightblue', edgecolor='black', alpha=0.7, density=True)
axes[1, 0].axvline(boot_ci_lower, color='blue', linestyle='--', linewidth=2, label='95% CI bounds')
axes[1, 0].axvline(boot_ci_upper, color='blue', linestyle='--', linewidth=2)
axes[1, 0].axvline(observed_diff, color='red', linewidth=2, label=f'Observed: {observed_diff:.2f}')
axes[1, 0].axvline(0, color='darkgreen', linestyle='--', linewidth=2, label='No difference')
axes[1, 0].set_xlabel('Difference in Means')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('95% Bootstrap Confidence Interval')
axes[1, 0].legend()

# Q-Q plot
stats.probplot(resample_diffs, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Q Plot: Bootstrap Differences')

plt.tight_layout()
plt.savefig('sleep_study_bootstrap.png', dpi=300, bbox_inches='tight')
print("Saved plot: sleep_study_bootstrap.png\n")

# ============================================================================
# R-squared Bootstrap Example
# ============================================================================

print("\n" + "="*80)
print("Bootstrapping a Single Statistic: R-squared")
print("="*80)
print()

# Load mtcars data (creating similar dataset)
mtcars_data = pd.DataFrame({
    'mpg': [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2,
            17.8, 16.4, 17.3, 15.2, 10.4, 10.4, 14.7, 32.4, 30.4, 33.9,
            21.5, 15.5, 15.2, 13.3, 19.2, 27.3, 26.0, 30.4, 15.8, 19.7, 15.0, 21.4],
    'wt': [2.620, 2.875, 2.320, 3.215, 3.440, 3.460, 3.570, 3.190, 3.150, 3.440,
           3.440, 4.070, 3.730, 3.780, 5.250, 5.424, 5.345, 2.200, 1.615, 1.835,
           2.465, 3.520, 3.435, 3.840, 3.845, 1.935, 2.140, 1.513, 3.170, 2.770, 3.570, 2.780],
    'disp': [160.0, 160.0, 108.0, 258.0, 360.0, 225.0, 360.0, 146.7, 140.8, 167.6,
             167.6, 275.8, 275.8, 275.8, 472.0, 460.0, 440.0, 78.7, 75.7, 71.1,
             120.1, 318.0, 304.0, 350.0, 400.0, 79.0, 120.3, 95.1, 351.0, 145.0, 301.0, 121.0]
})

print("Dataset: mtcars")
print(f"Variables: mpg (miles per gallon), wt (weight), disp (displacement)")
print(f"\nFirst few rows:\n{mtcars_data.head()}\n")

# Fit original model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = mtcars_data[['wt', 'disp']].values
y = mtcars_data['mpg'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
original_r2 = r2_score(y, y_pred)

print(f"Original R²: {original_r2:.4f}")
print(f"This means {original_r2*100:.2f}% of variance in mpg is explained by wt and disp.\n")

# Bootstrap R-squared
def rsq_bootstrap(data, indices):
    """Calculate R² for bootstrap sample."""
    boot_data = data.iloc[indices]
    X_boot = boot_data[['wt', 'disp']].values
    y_boot = boot_data['mpg'].values
    
    model_boot = LinearRegression()
    model_boot.fit(X_boot, y_boot)
    y_pred_boot = model_boot.predict(X_boot)
    return r2_score(y_boot, y_pred_boot)

n_boot = 1000
print(f"Performing {n_boot} bootstrap replications for R²...")
boot_r2 = np.zeros(n_boot)
for i in range(n_boot):
    indices = np.random.choice(len(mtcars_data), size=len(mtcars_data), replace=True)
    boot_r2[i] = rsq_bootstrap(mtcars_data, indices)
print("Bootstrap complete!\n")

print("Bootstrap Results:")
print(f"  Original R²: {original_r2:.4f}")
print(f"  Bootstrap mean R²: {np.mean(boot_r2):.4f}")
print(f"  Bootstrap bias: {np.mean(boot_r2) - original_r2:.4f}")
print(f"  Bootstrap SE: {np.std(boot_r2, ddof=1):.4f}\n")

# Bootstrap CI for R²
r2_ci_lower = np.percentile(boot_r2, 2.5)
r2_ci_upper = np.percentile(boot_r2, 97.5)

print("95% Bootstrap Confidence Interval for R²:")
print(f"  [{r2_ci_lower:.4f}, {r2_ci_upper:.4f}]")
print(f"  We are 95% confident that between {r2_ci_lower*100:.2f}% and {r2_ci_upper*100:.2f}%")
print(f"  of variance in MPG is explained by weight and displacement.\n")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram of R²
axes[0, 0].hist(boot_r2, bins=30, color='lightgreen', edgecolor='black', alpha=0.7, density=True)
axes[0, 0].axvline(original_r2, color='red', linewidth=3, label=f'Original: {original_r2:.4f}')
kde_r2 = gaussian_kde(boot_r2)
x_r2 = np.linspace(boot_r2.min(), boot_r2.max(), 100)
axes[0, 0].plot(x_r2, kde_r2(x_r2), 'darkgreen', linewidth=2, label='Bootstrap density')
axes[0, 0].set_xlabel('R-squared')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Bootstrap Distribution of R²')
axes[0, 0].legend()

# CI visualization
axes[0, 1].hist(boot_r2, bins=30, color='lightgray', edgecolor='black', alpha=0.7, density=True)
in_ci_r2 = (boot_r2 >= r2_ci_lower) & (boot_r2 <= r2_ci_upper)
axes[0, 1].hist(boot_r2[in_ci_r2], bins=30, color='lightgreen', edgecolor='black', alpha=0.7, density=True)
axes[0, 1].axvline(r2_ci_lower, color='blue', linestyle='--', linewidth=2)
axes[0, 1].axvline(r2_ci_upper, color='blue', linestyle='--', linewidth=2)
axes[0, 1].axvline(original_r2, color='red', linewidth=2)
axes[0, 1].set_xlabel('R-squared')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('95% Percentile CI for R²')

# Q-Q plot
stats.probplot(boot_r2, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot: Bootstrap R²')

# Sequence plot
axes[1, 1].plot(boot_r2, 'darkgreen', alpha=0.5, linewidth=0.5)
axes[1, 1].axhline(original_r2, color='red', linestyle='--', linewidth=2)
axes[1, 1].axhline(np.mean(boot_r2), color='blue', linestyle=':', linewidth=1)
axes[1, 1].set_xlabel('Bootstrap Replicate')
axes[1, 1].set_ylabel('R-squared')
axes[1, 1].set_title('Bootstrap R² Sequence')

plt.tight_layout()
plt.savefig('rsquared_bootstrap.png', dpi=300, bbox_inches='tight')
print("Saved plot: rsquared_bootstrap.png\n")

print("="*80)
print("Bootstrap Analysis Complete!")
print("="*80)
print("\nGenerated plots:")
print("  1. toy_collector_analysis.png")
print("  2. speed_of_light_bootstrap.png")
print("  3. sleep_study_bootstrap.png")
print("  4. rsquared_bootstrap.png")
