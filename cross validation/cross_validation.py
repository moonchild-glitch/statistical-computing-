# Agenda
# Suppose we have several different models for a particular data set. 
# How should we choose the best one? Naturally, we would want to select 
# the best performing model in order to choose the best. What is performance? 
# How to estimate performance? Thinking about these ideas, we'll consider 
# the following:
#
# - Model assessment and selection
# - Prediction error
# - Cross-validation
# - Smoothing example

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Model Assessment and Selection
# ============================================================================

# Basic problem: We have several different models so how can we choose the 
# best one? We could estimate the performance of each model in order to 
# choose the best one. What is performance? How to estimate performance?

# Model assessment: Evaluating how closely a particular model fits the data
# - How well does the model explain the observed data?
# - How accurate are the model's predictions?
# - Measures: R-squared, residual sum of squares, mean squared error, etc.

# Model selection: Choosing the best model among several different ones
# - Which model generalizes best to new, unseen data?
# - Balance between model complexity and goodness of fit
# - Avoid overfitting (too complex) and underfitting (too simple)

# Key insight: A model that fits the training data perfectly may perform 
# poorly on new data (overfitting). We need methods to estimate how well 
# a model will generalize.

# Model selection is ubiquitous. Where do we use model selection?
# - Linear regression (variable selection)
#   * Which predictors should be included in the model?
#   * Forward selection, backward elimination, stepwise regression
# - Smoothing (smoothing parameter selection)
#   * How much should we smooth the data?
#   * Choosing bandwidth, span, or degrees of freedom
# - Kernel density estimation (bandwidth selection)
#   * How wide should the kernel be?
#   * Trade-off between bias and variance

# ============================================================================
# Prediction Error
# ============================================================================

# Prediction error is one measure of performance of a model
# In short, we're interested in: How well does the model make predictions 
# about new data?

# Exact definition (e.g. squared error or other) depends on problem:
# - Squared error: E[(Y - f^(X))^2] for continuous outcomes
# - Classification error: P(Y ≠ f^(X)) for categorical outcomes
# - Absolute error: E[|Y - f^(X)|] for robust estimation
# - Log-likelihood: For probabilistic models

# The key is to estimate prediction error on NEW data, not the training data
# Training error typically underestimates true prediction error (optimism)

# ----------------------------------------------------------------------------
# Consider either a nonparametric (smoothing) or linear regression modeling 
# setting. In this case, the regression equation is fit using observed data
# (X1,Y1), ..., (Xn,Yn), resulting in a regression function f^n(·)
#
# The resulting model can be used for prediction at new X values resulting 
# in f^n(Xn+1)
#
# One measure of prediction error for regression is the Mean Squared 
# Prediction Error (MSPE), i.e.
#
#   MSPE = E|Yn+1 - f^n(Xn+1)|^2
#
# where the expectation is over new observations (Xn+1, Yn+1)

# ----------------------------------------------------------------------------
# Question: How do we estimate prediction error if we only have n observations?
#
# Answer: If we had m additional independent observations 
# (Xn+1,Yn+1), ..., (Xn+m,Yn+m), then we could estimate the MSPE by
#
#   MSPE^ = (1/m) * sum_{i=1}^m |Yn+i - f^n(Xn+i)|^2
#
# This is the test set approach:
# - Fit the model on the first n observations (training set)
# - Evaluate prediction error on the additional m observations (test set)
# - The test set provides an unbiased estimate of prediction error
#
# Problem: We often don't have additional data available for testing!
# Solution: Cross-validation - use the available data cleverly

# ----------------------------------------------------------------------------
# Notice, if we reuse the same data to estimate MSPE we have an in-sample 
# estimate of the MSPE:
#
#   (1/n) * sum_{i=1}^n |Yi - f^n(Xi)|^2
#
# But, as we saw in the lidar example, this is generally a bad (overly 
# optimistic) estimate of the MSPE. Using the same data for fitting and 
# estimating prediction error generally leads to bad estimates of prediction 
# error that are overly optimistic. This is related to the phenomenon known 
# as OVERFITTING.
#
# Why is in-sample error optimistic?
# - The model is specifically fit to minimize error on the training data
# - It "knows" the training data and can exploit its specific patterns
# - New data will have different patterns that the model hasn't seen
# - Result: training error < test error (on average)

# How do we estimate prediction error if we only have n observations?
# Answer: Cross-validation!

# ============================================================================
# Cross-Validation
# ============================================================================

# Cross-validation is a method for estimating prediction error
# Main idea is to split the data into two parts:
# - Training portion: used for fitting the model
# - Test portion: for validating the model or estimating the prediction error
#
# By holding out part of the data during training, we can get an honest 
# estimate of how well the model will perform on new data

# ----------------------------------------------------------------------------
# Leave-one-out cross validation (LOOCV)
# ----------------------------------------------------------------------------
# A special case of cross-validation is known as leave-one-out cross validation
# Simply K-fold cross-validation where K = n
#
# For the kth observation:
# 1. Fit the model using the remaining n-1 observations
# 2. Calculate the prediction error of the fitted model when predicting for 
#    the kth observation
#
# LOOCV error = (1/n) * sum_{i=1}^n |Yi - f^(-i)(Xi)|^2
# where f^(-i) is the model fit without the ith observation
#
# Advantages:
# - Uses maximum amount of data for training (n-1 observations)
# - Deterministic (no randomness in the splits)
# - Nearly unbiased estimate of prediction error
#
# Disadvantages:
# - Computationally expensive (fit model n times)
# - High variance (each training set is very similar)

# ----------------------------------------------------------------------------
# Pseudo-code for K-fold cross-validation
# ----------------------------------------------------------------------------
# The following is some pseudo-code for K-fold cross-validation:

# K = 10
# n = len(mydata)
# cv_error = np.zeros(K)
# 
# # Randomly split data into K subsets
# # - each observation gets a foldid between 1 and K
# foldid = np.random.choice(K, size=n, replace=True)
# 
# # Repeat K times
# for i in range(K):
#     # Fit using training set
#     f_hat = estimator(mydata[foldid != i])
#     
#     # Calculate prediction error on validation set
#     cv_error[i] = calc_error(f_hat, mydata[foldid == i])
# 
# cv_error_estimate = np.mean(cv_error)

# Common choices for K:
# - K = 5 or K = 10 (good balance between bias and variance)
# - K = n (leave-one-out, computationally expensive)
# - K = 2 (simple but high bias)

# ============================================================================
# Example: Lidar data
# ============================================================================

# LOESS (Locally Weighted Scatterplot Smoothing) implementation
def loess_smooth(x, y, x_pred, frac=0.75):
    """
    LOESS smoothing implementation
    
    Parameters:
    x: training x values
    y: training y values
    x_pred: x values to predict at
    frac: fraction of data to use for each local fit (span parameter)
    
    Returns:
    y_pred: predicted y values at x_pred
    """
    n = len(x)
    y_pred = np.zeros(len(x_pred))
    
    # Number of neighbors to use
    r = int(np.ceil(frac * n))
    
    for i, xp in enumerate(x_pred):
        # Calculate distances from prediction point
        distances = np.abs(x - xp)
        
        # Get the r nearest neighbors
        idx = np.argsort(distances)[:r]
        
        # Calculate weights using tricube kernel
        max_dist = np.max(distances[idx])
        if max_dist > 0:
            weights = (1 - (distances[idx] / max_dist) ** 3) ** 3
        else:
            weights = np.ones(r)
        
        # Weighted linear regression
        X_local = np.column_stack([np.ones(r), x[idx]])
        W = np.diag(weights)
        
        try:
            # Solve weighted least squares: (X'WX)^(-1) X'Wy
            beta = np.linalg.solve(X_local.T @ W @ X_local, X_local.T @ W @ y[idx])
            y_pred[i] = beta[0] + beta[1] * xp
        except:
            # If singular, use weighted mean
            y_pred[i] = np.sum(weights * y[idx]) / np.sum(weights)
    
    return y_pred

# Load lidar data (simulated since SemiPar package not available in Python)
# In practice, you would load from a CSV or other data source
# For demonstration, we'll create synthetic data similar to lidar
np.random.seed(42)
n_points = 221
range_vals = np.sort(np.random.uniform(390, 720, n_points))
# Create a non-linear relationship with noise
logratio = np.sin((range_vals - 390) / 50) * 0.3 + \
           np.exp(-(range_vals - 500)**2 / 5000) * 0.5 + \
           np.random.normal(0, 0.05, n_points)

lidar = pd.DataFrame({'range': range_vals, 'logratio': logratio})

print("Lidar data shape:", lidar.shape)
print(lidar.head())

# Using the lidar data again, we will:
# 1. Fit several different smooths corresponding to different choices of 
#    bandwidth (span)
# 2. Estimate the mean squared prediction error of each smooth
# 3. Choose the smooth with the smallest estimated MSPE

# Define a sequence of span values to try
s = np.arange(0.1, 1.1, 0.1)
K = 10
n = len(lidar)

# ============================================================================
# Example: Lidar Data - K-fold Cross-Validation
# ============================================================================

# Matrix to store cross-validation errors
# Rows: different folds, Columns: different span values
cv_error = np.zeros((K, len(s)))

# Randomly split data into K subsets
# Each observation gets a foldid between 0 and K-1
foldid = np.random.choice(K, size=n, replace=True)

for i in range(K):
    # Get training and validation indices
    train_idx = foldid != i
    val_idx = foldid == i
    
    # Training and validation data
    x_train = lidar.loc[train_idx, 'range'].values
    y_train = lidar.loc[train_idx, 'logratio'].values
    x_val = lidar.loc[val_idx, 'range'].values
    y_val = lidar.loc[val_idx, 'logratio'].values
    
    # Fit, predict, and calculate error for each bandwidth
    for j, span in enumerate(s):
        # Fit LOESS model to training set (all data except fold i)
        y_pred = loess_smooth(x_train, y_train, x_val, frac=span)
        
        # Calculate mean squared error on validation set
        cv_error[i, j] = np.mean((y_val - y_pred) ** 2)

# Columns of cv_error correspond to different bandwidths
# Average across all K folds to get final CV error estimate for each span
cv_error_estimate = np.mean(cv_error, axis=0)

# ============================================================================
# Cross-validation estimates of MSPE
# ============================================================================

# Plot the CV error estimates for different span values
plt.figure(figsize=(10, 6))
plt.plot(s, cv_error_estimate, 'o-', linewidth=2, markersize=8)
plt.xlabel('Span', fontsize=12)
plt.ylabel('CV Error Estimate', fontsize=12)
plt.title('Cross-Validation Error vs Span', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cv_error_vs_span.png', dpi=300, bbox_inches='tight')
plt.show()

# Find the span that minimizes the CV error
s_best = s[np.argmin(cv_error_estimate)]
print(f"Best span selected by cross-validation: {s_best:.2f}")
print(f"Minimum CV error: {np.min(cv_error_estimate):.6f}")

# ============================================================================
# Smoother selected by cross-validation
# ============================================================================

# Plot the data with the optimal smoother
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(lidar['range'], lidar['logratio'], alpha=0.5, s=20)
x_smooth = np.linspace(lidar['range'].min(), lidar['range'].max(), 200)
y_smooth = loess_smooth(lidar['range'].values, lidar['logratio'].values, 
                        x_smooth, frac=s_best)
plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label=f'span={s_best:.2f}')
plt.xlabel('Range', fontsize=12)
plt.ylabel('Log Ratio', fontsize=12)
plt.title(f'Lidar Data with Optimal Loess Smooth (span = {s_best:.2f})', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# Compare with other span choices
plt.subplot(1, 2, 2)
plt.scatter(lidar['range'], lidar['logratio'], alpha=0.5, s=20, label='Data')

# Span = 0.1 (too flexible)
y_smooth1 = loess_smooth(lidar['range'].values, lidar['logratio'].values, 
                         x_smooth, frac=0.1)
plt.plot(x_smooth, y_smooth1, 'b--', linewidth=2, label='span=0.1')

# Optimal span
y_smooth_opt = loess_smooth(lidar['range'].values, lidar['logratio'].values, 
                            x_smooth, frac=s_best)
plt.plot(x_smooth, y_smooth_opt, 'r-', linewidth=2, 
         label=f'span={s_best:.2f} (optimal)')

# Span = 1.0 (too smooth)
y_smooth3 = loess_smooth(lidar['range'].values, lidar['logratio'].values, 
                         x_smooth, frac=1.0)
plt.plot(x_smooth, y_smooth3, 'purple', linestyle='--', linewidth=2, label='span=1.0')

plt.xlabel('Range', fontsize=12)
plt.ylabel('Log Ratio', fontsize=12)
plt.title('Comparison of Different Span Values', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loess_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# Split data into training and test for visualization
# ============================================================================

# Randomly split data into K subsets
K = 10
n = len(lidar)
foldid = np.random.choice(K, size=n, replace=True)

# Split data into training and test data
lidar_train = lidar[foldid != 0]
lidar_test = lidar[foldid == 0]

# Example: Lidar data
plt.figure(figsize=(10, 6))
plt.scatter(lidar_train['range'], lidar_train['logratio'], alpha=0.6, s=30)
plt.xlabel('Range', fontsize=12)
plt.ylabel('Log Ratio', fontsize=12)
plt.title('Lidar Data (Training Set)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lidar_training_data.png', dpi=300, bbox_inches='tight')
plt.show()

# Nonparametric smoothing
# Benefits
# - Provides a flexible approach to representing data
# - Ease of use
# - Computations are relatively easy (sometimes)
#
# Disadvantages
# - No simple equation for a set of data
# - Less understood than parametric smoothers
# - Depends on a span parameter controlling the smoothness

# Example: Lidar data
# Can consider fitting a loess smooth to the data
x_train = lidar_train['range'].values
y_train = lidar_train['logratio'].values

# Visualize the loess smooth
x_smooth = np.linspace(x_train.min(), x_train.max(), 200)
y_smooth = loess_smooth(x_train, y_train, x_smooth, frac=0.75)

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, alpha=0.5, s=30, label='Training Data')
plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Loess Smooth (span=0.75)')
plt.xlabel('Range', fontsize=12)
plt.ylabel('Log Ratio', fontsize=12)
plt.title('Lidar Data with Loess Smooth', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lidar_loess_smooth.png', dpi=300, bbox_inches='tight')
plt.show()

# How to choose the span parameter?
# The span parameter controls the amount of smoothing:
# - Small span: flexible fit, follows data closely (may overfit)
# - Large span: smooth fit, averages over more data (may underfit)
# - Default span: 0.75 (moderate smoothing)

# Which model is the best? Why?
# Compare three different span values:

# Model 1: Very small span (0.02) - highly flexible, may overfit
y_pred1 = loess_smooth(x_train, y_train, x_train, frac=0.02)

# Model 2: Moderate span (0.3) - balanced flexibility
y_pred2 = loess_smooth(x_train, y_train, x_train, frac=0.3)

# Model 3: Large span (1) - very smooth, may underfit
y_pred3 = loess_smooth(x_train, y_train, x_train, frac=1.0)

# Visualize all three models
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(x_train, y_train, alpha=0.5, s=20)
axes[0, 0].plot(x_train, y_pred1, 'b-', linewidth=2)
axes[0, 0].set_title('Span = 0.02 (Too Flexible)', fontsize=12)
axes[0, 0].set_xlabel('Range')
axes[0, 0].set_ylabel('Log Ratio')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(x_train, y_train, alpha=0.5, s=20)
axes[0, 1].plot(x_train, y_pred2, 'g-', linewidth=2)
axes[0, 1].set_title('Span = 0.3 (Balanced)', fontsize=12)
axes[0, 1].set_xlabel('Range')
axes[0, 1].set_ylabel('Log Ratio')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(x_train, y_train, alpha=0.5, s=20)
axes[1, 0].plot(x_train, y_pred3, 'purple', linewidth=2)
axes[1, 0].set_title('Span = 1 (Too Smooth)', fontsize=12)
axes[1, 0].set_xlabel('Range')
axes[1, 0].set_ylabel('Log Ratio')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(x_train, y_train, alpha=0.5, s=20, label='Data')
axes[1, 1].plot(x_train, y_pred1, 'b--', linewidth=2, label='span=0.02')
axes[1, 1].plot(x_train, y_pred2, 'g-', linewidth=2, label='span=0.3')
axes[1, 1].plot(x_train, y_pred3, 'purple', linestyle='--', linewidth=2, label='span=1')
axes[1, 1].set_title('All Three Models', fontsize=12)
axes[1, 1].set_xlabel('Range')
axes[1, 1].set_ylabel('Log Ratio')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('span_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Example: Lidar data
# Compare models using squared error on training data
# Model 1: sum |Yi - f^(Xi)|^2 = 0 (essentially)
# Model 2: sum |Yi - f^(Xi)|^2 = 1.03
# Model 3: sum |Yi - f^(Xi)|^2 = 1.71
# Therefore, Model 1 has the smallest squared error over the data

# Sum of squared errors
sse1 = np.sum((y_train - y_pred1) ** 2)
sse2 = np.sum((y_train - y_pred2) ** 2)
sse3 = np.sum((y_train - y_pred3) ** 2)

print("\nSum of squared errors:")
print(f"Model 1 (span=0.02): {sse1:.6e}")
print(f"Model 2 (span=0.3):  {sse2:.6e}")
print(f"Model 3 (span=1.0):  {sse3:.6e}")

# Mean squared errors
mse1 = np.mean((y_train - y_pred1) ** 2)
mse2 = np.mean((y_train - y_pred2) ** 2)
mse3 = np.mean((y_train - y_pred3) ** 2)

print("\nMean squared errors:")
print(f"Model 1 (span=0.02): {mse1:.6e}")
print(f"Model 2 (span=0.3):  {mse2:.6e}")
print(f"Model 3 (span=1.0):  {mse3:.6e}")

# Example: Lidar data - Test set visualization
plt.figure(figsize=(10, 6))
plt.scatter(lidar_test['range'], lidar_test['logratio'], alpha=0.6, s=50, color='orange')
plt.xlabel('Range', fontsize=12)
plt.ylabel('Log Ratio', fontsize=12)
plt.title('New Lidar Data (Test Set)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lidar_test_data.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*80)
print("Cross-validation analysis complete!")
print("="*80)
