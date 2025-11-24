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

using Random
using Statistics
using DataFrames
using Plots
using LinearAlgebra
using StatsBase

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
# n = nrow(mydata)
# cv_error = zeros(K)
# 
# # Randomly split data into K subsets
# # - each observation gets a foldid between 1 and K
# foldid = sample(1:K, n, replace=true)
# 
# # Repeat K times
# for i in 1:K
#     # Fit using training set
#     f_hat = estimator(mydata[foldid .!= i, :])
#     
#     # Calculate prediction error on validation set
#     cv_error[i] = calc_error(f_hat, mydata[foldid .== i, :])
# end
# 
# cv_error_estimate = mean(cv_error)

# Common choices for K:
# - K = 5 or K = 10 (good balance between bias and variance)
# - K = n (leave-one-out, computationally expensive)
# - K = 2 (simple but high bias)

# ============================================================================
# Example: Lidar data
# ============================================================================

# LOESS (Locally Weighted Scatterplot Smoothing) implementation
function loess_smooth(x::Vector{Float64}, y::Vector{Float64}, 
                      x_pred::Vector{Float64}; frac::Float64=0.75)
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
    n = length(x)
    y_pred = zeros(length(x_pred))
    
    # Number of neighbors to use
    r = Int(ceil(frac * n))
    
    for (i, xp) in enumerate(x_pred)
        # Calculate distances from prediction point
        distances = abs.(x .- xp)
        
        # Get the r nearest neighbors
        idx = sortperm(distances)[1:r]
        
        # Calculate weights using tricube kernel
        max_dist = maximum(distances[idx])
        if max_dist > 0
            weights = (1 .- (distances[idx] ./ max_dist).^3).^3
        else
            weights = ones(r)
        end
        
        # Weighted linear regression
        X_local = hcat(ones(r), x[idx])
        W = Diagonal(weights)
        
        try
            # Solve weighted least squares: (X'WX)^(-1) X'Wy
            beta = (X_local' * W * X_local) \ (X_local' * W * y[idx])
            y_pred[i] = beta[1] + beta[2] * xp
        catch
            # If singular, use weighted mean
            y_pred[i] = sum(weights .* y[idx]) / sum(weights)
        end
    end
    
    return y_pred
end

# Load lidar data (simulated since SemiPar package not directly available)
# In practice, you would load from a CSV or other data source
# For demonstration, we'll create synthetic data similar to lidar
Random.seed!(42)
n_points = 221
range_vals = sort(rand(n_points) .* 330 .+ 390)
# Create a non-linear relationship with noise
logratio = sin.((range_vals .- 390) ./ 50) .* 0.3 .+ 
           exp.(-(range_vals .- 500).^2 ./ 5000) .* 0.5 .+ 
           randn(n_points) .* 0.05

lidar = DataFrame(range = range_vals, logratio = logratio)

println("Lidar data shape: ", size(lidar))
println(first(lidar, 5))

# Using the lidar data again, we will:
# 1. Fit several different smooths corresponding to different choices of 
#    bandwidth (span)
# 2. Estimate the mean squared prediction error of each smooth
# 3. Choose the smooth with the smallest estimated MSPE

# Define a sequence of span values to try
s = 0.1:0.1:1.0
K = 10
n = nrow(lidar)

# ============================================================================
# Example: Lidar Data - K-fold Cross-Validation
# ============================================================================

# Matrix to store cross-validation errors
# Rows: different folds, Columns: different span values
cv_error = zeros(K, length(s))

# Randomly split data into K subsets
# Each observation gets a foldid between 1 and K
foldid = sample(1:K, n, replace=true)

for i in 1:K
    # Get training and validation indices
    train_idx = foldid .!= i
    val_idx = foldid .== i
    
    # Training and validation data
    x_train = lidar.range[train_idx]
    y_train = lidar.logratio[train_idx]
    x_val = lidar.range[val_idx]
    y_val = lidar.logratio[val_idx]
    
    # Fit, predict, and calculate error for each bandwidth
    for (j, span) in enumerate(s)
        # Fit LOESS model to training set (all data except fold i)
        y_pred = loess_smooth(x_train, y_train, x_val, frac=span)
        
        # Calculate mean squared error on validation set
        cv_error[i, j] = mean((y_val .- y_pred).^2)
    end
end

# Columns of cv_error correspond to different bandwidths
# Average across all K folds to get final CV error estimate for each span
cv_error_estimate = vec(mean(cv_error, dims=1))

# ============================================================================
# Cross-validation estimates of MSPE
# ============================================================================

# Plot the CV error estimates for different span values
plot(collect(s), cv_error_estimate, 
     marker=:circle, markersize=6, linewidth=2,
     xlabel="Span", ylabel="CV Error Estimate",
     title="Cross-Validation Error vs Span",
     legend=false, grid=true)
savefig("cv_error_vs_span_julia.png")

# Find the span that minimizes the CV error
s_best = collect(s)[argmin(cv_error_estimate)]
println("\nBest span selected by cross-validation: ", round(s_best, digits=2))
println("Minimum CV error: ", round(minimum(cv_error_estimate), digits=6))

# ============================================================================
# Smoother selected by cross-validation
# ============================================================================

# Plot the data with the optimal smoother
p1 = scatter(lidar.range, lidar.logratio, alpha=0.5, markersize=3,
             xlabel="Range", ylabel="Log Ratio",
             title="Lidar Data with Optimal Loess Smooth (span = $s_best)",
             legend=false)
x_smooth = range(minimum(lidar.range), maximum(lidar.range), length=200)
y_smooth = loess_smooth(lidar.range, lidar.logratio, collect(x_smooth), frac=s_best)
plot!(p1, x_smooth, y_smooth, linewidth=2, color=:red, label="span=$s_best")

# Compare with other span choices
p2 = scatter(lidar.range, lidar.logratio, alpha=0.5, markersize=3,
             xlabel="Range", ylabel="Log Ratio",
             title="Comparison of Different Span Values",
             label="Data")

# Span = 0.1 (too flexible)
y_smooth1 = loess_smooth(lidar.range, lidar.logratio, collect(x_smooth), frac=0.1)
plot!(p2, x_smooth, y_smooth1, linewidth=2, color=:blue, 
      linestyle=:dash, label="span=0.1")

# Optimal span
y_smooth_opt = loess_smooth(lidar.range, lidar.logratio, collect(x_smooth), frac=s_best)
plot!(p2, x_smooth, y_smooth_opt, linewidth=2, color=:red, 
      label="span=$s_best (optimal)")

# Span = 1.0 (too smooth)
y_smooth3 = loess_smooth(lidar.range, lidar.logratio, collect(x_smooth), frac=1.0)
plot!(p2, x_smooth, y_smooth3, linewidth=2, color=:purple, 
      linestyle=:dash, label="span=1.0")

plot(p1, p2, layout=(1, 2), size=(1200, 500))
savefig("loess_comparison_julia.png")

# ============================================================================
# Split data into training and test for visualization
# ============================================================================

# Randomly split data into K subsets
K = 10
n = nrow(lidar)
foldid = sample(1:K, n, replace=true)

# Split data into training and test data
lidar_train = lidar[foldid .!= 1, :]
lidar_test = lidar[foldid .== 1, :]

# Example: Lidar data
scatter(lidar_train.range, lidar_train.logratio, 
        alpha=0.6, markersize=4,
        xlabel="Range", ylabel="Log Ratio",
        title="Lidar Data (Training Set)",
        legend=false, grid=true)
savefig("lidar_training_data_julia.png")

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
x_train = lidar_train.range
y_train = lidar_train.logratio

# Visualize the loess smooth
x_smooth = range(minimum(x_train), maximum(x_train), length=200)
y_smooth = loess_smooth(x_train, y_train, collect(x_smooth), frac=0.75)

scatter(x_train, y_train, alpha=0.5, markersize=4,
        xlabel="Range", ylabel="Log Ratio",
        title="Lidar Data with Loess Smooth",
        label="Training Data")
plot!(x_smooth, y_smooth, linewidth=2, color=:red, label="Loess Smooth (span=0.75)")
savefig("lidar_loess_smooth_julia.png")

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
p1 = scatter(x_train, y_train, alpha=0.5, markersize=3, legend=false,
             title="Span = 0.02 (Too Flexible)")
plot!(p1, x_train, y_pred1, linewidth=2, color=:blue)

p2 = scatter(x_train, y_train, alpha=0.5, markersize=3, legend=false,
             title="Span = 0.3 (Balanced)")
plot!(p2, x_train, y_pred2, linewidth=2, color=:green)

p3 = scatter(x_train, y_train, alpha=0.5, markersize=3, legend=false,
             title="Span = 1 (Too Smooth)")
plot!(p3, x_train, y_pred3, linewidth=2, color=:purple)

p4 = scatter(x_train, y_train, alpha=0.5, markersize=3, label="Data",
             title="All Three Models")
plot!(p4, x_train, y_pred1, linewidth=2, color=:blue, 
      linestyle=:dash, label="span=0.02")
plot!(p4, x_train, y_pred2, linewidth=2, color=:green, label="span=0.3")
plot!(p4, x_train, y_pred3, linewidth=2, color=:purple, 
      linestyle=:dash, label="span=1")

plot(p1, p2, p3, p4, layout=(2, 2), size=(1200, 900))
savefig("span_comparison_julia.png")

# Example: Lidar data
# Compare models using squared error on training data
# Model 1: sum |Yi - f^(Xi)|^2 = 0 (essentially)
# Model 2: sum |Yi - f^(Xi)|^2 = 1.03
# Model 3: sum |Yi - f^(Xi)|^2 = 1.71
# Therefore, Model 1 has the smallest squared error over the data

# Sum of squared errors
sse1 = sum((y_train .- y_pred1).^2)
sse2 = sum((y_train .- y_pred2).^2)
sse3 = sum((y_train .- y_pred3).^2)

println("\nSum of squared errors:")
println("Model 1 (span=0.02): ", @sprintf("%.6e", sse1))
println("Model 2 (span=0.3):  ", @sprintf("%.6e", sse2))
println("Model 3 (span=1.0):  ", @sprintf("%.6e", sse3))

# Mean squared errors
mse1 = mean((y_train .- y_pred1).^2)
mse2 = mean((y_train .- y_pred2).^2)
mse3 = mean((y_train .- y_pred3).^2)

println("\nMean squared errors:")
println("Model 1 (span=0.02): ", @sprintf("%.6e", mse1))
println("Model 2 (span=0.3):  ", @sprintf("%.6e", mse2))
println("Model 3 (span=1.0):  ", @sprintf("%.6e", mse3))

# Example: Lidar data - Test set visualization
scatter(lidar_test.range, lidar_test.logratio, 
        alpha=0.6, markersize=6, color=:orange,
        xlabel="Range", ylabel="Log Ratio",
        title="New Lidar Data (Test Set)",
        legend=false, grid=true)
savefig("lidar_test_data_julia.png")

println("\n" * "="^80)
println("Cross-validation analysis complete!")
println("="^80)
