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

# K <- 10
# n <- nrow(mydata)
# cv.error <- vector(length = K)
# 
# # Randomly split data into K subsets
# # - each observation gets a foldid between 1 and K
# foldid <- sample(rep(1:K, length = n))
# 
# # Repeat K times
# for(i in 1:K) {
#   # Fit using training set
#   f.hat <- estimator(mydata[foldid != i, ])
#   
#   # Calculate prediction error on validation set
#   cv.error[i] <- calc_error(f.hat, mydata[foldid == i, ])
# }
# 
# cv.error.estimate <- mean(cv.error)

# Common choices for K:
# - K = 5 or K = 10 (good balance between bias and variance)
# - K = n (leave-one-out, computationally expensive)
# - K = 2 (simple but high bias)

# ============================================================================
# Example: Lidar data
# Consider the lidar data from SemiPar package in R
library(SemiPar)
data(lidar)

# Using the lidar data again, we will:
# 1. Fit several different smooths corresponding to different choices of 
#    bandwidth (span)
# 2. Estimate the mean squared prediction error of each smooth
# 3. Choose the smooth with the smallest estimated MSPE

library(ggplot2)

# Define a sequence of span values to try
s = seq(from = 0.1, to = 1.0, by = 0.1)
K <- 10
n <- nrow(lidar)

# ============================================================================
# Example: Lidar Data - K-fold Cross-Validation
# ============================================================================

# Matrix to store cross-validation errors
# Rows: different folds, Columns: different span values
cv.error <- matrix(nrow = K, ncol = length(s))

# Randomly split data into K subsets
# Each observation gets a foldid between 1 and K
foldid <- sample(rep(1:K, length = n))

for(i in 1:K) {
  # Fit, predict, and calculate error for each bandwidth
  cv.error[i, ] <- sapply(s, function(span) {
    # Fit LOESS model to training set (all data except fold i)
    obj <- loess(logratio ~ range,
                 data = subset(lidar, foldid != i),
                 span = span,
                 control = loess.control(surface = 'direct'))
    # Predict and calculate error on the validation set (fold i)
    y.hat <- predict(obj, newdata = subset(lidar, foldid == i))
    pse <- mean((subset(lidar, foldid == i)$logratio - y.hat)^2)
    return(pse)
  }) 
}

# Columns of cv.error correspond to different bandwidths
# Average across all K folds to get final CV error estimate for each span
cv.error.estimate <- colMeans(cv.error)

# ============================================================================
# Cross-validation estimates of MSPE
# ============================================================================

# Plot the CV error estimates for different span values
qplot(s, cv.error.estimate, geom=c('line', 'point'), xlab='span',
      ylab='CV Error Estimate', main='Cross-Validation Error vs Span')

# Find the span that minimizes the CV error
s.best <- s[which.min(cv.error.estimate)]
cat("Best span selected by cross-validation:", s.best, "\n")
cat("Minimum CV error:", min(cv.error.estimate), "\n")

# ============================================================================
# Smoother selected by cross-validation
# ============================================================================

# Plot the data with the optimal smoother
qplot(range, logratio, data=lidar, 
      main=paste("Lidar Data with Optimal Loess Smooth (span =", s.best, ")")) + 
  geom_smooth(method = 'loess', span = s.best, se = FALSE, color = 'red', size = 1.2)

# Compare with other span choices
qplot(range, logratio, data=lidar, main="Comparison of Different Span Values") +
  geom_smooth(method = 'loess', span = 0.1, se = FALSE, 
              aes(color = "span=0.1"), size = 1) +
  geom_smooth(method = 'loess', span = s.best, se = FALSE, 
              aes(color = paste("span=", s.best, " (optimal)")), size = 1.2) +
  geom_smooth(method = 'loess', span = 1.0, se = FALSE, 
              aes(color = "span=1.0"), size = 1) +
  scale_color_manual(name = "Models",
                     values = c("span=0.1" = "blue", 
                               paste("span=", s.best, " (optimal)") = "red",
                               "span=1.0" = "purple"))

# ============================================================================
# Split data into training and test for visualization
# ============================================================================

# Randomly split data into K subsets
# Each observation gets a foldid between 1 and K
K <- 10
n <- nrow(lidar)
foldid <-sample(rep(1:K, length = n))

# Split data into training and test data
lidar.train <- subset(lidar, foldid != 1)
lidar.test <-  subset(lidar, foldid == 1)
attach(lidar.train)

# Example: Lidar data
plot(range,logratio, main="Lidar Data")

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
obj0 <- loess(logratio ~ range, data = lidar.train, 
              control = loess.control(surface = 'direct'))

# Visualize the loess smooth
plot(obj0, xlab="range", ylab="logratio", main="Lidar Data with Loess Smooth")
points(obj0$x, obj0$fitted, type="l", col="red", lwd=2)

# How to choose the span parameter?
# The span parameter controls the amount of smoothing:
# - Small span: flexible fit, follows data closely (may overfit)
# - Large span: smooth fit, averages over more data (may underfit)
# - Default span: 0.75 (moderate smoothing)

# Which model is the best? Why?
# Compare three different span values:

# Model 1: Very small span (0.02) - highly flexible, may overfit
obj1 <- loess(logratio ~ range, data = lidar.train, span = .02, 
              control = loess.control(surface = 'direct'))

# Model 2: Moderate span (0.3) - balanced flexibility
obj2 <- loess(logratio ~ range, data = lidar.train, span = .3, 
              control = loess.control(surface = 'direct'))

# Model 3: Large span (1) - very smooth, may underfit
obj3 <- loess(logratio ~ range, data = lidar.train, span = 1, 
              control = loess.control(surface = 'direct'))

# Visualize all three models
par(mfrow=c(2,2))
plot(logratio ~ range, data = lidar.train, main="Span = 0.02 (Too Flexible)")
lines(obj1$x, obj1$fitted, col="blue", lwd=2)

plot(logratio ~ range, data = lidar.train, main="Span = 0.3 (Balanced)")
lines(obj2$x, obj2$fitted, col="green", lwd=2)

plot(logratio ~ range, data = lidar.train, main="Span = 1 (Too Smooth)")
lines(obj3$x, obj3$fitted, col="purple", lwd=2)

plot(logratio ~ range, data = lidar.train, main="All Three Models")
lines(obj1$x, obj1$fitted, col="blue", lwd=2, lty=2)
lines(obj2$x, obj2$fitted, col="green", lwd=2)
lines(obj3$x, obj3$fitted, col="purple", lwd=2, lty=2)
legend("topright", legend=c("span=0.02", "span=0.3", "span=1"), 
       col=c("blue", "green", "purple"), lwd=2, lty=c(2,1,2))
par(mfrow=c(1,1))

# Example: Lidar data
# Compare models using squared error on training data
# Model 1: sum |Yi - f^(Xi)|^2 = 0 (essentially)
# Model 2: sum |Yi - f^(Xi)|^2 = 1.03
# Model 3: sum |Yi - f^(Xi)|^2 = 1.71
# Therefore, Model 1 has the smallest squared error over the data

# Sum of squared errors
c(sum((logratio - obj1$fitted)^2), sum((logratio - obj2$fitted)^2), 
  sum((logratio - obj3$fitted)^2))
## [1] 1.396971e-30 1.195109e+00 1.879064e+00

# Mean squared errors
c(mean((logratio - obj1$fitted)^2), mean((logratio - obj2$fitted)^2),
  mean((logratio - obj3$fitted)^2))
## [1] 7.055410e-33 6.035906e-03 9.490224e-03

# Example: Lidar data - Test set visualization
plot(lidar.test$range, lidar.test$logratio, xlab="range", 
     ylab="logratio", main="New Lidar Data")
