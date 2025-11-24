# Agenda
# In this session, we will explore methods for estimating probability 
# density functions from data. We'll cover the following topics:
#
# - Histograms
#   * Basic tool for visualizing distributions
#   * Choice of bin width and number of bins
#   * Advantages and limitations
#
# - Glivenko-Cantelli theorem
#   * Theoretical foundation for empirical distributions
#   * Convergence of empirical CDF to true CDF
#   * Uniform convergence properties
#
# - Error for density estimates
#   * Bias-variance tradeoff in density estimation
#   * Mean squared error and integrated squared error
#   * How to measure the quality of density estimates
#
# - Kernel density estimates
#   * Smooth alternative to histograms
#   * Choice of kernel function
#   * Bandwidth selection methods
#   * Properties and advantages over histograms
#
# - Bivariate density estimates
#   * Extension to two dimensions
#   * Kernel density estimation in 2D
#   * Visualization techniques (contour plots, 3D surfaces)
#   * Applications and interpretation

# ============================================================================
# Histograms
# ============================================================================

# Histograms are one of the first things learned in "Introduction to Statistics"
# Simple way of estimating a distribution
# Split the sample space up into bins
# Count how many samples fall into each bin
# If we hold the bins fixed and take more and more data, then the relative 
# frequency for each bin will converge on the bin's probability
#
# Key properties:
# - Easy to construct and interpret
# - No assumptions about the underlying distribution
# - Discrete approximation to a continuous density
#
# Limitations:
# - Sensitive to choice of bin width and bin placement
# - Can be misleading with poor bin choices
# - Not smooth (discontinuous at bin boundaries)
# - Difficult to compare across different sample sizes

# ============================================================================
# Example: Old Faithful Geyser Data
# ============================================================================

# Load the Old Faithful geyser eruption data
data(faithful)

# Set up histogram parameters
x0 <- 0           # Lower bound
x1 <- 8           # Upper bound
h <- 0.5          # Bin width

# Create breaks for histogram bins
my.breaks <- seq(from=x0, to=x1, by=h)

# Compute histogram (without plotting)
myhist <- hist(faithful$eruptions, breaks=my.breaks, right=F, plot=F)

# Plot the histogram as a step function showing density
pdf("plots/01_histogram_faithful.pdf")
plot(myhist$mids, myhist$density, type="s", xlab="Eruption length", 
     ylab="Frequency", main="Histogram of Eruption Lengths")
dev.off()

# The histogram shows the distribution of geyser eruption durations
# Notice the bimodal pattern: short eruptions (~2 min) and long eruptions (~4.5 min)

# ============================================================================
# Histograms - Bin Width Selection
# ============================================================================

# Bin width primarily controls the amount of smoothing, lots of guidance available

# Sturges' rule: Optimal width of class intervals is given by
#   h = R / (1 + log₂(n))
# where R is the sample range
# - Designed for data sampled from symmetric, unimodal populations
# - Simple and widely used
# - May produce too few bins for large datasets

# Scott's Normal reference rule: Specifies a bin width
#   h = 3.49 * σ̂ * n^(-1/3)
# where σ̂ is an estimate of the population standard deviation σ
# - Assumes data is approximately normal
# - Minimizes the integrated mean squared error
# - Works well for smooth, unimodal distributions

# Freedman-Diaconis rule: Specifies the bin width to be
#   h = 2 * IQR * n^(-1/3)
# where IQR is the sample inter-quartile range
# - More robust to outliers than Scott's rule
# - Uses IQR instead of standard deviation
# - Better for skewed or heavy-tailed distributions
#
# Note: All three rules suggest that bin width should decrease as n increases,
# but at a slower rate (n^(-1/3)) to balance bias and variance

# ============================================================================
# Histograms - Fundamental Questions
# ============================================================================

# Is learning the whole distribution non-parametrically even feasible?
# - Yes, but with important caveats
# - As sample size n → ∞, histogram converges to true density
# - However, convergence is slower than for parametric methods
# - Curse of dimensionality: difficulty increases exponentially with dimension
# - For univariate problems, non-parametric density estimation is practical
# - Trade-off: flexibility vs. statistical efficiency

# How can we measure error to deal with the bias-variance trade-off?
# - Common error measures for density estimates:
#   * Mean Squared Error (MSE) at a point: E[(f̂(x) - f(x))²]
#   * Integrated Squared Error (ISE): ∫(f̂(x) - f(x))² dx
#   * Mean Integrated Squared Error (MISE): E[∫(f̂(x) - f(x))² dx]
#
# - Bias-variance decomposition:
#   * MSE = Bias² + Variance
#   * Smaller bins → low bias, high variance (undersmoothing)
#   * Larger bins → high bias, low variance (oversmoothing)
#   * Optimal bin width balances these two sources of error
#
# - As sample size increases:
#   * Can use smaller bins (more detail)
#   * Both bias and variance decrease
#   * But must balance them appropriately

# ============================================================================
# Empirical CDF
# ============================================================================

# Learning the whole distribution is feasible
# Something even dumber than shrinking histograms will work
#
# Suppose we have one-dimensional samples x₁, ..., xₙ with CDF F
#
# Define the empirical cumulative distribution function on n samples as:
#
#   F̂ₙ(a) = (1/n) * Σᵢ₌₁ⁿ I(-∞ < xᵢ ≤ a)
#
# Just the fraction of the samples which are less than or equal to a
#
# Properties of the empirical CDF:
# - Non-parametric estimate of the true CDF
# - Step function with jumps of size 1/n at each observation
# - Unbiased: E[F̂ₙ(a)] = F(a) for all a
# - Consistent: F̂ₙ(a) → F(a) as n → ∞
# - No tuning parameters needed (unlike histograms)
# - Provides complete information about the distribution

# ============================================================================
# Glivenko-Cantelli Theorem
# ============================================================================

# Then the Glivenko-Cantelli theorem says:
#
#   max_a |F̂ₙ(a) - F(a)| → 0   as n → ∞
#
# So the empirical CDF converges to the true CDF everywhere, i.e., the 
# maximum gap between the two of them goes to zero
#
# Pitman (1979) calls this the "fundamental theorem of statistics"
#
# Key insights:
# - Uniform convergence: convergence holds simultaneously for all values a
# - Not just pointwise convergence at individual points
# - The worst-case error across the entire domain goes to zero
# - Can learn distributions just by collecting enough data
# - No assumptions about the form of F (completely non-parametric)
# - Provides theoretical justification for empirical methods
#
# Implications:
# - With enough data, we can learn any distribution
# - No need to assume parametric form (e.g., normal, exponential)
# - The empirical CDF is a strongly consistent estimator
# - Foundation for many statistical procedures (e.g., bootstrap, QQ-plots)

# ============================================================================
# From CDF to Density Estimation
# ============================================================================

# Can we use the empirical CDF to estimate a density?
# - Yes, but it's discrete and doesn't estimate a density well
# - The derivative of the empirical CDF is a sum of point masses
# - Usually we can expect to find some new samples between our old ones
# - So we want a non-zero density between our observations
# - Uniform distribution within each bin of a histogram doesn't have this issue
#
# Can we do better?

# ============================================================================
# Error for Density Estimates
# ============================================================================

# Yes, but what do we mean by "better" density estimates?
#
# Three ideas for measuring error between true density f(x) and estimate f̂(x):

# 1. Squared deviation from the true density should be small
#    Integrated Squared Error (ISE):
#
#    ∫(f(x) - f̂(x))² dx
#
#    - L₂ distance between densities
#    - Penalizes large errors heavily
#    - Most commonly used in theory
#    - Related to mean squared error

# 2. Absolute deviation from the true density should be small
#    Integrated Absolute Error (IAE):
#
#    ∫|f(x) - f̂(x)| dx
#
#    - L₁ distance between densities
#    - More robust to outliers
#    - Also known as total variation distance
#    - Easier to interpret (average absolute difference)

# 3. Average log-likelihood ratio should be kept low
#    Kullback-Leibler divergence:
#
#    ∫f(x) log(f(x)/f̂(x)) dx
#
#    - Information-theoretic measure
#    - Not symmetric: KL(f||f̂) ≠ KL(f̂||f)
#    - Penalizes putting probability mass where true density is low
#    - Related to maximum likelihood estimation
#    - Always non-negative, equals 0 when f = f̂
#
# In practice, we often use Mean Integrated Squared Error (MISE):
#    MISE = E[∫(f(x) - f̂(x))² dx]
# which averages ISE over all possible samples

# ============================================================================
# Error for Density Estimates - Detailed Discussion
# ============================================================================

# Squared deviation is similar to MSE criterion used in regression
# - Used most frequently since it's mathematically tractable
# - Allows for bias-variance decomposition
# - Leads to closed-form solutions in many cases

# Absolute deviation considers L₁ or total variation distance between the 
# true and the estimated density
# - Nice property that (1/2)∫|f(x) - f̂(x)| dx is exactly the maximum error 
#   in our estimate of the probability of any set
# - But it's tricky to work with, so we'll skip it
# - More robust but less convenient mathematically

# Minimizing the log-likelihood ratio is intimately connected both to 
# maximizing the likelihood and to minimizing entropy
# - Called Kullback-Leibler divergence or relative entropy
# - Natural from information theory perspective
# - Asymmetric measure (direction matters)

# ----------------------------------------------------------------------------
# Working with Integrated Squared Error
# ----------------------------------------------------------------------------

# Notice that:
#
#   ∫(f(x) - f̂(x))² dx = ∫f²(x) dx - 2∫f(x)f̂(x) dx + ∫f̂²(x) dx
#
# Breaking down each term:
#
# First term: ∫f²(x) dx
# - Doesn't depend on the estimate, so we can ignore it for purposes of 
#   optimization
# - It's a constant that doesn't affect which estimator is best
#
# Third term: ∫f̂²(x) dx
# - Only involves f̂(x), and is just an integral, which we can do numerically
# - Can be computed directly from the estimated density
# - Measures the "roughness" or variability of the estimate
#
# Second term: -2∫f(x)f̂(x) dx
# - Involves both the true and the estimated density
# - We can approximate it using Monte Carlo by:
#
#     -2∫f(x)f̂(x) dx ≈ -(2/n)Σᵢ₌₁ⁿ f̂(xᵢ)
#
# - Since the xᵢ are samples from f(x), the sample mean estimates the integral
# - This gives us a practical way to estimate prediction error
#
# Putting it together for optimization:
# To minimize ISE, we minimize:  ∫f̂²(x) dx - (2/n)Σᵢ₌₁ⁿ f̂(xᵢ)
# This is computable from the data and the estimate alone!

# ----------------------------------------------------------------------------
# Practical Error Measure
# ----------------------------------------------------------------------------

# Then our error measure is:
#
#   (2/n)Σᵢ₌₁ⁿ f̂(xᵢ) + ∫f̂²(x) dx
#
# (We've dropped the minus sign and constant terms for minimization)
#
# In fact, this error measure does not depend on having one-dimensional data
# - Extends naturally to multivariate densities
# - Same principle applies in higher dimensions
# - Computational cost increases with dimension
#
# For purposes of cross-validation, we can estimate f̂(x) on the training set 
# and then restrict the sum to points in the testing set
# - Split data into training and test sets
# - Fit density estimate on training data
# - Evaluate error on test data to avoid overfitting
# - Similar to cross-validation in regression

# ============================================================================
# Naive Estimator
# ============================================================================

# If a random variable X has probability density f, then:
#
#   f(x) = lim_{h→0} (1/(2h)) * P(x - h < X < x + h)
#
# This is the definition of density: probability per unit length
# - As the interval shrinks, the ratio approaches the density
# - Probability in small interval ≈ density × interval length
#
# Thus, a naive estimator would be:
#
#   f̂(x) = (1/(2nh)) * [# of xᵢ falling in (x - h, x + h)]
#
# How it works:
# - Count observations within distance h of x
# - Divide by total observations n
# - Divide by interval width 2h to get density (probability per unit length)
#
# Properties:
# - Simple and intuitive
# - Unbiased as n → ∞ and h → 0 appropriately
# - But discontinuous and not smooth
# - Choice of h (bandwidth) is critical
#   * Too small: noisy, high variance
#   * Too large: oversmoothed, high bias

# ----------------------------------------------------------------------------
# Naive Estimator - Alternative Formulation
# ----------------------------------------------------------------------------

# Or, equivalently:
#
#   f̂(x) = (1/n) Σᵢ₌₁ⁿ (1/h) * w((x - xᵢ)/h)
#
# where w is a weight function (kernel) defined as:
#
#   w(x) = { 1/2   if |x| < 1
#          { 0     otherwise
#
# This is the uniform (or "box") kernel
#
# In short, a naive estimate is constructed by:
# - Placing a box of width 2h and height 1/(2nh) on each observation
# - Summing to obtain the estimate
#
# Interpretation:
# - Each observation contributes a uniform "bump" of probability mass
# - The bump has total area 1/n (so all bumps sum to 1)
# - The bump is spread uniformly over an interval of width 2h
# - Height of each bump = (1/n) / (2h) = 1/(2nh)
#
# This formulation:
# - Generalizes easily to other kernel functions (not just boxes)
# - Makes the role of bandwidth h explicit
# - Shows how local averaging produces the density estimate
# - Each point gets equal weight within distance h, zero weight outside

# ============================================================================
# Example: Old Faithful Geyser Data - Naive Estimator
# ============================================================================

# Define the uniform (box) weight function
my.w <- function(x){
    if (abs(x) < 1) 
        w <- 1/2
    else
        w <- 0
    w
}

# Set up grid of points where we want to estimate the density
x <- seq(0, 6, 0.2)
m <- length(x)
n <- length(faithful$eruptions)
h <- 0.5  # Bandwidth

# Initialize density estimate vector
fhat <- rep(0, m)

# Compute naive density estimate at each grid point
for (i in 1:m){
    S <- 0
    # Sum contributions from all observations
    for (j in 1:n){
        # Kernel contribution from observation j to point x[i]
        S <- S + (1/h) * my.w((faithful$eruptions[j] - x[i]) / h)
    }
    # Average over all observations
    fhat[i] <- (1/n) * S
}

# Plot the naive density estimate
pdf("plots/02_naive_density_estimator.pdf")
plot(x, fhat, type="s", xlab="Eruption length", ylab="Density Estimate", 
     main="Naive Density Estimator")
dev.off()

# ============================================================================
# Limitations of Naive Estimator
# ============================================================================

# Not wholly satisfactory, from the point of view of using density estimates 
# for presentation
# - Estimate f̂ is a step function
# - Discontinuous at boundaries
# - Not smooth or visually appealing
# - Doesn't capture the smoothness we expect from continuous densities
#
# Solution: Use better kernel functions
#
# In the formula for the naive estimate, we can replace the weight function w 
# by another function K with more desirable properties
#
# Function K is called a kernel
#
# General kernel density estimate:
#   f̂(x) = (1/n) Σᵢ₌₁ⁿ (1/h) * K((x - xᵢ)/h)
#
# where K is a kernel function satisfying:
# - ∫K(u) du = 1  (integrates to 1, like a probability density)
# - K(u) ≥ 0 for all u  (non-negative)
# - K is symmetric: K(-u) = K(u)  (typically, but not always required)
# - K is smooth (unlike the box kernel)

# ============================================================================
# Kernel Density Estimates
# ============================================================================

# Resulting estimate is a kernel estimator:
#
#   f̂(x) = (1/n) Σᵢ₌₁ⁿ (1/h) * K((x - xᵢ)/h)
#
# where:
# - h is the window width, smoothing parameter, or bandwidth
# - K is the kernel function
# - n is the number of observations
# - xᵢ are the data points
#
# Key components:
#
# Bandwidth h:
# - Controls the amount of smoothing
# - Larger h → smoother estimate (more bias, less variance)
# - Smaller h → rougher estimate (less bias, more variance)
# - Critical choice for the quality of the estimate
#
# Kernel K:
# - Usually taken to be a probability density function itself
# - Common choices: normal (Gaussian), Epanechnikov, triangular, uniform
# - Normal density is popular: K(u) = (1/√(2π)) * exp(-u²/2)
# - Resulting estimate will inherit all the smoothness properties of K
#   * If K is continuous, f̂ is continuous
#   * If K is differentiable, f̂ is differentiable
#   * Smooth kernels produce smooth density estimates
#
# Interpretation:
# - Place a "bump" (scaled kernel) at each observation
# - Each bump has area 1/n
# - Each bump has width controlled by h
# - Sum all bumps to get the density estimate

# ============================================================================
# Kernel Function Implementation
# ============================================================================

# Define a flexible kernel function that supports multiple kernel types
my.w <- function(x, type="gaussian"){
  if(type == "gaussian"){
    # Gaussian (Normal) kernel
    w <- dnorm(x)
    return(w)
  }
  if(type == "naive"){
    # Uniform (Box) kernel
    if (abs(x) < 1) 
      w <- 1/2
    else
      w <- 0
    return(w)
  }
  # If neither type is specified, print error message
  print("You have asked for an undefined kernel.")
  return(NULL)
}

# This function allows easy comparison between different kernels
# - "gaussian": smooth, infinitely differentiable, most commonly used
# - "naive": uniform box kernel, simple but discontinuous
# - Can be extended to include other kernels (Epanechnikov, triangular, etc.)

# ============================================================================
# Example: Old Faithful Geyser Data - Gaussian Kernel
# ============================================================================

# Set up finer grid of points for smoother visualization
x <- seq(0, 6, 0.02)
m <- length(x)
n <- length(faithful$eruptions)
h <- 0.1  # Smaller bandwidth for more detail

# Initialize density estimate vector
fhat <- rep(0, m)

# Compute kernel density estimate at each grid point
for (i in 1:m){
    S <- 0
    # Sum contributions from all observations using Gaussian kernel
    for (j in 1:n){
        S <- S + (1/h) * my.w((faithful$eruptions[j] - x[i]) / h)
    }
    # Average over all observations
    fhat[i] <- (1/n) * S
}

# Note: By default, my.w uses the Gaussian kernel, producing a smooth estimate
# Compare this with the naive estimator's step function appearance

# Plot the kernel density estimate
pdf("plots/03_gaussian_kernel_density.pdf")
plot(x, fhat, type="l", xlab="Eruption length", ylab="Density Estimate", 
     main="Kernel Density Estimator (Gaussian)")
dev.off()

# ============================================================================
# Bandwidth Selection
# ============================================================================

# Choosing the bandwidth h is critical for kernel density estimation
# Too small → undersmoothing (high variance, low bias)
# Too large → oversmoothing (low variance, high bias)

# Method 1: Cross-validation
# - Use cross-validation to minimize prediction error
# - Split data into training and test sets
# - Try different bandwidth values
# - Choose h that minimizes cross-validated error
# - Problem: Can be time consuming, especially for large datasets
# - Advantage: Data-driven, no assumptions about true density

# Method 2: Gaussian reference rule (Rule-of-thumb bandwidth)
# - Optimal bandwidth for a Gaussian kernel to estimate a Gaussian distribution is:
#
#   h = 1.06 * σ / n^(1/5)
#
# where σ is the standard deviation of the data and n is the sample size
#
# Also called Silverman's rule of thumb
# - Simple and fast to compute
# - Works well when data is approximately normal
# - May not be optimal for multimodal or skewed distributions
# - When you call density() in R, this is basically what it does by default
#
# Alternative: Scott's rule
#   h = 1.059 * σ / n^(1/5)
# Very similar to Gaussian reference rule
#
# In practice:
# - Start with rule-of-thumb bandwidth as baseline
# - Adjust based on visual inspection
# - Use cross-validation for more careful analysis
# - For robust estimation, can use IQR instead of σ:
#   h = 0.9 * min(σ, IQR/1.34) / n^(1/5)

# ============================================================================
# Kernel Density Estimate Samples
# ============================================================================

# There are times when one wants to draw a random sample from the estimated 
# distribution
# - Easy with kernel density estimates, because each kernel is itself a 
#   probability density
# - The KDE is a mixture of n kernel densities
#
# Algorithm for sampling from a kernel density estimate:
#
# Suppose the kernel is Gaussian, that we have scalar observations x₁,...,xₙ
# and bandwidth h
#
# For a single draw:
# 1. Pick an integer i uniformly at random from 1 to n
# 2. Draw from N(xᵢ, h²)
#    Use: rnorm(1, x[i], h)
#
# For q draws:
#    rnorm(q, sample(x, q, replace=TRUE), h)
#
# Interpretation:
# - First, randomly select one of the n kernels (with equal probability 1/n)
# - Then, draw from that kernel centered at the selected observation
# - This is sampling from a mixture distribution
#
# Using a different kernel, we'd just need to use the random number generator 
# function for the corresponding distribution
# - For uniform kernel: runif with appropriate range
# - For Epanechnikov kernel: need custom sampler
# - For triangular kernel: need custom sampler

# ============================================================================
# Other Approaches
# ============================================================================

# Histograms and kernels are not the only possible way of estimating densities
#
# Alternative methods include:
#
# 1. Local polynomial density estimation
#    - Fit polynomials locally, similar to local regression
#    - More flexible than simple kernel smoothing
#    - Can adapt to varying smoothness
#
# 2. Series expansions
#    - Approximate density as sum of basis functions
#    - Examples: Fourier series, wavelets
#    - Good for certain functional forms
#
# 3. Splines
#    - Piecewise polynomial approximations
#    - Can enforce smoothness through penalties
#    - Flexible and computationally efficient
#
# 4. Penalized likelihood approaches
#    - Maximize likelihood subject to smoothness penalty
#    - Balance fit and smoothness explicitly
#    - Can control roughness through penalty parameter
#
# For some of these, avoid negative probability density estimates using the 
# log density
# - Model log(f(x)) instead of f(x)
# - Ensures f(x) = exp(log(f(x))) > 0 everywhere
# - Particularly useful for methods that might produce negative values
# - Common in penalized likelihood and spline approaches

# ============================================================================
# Density Estimation in R
# ============================================================================

# density() function is the most common built-in function for kernel density 
# estimation in R

# Basic usage:
#   density(x, ...)

# Full function signature:
# density(x, bw = "nrd0", adjust = 1,
#         kernel = c("gaussian", "epanechnikov", "rectangular",
#                    "triangular", "biweight",
#                    "cosine", "optcosine"),
#         weights = NULL, window = kernel, width,
#         give.Rkern = FALSE,
#         n = 512, from, to, cut = 3, na.rm = FALSE, ...)

# Key arguments:
# - x: numeric vector of data
# - bw: bandwidth selection method
#   * "nrd0" = rule-of-thumb (default, uses 1.06*σ*n^(-1/5))
#   * "nrd" = alternative rule-of-thumb
#   * "ucv" = unbiased cross-validation
#   * "bcv" = biased cross-validation
#   * Or specify a numeric value directly
# - adjust: multiplier for bandwidth (adjust > 1 for more smoothing)
# - kernel: choice of kernel function
#   * "gaussian" (default and most common)
#   * "epanechnikov" (theoretically optimal)
#   * "rectangular" (uniform/box kernel)
#   * "triangular", "biweight", "cosine", "optcosine"
# - n: number of equally spaced points at which density is estimated
# - from, to: range over which to compute density

# Example usage:
# dens <- density(faithful$eruptions)
# plot(dens)
# 
# # With custom bandwidth
# dens <- density(faithful$eruptions, bw = 0.1)
# 
# # With different kernel
# dens <- density(faithful$eruptions, kernel = "epanechnikov")

# Other R packages for density estimation:
# - ASH and KernSmooth are both fast, accurate, and well-maintained 
#   (Deng and Wickham, 2011)
# - ASH: Average Shifted Histogram (Härdle & Scott, 1992)
#   * Very fast for large datasets
#   * Uses binned approximation
# - KernSmooth: Kernel smoothing functions
#   * Provides more control over bandwidth selection
#   * Includes functions for local polynomial regression
# - ks: Kernel smoothing package with many advanced features
# - np: Nonparametric kernel methods (univariate and multivariate)

# ============================================================================
# Bivariate Density Estimation
# ============================================================================

# To construct a bivariate density histogram, it is necessary to define 
# two-dimensional bins and count the number of observations in each bin
#
# Extension to two dimensions:
# - Instead of intervals on the line, use rectangular bins in the plane
# - Count observations falling in each rectangular region
# - Divide by total count and bin area to get density
#
# Challenges:
# - Need to choose bin widths in both dimensions
# - Number of bins grows as product of bins in each dimension
# - Visualization more complex (3D surface, contour plot, heatmap)
# - Curse of dimensionality: need more data for good estimates

# Can use bin2d function in R to bin a bivariate data set
bin2d <- function(x, breaks1 = "Sturges", breaks2 = "Sturges"){
    # Compute histograms for each dimension separately
    histg1 <- hist(x[,1], breaks = breaks1, plot = FALSE)
    histg2 <- hist(x[,2], breaks = breaks2, plot = FALSE)
    
    # Extract break points for both dimensions
    brx <- histg1$breaks
    bry <- histg2$breaks
    
    # Create 2D frequency table by cutting data in both dimensions
    freq <- table(cut(x[,1], brx), cut(x[,2], bry))

    # Return results as a list
    return(list(call = match.call(), freq = freq,
                breaks1 = brx, breaks2 = bry,
                mids1 = histg1$mids, mids2 = histg2$mids))
}

# The function:
# - Takes a two-column matrix or data frame x
# - Uses histogram break rules for each dimension (default: Sturges' rule)
# - Returns frequency counts for each 2D bin
# - Also returns bin boundaries and midpoints for plotting
#
# Usage:
# data_2d <- cbind(x1, x2)  # Two-column matrix
# bins <- bin2d(data_2d)
# image(bins$mids1, bins$mids2, bins$freq)  # Heatmap
# contour(bins$mids1, bins$mids2, bins$freq)  # Contour plot

# ============================================================================
# Example: Bivariate Density Estimation - Iris Data
# ============================================================================

# Following example computes the bivariate frequency table
# Load the iris dataset
data(iris)

# Bin the first two variables (Sepal.Length and Sepal.Width) 
# for the first species (setosa, rows 1:50)
fit1 <- bin2d(iris[1:50, 1:2])

# After binning the data, the persp function plots the density histogram
# as a 3D perspective plot
pdf("plots/04_iris_bivariate_3d.pdf")
persp(x = fit1$mids1, y = fit1$mids2, z = fit1$freq, 
      shade = TRUE, theta = 45, phi = 30, ltheta = 60,
      xlab = "Sepal Length", ylab = "Sepal Width", zlab = "Frequency",
      main = "Bivariate Density Histogram - Iris Setosa")
dev.off()

# Arguments:
# - x, y: bin midpoints in each dimension
# - z: frequency matrix (heights of histogram bars)
# - shade: add shading to enhance 3D appearance
# - theta: azimuthal viewing angle (rotation around z-axis)
# - phi: colatitude viewing angle (elevation)
# - ltheta: azimuthal direction of light source

# Alternative visualizations:
# Heatmap
pdf("plots/05_iris_heatmap.pdf")
image(fit1$mids1, fit1$mids2, fit1$freq, 
      xlab = "Sepal Length", ylab = "Sepal Width",
      main = "Bivariate Density - Heatmap")
dev.off()

# Contour plot
pdf("plots/06_iris_contour.pdf")
contour(fit1$mids1, fit1$mids2, fit1$freq,
        xlab = "Sepal Length", ylab = "Sepal Width",
        main = "Bivariate Density - Contour Plot")
dev.off()

# ============================================================================
# Bivariate Kernel Methods
# ============================================================================

# Suppose the data is X₁, ..., Xₙ, where each Xᵢ ∈ ℝ²
#
# Kernel density estimates can be extended to a multivariate (bivariate) setting
#
# Let K(·) be a bivariate kernel (typically a bivariate density function), 
# then the bivariate kernel density estimate is:
#
#   f̂(X) = (1/(n·h^d)) * Σᵢ₌₁ⁿ K((X - Xᵢ)/h)
#
# where:
# - X is the point at which we estimate the density (a 2D vector)
# - Xᵢ are the observed data points (2D vectors)
# - h is the bandwidth (smoothing parameter)
# - d is the dimension (d = 2 for bivariate case)
# - K is the bivariate kernel function
# - n is the number of observations
#
# Common bivariate kernels:
#
# 1. Product kernel (most common):
#    K(u, v) = K₁(u) * K₂(v)
#    where K₁ and K₂ are univariate kernels
#    - Simple to implement
#    - Can use different bandwidths in each dimension
#
# 2. Bivariate Gaussian kernel:
#    K(X) = (1/(2π)) * exp(-||X||²/2)
#    where ||X||² = X₁² + X₂²
#    - Smooth and symmetric
#    - Most commonly used in practice
#
# 3. Radial kernels:
#    K(X) = c·k(||X||²)
#    where k is a univariate kernel and c is a normalizing constant
#    - Spherically symmetric
#    - Treat all directions equally
#
# Bandwidth selection in 2D:
# - Can use same bandwidth in both dimensions: h₁ = h₂ = h
# - Or different bandwidths: h₁ ≠ h₂ (for different scales)
# - Rule-of-thumb for Gaussian kernel: h = σ̂ * n^(-1/6)
#   Note: n^(-1/6) instead of n^(-1/5) due to curse of dimensionality
# - Cross-validation also applies in multivariate case
#
# Curse of dimensionality:
# - Need much more data in higher dimensions
# - Convergence rate slows: n^(-1/(d+4)) for d dimensions
# - Bandwidth decreases more slowly with sample size

# ============================================================================
# Example: Bivariate Normal Mixture
# ============================================================================

# Estimate the bivariate density when the data is generated from a mixture 
# model with three components with identical covariance Σ = I₂ (identity matrix)
# and different means:
#
# μ₁ = (0, 0)
# μ₂ = (1, 3)
# μ₃ = (4, -1)
#
# Mixture probabilities are p = (0.2, 0.3, 0.5)

# Load required library
library(MASS)  # For mvrnorm and kde2d functions

# Set random seed for reproducibility
set.seed(123)

# Generate mixture data
n <- 2000
p <- c(.2, .3, .5)  # Mixture probabilities
mu <- matrix(c(0, 1, 4, 0, 3, -1), 3, 2)  # Means: rows are components
Sigma <- diag(2)  # Identity covariance matrix

# Sample component labels according to mixture probabilities
i <- sample(1:3, replace = TRUE, prob = p, size = n)
k <- table(i)  # Count observations from each component

# Generate samples from each component
x1 <- mvrnorm(k[1], mu = mu[1,], Sigma)
x2 <- mvrnorm(k[2], mu = mu[2,], Sigma)
x3 <- mvrnorm(k[3], mu = mu[3,], Sigma)

# Combine into mixture data
X <- rbind(x1, x2, x3)
x <- X[,1]
y <- X[,2]

# Estimate density using kernel density estimation
# Bandwidths chosen to balance bias and variance
fhat <- kde2d(x, y, h = c(1.87, 1.84))

# Visualize the estimated density
pdf("plots/07_bivariate_mixture_density.pdf", width = 12, height = 6)
par(mfrow = c(1, 2))
contour(fhat)
persp(fhat, phi = 30, theta = 20, d = 5, xlab = "x")
dev.off()
par(mfrow = c(1, 1))

cat("\n=== All plots saved to plots/ directory ===\n")
cat("Generated files:\n")
cat("  01_histogram_faithful.pdf\n")
cat("  02_naive_density_estimator.pdf\n")
cat("  03_gaussian_kernel_density.pdf\n")
cat("  04_iris_bivariate_3d.pdf\n")
cat("  05_iris_heatmap.pdf\n")
cat("  06_iris_contour.pdf\n")
cat("  07_bivariate_mixture_density.pdf\n")
