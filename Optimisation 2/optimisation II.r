############################################
# OPTIMIZATION II
# Advanced Topics in Numerical Optimization
############################################

# AGENDA:
# - Gradient computation techniques
# - Matrix manipulation and apply functions
# - Numerical differentiation
# - Edge cases and numerical stability
# - Best practices for optimization

# Create plots directory
if(!dir.exists("../plots")) {
  dir.create("../plots", recursive=TRUE)
}

############################################
# GRADIENT COMPUTATION
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("GRADIENT COMPUTATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("Computing gradients is fundamental to optimization\n")
cat("We'll explore different approaches to numerical differentiation\n\n")

# Basic gradient function (component-wise)
gradient.basic <- function(f, x, deriv.steps, ...) {
  p <- length(x)
  stopifnot(length(deriv.steps) == p)
  gradient <- numeric(p)
  
  for(i in 1:p) {
    x.new <- x
    x.new[i] <- x[i] + deriv.steps[i]
    gradient[i] <- (f(x.new, ...) - f(x, ...)) / deriv.steps[i]
  }
  
  return(gradient)
}

cat("Basic gradient function (loop-based):\n")
cat("- Iterates through each component\n")
cat("- Computes finite difference for each dimension\n")
cat("- Can be slow for high-dimensional problems\n\n")

############################################
# BONUS EXAMPLE: IMPROVED GRADIENT
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("BONUS EXAMPLE: gradient() with Matrix Manipulation\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("Better: use matrix manipulation and apply\n\n")

gradient <- function(f, x, deriv.steps, ...) {
  p <- length(x)
  stopifnot(length(deriv.steps) == p)
  x.new <- matrix(rep(x, times=p), nrow=p) + diag(deriv.steps, nrow=p)
  f.new <- apply(x.new, 2, f, ...)
  gradient <- (f.new - f(x, ...)) / deriv.steps
  return(gradient)
}

cat("Improved gradient function:\n")
cat("- Clearer and half as long\n")
cat("- Uses matrix manipulation\n")
cat("- Vectorized computation with apply\n\n")

cat("Key features:\n")
cat("- Presumes that f takes a vector and returns a single number\n")
cat("- Any extra arguments to gradient will get passed to f\n\n")

cat("Check: Does this work when f is a function of a single number?\n\n")

############################################
# TEST THE GRADIENT FUNCTIONS
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("TESTING GRADIENT FUNCTIONS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Test function 1: Simple quadratic
f1 <- function(x) {
  sum(x^2)
}

# True gradient: 2*x
true.grad.f1 <- function(x) {
  2 * x
}

# Test at a point
x.test <- c(1, 2, 3)
deriv.steps <- rep(1e-5, 3)

grad.basic <- gradient.basic(f1, x.test, deriv.steps)
grad.matrix <- gradient(f1, x.test, deriv.steps)
grad.true <- true.grad.f1(x.test)

cat("Test Function 1: f(x) = sum(x^2)\n")
cat("Test point: (1, 2, 3)\n\n")
cat("Basic gradient:  ", sprintf("%.6f", grad.basic), "\n")
cat("Matrix gradient: ", sprintf("%.6f", grad.matrix), "\n")
cat("True gradient:   ", sprintf("%.6f", grad.true), "\n\n")

# Test function 2: Rosenbrock function
rosenbrock <- function(x) {
  100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
}

# True gradient
true.grad.rosenbrock <- function(x) {
  c(-400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
    200 * (x[2] - x[1]^2))
}

x.test2 <- c(0.5, 0.5)
deriv.steps2 <- rep(1e-5, 2)

grad.basic2 <- gradient.basic(rosenbrock, x.test2, deriv.steps2)
grad.matrix2 <- gradient(rosenbrock, x.test2, deriv.steps2)
grad.true2 <- true.grad.rosenbrock(x.test2)

cat("Test Function 2: Rosenbrock function\n")
cat("Test point: (0.5, 0.5)\n\n")
cat("Basic gradient:  ", sprintf("%.6f", grad.basic2), "\n")
cat("Matrix gradient: ", sprintf("%.6f", grad.matrix2), "\n")
cat("True gradient:   ", sprintf("%.6f", grad.true2), "\n\n")

# Test with single-variable function
f.single <- function(x) {
  x^2
}

x.single <- 2
step.single <- 1e-5

grad.single <- gradient(f.single, x.single, step.single)
cat("Single variable test: f(x) = x^2 at x=2\n")
cat("Computed gradient: ", sprintf("%.6f", grad.single), "\n")
cat("True gradient:     4.000000\n\n")

############################################
# VISUALIZE GRADIENT COMPUTATION
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("VISUALIZING GRADIENT COMPUTATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# 2D function for visualization
f.viz <- function(x) {
  x[1]^2 + 2*x[2]^2
}

# Create grid
x1 <- seq(-3, 3, length.out=50)
x2 <- seq(-3, 3, length.out=50)
z <- outer(x1, x2, function(a, b) {
  mapply(function(i, j) f.viz(c(i, j)), a, b)
})

# Plot function and gradient at several points
png("../plots/optimization_ii_gradient_field.png", width=800, height=800, res=100)
par(mar=c(5, 5, 4, 2))
contour(x1, x2, z, nlevels=20, xlab="x1", ylab="x2",
        main="Gradient Field: f(x) = x1^2 + 2*x2^2", 
        col="lightblue", lwd=2)

# Add gradient arrows at grid points
grid.points <- expand.grid(x1=seq(-2.5, 2.5, by=0.5),
                           x2=seq(-2.5, 2.5, by=0.5))

for(i in 1:nrow(grid.points)) {
  pt <- c(grid.points$x1[i], grid.points$x2[i])
  grad <- gradient(f.viz, pt, rep(1e-5, 2))
  grad.norm <- grad / sqrt(sum(grad^2))  # Normalize for visualization
  
  # Scale arrows
  arrow.scale <- 0.2
  arrows(pt[1], pt[2], 
         pt[1] - arrow.scale * grad.norm[1], 
         pt[2] - arrow.scale * grad.norm[2],
         length=0.08, col="red", lwd=1.5)
}

points(0, 0, pch=19, col="darkgreen", cex=2)
text(0, 0.3, "Minimum", col="darkgreen", font=2)
dev.off()

cat("Plot saved: optimization_ii_gradient_field.png\n\n")

############################################
# TIMING COMPARISON
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("TIMING COMPARISON\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Test with higher dimensional function
dim <- 100
x.large <- rnorm(dim)
deriv.steps.large <- rep(1e-5, dim)

f.large <- function(x) sum(x^2)

# Time basic version
time.basic <- system.time({
  for(i in 1:100) {
    grad.basic.large <- gradient.basic(f.large, x.large, deriv.steps.large)
  }
})

# Time matrix version
time.matrix <- system.time({
  for(i in 1:100) {
    grad.matrix.large <- gradient(f.large, x.large, deriv.steps.large)
  }
})

cat("Timing for 100-dimensional function (100 iterations):\n\n")
cat("Basic version (loop):   ", sprintf("%.4f", time.basic["elapsed"]), "seconds\n")
cat("Matrix version (apply): ", sprintf("%.4f", time.matrix["elapsed"]), "seconds\n")
cat("Speedup:                ", sprintf("%.2fx", time.basic["elapsed"] / time.matrix["elapsed"]), "\n\n")

############################################
# POTENTIAL ISSUES WITH GRADIENT
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("POTENTIAL ISSUES WITH GRADIENT FUNCTION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("The gradient function acts badly if:\n\n")

cat("1. f is only defined on a limited domain and we ask for the\n")
cat("   gradient somewhere near a boundary\n\n")

# Example: log function
f.log <- function(x) {
  if(any(x <= 0)) return(NA)
  sum(log(x))
}

x.boundary <- c(0.001, 1)
deriv.steps.boundary <- c(1e-3, 1e-3)

tryCatch({
  grad.boundary <- gradient(f.log, x.boundary, deriv.steps.boundary)
  cat("   Gradient near boundary: ", sprintf("%.6f", grad.boundary), "\n")
  cat("   Warning: May be inaccurate or produce NA!\n\n")
}, error = function(e) {
  cat("   Error near boundary:", e$message, "\n\n")
})

cat("2. Forces the user to choose deriv.steps\n")
cat("   - No automatic step size selection\n")
cat("   - User must understand numerical differentiation\n\n")

cat("3. Uses the same deriv.steps everywhere\n")
cat("   Example: f(x) = x^2 * sin(x)\n")
cat("   - May need different steps for different regions\n")
cat("   - Constant step size may be suboptimal\n\n")

# Example with different scales
f.mixed <- function(x) {
  x[1]^2 * sin(x[2])
}

x.mixed <- c(100, 0.1)
deriv.steps.mixed <- rep(1e-5, 2)

grad.mixed <- gradient(f.mixed, x.mixed, deriv.steps.mixed)
cat("   Example: f(x) = x1^2 * sin(x2) at (100, 0.1)\n")
cat("   Gradient with uniform step: ", sprintf("%.6f", grad.mixed), "\n")
cat("   (May have numerical issues due to scale differences)\n\n")

cat("4. ...and so on through much of a first course in numerical analysis\n\n")

############################################
# IMPROVED GRADIENT WITH ADAPTIVE STEPS
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("IMPROVED GRADIENT WITH ADAPTIVE STEPS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

gradient.adaptive <- function(f, x, eps=sqrt(.Machine$double.eps), ...) {
  p <- length(x)
  
  # Adaptive step size based on magnitude of x
  deriv.steps <- pmax(abs(x) * eps, eps)
  
  # Use matrix manipulation
  x.new <- matrix(rep(x, times=p), nrow=p) + diag(deriv.steps, nrow=p)
  f.new <- apply(x.new, 2, f, ...)
  gradient <- (f.new - f(x, ...)) / deriv.steps
  
  return(gradient)
}

cat("Adaptive gradient function:\n")
cat("- Automatically chooses step sizes\n")
cat("- Step size proportional to |x|\n")
cat("- Minimum step size to avoid underflow\n\n")

# Test on mixed scale function
grad.adaptive <- gradient.adaptive(f.mixed, x.mixed)
cat("Adaptive gradient on f(x) = x1^2 * sin(x2) at (100, 0.1):\n")
cat("  ", sprintf("%.6f", grad.adaptive), "\n\n")

############################################
# CENTRAL DIFFERENCE METHOD
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("CENTRAL DIFFERENCE METHOD\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("Forward difference:  f'(x) ≈ (f(x+h) - f(x)) / h\n")
cat("Central difference:  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)\n")
cat("\nCentral difference is more accurate (O(h^2) vs O(h))\n\n")

gradient.central <- function(f, x, deriv.steps, ...) {
  p <- length(x)
  stopifnot(length(deriv.steps) == p)
  
  # Forward perturbations
  x.forward <- matrix(rep(x, times=p), nrow=p) + diag(deriv.steps, nrow=p)
  # Backward perturbations
  x.backward <- matrix(rep(x, times=p), nrow=p) - diag(deriv.steps, nrow=p)
  
  f.forward <- apply(x.forward, 2, f, ...)
  f.backward <- apply(x.backward, 2, f, ...)
  
  gradient <- (f.forward - f.backward) / (2 * deriv.steps)
  return(gradient)
}

# Compare methods
x.compare <- c(1, 2)
steps.compare <- rep(1e-4, 2)

grad.forward <- gradient(rosenbrock, x.compare, steps.compare)
grad.central <- gradient.central(rosenbrock, x.compare, steps.compare)
grad.true <- true.grad.rosenbrock(x.compare)

cat("Comparison on Rosenbrock function at (1, 2):\n\n")
cat("Forward difference:  ", sprintf("%.8f", grad.forward), "\n")
cat("Central difference:  ", sprintf("%.8f", grad.central), "\n")
cat("True gradient:       ", sprintf("%.8f", grad.true), "\n\n")

# Error analysis
error.forward <- abs(grad.forward - grad.true)
error.central <- abs(grad.central - grad.true)

cat("Absolute errors:\n")
cat("Forward difference:  ", sprintf("%.2e", error.forward), "\n")
cat("Central difference:  ", sprintf("%.2e", error.central), "\n\n")

############################################
# STEP SIZE ANALYSIS
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("STEP SIZE ANALYSIS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Test different step sizes
f.simple <- function(x) x^2
x.test.step <- 1
true.deriv <- 2 * x.test.step

step.sizes <- 10^seq(-12, -1, by=0.5)
errors.forward <- numeric(length(step.sizes))
errors.central <- numeric(length(step.sizes))

for(i in 1:length(step.sizes)) {
  h <- step.sizes[i]
  
  # Forward difference
  grad.f <- (f.simple(x.test.step + h) - f.simple(x.test.step)) / h
  errors.forward[i] <- abs(grad.f - true.deriv)
  
  # Central difference
  grad.c <- (f.simple(x.test.step + h) - f.simple(x.test.step - h)) / (2*h)
  errors.central[i] <- abs(grad.c - true.deriv)
}

png("../plots/optimization_ii_step_size_analysis.png", width=800, height=600, res=100)
par(mar=c(5, 5, 4, 2))
plot(step.sizes, errors.forward, log="xy", type="b", col="blue", lwd=2,
     xlab="Step Size (h)", ylab="Absolute Error",
     main="Numerical Differentiation Error vs Step Size",
     pch=19, ylim=range(c(errors.forward, errors.central)))
lines(step.sizes, errors.central, type="b", col="red", lwd=2, pch=19)
legend("topright", legend=c("Forward Difference", "Central Difference"),
       col=c("blue", "red"), lwd=2, pch=19, bg="white")
grid(col="gray80")
dev.off()

cat("Plot saved: optimization_ii_step_size_analysis.png\n")
cat("Note: Central difference is more accurate for moderate step sizes\n\n")

############################################
# PRACTICAL RECOMMENDATIONS
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("PRACTICAL RECOMMENDATIONS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("1. Use adaptive step sizes based on scale of variables\n")
cat("   eps = sqrt(.Machine$double.eps) ≈ 1.5e-8 is a good default\n\n")

cat("2. Use central differences when possible (more accurate)\n\n")

cat("3. Check for boundary issues and constrained domains\n\n")

cat("4. For critical applications, compare with:\n")
cat("   - Automatic differentiation (e.g., numDeriv package)\n")
cat("   - Symbolic derivatives when available\n\n")

cat("5. Consider using existing packages:\n")
cat("   - numDeriv::grad() for numerical gradients\n")
cat("   - optimx package for optimization with gradients\n\n")

############################################
# USING numDeriv PACKAGE
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("USING THE numDeriv PACKAGE\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

if(require(numDeriv, quietly=TRUE)) {
  # Compare our methods with numDeriv
  grad.numDeriv <- grad(rosenbrock, x.compare)
  
  cat("Gradient comparison on Rosenbrock at (1, 2):\n\n")
  cat("Our forward:     ", sprintf("%.8f", grad.forward), "\n")
  cat("Our central:     ", sprintf("%.8f", grad.central), "\n")
  cat("numDeriv::grad():", sprintf("%.8f", grad.numDeriv), "\n")
  cat("True gradient:   ", sprintf("%.8f", grad.true), "\n\n")
  
  cat("numDeriv uses Richardson extrapolation for high accuracy\n\n")
} else {
  cat("numDeriv package not installed\n")
  cat("Install with: install.packages('numDeriv')\n\n")
}

############################################
# APPLICATION TO OPTIMIZATION
############################################

cat("\n", paste(rep("=", 50), collapse=""), "\n")
cat("APPLICATION TO OPTIMIZATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

cat("Gradients are essential for gradient-based optimization methods:\n")
cat("- Gradient descent\n")
cat("- Conjugate gradient\n")
cat("- Newton's method\n")
cat("- Quasi-Newton methods (BFGS, L-BFGS)\n\n")

# Simple gradient descent example
gradient.descent <- function(f, x0, gradient.fn, 
                            alpha=0.01, max.iter=1000, tol=1e-6) {
  x <- x0
  path <- matrix(nrow=max.iter+1, ncol=length(x0))
  path[1, ] <- x
  
  for(i in 1:max.iter) {
    grad <- gradient.fn(f, x, rep(1e-5, length(x)))
    x.new <- x - alpha * grad
    path[i+1, ] <- x.new
    
    if(sqrt(sum((x.new - x)^2)) < tol) {
      path <- path[1:(i+1), , drop=FALSE]
      break
    }
    x <- x.new
  }
  
  return(list(x=x, path=path, iterations=i))
}

# Optimize Rosenbrock function
x.start <- c(-1, 1)
result <- gradient.descent(rosenbrock, x.start, gradient.central, 
                           alpha=0.001, max.iter=5000)

cat("Gradient Descent on Rosenbrock Function:\n")
cat("Starting point: (", paste(sprintf("%.2f", x.start), collapse=", "), ")\n")
cat("Final point:    (", paste(sprintf("%.6f", result$x), collapse=", "), ")\n")
cat("True minimum:   (1.000000, 1.000000)\n")
cat("Iterations:     ", result$iterations, "\n\n")

# Plot optimization path
png("../plots/optimization_ii_gradient_descent.png", width=800, height=800, res=100)
par(mar=c(5, 5, 4, 2))

# Create contour plot
x1.opt <- seq(-1.5, 1.5, length.out=100)
x2.opt <- seq(-0.5, 1.5, length.out=100)
z.opt <- outer(x1.opt, x2.opt, function(a, b) {
  mapply(function(i, j) rosenbrock(c(i, j)), a, b)
})

contour(x1.opt, x2.opt, z.opt, nlevels=30, xlab="x1", ylab="x2",
        main="Gradient Descent on Rosenbrock Function",
        col="lightblue", lwd=1.5)

# Plot path
lines(result$path[, 1], result$path[, 2], col="red", lwd=2)
points(result$path[1, 1], result$path[1, 2], pch=19, col="darkgreen", cex=2)
points(result$path[nrow(result$path), 1], result$path[nrow(result$path), 2], 
       pch=19, col="darkred", cex=2)
points(1, 1, pch=4, col="blue", cex=2, lwd=3)

legend("topright", legend=c("Start", "End", "True Minimum", "Path"),
       col=c("darkgreen", "darkred", "blue", "red"),
       pch=c(19, 19, 4, NA), lty=c(NA, NA, NA, 1), lwd=c(NA, NA, 3, 2),
       bg="white")
dev.off()

cat("Plot saved: optimization_ii_gradient_descent.png\n\n")

############################################
# SUMMARY
############################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("SUMMARY: OPTIMIZATION II\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("Key Takeaways:\n\n")

cat("✓ Matrix manipulation makes gradient computation clearer and faster\n")
cat("  - Use apply() instead of loops when possible\n")
cat("  - Vectorization improves performance\n\n")

cat("✓ Numerical differentiation requires careful consideration:\n")
cat("  - Step size selection is critical\n")
cat("  - Central differences more accurate than forward\n")
cat("  - Adaptive steps handle different scales\n\n")

cat("✓ Common pitfalls:\n")
cat("  - Boundary issues with constrained domains\n")
cat("  - Fixed step sizes for all variables\n")
cat("  - Not accounting for function scale\n\n")

cat("✓ Best practices:\n")
cat("  - Use existing packages (numDeriv) for production code\n")
cat("  - Validate numerical gradients when possible\n")
cat("  - Consider automatic differentiation\n\n")

cat("✓ Gradients enable powerful optimization methods:\n")
cat("  - Gradient descent and variants\n")
cat("  - Newton and quasi-Newton methods\n")
cat("  - Constrained optimization algorithms\n\n")

cat(paste(rep("=", 60), collapse=""), "\n")
cat("OPTIMIZATION II TUTORIAL COMPLETE\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

plot.count <- length(list.files("../plots", pattern="^optimization_ii.*\\.png$"))
cat("Generated", plot.count, "plots\n\n")

cat("For more advanced topics, see:\n")
cat("- optim() and nlm() documentation\n")
cat("- optimx package for unified optimization interface\n")
cat("- numDeriv package for accurate numerical derivatives\n")
cat("- BB package for large-scale optimization\n\n")

cat("Thank you for completing this tutorial!\n")
