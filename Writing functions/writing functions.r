# ============================================
# WRITING FUNCTIONS IN R
# ============================================
# 
# AGENDA:
# - Defining functions: Tying related commands into bundles
# - Interfaces: Controlling what the function can see and do
# - Example: Parameter estimation code
# - Multiple functions
# - Recursion: Making hard problems simpler
# ============================================

# ============================================
# WHY FUNCTIONS?
# ============================================
# Data structures tie related values into one object
# Functions tie related commands into one object
# 
# In both cases: easier to understand, easier to work with, 
# easier to build into larger things
#
# Benefits of functions:
# 1. Organize code into reusable chunks
# 2. Avoid repetition (DRY principle: Don't Repeat Yourself)
# 3. Make code more readable and maintainable
# 4. Abstract complex operations into simple interfaces
# 5. Test and debug code more easily
# 6. Build complex programs from simple, well-tested components

# ============================================
# DEFINING FUNCTIONS
# ============================================
# Basic syntax:
# function_name <- function(argument1, argument2, ...) {
#   # Function body
#   # Code to execute
#   return(result)  # Optional: last expression is returned by default
# }

# Example: Cubic function
cube <- function(x) x ^ 3

# Display the function
cat("Function definition:\n")
print(cube)

# Test with a single value
cat("\ncube(3) =", cube(3), "\n")

# Test with a vector
cat("\ncube(1:10) =", cube(1:10), "\n")

# Test with a matrix
cat("\ncube(matrix(1:8, 2, 4)):\n")
print(cube(matrix(1:8, 2, 4)))

cat("\nmatrix(cube(1:8), 2, 4):\n")
print(matrix(cube(1:8), 2, 4))

# Works with arrays too
# cube(array(1:24, c(2, 3, 4))) # cube each element in an array

# Check the mode/type of the function
cat("\nmode(cube) =", mode(cube), "\n")

# ============================================
# ROBUST LOSS FUNCTION EXAMPLE
# ============================================
# "Robust" loss function, for outlier-resistant regression
# Inputs: vector of numbers (x)
# Outputs: vector with x^2 for small entries, 2|x|-1 for large ones

psi.1 <- function(x) {
  psi <- ifelse(x^2 > 1, 2*abs(x)-1, x^2)
  return(psi)
}

# Our functions get used just like the built-in ones:
cat("\n=== ROBUST LOSS FUNCTION ===\n")
z <- c(-0.5, -5, 0.9, 9)
cat("z =", z, "\n")
cat("psi.1(z) =", psi.1(z), "\n")

# ============================================
# FUNCTION ANATOMY
# ============================================
# Go back to the declaration and look at the parts:
#
# psi.1 <- function(x) {
#   psi <- ifelse(x^2 > 1, 2*abs(x)-1, x^2)
#   return(psi)
# }
#
# INTERFACES: the inputs (arguments) and outputs (return value)
# - Calls other functions: ifelse(), abs(), operators ^ and >, 
#   and could also call other functions we've written
# - return() says what the output is
# - Alternatively, return the last evaluation (implicit return)
# - Comments: Not required by R, but a good idea

# ============================================
# WHAT SHOULD BE A FUNCTION?
# ============================================
# Things you're going to re-run, especially if it will be re-run with changes
# Chunks of code you keep highlighting and hitting return on
# Chunks of code which are small parts of bigger analyses
# Chunks which are very similar to other chunks

# ============================================
# NAMED AND DEFAULT ARGUMENTS
# ============================================
# "Robust" loss function with adjustable crossover scale
# Inputs: vector of numbers (x), scale for crossover (c)
# Outputs: vector with x^2 for small entries, 2c|x|-c^2 for large ones

psi.2 <- function(x, c=1) {
  psi <- ifelse(x^2 > c^2, 2*c*abs(x)-c^2, x^2)
  return(psi)
}

cat("\n=== NAMED AND DEFAULT ARGUMENTS ===\n")

# Default value c=1 makes psi.2 equivalent to psi.1
cat("identical(psi.1(z), psi.2(z,c=1)):", identical(psi.1(z), psi.2(z,c=1)), "\n")

# Default values get used if names are missing:
cat("identical(psi.2(z,c=1), psi.2(z)):", identical(psi.2(z,c=1), psi.2(z)), "\n")

# Named arguments can go in any order when explicitly tagged:
cat("identical(psi.2(x=z,c=2), psi.2(c=2,x=z)):", identical(psi.2(x=z,c=2), psi.2(c=2,x=z)), "\n")

# ============================================
# CHECKING ARGUMENTS
# ============================================
cat("\n=== CHECKING ARGUMENTS ===\n")

# Problem: Odd behavior when arguments aren't as we expect
cat("\nProblem - vector c causes element-wise comparison:\n")
cat("psi.2(x=z, c=c(1,1,1,10)) =", psi.2(x=z, c=c(1,1,1,10)), "\n")

cat("\nProblem - negative c gives nonsensical results:\n")
cat("psi.2(x=z, c=-1) =", psi.2(x=z, c=-1), "\n")

# Solution: Put little sanity checks into the code
# "Robust" loss function with argument validation
# Inputs: vector of numbers (x), scale for crossover (c)
# Outputs: vector with x^2 for small entries, 2c|x|-c^2 for large ones

psi.3 <- function(x, c=1) {
  # Scale should be a single positive number
  stopifnot(length(c) == 1, c > 0)
  psi <- ifelse(x^2 > c^2, 2*c*abs(x)-c^2, x^2)
  return(psi)
}

cat("\nSolution - stopifnot() validates arguments:\n")
cat("psi.3(z, c=2) =", psi.3(z, c=2), "\n")

# Arguments to stopifnot() are a series of expressions which should all be TRUE
# Execution halts, with error message, at first FALSE (try it!)
cat("\nTry uncommenting these to see the errors:\n")
cat("# psi.3(z, c=c(1,1,1,10))  # Error: length(c) == 1 is not TRUE\n")
cat("# psi.3(z, c=-1)           # Error: c > 0 is not TRUE\n")

# Example 1: Simple function with no arguments
hello <- function() {
  print("Hello, World!")
}

# Test the function
hello()

# Example 2: Function with one argument
square <- function(x) {
  return(x ^ 2)
}

cat("\nSquare of 5:", square(5), "\n")
cat("Square of 10:", square(10), "\n")

# Example 3: Function with multiple arguments
add <- function(a, b) {
  result <- a + b
  return(result)
}

cat("\n3 + 7 =", add(3, 7), "\n")

# Example 4: Function with default arguments
greet <- function(name = "User", greeting = "Hello") {
  message <- paste(greeting, name, "!")
  return(message)
}

cat("\n", greet(), "\n")
cat(greet("Alice"), "\n")
cat(greet("Bob", "Good morning"), "\n")

# Example 5: Implicit return (last expression is returned)
multiply <- function(x, y) {
  x * y  # No explicit return statement needed
}

cat("\n4 * 6 =", multiply(4, 6), "\n")

# ============================================
# FUNCTION INTERFACES
# ============================================
# Functions have their own environment and scope
# They can access variables from:
# 1. Their arguments
# 2. Variables defined inside the function
# 3. Variables in parent environments (but be careful!)

cat("\n=== WHAT THE FUNCTION CAN SEE AND DO ===\n")

# Key principles:
# - Each function has its own environment
# - Names here over-ride names in the global environment
# - Internal environment starts with the named arguments
# - Assignments inside the function only change the internal environment
#   (There are ways around this, but they are difficult and best avoided)
# - Names undefined in the function are looked for in the environment 
#   the function gets called from, not the environment of definition

# Example: Variable scope
x <- 10  # Global variable

test_scope <- function() {
  x <- 5  # Local variable (doesn't affect global x)
  cat("Inside function: x =", x, "\n")
  return(x)
}

cat("\nBefore function call: x =", x, "\n")
result <- test_scope()
cat("After function call: x =", x, "\n")
cat("Function returned:", result, "\n")

# ============================================
# INTERNAL ENVIRONMENT EXAMPLES
# ============================================
cat("\n=== INTERNAL ENVIRONMENT EXAMPLES ===\n")

# Example 1: Function creates its own local x
x <- 7
y <- c("A", "C", "G", "T", "U")

adder <- function(y) { 
  x <- x + y  # Local x, doesn't affect global x
  return(x) 
}

cat("\nBefore adder: x =", x, "\n")
cat("adder(1) =", adder(1), "\n")
cat("After adder: x =", x, "\n")
cat("y =", y, "\n")

# Example 2: Function uses global constant
circle.area <- function(r) { 
  return(pi * r^2) 
}

cat("\nWith built-in pi:\n")
cat("circle.area(c(1,2,3)) =", circle.area(c(1, 2, 3)), "\n")

# Override pi in global environment
truepi <- pi
pi <- 3

cat("\nWith pi = 3:\n")
cat("circle.area(c(1,2,3)) =", circle.area(c(1, 2, 3)), "\n")

# Restore sanity
pi <- truepi

cat("\nWith pi restored:\n")
cat("circle.area(c(1,2,3)) =", circle.area(c(1, 2, 3)), "\n")

# ============================================
# RESPECT THE INTERFACES
# ============================================
cat("\n=== RESPECT THE INTERFACES ===\n")
cat("Interfaces mark out a controlled inner environment for our code\n")
cat("Interact with the rest of the system only at the interface\n\n")
cat("Advice: arguments explicitly give the function all the information\n")
cat("  - Reduces risk of confusion and error\n")
cat("  - Exception: true universals like π\n\n")
cat("Likewise, output should only be through the return value\n")

# ============================================
# EXAMPLE: FITTING A MODEL
# ============================================
cat("\n=== FITTING A MODEL ===\n")
cat("Fact: bigger cities tend to produce more economically per capita\n\n")
cat("A proposed statistical model (Geoffrey West et al.):\n")
cat("Y = y0 * N^a + noise\n")
cat("where Y is the per-capita gross metropolitan product of a city,\n")
cat("N is its population, and y0 and a are parameters\n")

# Create sample GMP data (since the URL is not accessible)
# This simulates the structure of the original dataset
set.seed(42)
n_cities <- 366
pop <- 10^runif(n_cities, 4.5, 7.5)  # Population from ~30K to ~30M
true_a <- 0.125  # True scaling exponent
true_y0 <- 6611  # True baseline
pcgmp <- true_y0 * pop^true_a * exp(rnorm(n_cities, 0, 0.1))  # Add noise
gmp_total <- pcgmp * pop

gmp <- data.frame(
  gmp = gmp_total,
  pcgmp = pcgmp
)
gmp$pop <- gmp$gmp / gmp$pcgmp

cat("\nData created: gmp dataset with", nrow(gmp), "cities\n")
cat("(Simulated data based on the power-law model)\n")

# Plot the data with initial guess
plot(pcgmp ~ pop, data = gmp, log = "x", 
     xlab = "Population", 
     ylab = "Per-Capita Economic Output ($/person-year)",
     main = "US Metropolitan Areas, 2006")
curve(6611 * x^(1/8), add = TRUE, col = "blue")

cat("Plot created with initial curve: y = 6611 * pop^(1/8)\n")

# ============================================
# ITERATIVE OPTIMIZATION APPROACH
# ============================================
cat("\n=== OPTIMIZATION APPROACH ===\n")
cat("Take y0 = 6611 for now and estimate a by minimizing mean squared error\n")
cat("Approximate the derivative of error w.r.t a and move against it\n\n")
cat("MSE(a) = (1/n) * sum((Y_i - y0*N_i^a)^2)\n")
cat("MSE'(a) ≈ (MSE(a+h) - MSE(a)) / h\n")
cat("a_{t+1} = a_t - step_scale * MSE'(a)\n")

# ============================================
# AN ACTUAL FIRST ATTEMPT AT CODE
# ============================================
cat("\n=== FIRST ATTEMPT (Not as a function) ===\n")

maximum.iterations <- 100
deriv.step <- 1/1000
step.scale <- 1e-12
stopping.deriv <- 1/100
iteration <- 0
deriv <- Inf
a <- 0.15

while ((iteration < maximum.iterations) && (deriv > stopping.deriv)) {
  iteration <- iteration + 1
  mse.1 <- mean((gmp$pcgmp - 6611*gmp$pop^a)^2)
  mse.2 <- mean((gmp$pcgmp - 6611*gmp$pop^(a+deriv.step))^2)
  deriv <- (mse.2 - mse.1)/deriv.step
  a <- a - step.scale*deriv
}

cat("Result: a =", a, ", iterations =", iteration, "\n")

# ============================================
# WHAT'S WRONG WITH THIS?
# ============================================
cat("\n=== WHAT'S WRONG WITH THIS? ===\n")
cat("1. Not encapsulated: Re-run by cutting and pasting code\n")
cat("   - How much of it?\n")
cat("   - Hard to make part of something larger\n\n")
cat("2. Inflexible: To change initial guess at a, have to edit, cut, paste, re-run\n\n")
cat("3. Error-prone: To change data set, have to edit, cut, paste, re-run,\n")
cat("   and hope all edits are consistent\n\n")
cat("4. Hard to fix: Should stop when absolute value of derivative is small,\n")
cat("   but this stops when large and negative\n")
cat("   Imagine having five copies and needing to fix same bug on each\n\n")
cat("We will turn this into a function and then improve it\n")

# ============================================
# VERSION 1: BASIC FUNCTION WITH LOGIC FIX
# ============================================
cat("\n=== VERSION 1: Basic Function ===\n")

estimate.scaling.exponent.1 <- function(a) {
  maximum.iterations <- 100
  deriv.step <- 1/1000
  step.scale <- 1e-12
  stopping.deriv <- 1/100
  iteration <- 0
  deriv <- Inf
  while ((iteration < maximum.iterations) && (abs(deriv) > stopping.deriv)) {
    iteration <- iteration + 1
    mse.1 <- mean((gmp$pcgmp - 6611*gmp$pop^a)^2)
    mse.2 <- mean((gmp$pcgmp - 6611*gmp$pop^(a+deriv.step))^2)
    deriv <- (mse.2 - mse.1)/deriv.step
    a <- a - step.scale*deriv
  }
  fit <- list(a=a, iterations=iteration,
    converged=(iteration < maximum.iterations))
  return(fit)
}

result1 <- estimate.scaling.exponent.1(0.15)
cat("estimate.scaling.exponent.1(0.15):\n")
cat("  a =", result1$a, "\n")
cat("  iterations =", result1$iterations, "\n")
cat("  converged =", result1$converged, "\n")

# ============================================
# VERSION 2: MAKE MAGIC NUMBERS INTO DEFAULTS
# ============================================
cat("\n=== VERSION 2: Magic Numbers → Defaults ===\n")
cat("Problem: All those magic numbers!\n")
cat("Solution: Make them defaults\n\n")

estimate.scaling.exponent.2 <- function(a, y0=6611,
  maximum.iterations=100, deriv.step = .001,
  step.scale = 1e-12, stopping.deriv = .01) {
  iteration <- 0
  deriv <- Inf
  while ((iteration < maximum.iterations) && (abs(deriv) > stopping.deriv)) {
    iteration <- iteration + 1
    mse.1 <- mean((gmp$pcgmp - y0*gmp$pop^a)^2)
    mse.2 <- mean((gmp$pcgmp - y0*gmp$pop^(a+deriv.step))^2)
    deriv <- (mse.2 - mse.1)/deriv.step
    a <- a - step.scale*deriv
  }
  fit <- list(a=a, iterations=iteration,
    converged=(iteration < maximum.iterations))
  return(fit)
}

result2 <- estimate.scaling.exponent.2(0.15)
cat("estimate.scaling.exponent.2(0.15):\n")
cat("  a =", result2$a, "\n")
cat("  iterations =", result2$iterations, "\n")

# ============================================
# VERSION 3: ELIMINATE REPEATED CODE
# ============================================
cat("\n=== VERSION 3: Declare Helper Function ===\n")
cat("Problem: Why type out the same calculation of MSE twice?\n")
cat("Solution: Declare a function\n\n")

estimate.scaling.exponent.3 <- function(a, y0=6611,
  maximum.iterations=100, deriv.step = .001,
  step.scale = 1e-12, stopping.deriv = .01) {
  iteration <- 0
  deriv <- Inf
  mse <- function(a) { mean((gmp$pcgmp - y0*gmp$pop^a)^2) }
  while ((iteration < maximum.iterations) && (abs(deriv) > stopping.deriv)) {
    iteration <- iteration + 1
    deriv <- (mse(a+deriv.step) - mse(a))/deriv.step
    a <- a - step.scale*deriv
  }
  fit <- list(a=a, iterations=iteration,
    converged=(iteration < maximum.iterations))
  return(fit)
}

cat("mse() declared inside the function:\n")
cat("  - Can see y0\n")
cat("  - Not added to global environment\n\n")

result3 <- estimate.scaling.exponent.3(0.15)
cat("estimate.scaling.exponent.3(0.15):\n")
cat("  a =", result3$a, "\n")

# ============================================
# VERSION 4: FLEXIBLE DATA INPUTS
# ============================================
cat("\n=== VERSION 4: Flexible Data Inputs ===\n")
cat("Problem: Locked in to using specific columns of gmp\n")
cat("         Shouldn't have to re-write to compare two data sets\n")
cat("Solution: More arguments, with defaults\n\n")

estimate.scaling.exponent.4 <- function(a, y0=6611,
  response=gmp$pcgmp, predictor = gmp$pop,
  maximum.iterations=100, deriv.step = .001,
  step.scale = 1e-12, stopping.deriv = .01) {
  iteration <- 0
  deriv <- Inf
  mse <- function(a) { mean((response - y0*predictor^a)^2) }
  while ((iteration < maximum.iterations) && (abs(deriv) > stopping.deriv)) {
    iteration <- iteration + 1
    deriv <- (mse(a+deriv.step) - mse(a))/deriv.step
    a <- a - step.scale*deriv
  }
  fit <- list(a=a, iterations=iteration,
    converged=(iteration < maximum.iterations))
  return(fit)
}

result4 <- estimate.scaling.exponent.4(0.15)
cat("estimate.scaling.exponent.4(0.15):\n")
cat("  a =", result4$a, "\n")

# ============================================
# VERSION 5: RESPECTING THE INTERFACES
# ============================================
cat("\n=== VERSION 5: while() → for() Loop ===\n")
cat("Respecting the interfaces:\n")
cat("We could turn the while() loop into a for() loop,\n")
cat("and nothing outside the function would care\n\n")

estimate.scaling.exponent.5 <- function(a, y0=6611,
  response=gmp$pcgmp, predictor = gmp$pop,
  maximum.iterations=100, deriv.step = .001,
  step.scale = 1e-12, stopping.deriv = .01) {
  mse <- function(a) { mean((response - y0*predictor^a)^2) }
  for (iteration in 1:maximum.iterations) {
    deriv <- (mse(a+deriv.step) - mse(a))/deriv.step
    a <- a - step.scale*deriv
    if (abs(deriv) <= stopping.deriv) { break() }
  }
  fit <- list(a=a, iterations=iteration,
    converged=(iteration < maximum.iterations))
  return(fit)
}

result5 <- estimate.scaling.exponent.5(0.15)
cat("estimate.scaling.exponent.5(0.15):\n")
cat("  a =", result5$a, "\n")
cat("  iterations =", result5$iterations, "\n")

# ============================================
# WHAT HAVE WE DONE?
# ============================================
cat("\n=== WHAT HAVE WE DONE? ===\n")
cat("The final code is:\n")
cat("  - Shorter\n")
cat("  - Clearer\n")
cat("  - More flexible\n")
cat("  - More re-usable\n\n")

cat("EXERCISES:\n")
cat("1. Run the code with default values to get an estimate of a;\n")
cat("   plot the curve along with the data points\n\n")
cat("2. Randomly remove one data point — how much does the estimate change?\n\n")
cat("3. Run the code from multiple starting points —\n")
cat("   how different are the estimates of a?\n\n")

# Exercise 1: Plot with estimated curve
final.fit <- estimate.scaling.exponent.5(0.15)
cat("Final estimate: a =", final.fit$a, "\n\n")

plot(pcgmp ~ pop, data = gmp, log = "x", 
     xlab = "Population", 
     ylab = "Per-Capita Economic Output ($/person-year)",
     main = "US Metropolitan Areas, 2006")
curve(6611 * x^(1/8), add = TRUE, col = "blue", lty = 2, lwd = 2)
curve(6611 * x^final.fit$a, add = TRUE, col = "red", lwd = 2)
legend("bottomright", 
       legend = c("Initial guess (a=1/8)", paste("Estimated (a=", round(final.fit$a, 4), ")")),
       col = c("blue", "red"), 
       lty = c(2, 1), 
       lwd = 2)

cat("Plot created comparing initial guess vs estimated curve\n")

# ============================================
# HOW WE EXTEND FUNCTIONS
# ============================================
cat("\n=== HOW WE EXTEND FUNCTIONS ===\n")
cat("Two main approaches:\n")
cat("1. Multiple functions: Doing different things to the same object\n")
cat("2. Sub-functions: Breaking up big jobs into small ones\n")

# ============================================
# WHY MULTIPLE FUNCTIONS?
# ============================================
cat("\n=== WHY MULTIPLE FUNCTIONS? ===\n")

cat("\nMeta-problems:\n")
cat("  - You've got more than one problem\n")
cat("  - Your problem is too hard to solve in one step\n")
cat("  - You keep solving the same problems\n\n")

cat("Meta-solutions:\n")
cat("  - Write multiple functions, which rely on each other\n")
cat("  - Split your problem, and write functions for the pieces\n")
cat("  - Solve the recurring problems once, and re-use the solutions\n")

# ============================================
# WRITING MULTIPLE RELATED FUNCTIONS
# ============================================
cat("\n=== WRITING MULTIPLE RELATED FUNCTIONS ===\n")
cat("Statisticians want to do lots of things with their models:\n")
cat("  estimate, predict, visualize, test, compare, simulate, uncertainty, ...\n\n")
cat("Write multiple functions to do these things:\n")
cat("  - Make the model one object\n")
cat("  - Assume it has certain components\n")

# ============================================
# CONSISTENT INTERFACES
# ============================================
cat("\n=== CONSISTENT INTERFACES ===\n")
cat("Functions for the same kind of object should:\n")
cat("  - Use the same arguments\n")
cat("  - Presume the same structure\n\n")
cat("Functions for the same kind of task should:\n")
cat("  - Use the same arguments\n")
cat("  - Return the same sort of value (to the extent possible)\n")

# ============================================
# KEEP RELATED THINGS TOGETHER
# ============================================
cat("\n=== KEEP RELATED THINGS TOGETHER ===\n")
cat("  - Put all the related functions in a single file\n")
cat("  - Source them together\n")
cat("  - Use comments to note dependencies\n")

# ============================================
# POWER-LAW SCALING MODEL
# ============================================
cat("\n=== POWER-LAW SCALING MODEL ===\n")
cat("Remember the model: Y = y0 * N^a + noise\n")
cat("(output per person) = (baseline) * (population)^(scaling exponent) + noise\n\n")
cat("Estimated parameters a, y0 by minimizing the mean squared error\n")

# Modified estimation function that returns both a and y0
estimate.scaling.exponent <- function(a, y0=6611,
  response=gmp$pcgmp, predictor = gmp$pop,
  maximum.iterations=100, deriv.step = .001,
  step.scale = 1e-12, stopping.deriv = .01) {
  mse <- function(a) { mean((response - y0*predictor^a)^2) }
  for (iteration in 1:maximum.iterations) {
    deriv <- (mse(a+deriv.step) - mse(a))/deriv.step
    a <- a - step.scale*deriv
    if (abs(deriv) <= stopping.deriv) { break() }
  }
  fit <- list(a=a, y0=y0, iterations=iteration,
    converged=(iteration < maximum.iterations))
  return(fit)
}

cat("\nExercise: Modified estimation code returns list with components a and y0\n")
plm.fit <- estimate.scaling.exponent(0.15)
cat("Fitted model: a =", plm.fit$a, ", y0 =", plm.fit$y0, "\n")

# ============================================
# PREDICTING FROM A FITTED MODEL
# ============================================
cat("\n=== PREDICTING FROM A FITTED MODEL ===\n")

# Predict response values from a power-law scaling model
# Inputs: fitted power-law model (object), vector of values at which to make
  # predictions at (newdata)
# Outputs: vector of predicted response values
predict.plm <- function(object, newdata) {
  # Check that object has the right components
  stopifnot("a" %in% names(object), "y0" %in% names(object))
  a <- object$a
  y0 <- object$y0
  # Sanity check the inputs
  stopifnot(is.numeric(a), length(a)==1)
  stopifnot(is.numeric(y0), length(y0)==1)
  stopifnot(is.numeric(newdata))
  return(y0*newdata^a)  # Actual calculation and return
}

# Test prediction function
test.cities <- c(1e5, 5e5, 1e6, 5e6)
predictions <- predict.plm(plm.fit, test.cities)
cat("\nPredictions for cities of size", test.cities, ":\n")
cat("Per-capita output:", round(predictions, 2), "\n")

# ============================================
# PLOTTING A FITTED MODEL (VERSION 1)
# ============================================
cat("\n=== PLOTTING A FITTED MODEL (VERSION 1) ===\n")

# Plot fitted curve from power law model over specified range
# Inputs: list containing parameters (plm), start and end of range (from, to)
# Outputs: TRUE, silently, if successful
# Side-effect: Makes the plot
plot.plm.1 <- function(plm, from, to) {
  # Take sanity-checking of parameters as read
  y0 <- plm$y0 # Extract parameters
  a <- plm$a
  f <- function(x) { return(y0*x^a) }
  curve(f(x), from=from, to=to)
  # Return with no visible value on the terminal
  invisible(TRUE)
}

cat("plot.plm.1: Simple plotting function with from/to arguments\n")

# ============================================
# PLOTTING WITH ... (VERSION 2)
# ============================================
cat("\n=== PLOTTING WITH ... ARGUMENT (VERSION 2) ===\n")
cat("When one function calls another, use ... as a meta-argument\n")
cat("to pass along unspecified inputs to the called function\n\n")

plot.plm.2 <- function(plm, ...) {
  y0 <- plm$y0
  a <- plm$a
  f <- function(x) { return(y0*x^a) }
  # from and to are possible arguments to curve()
  curve(f(x), ...)
  invisible(TRUE)
}

cat("plot.plm.2: Uses ... to pass arguments to curve()\n")

# ============================================
# SUB-FUNCTIONS
# ============================================
cat("\n=== SUB-FUNCTIONS ===\n")
cat("Solve big problems by dividing them into a few sub-problems\n\n")
cat("Benefits:\n")
cat("  - Easier to understand, get the big picture at a glance\n")
cat("  - Easier to fix, improve and modify\n")
cat("  - Easier to design\n")
cat("  - Easier to re-use solutions to recurring sub-problems\n\n")
cat("Rule of thumb: A function longer than a page is probably too long\n")

# ============================================
# SUB-FUNCTIONS OR SEPARATE FUNCTIONS?
# ============================================
cat("\n=== SUB-FUNCTIONS OR SEPARATE FUNCTIONS? ===\n")
cat("\nDefining a function inside another function:\n")
cat("  Pros: Simpler code, access to local variables, doesn't clutter workspace\n")
cat("  Cons: Gets re-declared each time, can't access in global environment\n\n")
cat("Alternative: Declare the function in the same file, source them together\n\n")
cat("Rule of thumb: If you find yourself writing the same code in multiple\n")
cat("places, make it a separate function\n")

# ============================================
# PLOTTING WITH PREDICTION (VERSION 3)
# ============================================
cat("\n=== PLOTTING USING PREDICTION FUNCTION (VERSION 3) ===\n")
cat("Our old plotting function calculated the fitted values\n")
cat("But so does our prediction function!\n\n")

plot.plm.3 <- function(plm, from, to, n=101, ...) {
  x <- seq(from=from, to=to, length.out=n)
  y <- predict.plm(object=plm, newdata=x)
  plot(x, y, ...)
  invisible(TRUE)
}

cat("plot.plm.3: Re-uses predict.plm() instead of recalculating\n")
cat("This demonstrates the DRY principle: Don't Repeat Yourself\n")

# Create a demonstration plot
plot.plm.3(plm.fit, from=1e5, to=1e7, 
           xlab="Population", ylab="Per-Capita Output",
           main="Power-Law Model Fit", type="l", col="blue", lwd=2, log="x")
points(gmp$pop, gmp$pcgmp, pch=20, col="red")
legend("bottomright", legend=c("Fitted curve", "Data"), 
       col=c("blue", "red"), lwd=c(2, NA), pch=c(NA, 20))

cat("Created plot using plot.plm.3()\n")

# ============================================
# RECURSION REVISITED
# ============================================
cat("\n=== RECURSION (REVISITED) ===\n")
cat("Reduce the problem to an easier one of the same form\n\n")

# Factorial using recursion
my.factorial <- function(n) {
  if (n == 1) {
    return(1)
  } else {
    return(n*my.factorial(n-1))
  }
}

cat("Factorial (recursive):\n")
for (i in 1:6) {
  cat("my.factorial(", i, ") =", my.factorial(i), "\n")
}

# Fibonacci with multiple recursive calls
fib <- function(n) {
  if ((n==1) || (n==0)) {
   return(1)
  } else {
   return(fib(n-1) + fib(n-2))
  }
}

cat("\nFibonacci (recursive):\n")
for (i in 0:8) {
  cat("fib(", i, ") =", fib(i), "\n")
}

cat("\nExercise: Convince yourself that any loop can be replaced by recursion;\n")
cat("can you always replace recursion with a loop?\n")

# ============================================
# SUMMARY
# ============================================
cat("\n=== SUMMARY ===\n")
cat("1. Functions bundle related commands together into objects:\n")
cat("   - Easier to re-run, re-use, combine, and modify\n")
cat("   - Less risk of error, easier to think about\n\n")

cat("2. Interfaces control what the function can see and change:\n")
cat("   - Arguments and environment (what it can see)\n")
cat("   - Its internals and return value (what it can change)\n\n")

cat("3. Calling functions we define works like built-in functions:\n")
cat("   - Named arguments, defaults\n\n")

cat("4. Multiple functions let us do multiple related jobs:\n")
cat("   - Either on the same object or on similar ones\n\n")

cat("5. Sub-functions let us break big problems into smaller ones:\n")
cat("   - Re-use the solutions to the smaller ones\n\n")

cat("6. Recursion is a powerful way of making hard problems simpler\n")

cat("\n=== Writing Functions Tutorial Complete ===\n")

# Example: Accessing global variables (not recommended)
y <- 100

access_global <- function() {
  cat("Accessing global y:", y, "\n")
}

access_global()

# Example: Modifying global variables with <<- (use sparingly!)
counter <- 0

increment_counter <- function() {
  counter <<- counter + 1  # Modifies global counter
  cat("Counter is now:", counter, "\n")
}

cat("\nInitial counter:", counter, "\n")
increment_counter()
increment_counter()
increment_counter()

# ============================================
# FUNCTION ARGUMENTS
# ============================================

# Example: Named arguments
divide <- function(numerator, denominator) {
  if (denominator == 0) {
    stop("Cannot divide by zero!")
  }
  return(numerator / denominator)
}

cat("\nUsing positional arguments:\n")
cat("10 / 2 =", divide(10, 2), "\n")

cat("\nUsing named arguments:\n")
cat("10 / 2 =", divide(numerator = 10, denominator = 2), "\n")
cat("10 / 2 =", divide(denominator = 2, numerator = 10), "\n")

# Example: Variable number of arguments with ...
print_all <- function(...) {
  args <- list(...)
  cat("Number of arguments:", length(args), "\n")
  for (i in seq_along(args)) {
    cat("Argument", i, ":", args[[i]], "\n")
  }
}

cat("\n")
print_all(1, 2, 3)
cat("\n")
print_all("a", "b", "c", "d")

# Example: Passing ... to another function
my_sum <- function(...) {
  sum(...)  # Pass all arguments to sum()
}

cat("\nSum using ...:\n")
cat("my_sum(1, 2, 3, 4, 5) =", my_sum(1, 2, 3, 4, 5), "\n")

# ============================================
# PARAMETER ESTIMATION EXAMPLE
# ============================================
# Let's write functions for statistical parameter estimation

# Function to calculate sample mean
sample_mean <- function(x) {
  if (length(x) == 0) {
    stop("Cannot calculate mean of empty vector")
  }
  return(sum(x) / length(x))
}

# Function to calculate sample variance
sample_variance <- function(x, unbiased = TRUE) {
  if (length(x) < 2) {
    stop("Need at least 2 observations for variance")
  }
  
  n <- length(x)
  mean_x <- sample_mean(x)
  squared_deviations <- (x - mean_x) ^ 2
  
  if (unbiased) {
    return(sum(squared_deviations) / (n - 1))  # Unbiased estimator
  } else {
    return(sum(squared_deviations) / n)  # Biased estimator
  }
}

# Function to calculate sample standard deviation
sample_sd <- function(x, unbiased = TRUE) {
  return(sqrt(sample_variance(x, unbiased)))
}

# Function to calculate standard error of the mean
standard_error <- function(x) {
  n <- length(x)
  return(sample_sd(x) / sqrt(n))
}

# Function to calculate confidence interval for mean
confidence_interval <- function(x, confidence_level = 0.95) {
  n <- length(x)
  mean_x <- sample_mean(x)
  se <- standard_error(x)
  
  # Use t-distribution for small samples
  alpha <- 1 - confidence_level
  t_value <- qt(1 - alpha/2, df = n - 1)
  
  margin_error <- t_value * se
  lower <- mean_x - margin_error
  upper <- mean_x + margin_error
  
  return(list(
    mean = mean_x,
    lower = lower,
    upper = upper,
    confidence_level = confidence_level
  ))
}

# Test parameter estimation functions
cat("\n=== PARAMETER ESTIMATION EXAMPLE ===\n")
data <- c(23, 25, 27, 29, 31, 28, 26, 24, 30, 32)

cat("\nData:", data, "\n")
cat("Sample mean:", sample_mean(data), "\n")
cat("Sample variance:", sample_variance(data), "\n")
cat("Sample SD:", sample_sd(data), "\n")
cat("Standard error:", standard_error(data), "\n")

ci <- confidence_interval(data, 0.95)
cat("\n95% Confidence Interval:\n")
cat("  Mean:", ci$mean, "\n")
cat("  Lower bound:", ci$lower, "\n")
cat("  Upper bound:", ci$upper, "\n")

# Compare with built-in functions
cat("\nComparison with built-in functions:\n")
cat("mean():", mean(data), "\n")
cat("var():", var(data), "\n")
cat("sd():", sd(data), "\n")

# ============================================
# MULTIPLE FUNCTIONS WORKING TOGETHER
# ============================================
# Let's create a set of functions for linear regression

# Function to calculate slope and intercept
linear_regression <- function(x, y) {
  if (length(x) != length(y)) {
    stop("x and y must have the same length")
  }
  
  n <- length(x)
  mean_x <- mean(x)
  mean_y <- mean(y)
  
  # Calculate slope
  numerator <- sum((x - mean_x) * (y - mean_y))
  denominator <- sum((x - mean_x) ^ 2)
  slope <- numerator / denominator
  
  # Calculate intercept
  intercept <- mean_y - slope * mean_x
  
  return(list(slope = slope, intercept = intercept))
}

# Function to make predictions
predict_linear <- function(model, new_x) {
  predictions <- model$intercept + model$slope * new_x
  return(predictions)
}

# Function to calculate R-squared
r_squared <- function(x, y, model) {
  predictions <- predict_linear(model, x)
  residuals <- y - predictions
  ss_residual <- sum(residuals ^ 2)
  ss_total <- sum((y - mean(y)) ^ 2)
  r2 <- 1 - (ss_residual / ss_total)
  return(r2)
}

# Function to calculate residuals
calculate_residuals <- function(x, y, model) {
  predictions <- predict_linear(model, x)
  residuals <- y - predictions
  return(residuals)
}

# Function to print regression summary
print_regression_summary <- function(x, y, model) {
  cat("\n=== LINEAR REGRESSION SUMMARY ===\n")
  cat("Slope:", model$slope, "\n")
  cat("Intercept:", model$intercept, "\n")
  cat("R-squared:", r_squared(x, y, model), "\n")
  
  residuals <- calculate_residuals(x, y, model)
  cat("\nResidual statistics:\n")
  cat("  Min:", min(residuals), "\n")
  cat("  Max:", max(residuals), "\n")
  cat("  Mean:", mean(residuals), "\n")
  cat("  SD:", sd(residuals), "\n")
}

# Test linear regression functions
cat("\n=== MULTIPLE FUNCTIONS EXAMPLE ===\n")
x <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
y <- c(2.1, 4.3, 6.2, 7.9, 10.1, 12.3, 14.0, 16.2, 18.1, 20.0)

model <- linear_regression(x, y)
print_regression_summary(x, y, model)

# Make predictions
new_x <- c(11, 12, 13)
predictions <- predict_linear(model, new_x)
cat("\nPredictions for x =", new_x, ":\n")
cat("y =", predictions, "\n")

# Compare with built-in lm()
cat("\nComparison with lm():\n")
lm_model <- lm(y ~ x)
cat("Built-in slope:", coef(lm_model)[2], "\n")
cat("Built-in intercept:", coef(lm_model)[1], "\n")
cat("Built-in R-squared:", summary(lm_model)$r.squared, "\n")

# ============================================
# RECURSION
# ============================================
# Recursion: A function that calls itself
# Useful for problems that can be broken down into smaller subproblems

# Example 1: Factorial
# n! = n × (n-1) × (n-2) × ... × 2 × 1
# Recursive definition: n! = n × (n-1)!
factorial <- function(n) {
  # Base case
  if (n <= 1) {
    return(1)
  }
  # Recursive case
  return(n * factorial(n - 1))
}

cat("\n=== RECURSION EXAMPLES ===\n")
cat("\nFactorial:\n")
for (i in 0:6) {
  cat(i, "! =", factorial(i), "\n")
}

# Example 2: Fibonacci sequence
# F(0) = 0, F(1) = 1
# F(n) = F(n-1) + F(n-2)
fibonacci <- function(n) {
  # Base cases
  if (n == 0) return(0)
  if (n == 1) return(1)
  
  # Recursive case
  return(fibonacci(n - 1) + fibonacci(n - 2))
}

cat("\nFibonacci sequence:\n")
for (i in 0:10) {
  cat("F(", i, ") =", fibonacci(i), "\n")
}

# Example 3: Sum of list (recursive)
recursive_sum <- function(vec) {
  # Base case: empty vector
  if (length(vec) == 0) {
    return(0)
  }
  
  # Base case: single element
  if (length(vec) == 1) {
    return(vec[1])
  }
  
  # Recursive case: first element + sum of rest
  return(vec[1] + recursive_sum(vec[-1]))
}

cat("\nRecursive sum:\n")
test_vec <- c(1, 2, 3, 4, 5)
cat("Sum of", test_vec, "=", recursive_sum(test_vec), "\n")
cat("Compare with sum():", sum(test_vec), "\n")

# Example 4: Greatest Common Divisor (Euclidean algorithm)
gcd <- function(a, b) {
  # Base case
  if (b == 0) {
    return(abs(a))
  }
  
  # Recursive case
  return(gcd(b, a %% b))
}

cat("\nGreatest Common Divisor:\n")
cat("GCD(48, 18) =", gcd(48, 18), "\n")
cat("GCD(100, 35) =", gcd(100, 35), "\n")
cat("GCD(17, 19) =", gcd(17, 19), "\n")

# Example 5: Power function (recursive)
power <- function(base, exponent) {
  # Base cases
  if (exponent == 0) return(1)
  if (exponent == 1) return(base)
  
  # Recursive case for positive exponents
  if (exponent > 0) {
    return(base * power(base, exponent - 1))
  }
  
  # Handle negative exponents
  return(1 / power(base, -exponent))
}

cat("\nPower function:\n")
cat("2^5 =", power(2, 5), "\n")
cat("3^4 =", power(3, 4), "\n")
cat("5^0 =", power(5, 0), "\n")
cat("2^(-3) =", power(2, -3), "\n")

# Example 6: Binary search (recursive)
binary_search <- function(vec, target, low = 1, high = length(vec)) {
  # Base case: element not found
  if (low > high) {
    return(-1)
  }
  
  # Calculate middle index
  mid <- floor((low + high) / 2)
  
  # Base case: element found
  if (vec[mid] == target) {
    return(mid)
  }
  
  # Recursive cases
  if (vec[mid] > target) {
    # Search in left half
    return(binary_search(vec, target, low, mid - 1))
  } else {
    # Search in right half
    return(binary_search(vec, target, mid + 1, high))
  }
}

cat("\nBinary search:\n")
sorted_vec <- c(2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78)
cat("Searching in:", sorted_vec, "\n")
cat("Index of 23:", binary_search(sorted_vec, 23), "\n")
cat("Index of 67:", binary_search(sorted_vec, 67), "\n")
cat("Index of 2:", binary_search(sorted_vec, 2), "\n")
cat("Index of 100 (not found):", binary_search(sorted_vec, 100), "\n")

# Example 7: Merge sort (recursive)
merge <- function(left, right) {
  result <- c()
  i <- 1
  j <- 1
  
  # Merge two sorted vectors
  while (i <= length(left) && j <= length(right)) {
    if (left[i] <= right[j]) {
      result <- c(result, left[i])
      i <- i + 1
    } else {
      result <- c(result, right[j])
      j <- j + 1
    }
  }
  
  # Append remaining elements
  if (i <= length(left)) {
    result <- c(result, left[i:length(left)])
  }
  if (j <= length(right)) {
    result <- c(result, right[j:length(right)])
  }
  
  return(result)
}

merge_sort <- function(vec) {
  # Base case: vector with 0 or 1 element is already sorted
  if (length(vec) <= 1) {
    return(vec)
  }
  
  # Divide the vector into two halves
  mid <- floor(length(vec) / 2)
  left <- vec[1:mid]
  right <- vec[(mid + 1):length(vec)]
  
  # Recursively sort both halves
  left <- merge_sort(left)
  right <- merge_sort(right)
  
  # Merge the sorted halves
  return(merge(left, right))
}

cat("\nMerge sort:\n")
unsorted <- c(38, 27, 43, 3, 9, 82, 10)
cat("Unsorted:", unsorted, "\n")
sorted <- merge_sort(unsorted)
cat("Sorted:", sorted, "\n")

# ============================================
# ADVANCED RECURSION: TOWER OF HANOI
# ============================================
# Tower of Hanoi: Move n disks from source to destination using auxiliary peg
# Rules: 
# 1. Only one disk can be moved at a time
# 2. A larger disk cannot be placed on a smaller disk

tower_of_hanoi <- function(n, source = "A", destination = "C", auxiliary = "B") {
  if (n == 1) {
    cat("Move disk 1 from", source, "to", destination, "\n")
    return(1)
  }
  
  # Move n-1 disks from source to auxiliary
  moves <- tower_of_hanoi(n - 1, source, auxiliary, destination)
  
  # Move the largest disk from source to destination
  cat("Move disk", n, "from", source, "to", destination, "\n")
  moves <- moves + 1
  
  # Move n-1 disks from auxiliary to destination
  moves <- moves + tower_of_hanoi(n - 1, auxiliary, destination, source)
  
  return(moves)
}

cat("\n=== TOWER OF HANOI ===\n")
cat("\nSolving Tower of Hanoi with 3 disks:\n")
total_moves <- tower_of_hanoi(3)
cat("\nTotal moves:", total_moves, "\n")
cat("Formula: 2^n - 1 =", 2^3 - 1, "\n")

# ============================================
# MEMOIZATION: OPTIMIZING RECURSIVE FUNCTIONS
# ============================================
# Fibonacci with memoization (caching results)

# Create environment to store computed values
fib_cache <- new.env()

fibonacci_memo <- function(n) {
  # Check if result is already cached
  if (exists(as.character(n), envir = fib_cache)) {
    return(get(as.character(n), envir = fib_cache))
  }
  
  # Base cases
  if (n == 0) return(0)
  if (n == 1) return(1)
  
  # Compute and cache result
  result <- fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
  assign(as.character(n), result, envir = fib_cache)
  
  return(result)
}

cat("\n=== MEMOIZATION ===\n")
cat("\nFibonacci with memoization (much faster for large n):\n")

# Time comparison
cat("\nWithout memoization (n=30):\n")
start_time <- Sys.time()
result1 <- fibonacci(30)
end_time <- Sys.time()
cat("Result:", result1, "\n")
cat("Time:", end_time - start_time, "\n")

cat("\nWith memoization (n=30):\n")
start_time <- Sys.time()
result2 <- fibonacci_memo(30)
end_time <- Sys.time()
cat("Result:", result2, "\n")
cat("Time:", end_time - start_time, "\n")

# Can compute much larger values with memoization
cat("\nFibonacci(100) with memoization:", fibonacci_memo(100), "\n")

# ============================================
# SUMMARY
# ============================================
cat("\n=== SUMMARY ===\n")
cat("1. Functions organize code into reusable, testable units\n")
cat("2. Functions have their own scope and can access parent environments\n")
cat("3. Use named arguments for clarity, default values for flexibility\n")
cat("4. Multiple functions can work together to solve complex problems\n")
cat("5. Recursion is powerful for problems with recursive structure\n")
cat("6. Base cases are crucial for recursive functions to terminate\n")
cat("7. Memoization can dramatically improve recursive function performance\n")
cat("\n=== Functions Tutorial Complete ===\n")
