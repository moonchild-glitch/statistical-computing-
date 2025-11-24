# R script for "Optimization I"

# ============================================
# OPTIMIZATION I
# ============================================
# 
# AGENDA:
# - Functions are objects: can be arguments for or returned by other functions
# - Example: curve()
# - Optimization via gradient descent, Newton's method, Nelder-Mead, …
# - Curve-fitting by optimizing
# ============================================

# Load required packages
if (!require("numDeriv")) install.packages("numDeriv")
if (!require("MASS")) install.packages("MASS")

library(numDeriv)
library(MASS)

# Create plots directory if it doesn't exist
if (!dir.exists("../plots")) {
  dir.create("../plots")
}

# ============================================
# FUNCTIONS AS OBJECTS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("FUNCTIONS AS OBJECTS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

class(sin)
print(paste("class(sin):", class(sin)))

class(sample)
print(paste("class(sample):", class(sample)))

resample <- function(x) { sample(x, size=length(x), replace=TRUE) }
print(paste("class(resample):", class(resample)))

print(paste("typeof(resample):", typeof(resample)))
print(paste("typeof(sample):", typeof(sample)))
print(paste("typeof(sin):", typeof(sin)))

# Functions can be passed as arguments
result <- sapply((-2):2,function(log.ratio){exp(log.ratio)/(1+exp(log.ratio))})
print("Logistic transformation results:")
print(result)

# ============================================
# NUMERICAL DERIVATIVES WITH grad()
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("NUMERICAL DERIVATIVES\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# grad() from numDeriv computes numerical derivatives
# Test with cosine function
just_a_phase <- runif(n=1,min=-pi,max=pi)
derivative_check <- all.equal(grad(func=cos,x=just_a_phase),-sin(just_a_phase))
print(paste("grad(cos) == -sin:", derivative_check))

# Works with vectors too
phases <- runif(n=10,min=-pi,max=pi)
vec_derivative_check <- all.equal(grad(func=cos,x=phases),-sin(phases))
print(paste("grad(cos) on vectors:", vec_derivative_check))

# Multivariable functions
grad_result <- grad(func=function(x){x[1]^2+x[2]^3}, x=c(1,-1))
print("Gradient of x[1]^2 + x[2]^3 at (1,-1):")
print(grad_result)

# ============================================
# GRADIENT DESCENT IMPLEMENTATION
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("GRADIENT DESCENT\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

gradient.descent <- function(f,x,max.iterations,step.scale,
  stopping.deriv,...) {
  for (iteration in 1:max.iterations) {
    gradient <- grad(f,x,...)
    if(all(abs(gradient) < stopping.deriv)) { break() }
    x <- x - step.scale*gradient
  }
  fit <- list(argmin=x,final.gradient=gradient,final.value=f(x,...),
    iterations=iteration)
  return(fit)
}

print("Gradient descent function defined")

# ============================================
# FUNCTIONS RETURNING FUNCTIONS
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("FUNCTIONS RETURNING FUNCTIONS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

make.linear.predictor <- function(x,y) {
  linear.fit <- lm(y~x)
  predictor <- function(x) {
   return(predict(object=linear.fit,newdata=data.frame(x=x)))
  }
  return(predictor)
}

# Example with cats data
data(cats)
vet_predictor <- make.linear.predictor(x=cats$Bwt,y=cats$Hwt)
rm(cats)            # Data set goes away
prediction <- vet_predictor(3.5)  # My cat's body mass in kilograms
print(paste("Predicted heart weight for 3.5kg cat:", round(prediction, 2), "grams"))

# ============================================
# CURVE() FUNCTION FOR PLOTTING
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("CURVE() FUNCTION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# curve() plots a function over a range
png("../plots/optimization_curve1.png", width=800, height=600)
curve(x^2 * sin(x), from=-10, to=10, main="x^2 * sin(x)", 
      xlab="x", ylab="f(x)", lwd=2, col="blue")
dev.off()
print("Plot saved: optimization_curve1.png")

# Robust loss function
psi <- function(x,c=1) {ifelse(abs(x)>c,2*c*abs(x)-c^2,x^2)}

png("../plots/optimization_psi1.png", width=800, height=600)
curve(psi(x,c=10),from=-20,to=20, main="Robust Loss Function (c=10)",
      xlab="x", ylab="psi(x)", lwd=2, col="darkgreen")
dev.off()
print("Plot saved: optimization_psi1.png")

png("../plots/optimization_psi2.png", width=800, height=600)
curve(psi(x=10,c=x),from=-20,to=20, main="Robust Loss Function (x=10, varying c)",
      xlab="c", ylab="psi(10,c)", lwd=2, col="purple")
dev.off()
print("Plot saved: optimization_psi2.png")

# ============================================
# GMP DATA AND MSE FUNCTION
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("GMP DATA AND OPTIMIZATION\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Create synthetic GMP data (same as in writing functions tutorial)
set.seed(42)
n_cities <- 366
pop <- 10^runif(n_cities, 4.5, 7.5)
true_a <- 0.125
true_y0 <- 6611
pcgmp <- true_y0 * pop^true_a * exp(rnorm(n_cities, 0, 0.1))
gmp_total <- pcgmp * pop
gmp <- data.frame(gmp = gmp_total, pcgmp = pcgmp, pop = pop)

print(paste("Created GMP dataset with", nrow(gmp), "cities"))

# Mean squared error function
mse <- function(y0,a,Y=gmp$pcgmp,N=gmp$pop) {
   mean((Y - y0*(N^a))^2)
}

# Test MSE function
test_values <- sapply(seq(from=0.10,to=0.15,by=0.01),mse,y0=6611)
print("MSE values for a from 0.10 to 0.15:")
print(test_values)

print(paste("MSE at y0=6611, a=0.10:", mse(6611,0.10)))

# ============================================
# MAKING FUNCTIONS PLOTTABLE
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("VECTORIZING FUNCTIONS\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")

# Method 1: Wrapper with sapply
mse.plottable <- function(a,...){ return(sapply(a,mse,...)) }
result <- mse.plottable(seq(from=0.10,to=0.15,by=0.01),y0=6611)
print("MSE via plottable wrapper:")
print(result)

png("../plots/optimization_mse1.png", width=800, height=600)
curve(mse.plottable(a=x,y0=6611),from=0.10,to=0.20,xlab="a",ylab="MSE",
      main="MSE vs Scaling Exponent", lwd=2, col="red")
curve(mse.plottable(a=x,y0=5100),add=TRUE,col="blue", lwd=2)
legend("topright", legend=c("y0=6611", "y0=5100"), 
       col=c("red","blue"), lwd=2)
dev.off()
print("Plot saved: optimization_mse1.png")

# Method 2: Vectorize() function
mse.vec <- Vectorize(mse, vectorize.args=c("y0","a"))
result_vec <- mse.vec(a=seq(from=0.10,to=0.15,by=0.01),y0=6611)
print("MSE via Vectorize():")
print(result_vec)

result_multi <- mse.vec(a=1/8,y0=c(5000,6000,7000))
print("MSE at a=1/8 for different y0 values:")
print(result_multi)

png("../plots/optimization_mse2.png", width=800, height=600)
curve(mse.vec(a=x,y0=6611),from=0.10,to=0.20,xlab="a",ylab="MSE",
      main="MSE vs Scaling Exponent (Vectorized)", lwd=2, col="red")
curve(mse.vec(a=x,y0=5100),add=TRUE,col="blue", lwd=2)
legend("topright", legend=c("y0=6611", "y0=5100"), 
       col=c("red","blue"), lwd=2)
dev.off()
print("Plot saved: optimization_mse2.png")

# ============================================
# SUMMARY
# ============================================
cat("\n")
cat(paste(rep("=", 50), collapse=""), "\n")
cat("SUMMARY\n")
cat(paste(rep("=", 50), collapse=""), "\n\n")
print("1. Functions are first-class objects in R")
print("2. Functions can take other functions as arguments")
print("3. Functions can return other functions")
print("4. curve() is a convenient way to plot functions")
print("5. Vectorize() and sapply() make functions work with vectors")
print("6. Optimization involves finding function minima/maxima")
print("7. Gradient descent is one optimization method")

cat("\nOptimization I Tutorial Complete\n")
print(paste("Generated", length(list.files("../plots", pattern="optimization_.*\\.png")), "plots"))

