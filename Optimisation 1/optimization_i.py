"""
============================================
OPTIMIZATION I
============================================

AGENDA:
- Functions are objects: can be arguments for or returned by other functions
- Example: plotting functions
- Optimization via gradient descent, Newton's method, Nelder-Mead, …
- Curve-fitting by optimizing
============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.misc import derivative
import pandas as pd
from functools import wraps
import os

# Create plots directory if it doesn't exist
if not os.path.exists("../plots"):
    os.makedirs("../plots")

# ============================================
# FUNCTIONS AS OBJECTS
# ============================================
print("\n" + "=" * 50)
print("FUNCTIONS AS OBJECTS")
print("=" * 50 + "\n")

print(f"type(np.sin): {type(np.sin)}")
print(f"type(np.random.choice): {type(np.random.choice)}")

def resample(x):
    """Resample with replacement"""
    return np.random.choice(x, size=len(x), replace=True)

print(f"type(resample): {type(resample)}")

# Functions can be passed as arguments
def apply_function(func, values):
    """Apply a function to each value"""
    return [func(v) for v in values]

# Logistic transformation
log_ratios = np.arange(-2, 3)
logistic = lambda x: np.exp(x) / (1 + np.exp(x))
result = apply_function(logistic, log_ratios)
print(f"\nLogistic transformation results:")
print(np.array(result))

# ============================================
# NUMERICAL DERIVATIVES
# ============================================
print("\n" + "=" * 50)
print("NUMERICAL DERIVATIVES")
print("=" * 50 + "\n")

# scipy.misc.derivative computes numerical derivatives
np.random.seed(42)
just_a_phase = np.random.uniform(-np.pi, np.pi)
derivative_check = np.allclose(derivative(np.cos, just_a_phase, dx=1e-8), -np.sin(just_a_phase))
print(f"derivative(cos) ≈ -sin: {derivative_check}")

# Multivariable functions - use numerical gradient
def multivariable_func(x):
    """f(x) = x[0]^2 + x[1]^3"""
    return x[0]**2 + x[1]**3

def numerical_gradient(f, x, h=1e-8):
    """Compute numerical gradient of f at x"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        x_minus = x.copy()
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

grad_result = numerical_gradient(multivariable_func, np.array([1.0, -1.0]))
print(f"Gradient of x[0]^2 + x[1]^3 at (1,-1): {grad_result}")

# ============================================
# GRADIENT DESCENT IMPLEMENTATION
# ============================================
print("\n" + "=" * 50)
print("GRADIENT DESCENT")
print("=" * 50 + "\n")

def gradient_descent(f, x, max_iterations=100, step_scale=0.01, stopping_deriv=0.01):
    """
    Minimize function f using gradient descent
    
    Parameters:
    - f: function to minimize
    - x: initial point
    - max_iterations: maximum number of iterations
    - step_scale: step size for gradient descent
    - stopping_deriv: threshold for stopping criterion
    
    Returns:
    - dict with argmin, final_gradient, final_value, iterations
    """
    x = np.array(x, dtype=float)
    
    for iteration in range(1, max_iterations + 1):
        gradient = numerical_gradient(f, x)
        if np.all(np.abs(gradient) < stopping_deriv):
            break
        x = x - step_scale * gradient
    
    return {
        'argmin': x,
        'final_gradient': gradient,
        'final_value': f(x),
        'iterations': iteration,
        'converged': iteration < max_iterations
    }

print("Gradient descent function defined")

# ============================================
# FUNCTIONS RETURNING FUNCTIONS
# ============================================
print("\n" + "=" * 50)
print("FUNCTIONS RETURNING FUNCTIONS")
print("=" * 50 + "\n")

def make_linear_predictor(x, y):
    """
    Create a linear predictor function from training data
    
    Returns a function that predicts y from x
    """
    # Fit linear model
    coeffs = np.polyfit(x, y, 1)
    
    def predictor(x_new):
        return np.polyval(coeffs, x_new)
    
    return predictor

# Example with synthetic cat data
np.random.seed(42)
cat_body_weight = np.random.uniform(2.0, 4.0, 50)
cat_heart_weight = 4.0 * cat_body_weight + np.random.normal(0, 0.5, 50)

vet_predictor = make_linear_predictor(cat_body_weight, cat_heart_weight)
prediction = vet_predictor(3.5)
print(f"Predicted heart weight for 3.5kg cat: {prediction:.2f} grams")

# ============================================
# PLOTTING FUNCTIONS
# ============================================
print("\n" + "=" * 50)
print("PLOTTING FUNCTIONS")
print("=" * 50 + "\n")

# Plot a function over a range
x = np.linspace(-10, 10, 1000)
y = x**2 * np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('x² * sin(x)')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_curve1_py.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: optimization_curve1_py.png")

# Robust loss function
def psi(x, c=1):
    """Robust loss function"""
    x = np.asarray(x)
    return np.where(np.abs(x) > c, 2*c*np.abs(x) - c**2, x**2)

x = np.linspace(-20, 20, 1000)
plt.figure(figsize=(10, 6))
plt.plot(x, psi(x, c=10), 'g-', linewidth=2)
plt.xlabel('x')
plt.ylabel('ψ(x)')
plt.title('Robust Loss Function (c=10)')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_psi1_py.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: optimization_psi1_py.png")

c_values = np.linspace(-20, 20, 1000)
plt.figure(figsize=(10, 6))
plt.plot(c_values, psi(10, c=c_values), 'purple', linewidth=2)
plt.xlabel('c')
plt.ylabel('ψ(10, c)')
plt.title('Robust Loss Function (x=10, varying c)')
plt.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_psi2_py.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: optimization_psi2_py.png")

# ============================================
# GMP DATA AND MSE FUNCTION
# ============================================
print("\n" + "=" * 50)
print("GMP DATA AND OPTIMIZATION")
print("=" * 50 + "\n")

# Create synthetic GMP data
np.random.seed(42)
n_cities = 366
pop = 10**np.random.uniform(4.5, 7.5, n_cities)
true_a = 0.125
true_y0 = 6611
pcgmp = true_y0 * pop**true_a * np.exp(np.random.normal(0, 0.1, n_cities))
gmp_total = pcgmp * pop

gmp = pd.DataFrame({
    'gmp': gmp_total,
    'pcgmp': pcgmp,
    'pop': pop
})

print(f"Created GMP dataset with {len(gmp)} cities")

# Mean squared error function
def mse(y0, a, Y=None, N=None):
    """Calculate mean squared error for power law model"""
    if Y is None:
        Y = gmp['pcgmp'].values
    if N is None:
        N = gmp['pop'].values
    return np.mean((Y - y0 * N**a)**2)

# Test MSE function
test_values = [mse(6611, a) for a in np.arange(0.10, 0.16, 0.01)]
print(f"MSE values for a from 0.10 to 0.15:")
print(test_values)

print(f"MSE at y0=6611, a=0.10: {mse(6611, 0.10)}")

# ============================================
# VECTORIZING FUNCTIONS
# ============================================
print("\n" + "=" * 50)
print("VECTORIZING FUNCTIONS")
print("=" * 50 + "\n")

# Method 1: Wrapper that vectorizes
def mse_plottable(a, y0):
    """Vectorized MSE function"""
    if np.isscalar(a):
        return mse(y0, a)
    return np.array([mse(y0, a_val) for a_val in a])

result = mse_plottable(np.arange(0.10, 0.16, 0.01), y0=6611)
print("MSE via plottable wrapper:")
print(result)

a_range = np.linspace(0.10, 0.20, 100)
plt.figure(figsize=(10, 6))
plt.plot(a_range, mse_plottable(a_range, y0=6611), 'r-', linewidth=2, label='y0=6611')
plt.plot(a_range, mse_plottable(a_range, y0=5100), 'b-', linewidth=2, label='y0=5100')
plt.xlabel('a')
plt.ylabel('MSE')
plt.title('MSE vs Scaling Exponent')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_mse1_py.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: optimization_mse1_py.png")

# Method 2: Using numpy vectorize
mse_vec = np.vectorize(mse, excluded=['Y', 'N'])
result_vec = mse_vec(6611, np.arange(0.10, 0.16, 0.01))
print("MSE via np.vectorize():")
print(result_vec)

result_multi = mse_vec([5000, 6000, 7000], 1/8)
print("MSE at a=1/8 for different y0 values:")
print(result_multi)

plt.figure(figsize=(10, 6))
plt.plot(a_range, mse_vec(6611, a_range), 'r-', linewidth=2, label='y0=6611')
plt.plot(a_range, mse_vec(5100, a_range), 'b-', linewidth=2, label='y0=5100')
plt.xlabel('a')
plt.ylabel('MSE')
plt.title('MSE vs Scaling Exponent (Vectorized)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_mse2_py.png', dpi=100, bbox_inches='tight')
plt.close()
print("Plot saved: optimization_mse2_py.png")

# ============================================
# USING SCIPY OPTIMIZE
# ============================================
print("\n" + "=" * 50)
print("SCIPY OPTIMIZATION")
print("=" * 50 + "\n")

# Optimize using scipy
def objective(a):
    """Objective function for optimization"""
    return mse(6611, a)

result_opt = optimize.minimize_scalar(objective, bounds=(0.10, 0.20), method='bounded')
print(f"Optimal a using scipy: {result_opt.x:.6f}")
print(f"Minimum MSE: {result_opt.fun:.2f}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50 + "\n")
print("1. Functions are first-class objects in Python")
print("2. Functions can take other functions as arguments")
print("3. Functions can return other functions (closures)")
print("4. matplotlib provides flexible function plotting")
print("5. np.vectorize() makes functions work with arrays")
print("6. scipy.optimize provides robust optimization methods")
print("7. Gradient descent and other methods find function minima")

print("\nOptimization I Tutorial Complete")
plot_count = len([f for f in os.listdir("../plots") if f.startswith("optimization_") and f.endswith("_py.png")])
print(f"Generated {plot_count} plots")
