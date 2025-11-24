#!/usr/bin/env python3
"""
============================================
OPTIMIZATION II
Advanced Topics in Numerical Optimization
============================================

AGENDA:
- Gradient computation techniques
- Matrix manipulation and vectorization
- Numerical differentiation
- Edge cases and numerical stability
- Best practices for optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.misc import derivative
import time
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Create plots directory
os.makedirs("../plots", exist_ok=True)

############################################
# GRADIENT COMPUTATION
############################################

print("\n" + "="*50)
print("GRADIENT COMPUTATION")
print("="*50 + "\n")

print("Computing gradients is fundamental to optimization")
print("We'll explore different approaches to numerical differentiation\n")

# Basic gradient function (component-wise)
def gradient_basic(f, x, deriv_steps, *args):
    """Basic gradient computation using loops"""
    p = len(x)
    assert len(deriv_steps) == p, "deriv_steps must have same length as x"
    gradient = np.zeros(p)
    
    for i in range(p):
        x_new = x.copy()
        x_new[i] = x[i] + deriv_steps[i]
        gradient[i] = (f(x_new, *args) - f(x, *args)) / deriv_steps[i]
    
    return gradient

print("Basic gradient function (loop-based):")
print("- Iterates through each component")
print("- Computes finite difference for each dimension")
print("- Can be slow for high-dimensional problems\n")

############################################
# BONUS EXAMPLE: IMPROVED GRADIENT
############################################

print("\n" + "="*50)
print("BONUS EXAMPLE: gradient() with Matrix Manipulation")
print("="*50 + "\n")

print("Better: use matrix manipulation and vectorization\n")

def gradient(f, x, deriv_steps, *args):
    """
    Improved gradient computation using matrix manipulation
    - Clearer and more efficient
    - Uses vectorized operations
    - Presumes that f takes a vector and returns a single number
    - Any extra arguments to gradient will get passed to f
    """
    p = len(x)
    assert len(deriv_steps) == p, "deriv_steps must have same length as x"
    
    # Create matrix of perturbed points
    x_new = np.tile(x, (p, 1)) + np.diag(deriv_steps)
    
    # Evaluate function at all perturbed points
    f_new = np.array([f(x_new[i], *args) for i in range(p)])
    
    # Compute gradient
    gradient = (f_new - f(x, *args)) / deriv_steps
    return gradient

print("Improved gradient function:")
print("- Clearer and more concise")
print("- Uses matrix manipulation")
print("- Vectorized computation\n")

print("Key features:")
print("- Presumes that f takes a vector and returns a single number")
print("- Any extra arguments to gradient will get passed to f\n")

print("Check: Does this work when f is a function of a single number?\n")

############################################
# TEST THE GRADIENT FUNCTIONS
############################################

print("\n" + "="*50)
print("TESTING GRADIENT FUNCTIONS")
print("="*50 + "\n")

# Test function 1: Simple quadratic
def f1(x):
    return np.sum(x**2)

# True gradient: 2*x
def true_grad_f1(x):
    return 2 * x

# Test at a point
x_test = np.array([1.0, 2.0, 3.0])
deriv_steps = np.full(3, 1e-5)

grad_basic = gradient_basic(f1, x_test, deriv_steps)
grad_matrix = gradient(f1, x_test, deriv_steps)
grad_true = true_grad_f1(x_test)

print("Test Function 1: f(x) = sum(x^2)")
print("Test point: (1, 2, 3)\n")
print(f"Basic gradient:   {grad_basic}")
print(f"Matrix gradient:  {grad_matrix}")
print(f"True gradient:    {grad_true}\n")

# Test function 2: Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# True gradient
def true_grad_rosenbrock(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

x_test2 = np.array([0.5, 0.5])
deriv_steps2 = np.full(2, 1e-5)

grad_basic2 = gradient_basic(rosenbrock, x_test2, deriv_steps2)
grad_matrix2 = gradient(rosenbrock, x_test2, deriv_steps2)
grad_true2 = true_grad_rosenbrock(x_test2)

print("Test Function 2: Rosenbrock function")
print("Test point: (0.5, 0.5)\n")
print(f"Basic gradient:   {grad_basic2}")
print(f"Matrix gradient:  {grad_matrix2}")
print(f"True gradient:    {grad_true2}\n")

# Test with single-variable function
def f_single(x):
    return x[0]**2 if isinstance(x, np.ndarray) else x**2

x_single = np.array([2.0])
step_single = np.array([1e-5])

grad_single = gradient(f_single, x_single, step_single)
print("Single variable test: f(x) = x^2 at x=2")
print(f"Computed gradient: {grad_single[0]:.6f}")
print("True gradient:     4.000000\n")

############################################
# VISUALIZE GRADIENT COMPUTATION
############################################

print("\n" + "="*50)
print("VISUALIZING GRADIENT COMPUTATION")
print("="*50 + "\n")

# 2D function for visualization
def f_viz(x):
    return x[0]**2 + 2*x[1]**2

# Create grid
x1 = np.linspace(-3, 3, 50)
x2 = np.linspace(-3, 3, 50)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + 2*X2**2

# Plot function and gradient at several points
fig, ax = plt.subplots(figsize=(10, 10))
contour = ax.contour(X1, X2, Z, levels=20, colors='lightblue', linewidths=2)
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Gradient Field: f(x) = x1² + 2·x2²', fontsize=14)

# Add gradient arrows at grid points
grid_x1 = np.arange(-2.5, 3.0, 0.5)
grid_x2 = np.arange(-2.5, 3.0, 0.5)

for gx1 in grid_x1:
    for gx2 in grid_x2:
        pt = np.array([gx1, gx2])
        grad = gradient(f_viz, pt, np.array([1e-5, 1e-5]))
        grad_norm = grad / np.linalg.norm(grad)  # Normalize for visualization
        
        # Scale arrows
        arrow_scale = 0.2
        ax.arrow(pt[0], pt[1], 
                -arrow_scale * grad_norm[0], 
                -arrow_scale * grad_norm[1],
                head_width=0.08, head_length=0.08, fc='red', ec='red', lw=1.5)

ax.plot(0, 0, 'o', color='darkgreen', markersize=12, label='Minimum')
ax.text(0, 0.3, 'Minimum', color='darkgreen', fontsize=12, fontweight='bold', ha='center')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_ii_gradient_field.png', dpi=100, bbox_inches='tight')
plt.close()

print("Plot saved: optimization_ii_gradient_field.png\n")

############################################
# TIMING COMPARISON
############################################

print("\n" + "="*50)
print("TIMING COMPARISON")
print("="*50 + "\n")

# Test with higher dimensional function
dim = 100
x_large = np.random.randn(dim)
deriv_steps_large = np.full(dim, 1e-5)

def f_large(x):
    return np.sum(x**2)

# Time basic version
start_basic = time.time()
for _ in range(100):
    grad_basic_large = gradient_basic(f_large, x_large, deriv_steps_large)
time_basic = time.time() - start_basic

# Time matrix version
start_matrix = time.time()
for _ in range(100):
    grad_matrix_large = gradient(f_large, x_large, deriv_steps_large)
time_matrix = time.time() - start_matrix

print("Timing for 100-dimensional function (100 iterations):\n")
print(f"Basic version (loop):      {time_basic:.4f} seconds")
print(f"Matrix version (vectorized): {time_matrix:.4f} seconds")
print(f"Speedup:                   {time_basic / time_matrix:.2f}x\n")

############################################
# POTENTIAL ISSUES WITH GRADIENT
############################################

print("\n" + "="*50)
print("POTENTIAL ISSUES WITH GRADIENT FUNCTION")
print("="*50 + "\n")

print("The gradient function acts badly if:\n")

print("1. f is only defined on a limited domain and we ask for the")
print("   gradient somewhere near a boundary\n")

# Example: log function
def f_log(x):
    if np.any(x <= 0):
        return np.nan
    return np.sum(np.log(x))

x_boundary = np.array([0.001, 1.0])
deriv_steps_boundary = np.array([1e-3, 1e-3])

try:
    grad_boundary = gradient(f_log, x_boundary, deriv_steps_boundary)
    print(f"   Gradient near boundary: {grad_boundary}")
    print("   Warning: May be inaccurate or produce NaN!\n")
except Exception as e:
    print(f"   Error near boundary: {e}\n")

print("2. Forces the user to choose deriv_steps")
print("   - No automatic step size selection")
print("   - User must understand numerical differentiation\n")

print("3. Uses the same deriv_steps everywhere")
print("   Example: f(x) = x1² * sin(x2)")
print("   - May need different steps for different regions")
print("   - Constant step size may be suboptimal\n")

# Example with different scales
def f_mixed(x):
    return x[0]**2 * np.sin(x[1])

x_mixed = np.array([100.0, 0.1])
deriv_steps_mixed = np.full(2, 1e-5)

grad_mixed = gradient(f_mixed, x_mixed, deriv_steps_mixed)
print("   Example: f(x) = x1² * sin(x2) at (100, 0.1)")
print(f"   Gradient with uniform step: {grad_mixed}")
print("   (May have numerical issues due to scale differences)\n")

print("4. ...and so on through much of a first course in numerical analysis\n")

############################################
# IMPROVED GRADIENT WITH ADAPTIVE STEPS
############################################

print("\n" + "="*50)
print("IMPROVED GRADIENT WITH ADAPTIVE STEPS")
print("="*50 + "\n")

def gradient_adaptive(f, x, eps=None, *args):
    """
    Adaptive gradient with automatic step size selection
    """
    if eps is None:
        eps = np.sqrt(np.finfo(float).eps)  # ~1.5e-8
    
    p = len(x)
    
    # Adaptive step size based on magnitude of x
    deriv_steps = np.maximum(np.abs(x) * eps, eps)
    
    # Use matrix manipulation
    x_new = np.tile(x, (p, 1)) + np.diag(deriv_steps)
    f_new = np.array([f(x_new[i], *args) for i in range(p)])
    gradient = (f_new - f(x, *args)) / deriv_steps
    
    return gradient

print("Adaptive gradient function:")
print("- Automatically chooses step sizes")
print("- Step size proportional to |x|")
print("- Minimum step size to avoid underflow\n")

# Test on mixed scale function
grad_adaptive = gradient_adaptive(f_mixed, x_mixed)
print("Adaptive gradient on f(x) = x1² * sin(x2) at (100, 0.1):")
print(f"  {grad_adaptive}\n")

############################################
# CENTRAL DIFFERENCE METHOD
############################################

print("\n" + "="*50)
print("CENTRAL DIFFERENCE METHOD")
print("="*50 + "\n")

print("Forward difference:  f'(x) ≈ (f(x+h) - f(x)) / h")
print("Central difference:  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)")
print("\nCentral difference is more accurate (O(h²) vs O(h))\n")

def gradient_central(f, x, deriv_steps, *args):
    """Central difference gradient computation"""
    p = len(x)
    assert len(deriv_steps) == p, "deriv_steps must have same length as x"
    
    # Forward perturbations
    x_forward = np.tile(x, (p, 1)) + np.diag(deriv_steps)
    # Backward perturbations
    x_backward = np.tile(x, (p, 1)) - np.diag(deriv_steps)
    
    f_forward = np.array([f(x_forward[i], *args) for i in range(p)])
    f_backward = np.array([f(x_backward[i], *args) for i in range(p)])
    
    gradient = (f_forward - f_backward) / (2 * deriv_steps)
    return gradient

# Compare methods
x_compare = np.array([1.0, 2.0])
steps_compare = np.full(2, 1e-4)

grad_forward = gradient(rosenbrock, x_compare, steps_compare)
grad_central = gradient_central(rosenbrock, x_compare, steps_compare)
grad_true = true_grad_rosenbrock(x_compare)

print("Comparison on Rosenbrock function at (1, 2):\n")
print(f"Forward difference:  {grad_forward}")
print(f"Central difference:  {grad_central}")
print(f"True gradient:       {grad_true}\n")

# Error analysis
error_forward = np.abs(grad_forward - grad_true)
error_central = np.abs(grad_central - grad_true)

print("Absolute errors:")
print(f"Forward difference:  {error_forward}")
print(f"Central difference:  {error_central}\n")

############################################
# STEP SIZE ANALYSIS
############################################

print("\n" + "="*50)
print("STEP SIZE ANALYSIS")
print("="*50 + "\n")

# Test different step sizes
def f_simple(x):
    return x[0]**2 if isinstance(x, np.ndarray) else x**2

x_test_step = np.array([1.0])
true_deriv = 2.0

step_sizes = 10**np.arange(-12, -0.5, 0.5)
errors_forward = np.zeros(len(step_sizes))
errors_central = np.zeros(len(step_sizes))

for i, h in enumerate(step_sizes):
    # Forward difference
    grad_f = (f_simple(x_test_step + h) - f_simple(x_test_step)) / h
    errors_forward[i] = np.abs(grad_f - true_deriv)
    
    # Central difference
    grad_c = (f_simple(x_test_step + h) - f_simple(x_test_step - h)) / (2*h)
    errors_central[i] = np.abs(grad_c - true_deriv)

fig, ax = plt.subplots(figsize=(10, 7))
ax.loglog(step_sizes, errors_forward, 'o-', color='blue', linewidth=2, 
          markersize=8, label='Forward Difference')
ax.loglog(step_sizes, errors_central, 's-', color='red', linewidth=2, 
          markersize=8, label='Central Difference')
ax.set_xlabel('Step Size (h)', fontsize=12)
ax.set_ylabel('Absolute Error', fontsize=12)
ax.set_title('Numerical Differentiation Error vs Step Size', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_ii_step_size_analysis.png', dpi=100, bbox_inches='tight')
plt.close()

print("Plot saved: optimization_ii_step_size_analysis.png")
print("Note: Central difference is more accurate for moderate step sizes\n")

############################################
# PRACTICAL RECOMMENDATIONS
############################################

print("\n" + "="*50)
print("PRACTICAL RECOMMENDATIONS")
print("="*50 + "\n")

print("1. Use adaptive step sizes based on scale of variables")
print("   eps = sqrt(machine_epsilon) ≈ 1.5e-8 is a good default\n")

print("2. Use central differences when possible (more accurate)\n")

print("3. Check for boundary issues and constrained domains\n")

print("4. For critical applications, compare with:")
print("   - Automatic differentiation (e.g., JAX, PyTorch)")
print("   - Symbolic derivatives (e.g., SymPy)\n")

print("5. Consider using existing packages:")
print("   - scipy.optimize.approx_fprime() for numerical gradients")
print("   - numdifftools for advanced numerical differentiation\n")

############################################
# USING SCIPY
############################################

print("\n" + "="*50)
print("USING SCIPY FOR GRADIENTS")
print("="*50 + "\n")

from scipy.optimize import approx_fprime

# Compare our methods with scipy
grad_scipy = approx_fprime(x_compare, rosenbrock, 1e-8)

print("Gradient comparison on Rosenbrock at (1, 2):\n")
print(f"Our forward:         {grad_forward}")
print(f"Our central:         {grad_central}")
print(f"scipy.approx_fprime: {grad_scipy}")
print(f"True gradient:       {grad_true}\n")

############################################
# APPLICATION TO OPTIMIZATION
############################################

print("\n" + "="*50)
print("APPLICATION TO OPTIMIZATION")
print("="*50 + "\n")

print("Gradients are essential for gradient-based optimization methods:")
print("- Gradient descent")
print("- Conjugate gradient")
print("- Newton's method")
print("- Quasi-Newton methods (BFGS, L-BFGS)\n")

# Simple gradient descent example
def gradient_descent(f, x0, gradient_fn, alpha=0.01, max_iter=1000, tol=1e-6):
    """Simple gradient descent optimization"""
    x = x0.copy()
    path = [x.copy()]
    
    for i in range(max_iter):
        grad = gradient_fn(f, x, np.full(len(x), 1e-5))
        x_new = x - alpha * grad
        path.append(x_new.copy())
        
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    
    return {'x': x, 'path': np.array(path), 'iterations': i+1}

# Optimize Rosenbrock function
x_start = np.array([-1.0, 1.0])
result = gradient_descent(rosenbrock, x_start, gradient_central, 
                         alpha=0.001, max_iter=5000)

print("Gradient Descent on Rosenbrock Function:")
print(f"Starting point: ({x_start[0]:.2f}, {x_start[1]:.2f})")
print(f"Final point:    ({result['x'][0]:.6f}, {result['x'][1]:.6f})")
print("True minimum:   (1.000000, 1.000000)")
print(f"Iterations:     {result['iterations']}\n")

# Plot optimization path
fig, ax = plt.subplots(figsize=(10, 10))

# Create contour plot
x1_opt = np.linspace(-1.5, 1.5, 100)
x2_opt = np.linspace(-0.5, 1.5, 100)
X1_opt, X2_opt = np.meshgrid(x1_opt, x2_opt)
Z_opt = 100 * (X2_opt - X1_opt**2)**2 + (1 - X1_opt)**2

contour = ax.contour(X1_opt, X2_opt, Z_opt, levels=30, colors='lightblue', linewidths=1.5)
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Gradient Descent on Rosenbrock Function', fontsize=14)

# Plot path
path = result['path']
ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=2, label='Path')
ax.plot(path[0, 0], path[0, 1], 'o', color='darkgreen', markersize=12, label='Start')
ax.plot(path[-1, 0], path[-1, 1], 'o', color='darkred', markersize=12, label='End')
ax.plot(1, 1, 'x', color='blue', markersize=12, markeredgewidth=3, label='True Minimum')

ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('../plots/optimization_ii_gradient_descent.png', dpi=100, bbox_inches='tight')
plt.close()

print("Plot saved: optimization_ii_gradient_descent.png\n")

############################################
# SUMMARY
############################################

print("\n" + "="*60)
print("SUMMARY: OPTIMIZATION II")
print("="*60 + "\n")

print("Key Takeaways:\n")

print("✓ Matrix manipulation makes gradient computation clearer and faster")
print("  - Use vectorization instead of loops when possible")
print("  - NumPy array operations improve performance\n")

print("✓ Numerical differentiation requires careful consideration:")
print("  - Step size selection is critical")
print("  - Central differences more accurate than forward")
print("  - Adaptive steps handle different scales\n")

print("✓ Common pitfalls:")
print("  - Boundary issues with constrained domains")
print("  - Fixed step sizes for all variables")
print("  - Not accounting for function scale\n")

print("✓ Best practices:")
print("  - Use existing packages (scipy.optimize) for production code")
print("  - Validate numerical gradients when possible")
print("  - Consider automatic differentiation\n")

print("✓ Gradients enable powerful optimization methods:")
print("  - Gradient descent and variants")
print("  - Newton and quasi-Newton methods")
print("  - Constrained optimization algorithms\n")

print("="*60)
print("OPTIMIZATION II TUTORIAL COMPLETE")
print("="*60 + "\n")

plot_count = len([f for f in os.listdir('../plots') if f.startswith('optimization_ii') and f.endswith('.png')])
print(f"Generated {plot_count} plots\n")

print("For more advanced topics, see:")
print("- scipy.optimize documentation")
print("- JAX for automatic differentiation")
print("- PyTorch/TensorFlow for deep learning optimization")
print("- numdifftools for advanced numerical derivatives\n")

print("Thank you for completing this tutorial!")
