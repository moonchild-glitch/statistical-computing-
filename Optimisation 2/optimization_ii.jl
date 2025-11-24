#!/usr/bin/env julia
"""
============================================
OPTIMIZATION II
Advanced Topics in Numerical Optimization
============================================

AGENDA:
- Gradient computation techniques
- Matrix manipulation and broadcasting
- Numerical differentiation
- Edge cases and numerical stability
- Best practices for optimization
"""

using LinearAlgebra
using Plots
using Statistics
using Printf

# Set plotting defaults
gr()
Plots.default(size=(800, 600), dpi=100)

# Create plots directory
mkpath("../plots")

############################################
# GRADIENT COMPUTATION
############################################

println("\n" * "="^50)
println("GRADIENT COMPUTATION")
println("="^50 * "\n")

println("Computing gradients is fundamental to optimization")
println("We'll explore different approaches to numerical differentiation\n")

# Basic gradient function (component-wise)
function gradient_basic(f::Function, x::Vector{<:Real}, deriv_steps::Vector{<:Real}, args...)
    """Basic gradient computation using loops"""
    p = length(x)
    @assert length(deriv_steps) == p "deriv_steps must have same length as x"
    gradient = zeros(p)
    
    for i in 1:p
        x_new = copy(x)
        x_new[i] = x[i] + deriv_steps[i]
        gradient[i] = (f(x_new, args...) - f(x, args...)) / deriv_steps[i]
    end
    
    return gradient
end

println("Basic gradient function (loop-based):")
println("- Iterates through each component")
println("- Computes finite difference for each dimension")
println("- Can be slow for high-dimensional problems\n")

############################################
# BONUS EXAMPLE: IMPROVED GRADIENT
############################################

println("\n" * "="^50)
println("BONUS EXAMPLE: gradient() with Matrix Manipulation")
println("="^50 * "\n")

println("Better: use matrix manipulation and broadcasting\n")

function gradient(f::Function, x::Vector{<:Real}, deriv_steps::Vector{<:Real}, args...)
    """
    Improved gradient computation using matrix manipulation
    - Clearer and more efficient
    - Uses broadcasting and matrix operations
    - Presumes that f takes a vector and returns a single number
    - Any extra arguments to gradient will get passed to f
    """
    p = length(x)
    @assert length(deriv_steps) == p "deriv_steps must have same length as x"
    
    # Create matrix of perturbed points
    x_new = repeat(x', p, 1) .+ diagm(deriv_steps)
    
    # Evaluate function at all perturbed points
    f_new = [f(x_new[i, :], args...) for i in 1:p]
    
    # Compute gradient
    grad = (f_new .- f(x, args...)) ./ deriv_steps
    return grad
end

println("Improved gradient function:")
println("- Clearer and more concise")
println("- Uses matrix manipulation")
println("- Broadcasting for efficient computation\n")

println("Key features:")
println("- Presumes that f takes a vector and returns a single number")
println("- Any extra arguments to gradient will get passed to f\n")

println("Check: Does this work when f is a function of a single number?\n")

############################################
# TEST THE GRADIENT FUNCTIONS
############################################

println("\n" * "="^50)
println("TESTING GRADIENT FUNCTIONS")
println("="^50 * "\n")

# Test function 1: Simple quadratic
f1(x) = sum(x.^2)

# True gradient: 2*x
true_grad_f1(x) = 2 .* x

# Test at a point
x_test = [1.0, 2.0, 3.0]
deriv_steps = fill(1e-5, 3)

grad_basic = gradient_basic(f1, x_test, deriv_steps)
grad_matrix = gradient(f1, x_test, deriv_steps)
grad_true = true_grad_f1(x_test)

println("Test Function 1: f(x) = sum(x^2)")
println("Test point: (1, 2, 3)\n")
println("Basic gradient:   ", grad_basic)
println("Matrix gradient:  ", grad_matrix)
println("True gradient:    ", grad_true, "\n")

# Test function 2: Rosenbrock function
function rosenbrock(x)
    return 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2
end

# True gradient
function true_grad_rosenbrock(x)
    return [
        -400 * x[1] * (x[2] - x[1]^2) - 2 * (1 - x[1]),
        200 * (x[2] - x[1]^2)
    ]
end

x_test2 = [0.5, 0.5]
deriv_steps2 = fill(1e-5, 2)

grad_basic2 = gradient_basic(rosenbrock, x_test2, deriv_steps2)
grad_matrix2 = gradient(rosenbrock, x_test2, deriv_steps2)
grad_true2 = true_grad_rosenbrock(x_test2)

println("Test Function 2: Rosenbrock function")
println("Test point: (0.5, 0.5)\n")
println("Basic gradient:   ", grad_basic2)
println("Matrix gradient:  ", grad_matrix2)
println("True gradient:    ", grad_true2, "\n")

# Test with single-variable function
f_single(x) = x[1]^2

x_single = [2.0]
step_single = [1e-5]

grad_single = gradient(f_single, x_single, step_single)
println("Single variable test: f(x) = x^2 at x=2")
@printf("Computed gradient: %.6f\n", grad_single[1])
println("True gradient:     4.000000\n")

############################################
# VISUALIZE GRADIENT COMPUTATION
############################################

println("\n" * "="^50)
println("VISUALIZING GRADIENT COMPUTATION")
println("="^50 * "\n")

# 2D function for visualization
f_viz(x) = x[1]^2 + 2*x[2]^2

# Create grid
x1 = range(-3, 3, length=50)
x2 = range(-3, 3, length=50)
Z = [f_viz([i, j]) for j in x2, i in x1]

# Plot function and gradient at several points
p = contour(x1, x2, Z, levels=20, color=:lightblue, linewidth=2,
            xlabel="x1", ylabel="x2", 
            title="Gradient Field: f(x) = x1² + 2·x2²",
            aspect_ratio=:equal, size=(1000, 1000))

# Add gradient arrows at grid points
grid_x1 = -2.5:0.5:2.5
grid_x2 = -2.5:0.5:2.5

for gx1 in grid_x1
    for gx2 in grid_x2
        pt = [gx1, gx2]
        grad = gradient(f_viz, pt, [1e-5, 1e-5])
        grad_norm = grad ./ norm(grad)  # Normalize for visualization
        
        # Scale arrows
        arrow_scale = 0.2
        quiver!([pt[1]], [pt[2]], 
               quiver=([-arrow_scale * grad_norm[1]], [-arrow_scale * grad_norm[2]]),
               color=:red, linewidth=1.5, arrow=true)
    end
end

scatter!([0], [0], markersize=10, color=:darkgreen, label="Minimum", 
         markerstrokewidth=0)
annotate!(0, 0.3, text("Minimum", :darkgreen, 12, :center, :bold))

savefig("../plots/optimization_ii_gradient_field.png")
println("Plot saved: optimization_ii_gradient_field.png\n")

############################################
# TIMING COMPARISON
############################################

println("\n" * "="^50)
println("TIMING COMPARISON")
println("="^50 * "\n")

# Test with higher dimensional function
dim = 100
x_large = randn(dim)
deriv_steps_large = fill(1e-5, dim)

f_large(x) = sum(x.^2)

# Time basic version
time_basic = @elapsed begin
    for _ in 1:100
        grad_basic_large = gradient_basic(f_large, x_large, deriv_steps_large)
    end
end

# Time matrix version
time_matrix = @elapsed begin
    for _ in 1:100
        grad_matrix_large = gradient(f_large, x_large, deriv_steps_large)
    end
end

println("Timing for 100-dimensional function (100 iterations):\n")
@printf("Basic version (loop):      %.4f seconds\n", time_basic)
@printf("Matrix version (broadcast): %.4f seconds\n", time_matrix)
@printf("Speedup:                   %.2fx\n\n", time_basic / time_matrix)

############################################
# POTENTIAL ISSUES WITH GRADIENT
############################################

println("\n" * "="^50)
println("POTENTIAL ISSUES WITH GRADIENT FUNCTION")
println("="^50 * "\n")

println("The gradient function acts badly if:\n")

println("1. f is only defined on a limited domain and we ask for the")
println("   gradient somewhere near a boundary\n")

# Example: log function
function f_log(x)
    if any(x .<= 0)
        return NaN
    end
    return sum(log.(x))
end

x_boundary = [0.001, 1.0]
deriv_steps_boundary = [1e-3, 1e-3]

try
    grad_boundary = gradient(f_log, x_boundary, deriv_steps_boundary)
    println("   Gradient near boundary: ", grad_boundary)
    println("   Warning: May be inaccurate or produce NaN!\n")
catch e
    println("   Error near boundary: ", e, "\n")
end

println("2. Forces the user to choose deriv_steps")
println("   - No automatic step size selection")
println("   - User must understand numerical differentiation\n")

println("3. Uses the same deriv_steps everywhere")
println("   Example: f(x) = x1² * sin(x2)")
println("   - May need different steps for different regions")
println("   - Constant step size may be suboptimal\n")

# Example with different scales
f_mixed(x) = x[1]^2 * sin(x[2])

x_mixed = [100.0, 0.1]
deriv_steps_mixed = fill(1e-5, 2)

grad_mixed = gradient(f_mixed, x_mixed, deriv_steps_mixed)
println("   Example: f(x) = x1² * sin(x2) at (100, 0.1)")
println("   Gradient with uniform step: ", grad_mixed)
println("   (May have numerical issues due to scale differences)\n")

println("4. ...and so on through much of a first course in numerical analysis\n")

############################################
# IMPROVED GRADIENT WITH ADAPTIVE STEPS
############################################

println("\n" * "="^50)
println("IMPROVED GRADIENT WITH ADAPTIVE STEPS")
println("="^50 * "\n")

function gradient_adaptive(f::Function, x::Vector{<:Real}, eps=nothing, args...)
    """
    Adaptive gradient with automatic step size selection
    """
    if eps === nothing
        eps = sqrt(eps(Float64))  # ~1.5e-8
    end
    
    p = length(x)
    
    # Adaptive step size based on magnitude of x
    deriv_steps = max.(abs.(x) .* eps, eps)
    
    # Use matrix manipulation
    x_new = repeat(x', p, 1) .+ diagm(deriv_steps)
    f_new = [f(x_new[i, :], args...) for i in 1:p]
    grad = (f_new .- f(x, args...)) ./ deriv_steps
    
    return grad
end

println("Adaptive gradient function:")
println("- Automatically chooses step sizes")
println("- Step size proportional to |x|")
println("- Minimum step size to avoid underflow\n")

# Test on mixed scale function
grad_adaptive = gradient_adaptive(f_mixed, x_mixed)
println("Adaptive gradient on f(x) = x1² * sin(x2) at (100, 0.1):")
println("  ", grad_adaptive, "\n")

############################################
# CENTRAL DIFFERENCE METHOD
############################################

println("\n" * "="^50)
println("CENTRAL DIFFERENCE METHOD")
println("="^50 * "\n")

println("Forward difference:  f'(x) ≈ (f(x+h) - f(x)) / h")
println("Central difference:  f'(x) ≈ (f(x+h) - f(x-h)) / (2h)")
println("\nCentral difference is more accurate (O(h²) vs O(h))\n")

function gradient_central(f::Function, x::Vector{<:Real}, deriv_steps::Vector{<:Real}, args...)
    """Central difference gradient computation"""
    p = length(x)
    @assert length(deriv_steps) == p "deriv_steps must have same length as x"
    
    # Forward perturbations
    x_forward = repeat(x', p, 1) .+ diagm(deriv_steps)
    # Backward perturbations
    x_backward = repeat(x', p, 1) .- diagm(deriv_steps)
    
    f_forward = [f(x_forward[i, :], args...) for i in 1:p]
    f_backward = [f(x_backward[i, :], args...) for i in 1:p]
    
    grad = (f_forward .- f_backward) ./ (2 .* deriv_steps)
    return grad
end

# Compare methods
x_compare = [1.0, 2.0]
steps_compare = fill(1e-4, 2)

grad_forward = gradient(rosenbrock, x_compare, steps_compare)
grad_central = gradient_central(rosenbrock, x_compare, steps_compare)
grad_true = true_grad_rosenbrock(x_compare)

println("Comparison on Rosenbrock function at (1, 2):\n")
println("Forward difference:  ", grad_forward)
println("Central difference:  ", grad_central)
println("True gradient:       ", grad_true, "\n")

# Error analysis
error_forward = abs.(grad_forward .- grad_true)
error_central = abs.(grad_central .- grad_true)

println("Absolute errors:")
println("Forward difference:  ", error_forward)
println("Central difference:  ", error_central, "\n")

############################################
# STEP SIZE ANALYSIS
############################################

println("\n" * "="^50)
println("STEP SIZE ANALYSIS")
println("="^50 * "\n")

# Test different step sizes
f_simple(x) = x[1]^2
x_test_step = [1.0]
true_deriv = 2.0

step_sizes = 10.0 .^ (-12:0.5:-1)
errors_forward = zeros(length(step_sizes))
errors_central = zeros(length(step_sizes))

for (i, h) in enumerate(step_sizes)
    # Forward difference
    grad_f = (f_simple(x_test_step .+ h) - f_simple(x_test_step)) / h
    errors_forward[i] = abs(grad_f - true_deriv)
    
    # Central difference
    grad_c = (f_simple(x_test_step .+ h) - f_simple(x_test_step .- h)) / (2*h)
    errors_central[i] = abs(grad_c - true_deriv)
end

p = plot(step_sizes, errors_forward, xscale=:log10, yscale=:log10,
         marker=:circle, linewidth=2, color=:blue, label="Forward Difference",
         xlabel="Step Size (h)", ylabel="Absolute Error",
         title="Numerical Differentiation Error vs Step Size",
         size=(1000, 700), legend=:topright)
plot!(step_sizes, errors_central, marker=:square, linewidth=2, 
      color=:red, label="Central Difference")

savefig("../plots/optimization_ii_step_size_analysis.png")
println("Plot saved: optimization_ii_step_size_analysis.png")
println("Note: Central difference is more accurate for moderate step sizes\n")

############################################
# PRACTICAL RECOMMENDATIONS
############################################

println("\n" * "="^50)
println("PRACTICAL RECOMMENDATIONS")
println("="^50 * "\n")

println("1. Use adaptive step sizes based on scale of variables")
println("   eps = sqrt(eps(Float64)) ≈ 1.5e-8 is a good default\n")

println("2. Use central differences when possible (more accurate)\n")

println("3. Check for boundary issues and constrained domains\n")

println("4. For critical applications, compare with:")
println("   - Automatic differentiation (e.g., ForwardDiff.jl, Zygote.jl)")
println("   - Symbolic derivatives (e.g., Symbolics.jl)\n")

println("5. Consider using existing packages:")
println("   - FiniteDiff.jl for numerical gradients")
println("   - Optim.jl for optimization with autodiff support\n")

############################################
# USING FiniteDiff.jl
############################################

println("\n" * "="^50)
println("USING FiniteDiff.jl PACKAGE")
println("="^50 * "\n")

try
    using FiniteDiff
    
    # Compare our methods with FiniteDiff
    grad_finitediff = FiniteDiff.finite_difference_gradient(rosenbrock, x_compare)
    
    println("Gradient comparison on Rosenbrock at (1, 2):\n")
    println("Our forward:        ", grad_forward)
    println("Our central:        ", grad_central)
    println("FiniteDiff.jl:      ", grad_finitediff)
    println("True gradient:      ", grad_true, "\n")
    
    println("FiniteDiff.jl uses advanced techniques for high accuracy\n")
catch
    println("FiniteDiff package not installed")
    println("Install with: using Pkg; Pkg.add(\"FiniteDiff\")\n")
end

############################################
# APPLICATION TO OPTIMIZATION
############################################

println("\n" * "="^50)
println("APPLICATION TO OPTIMIZATION")
println("="^50 * "\n")

println("Gradients are essential for gradient-based optimization methods:")
println("- Gradient descent")
println("- Conjugate gradient")
println("- Newton's method")
println("- Quasi-Newton methods (BFGS, L-BFGS)\n")

# Simple gradient descent example
function gradient_descent(f::Function, x0::Vector{<:Real}, gradient_fn::Function;
                         alpha=0.01, max_iter=1000, tol=1e-6)
    """Simple gradient descent optimization"""
    x = copy(x0)
    path = [copy(x)]
    
    for i in 1:max_iter
        grad = gradient_fn(f, x, fill(1e-5, length(x)))
        x_new = x .- alpha .* grad
        push!(path, copy(x_new))
        
        if norm(x_new .- x) < tol
            return (x=x_new, path=hcat(path...)', iterations=i)
        end
        x = x_new
    end
    
    return (x=x, path=hcat(path...)', iterations=max_iter)
end

# Optimize Rosenbrock function
x_start = [-1.0, 1.0]
result = gradient_descent(rosenbrock, x_start, gradient_central,
                         alpha=0.001, max_iter=5000)

println("Gradient Descent on Rosenbrock Function:")
@printf("Starting point: (%.2f, %.2f)\n", x_start[1], x_start[2])
@printf("Final point:    (%.6f, %.6f)\n", result.x[1], result.x[2])
println("True minimum:   (1.000000, 1.000000)")
@printf("Iterations:     %d\n\n", result.iterations)

# Plot optimization path
x1_opt = range(-1.5, 1.5, length=100)
x2_opt = range(-0.5, 1.5, length=100)
Z_opt = [rosenbrock([i, j]) for j in x2_opt, i in x1_opt]

p = contour(x1_opt, x2_opt, Z_opt, levels=30, color=:lightblue, linewidth=1.5,
            xlabel="x1", ylabel="x2",
            title="Gradient Descent on Rosenbrock Function",
            size=(1000, 1000))

# Plot path
path = result.path
plot!(path[:, 1], path[:, 2], color=:red, linewidth=2, label="Path")
scatter!([path[1, 1]], [path[1, 2]], markersize=10, color=:darkgreen, 
         label="Start", markerstrokewidth=0)
scatter!([path[end, 1]], [path[end, 2]], markersize=10, color=:darkred,
         label="End", markerstrokewidth=0)
scatter!([1], [1], marker=:x, markersize=10, color=:blue, 
         markerstrokewidth=3, label="True Minimum")

savefig("../plots/optimization_ii_gradient_descent.png")
println("Plot saved: optimization_ii_gradient_descent.png\n")

############################################
# SUMMARY
############################################

println("\n" * "="^60)
println("SUMMARY: OPTIMIZATION II")
println("="^60 * "\n")

println("Key Takeaways:\n")

println("✓ Matrix manipulation makes gradient computation clearer and faster")
println("  - Use broadcasting instead of loops when possible")
println("  - Julia's array operations are highly optimized\n")

println("✓ Numerical differentiation requires careful consideration:")
println("  - Step size selection is critical")
println("  - Central differences more accurate than forward")
println("  - Adaptive steps handle different scales\n")

println("✓ Common pitfalls:")
println("  - Boundary issues with constrained domains")
println("  - Fixed step sizes for all variables")
println("  - Not accounting for function scale\n")

println("✓ Best practices:")
println("  - Use existing packages (Optim.jl, FiniteDiff.jl) for production code")
println("  - Validate numerical gradients when possible")
println("  - Consider automatic differentiation (ForwardDiff.jl)\n")

println("✓ Gradients enable powerful optimization methods:")
println("  - Gradient descent and variants")
println("  - Newton and quasi-Newton methods")
println("  - Constrained optimization algorithms\n")

println("="^60)
println("OPTIMIZATION II TUTORIAL COMPLETE")
println("="^60 * "\n")

plot_files = filter(f -> startswith(f, "optimization_ii") && endswith(f, ".png"),
                   readdir("../plots"))
println("Generated $(length(plot_files)) plots\n")

println("For more advanced topics, see:")
println("- Optim.jl documentation")
println("- ForwardDiff.jl for automatic differentiation")
println("- Zygote.jl for reverse-mode autodiff")
println("- JuMP.jl for constrained optimization\n")

println("Thank you for completing this tutorial!")
