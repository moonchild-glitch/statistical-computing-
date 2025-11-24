# ============================================
# OPTIMIZATION I
# ============================================
# 
# AGENDA:
# - Functions are objects: can be arguments for or returned by other functions
# - Example: plotting functions
# - Optimization via gradient descent, Newton's method, Nelder-Mead, …
# - Curve-fitting by optimizing
# ============================================

using Statistics
using Plots
using DataFrames
using Random
using Optim

# Create plots directory if it doesn't exist
if !isdir("../plots")
    mkdir("../plots")
end

# ============================================
# FUNCTIONS AS OBJECTS
# ============================================
println("\n" * "=" ^ 50)
println("FUNCTIONS AS OBJECTS")
println("=" ^ 50 * "\n")

println("typeof(sin): ", typeof(sin))
println("typeof(rand): ", typeof(rand))

function resample(x)
    """Resample with replacement"""
    return rand(x, length(x))
end

println("typeof(resample): ", typeof(resample))

# Functions can be passed as arguments
function apply_function(func, values)
    """Apply a function to each value"""
    return [func(v) for v in values]
end

# Logistic transformation
log_ratios = -2:2
logistic(x) = exp(x) / (1 + exp(x))
result = apply_function(logistic, log_ratios)
println("\nLogistic transformation results:")
println(result)

# ============================================
# NUMERICAL DERIVATIVES
# ============================================
println("\n" * "=" ^ 50)
println("NUMERICAL DERIVATIVES")
println("=" ^ 50 * "\n")

# Numerical derivative function
function numerical_derivative(f, x; h=1e-8)
    """Compute numerical derivative of f at x"""
    return (f(x + h) - f(x - h)) / (2h)
end

Random.seed!(42)
just_a_phase = rand() * 2π - π
derivative_check = isapprox(numerical_derivative(cos, just_a_phase), -sin(just_a_phase))
println("derivative(cos) ≈ -sin: ", derivative_check)

# Multivariable functions - numerical gradient
function multivariable_func(x)
    """f(x) = x[1]^2 + x[2]^3"""
    return x[1]^2 + x[2]^3
end

function numerical_gradient(f, x; h=1e-8)
    """Compute numerical gradient of f at x"""
    grad = zeros(length(x))
    for i in 1:length(x)
        x_plus = copy(x)
        x_minus = copy(x)
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2h)
    end
    return grad
end

grad_result = numerical_gradient(multivariable_func, [1.0, -1.0])
println("Gradient of x[1]^2 + x[2]^3 at (1,-1): ", grad_result)

# ============================================
# GRADIENT DESCENT IMPLEMENTATION
# ============================================
println("\n" * "=" ^ 50)
println("GRADIENT DESCENT")
println("=" ^ 50 * "\n")

function gradient_descent(f, x; max_iterations=100, step_scale=0.01, stopping_deriv=0.01)
    """
    Minimize function f using gradient descent
    
    Parameters:
    - f: function to minimize
    - x: initial point
    - max_iterations: maximum number of iterations
    - step_scale: step size for gradient descent
    - stopping_deriv: threshold for stopping criterion
    
    Returns:
    - Dict with argmin, final_gradient, final_value, iterations
    """
    x = copy(x)
    local gradient
    local iteration
    
    for iter in 1:max_iterations
        iteration = iter
        gradient = numerical_gradient(f, x)
        if all(abs.(gradient) .< stopping_deriv)
            break
        end
        x = x .- step_scale .* gradient
    end
    
    return Dict(
        :argmin => x,
        :final_gradient => gradient,
        :final_value => f(x),
        :iterations => iteration,
        :converged => iteration < max_iterations
    )
end

println("Gradient descent function defined")

# ============================================
# FUNCTIONS RETURNING FUNCTIONS
# ============================================
println("\n" * "=" ^ 50)
println("FUNCTIONS RETURNING FUNCTIONS")
println("=" ^ 50 * "\n")

function make_linear_predictor(x, y)
    """
    Create a linear predictor function from training data
    
    Returns a function that predicts y from x
    """
    # Fit linear model: y = a*x + b
    n = length(x)
    x_mean = mean(x)
    y_mean = mean(y)
    
    a = sum((x .- x_mean) .* (y .- y_mean)) / sum((x .- x_mean).^2)
    b = y_mean - a * x_mean
    
    function predictor(x_new)
        return a .* x_new .+ b
    end
    
    return predictor
end

# Example with synthetic cat data
Random.seed!(42)
cat_body_weight = rand(Uniform(2.0, 4.0), 50)
cat_heart_weight = 4.0 .* cat_body_weight .+ randn(50) .* 0.5

vet_predictor = make_linear_predictor(cat_body_weight, cat_heart_weight)
prediction = vet_predictor(3.5)
println("Predicted heart weight for 3.5kg cat: ", round(prediction, digits=2), " grams")

# ============================================
# PLOTTING FUNCTIONS
# ============================================
println("\n" * "=" ^ 50)
println("PLOTTING FUNCTIONS")
println("=" ^ 50 * "\n")

# Plot a function over a range
x_range = range(-10, 10, length=1000)
y_vals = x_range.^2 .* sin.(x_range)

plot(x_range, y_vals, linewidth=2, color=:blue,
     xlabel="x", ylabel="f(x)", title="x² * sin(x)",
     label="", grid=true)
savefig("../plots/optimization_curve1_jl.png")
println("Plot saved: optimization_curve1_jl.png")

# Robust loss function
function psi(x; c=1)
    """Robust loss function"""
    return ifelse.(abs.(x) .> c, 2 .* c .* abs.(x) .- c^2, x.^2)
end

x_range = range(-20, 20, length=1000)
plot(x_range, psi(x_range, c=10), linewidth=2, color=:green,
     xlabel="x", ylabel="ψ(x)", title="Robust Loss Function (c=10)",
     label="", grid=true)
savefig("../plots/optimization_psi1_jl.png")
println("Plot saved: optimization_psi1_jl.png")

c_values = range(-20, 20, length=1000)
plot(c_values, psi(10, c=c_values), linewidth=2, color=:purple,
     xlabel="c", ylabel="ψ(10, c)", title="Robust Loss Function (x=10, varying c)",
     label="", grid=true)
savefig("../plots/optimization_psi2_jl.png")
println("Plot saved: optimization_psi2_jl.png")

# ============================================
# GMP DATA AND MSE FUNCTION
# ============================================
println("\n" * "=" ^ 50)
println("GMP DATA AND OPTIMIZATION")
println("=" ^ 50 * "\n")

# Create synthetic GMP data
Random.seed!(42)
n_cities = 366
pop = 10 .^ rand(Uniform(4.5, 7.5), n_cities)
true_a = 0.125
true_y0 = 6611
pcgmp = true_y0 .* pop.^true_a .* exp.(randn(n_cities) .* 0.1)
gmp_total = pcgmp .* pop

gmp = DataFrame(
    gmp = gmp_total,
    pcgmp = pcgmp,
    pop = pop
)

println("Created GMP dataset with $(nrow(gmp)) cities")

# Mean squared error function
function mse(y0, a; Y=gmp.pcgmp, N=gmp.pop)
    """Calculate mean squared error for power law model"""
    return mean((Y .- y0 .* N.^a).^2)
end

# Test MSE function
test_values = [mse(6611, a) for a in 0.10:0.01:0.15]
println("MSE values for a from 0.10 to 0.15:")
println(test_values)

println("MSE at y0=6611, a=0.10: ", mse(6611, 0.10))

# ============================================
# VECTORIZING FUNCTIONS
# ============================================
println("\n" * "=" ^ 50)
println("VECTORIZING FUNCTIONS")
println("=" ^ 50 * "\n")

# Method 1: Broadcasting works naturally in Julia
function mse_plottable(a, y0)
    """Vectorized MSE function"""
    if isa(a, Number)
        return mse(y0, a)
    end
    return [mse(y0, a_val) for a_val in a]
end

result = mse_plottable(0.10:0.01:0.15, 6611)
println("MSE via plottable wrapper:")
println(result)

a_range = range(0.10, 0.20, length=100)
plot(a_range, mse_plottable(a_range, 6611), linewidth=2, color=:red, label="y0=6611",
     xlabel="a", ylabel="MSE", title="MSE vs Scaling Exponent", grid=true)
plot!(a_range, mse_plottable(a_range, 5100), linewidth=2, color=:blue, label="y0=5100")
savefig("../plots/optimization_mse1_jl.png")
println("Plot saved: optimization_mse1_jl.png")

# Method 2: Broadcasting with dot syntax
a_test = collect(0.10:0.01:0.15)
result_vec = [mse(6611, a_val) for a_val in a_test]
println("MSE via list comprehension:")
println(result_vec)

y0_values = [5000, 6000, 7000]
result_multi = [mse(y0_val, 1/8) for y0_val in y0_values]
println("MSE at a=1/8 for different y0 values:")
println(result_multi)

plot(a_range, [mse(6611, a_val) for a_val in a_range], linewidth=2, color=:red, label="y0=6611",
     xlabel="a", ylabel="MSE", title="MSE vs Scaling Exponent (Vectorized)", grid=true)
plot!(a_range, [mse(5100, a_val) for a_val in a_range], linewidth=2, color=:blue, label="y0=5100")
savefig("../plots/optimization_mse2_jl.png")
println("Plot saved: optimization_mse2_jl.png")

# ============================================
# USING OPTIM.JL
# ============================================
println("\n" * "=" ^ 50)
println("OPTIM.JL OPTIMIZATION")
println("=" ^ 50 * "\n")

# Optimize using Optim.jl
objective(a) = mse(6611, a[1])

result_opt = optimize(objective, [0.10], [0.20], [0.15], Fminbox(GradientDescent()))
println("Optimal a using Optim: ", Optim.minimizer(result_opt)[1])
println("Minimum MSE: ", Optim.minimum(result_opt))

# Alternative: use univariate optimization
objective_scalar(a) = mse(6611, a)
result_scalar = optimize(objective_scalar, 0.10, 0.20)
println("Optimal a (scalar): ", Optim.minimizer(result_scalar))

# ============================================
# SUMMARY
# ============================================
println("\n" * "=" ^ 50)
println("SUMMARY")
println("=" ^ 50 * "\n")
println("1. Functions are first-class objects in Julia")
println("2. Functions can take other functions as arguments")
println("3. Functions can return other functions (closures)")
println("4. Plots.jl provides flexible function plotting")
println("5. Broadcasting with . operator makes functions work with arrays")
println("6. Optim.jl provides robust optimization methods")
println("7. Gradient descent and other methods find function minima")

println("\nOptimization I Tutorial Complete")
plot_files = filter(f -> startswith(f, "optimization_") && endswith(f, "_jl.png"), 
                    readdir("../plots"))
println("Generated $(length(plot_files)) plots")
