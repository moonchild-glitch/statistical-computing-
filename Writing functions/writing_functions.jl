# ============================================
# WRITING FUNCTIONS IN JULIA
# ============================================
# 
# AGENDA:
# - Defining functions: Tying related commands into bundles
# - Interfaces: Controlling what the function can see and do
# - Example: Parameter estimation code
# - Multiple functions
# - Recursion: Making hard problems simpler
# ============================================

using Statistics
using Plots
using DataFrames
using Random

# ============================================
# WHY FUNCTIONS?
# ============================================
println("=" ^ 50)
println("WHY FUNCTIONS?")
println("=" ^ 50)
println("Data structures tie related values into one object")
println("Functions tie related commands into one object")
println()
println("In both cases: easier to understand, easier to work with,")
println("easier to build into larger things")
println()
println("Benefits of functions:")
println("1. Organize code into reusable chunks")
println("2. Avoid repetition (DRY principle: Don't Repeat Yourself)")
println("3. Make code more readable and maintainable")
println("4. Abstract complex operations into simple interfaces")
println("5. Test and debug code more easily")
println("6. Build complex programs from simple, well-tested components")

# ============================================
# DEFINING FUNCTIONS
# ============================================
println("\n" * "=" ^ 50)
println("DEFINING FUNCTIONS")
println("=" ^ 50)

# Example: Cubic function
cube(x) = x .^ 3

# Display the function
println("\nFunction definition:")
println("cube = ", cube)

# Test with a single value
println("\ncube(3) = ", cube(3))

# Test with an array
println("cube(1:10) = ", cube(1:10))

# Test with a matrix
matrix = reshape(1:8, 2, 4)
println("\ncube(matrix(1:8, 2, 4)):")
println(cube(matrix))

println("\ntypeof(cube) = ", typeof(cube))

# ============================================
# ROBUST LOSS FUNCTION EXAMPLE
# ============================================
println("\n" * "=" ^ 50)
println("ROBUST LOSS FUNCTION")
println("=" ^ 50)

# "Robust" loss function, for outlier-resistant regression
# Inputs: array of numbers (x)
# Outputs: array with x^2 for small entries, 2|x|-1 for large ones
function psi_1(x)
    psi = ifelse.(x.^2 .> 1, 2 .* abs.(x) .- 1, x.^2)
    return psi
end

# Our functions get used just like the built-in ones:
z = [-0.5, -5, 0.9, 9]
println("z = ", z)
println("psi_1(z) = ", psi_1(z))

# ============================================
# FUNCTION ANATOMY
# ============================================
println("\n" * "=" ^ 50)
println("FUNCTION ANATOMY")
println("=" ^ 50)
println("INTERFACES: the inputs (arguments) and outputs (return value)")
println("- Calls other functions: ifelse(), abs(), operators ^ and >")
println("- return statement says what the output is")
println("- Comments: Not required by Julia, but a good idea")

# ============================================
# WHAT SHOULD BE A FUNCTION?
# ============================================
println("\n" * "=" ^ 50)
println("WHAT SHOULD BE A FUNCTION?")
println("=" ^ 50)
println("- Things you're going to re-run, especially with changes")
println("- Chunks of code you keep highlighting and running")
println("- Chunks of code which are small parts of bigger analyses")
println("- Chunks which are very similar to other chunks")

# ============================================
# NAMED AND DEFAULT ARGUMENTS
# ============================================
println("\n" * "=" ^ 50)
println("NAMED AND DEFAULT ARGUMENTS")
println("=" ^ 50)

# "Robust" loss function with adjustable crossover scale
function psi_2(x; c=1)
    psi = ifelse.(x.^2 .> c^2, 2 .* c .* abs.(x) .- c^2, x.^2)
    return psi
end

# Default value c=1 makes psi_2 equivalent to psi_1
println("psi_1(z) == psi_2(z, c=1): ", psi_1(z) == psi_2(z, c=1))

# Default values get used if names are missing:
println("psi_2(z, c=1) == psi_2(z): ", psi_2(z, c=1) == psi_2(z))

# Named arguments can go in any order when explicitly tagged:
println("psi_2(z, c=2) == psi_2(z, c=2): ", psi_2(z, c=2) == psi_2(z, c=2))

# ============================================
# CHECKING ARGUMENTS
# ============================================
println("\n" * "=" ^ 50)
println("CHECKING ARGUMENTS")
println("=" ^ 50)

# Problem: Odd behavior when arguments aren't as we expect
println("\nProblem - array c causes element-wise comparison:")
println("psi_2(z, c=[1,1,1,10]) = ", psi_2(z, c=[1,1,1,10]))

println("\nProblem - negative c gives nonsensical results:")
println("psi_2(z, c=-1) = ", psi_2(z, c=-1))

# Solution: Put little sanity checks into the code
function psi_3(x; c=1)
    # Scale should be a single positive number
    if !(c isa Real && c > 0)
        error("c must be a single positive number")
    end
    psi = ifelse.(x.^2 .> c^2, 2 .* c .* abs.(x) .- c^2, x.^2)
    return psi
end

println("\nSolution - validation in function:")
println("psi_3(z, c=2) = ", psi_3(z, c=2))

println("\nTry uncommenting these to see the errors:")
println("# psi_3(z, c=[1,1,1,10])  # Error: c must be a single positive number")
println("# psi_3(z, c=-1)           # Error: c must be a single positive number")

# ============================================
# SIMPLE FUNCTION EXAMPLES
# ============================================
println("\n" * "=" ^ 50)
println("SIMPLE FUNCTION EXAMPLES")
println("=" ^ 50)

# Example 1: Simple function with no arguments
function hello()
    println("Hello, World!")
end

hello()

# Example 2: Function with one argument
square(x) = x ^ 2

println("\nSquare of 5: ", square(5))
println("Square of 10: ", square(10))

# Example 3: Function with multiple arguments
function add(a, b)
    result = a + b
    return result
end

println("\n3 + 7 = ", add(3, 7))

# Example 4: Function with default arguments
function greet(name="User"; greeting="Hello")
    message = "$greeting $name!"
    return message
end

println("\n", greet())
println(greet("Alice"))
println(greet("Bob", greeting="Good morning"))

# ============================================
# FUNCTION SCOPE
# ============================================
println("\n" * "=" ^ 50)
println("WHAT THE FUNCTION CAN SEE AND DO")
println("=" ^ 50)
println("Key principles:")
println("- Each function has its own scope (local namespace)")
println("- Names here over-ride names in the global scope")
println("- Local scope starts with the named arguments")
println("- Assignments inside the function only change the local scope")
println("- Names undefined in the function are looked for in enclosing scopes")

# Example: Variable scope
x = 10

function test_scope()
    x = 5  # Local variable
    println("Inside function: x = ", x)
    return x
end

println("\nBefore function call: x = ", x)
result = test_scope()
println("After function call: x = ", x)
println("Function returned: ", result)

# ============================================
# INTERNAL ENVIRONMENT EXAMPLES
# ============================================
println("\n" * "=" ^ 50)
println("INTERNAL ENVIRONMENT EXAMPLES")
println("=" ^ 50)

# Example 1: Function creates its own local x
x_global = 7
y_global = ["A", "C", "G", "T", "U"]

function adder(y)
    x_local = x_global + y  # Uses global x_global, creates local x_local
    return x_local
end

println("\nBefore adder: x_global = ", x_global)
println("adder(1) = ", adder(1))
println("After adder: x_global = ", x_global)
println("y_global = ", y_global)

# Example 2: Function uses global constant
function circle_area(r)
    return π .* r.^2
end

println("\nWith built-in π:")
println("circle_area([1,2,3]) = ", circle_area([1,2,3]))

# Note: In Julia, π is a constant and cannot be reassigned like in Python

# ============================================
# RESPECT THE INTERFACES
# ============================================
println("\n" * "=" ^ 50)
println("RESPECT THE INTERFACES")
println("=" ^ 50)
println("Interfaces mark out a controlled inner environment for our code")
println("Interact with the rest of the system only at the interface")
println()
println("Advice: arguments explicitly give the function all the information")
println("  - Reduces risk of confusion and error")
println("  - Exception: true universals like π")
println()
println("Likewise, output should only be through the return value")

# ============================================
# FITTING A MODEL
# ============================================
println("\n" * "=" ^ 50)
println("FITTING A MODEL")
println("=" ^ 50)
println("Fact: bigger cities tend to produce more economically per capita")
println()
println("A proposed statistical model (Geoffrey West et al.):")
println("Y = y0 * N^a + noise")
println("where Y is the per-capita gross metropolitan product of a city,")
println("N is its population, and y0 and a are parameters")

# Create sample GMP data
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

println("\nData created: gmp dataset with $(nrow(gmp)) cities")
println("(Simulated data based on the power-law model)")

# Plot the data with initial guess
scatter(gmp.pop, gmp.pcgmp, alpha=0.5, label="Data", 
        xscale=:log10, xlabel="Population", 
        ylabel="Per-Capita Economic Output (\$/person-year)",
        title="US Metropolitan Areas (Simulated)",
        markersize=3)
x_range = 10 .^ range(4.5, 7.5, length=100)
plot!(x_range, 6611 .* x_range.^(1/8), linewidth=2, 
      label="Initial guess (a=1/8)", legend=:topleft)
savefig("gmp_initial_julia.png")

println("Plot created with initial curve: y = 6611 * pop^(1/8)")

# ============================================
# OPTIMIZATION APPROACH
# ============================================
println("\n" * "=" ^ 50)
println("OPTIMIZATION APPROACH")
println("=" ^ 50)
println("Take y0 = 6611 for now and estimate a by minimizing mean squared error")
println("Approximate the derivative of error w.r.t a and move against it")
println()
println("MSE(a) = (1/n) * sum((Y_i - y0*N_i^a)^2)")
println("MSE'(a) ≈ (MSE(a+h) - MSE(a)) / h")
println("a_{t+1} = a_t - step_scale * MSE'(a)")

# ============================================
# VERSION 1: BASIC FUNCTION
# ============================================
println("\n" * "=" ^ 50)
println("VERSION 1: Basic Function")
println("=" ^ 50)

function estimate_scaling_exponent_1(a, gmp_data)
    maximum_iterations = 100
    deriv_step = 1/1000
    step_scale = 1e-12
    stopping_deriv = 1/100
    iteration = 0
    deriv = Inf
    
    while (iteration < maximum_iterations) && (abs(deriv) > stopping_deriv)
        iteration += 1
        mse_1 = mean((gmp_data.pcgmp .- 6611 .* gmp_data.pop.^a).^2)
        mse_2 = mean((gmp_data.pcgmp .- 6611 .* gmp_data.pop.^(a+deriv_step)).^2)
        deriv = (mse_2 - mse_1) / deriv_step
        a = a - step_scale * deriv
    end
    
    fit = Dict(
        :a => a,
        :iterations => iteration,
        :converged => iteration < maximum_iterations
    )
    return fit
end

result1 = estimate_scaling_exponent_1(0.15, gmp)
println("estimate_scaling_exponent_1(0.15):")
println("  a = ", result1[:a])
println("  iterations = ", result1[:iterations])
println("  converged = ", result1[:converged])

# ============================================
# VERSION 2: MAGIC NUMBERS → DEFAULTS
# ============================================
println("\n" * "=" ^ 50)
println("VERSION 2: Magic Numbers → Defaults")
println("=" ^ 50)

function estimate_scaling_exponent_2(a; y0=6611, gmp_data=gmp,
                                     maximum_iterations=100, deriv_step=0.001,
                                     step_scale=1e-12, stopping_deriv=0.01)
    iteration = 0
    deriv = Inf
    
    while (iteration < maximum_iterations) && (abs(deriv) > stopping_deriv)
        iteration += 1
        mse_1 = mean((gmp_data.pcgmp .- y0 .* gmp_data.pop.^a).^2)
        mse_2 = mean((gmp_data.pcgmp .- y0 .* gmp_data.pop.^(a+deriv_step)).^2)
        deriv = (mse_2 - mse_1) / deriv_step
        a = a - step_scale * deriv
    end
    
    fit = Dict(
        :a => a,
        :y0 => y0,
        :iterations => iteration,
        :converged => iteration < maximum_iterations
    )
    return fit
end

result2 = estimate_scaling_exponent_2(0.15)
println("estimate_scaling_exponent_2(0.15):")
println("  a = ", result2[:a])
println("  iterations = ", result2[:iterations])

# ============================================
# VERSION 3: DECLARE HELPER FUNCTION
# ============================================
println("\n" * "=" ^ 50)
println("VERSION 3: Declare Helper Function")
println("=" ^ 50)

function estimate_scaling_exponent_3(a; y0=6611, gmp_data=gmp,
                                     maximum_iterations=100, deriv_step=0.001,
                                     step_scale=1e-12, stopping_deriv=0.01)
    iteration = 0
    deriv = Inf
    
    function mse(a_val)
        return mean((gmp_data.pcgmp .- y0 .* gmp_data.pop.^a_val).^2)
    end
    
    while (iteration < maximum_iterations) && (abs(deriv) > stopping_deriv)
        iteration += 1
        deriv = (mse(a+deriv_step) - mse(a)) / deriv_step
        a = a - step_scale * deriv
    end
    
    fit = Dict(
        :a => a,
        :y0 => y0,
        :iterations => iteration,
        :converged => iteration < maximum_iterations
    )
    return fit
end

println("mse() declared inside the function:")
println("  - Can see y0 and gmp_data")
println("  - Not added to global namespace")

result3 = estimate_scaling_exponent_3(0.15)
println("\nestimate_scaling_exponent_3(0.15):")
println("  a = ", result3[:a])

# ============================================
# VERSION 4: FLEXIBLE DATA INPUTS
# ============================================
println("\n" * "=" ^ 50)
println("VERSION 4: Flexible Data Inputs")
println("=" ^ 50)

function estimate_scaling_exponent_4(a; y0=6611, response=gmp.pcgmp, predictor=gmp.pop,
                                     maximum_iterations=100, deriv_step=0.001,
                                     step_scale=1e-12, stopping_deriv=0.01)
    iteration = 0
    deriv = Inf
    
    function mse(a_val)
        return mean((response .- y0 .* predictor.^a_val).^2)
    end
    
    while (iteration < maximum_iterations) && (abs(deriv) > stopping_deriv)
        iteration += 1
        deriv = (mse(a+deriv_step) - mse(a)) / deriv_step
        a = a - step_scale * deriv
    end
    
    fit = Dict(
        :a => a,
        :y0 => y0,
        :iterations => iteration,
        :converged => iteration < maximum_iterations
    )
    return fit
end

result4 = estimate_scaling_exponent_4(0.15)
println("estimate_scaling_exponent_4(0.15):")
println("  a = ", result4[:a])

# ============================================
# VERSION 5: while() → for() Loop
# ============================================
println("\n" * "=" ^ 50)
println("VERSION 5: while() → for() Loop")
println("=" ^ 50)

function estimate_scaling_exponent_5(a; y0=6611, response=gmp.pcgmp, predictor=gmp.pop,
                                     maximum_iterations=100, deriv_step=0.001,
                                     step_scale=1e-12, stopping_deriv=0.01)
    function mse(a_val)
        return mean((response .- y0 .* predictor.^a_val).^2)
    end
    
    iteration = 0
    for iter in 1:maximum_iterations
        iteration = iter
        deriv = (mse(a+deriv_step) - mse(a)) / deriv_step
        a = a - step_scale * deriv
        if abs(deriv) <= stopping_deriv
            break
        end
    end
    
    fit = Dict(
        :a => a,
        :y0 => y0,
        :iterations => iteration,
        :converged => iteration < maximum_iterations
    )
    return fit
end

result5 = estimate_scaling_exponent_5(0.15)
println("estimate_scaling_exponent_5(0.15):")
println("  a = ", result5[:a])
println("  iterations = ", result5[:iterations])

# ============================================
# HOW WE EXTEND FUNCTIONS
# ============================================
println("\n" * "=" ^ 50)
println("HOW WE EXTEND FUNCTIONS")
println("=" ^ 50)
println("Two main approaches:")
println("1. Multiple functions: Doing different things to the same object")
println("2. Sub-functions: Breaking up big jobs into small ones")

# ============================================
# PREDICTING FROM A FITTED MODEL
# ============================================
println("\n" * "=" ^ 50)
println("PREDICTING FROM A FITTED MODEL")
println("=" ^ 50)

function predict_plm(model, newdata)
    """
    Predict response values from a power-law scaling model
    
    Parameters:
    - model: dict with :a and :y0 keys
    - newdata: array of predictor values
    
    Returns:
    - array of predicted values
    """
    if !haskey(model, :a) || !haskey(model, :y0)
        error("model must contain :a and :y0 keys")
    end
    
    a = model[:a]
    y0 = model[:y0]
    
    return y0 .* newdata.^a
end

# Test prediction function
test_cities = [1e5, 5e5, 1e6, 5e6]
predictions = predict_plm(result5, test_cities)
println("\nPredictions for cities of size ", test_cities)
println("Per-capita output: ", predictions)

# ============================================
# PLOTTING FUNCTIONS
# ============================================
println("\n" * "=" ^ 50)
println("PLOTTING FUNCTIONS")
println("=" ^ 50)

function plot_plm!(model, from_val, to_val; n=101, kwargs...)
    """Plot fitted curve from power law model"""
    x = range(from_val, to_val, length=n)
    y = predict_plm(model, x)
    plot!(x, y; kwargs...)
end

# Create demonstration plot
scatter(gmp.pop, gmp.pcgmp, alpha=0.5, markersize=3, 
        color=:red, label="Data", xscale=:log10,
        xlabel="Population", ylabel="Per-Capita Output",
        title="Power-Law Model Fit")
x_range = 10 .^ range(4.5, 7.5, length=100)
plot!(x_range, 6611 .* x_range.^(1/8), linewidth=2, 
      linestyle=:dash, color=:blue, label="Initial (a=1/8)")
plot_plm!(result5, 10^4.5, 10^7.5, linewidth=2, 
          color=:green, label="Estimated (a=$(round(result5[:a], digits=4)))")
savefig("gmp_fitted_julia.png")

println("Created plot using plot_plm!()")

# ============================================
# RECURSION
# ============================================
println("\n" * "=" ^ 50)
println("RECURSION")
println("=" ^ 50)

# Factorial
function factorial_rec(n)
    if n <= 1
        return 1
    end
    return n * factorial_rec(n - 1)
end

println("\nFactorial:")
for i in 0:6
    println("$i! = ", factorial_rec(i))
end

# Fibonacci
function fibonacci(n)
    if n == 0
        return 0
    end
    if n == 1
        return 1
    end
    return fibonacci(n - 1) + fibonacci(n - 2)
end

println("\nFibonacci sequence:")
for i in 0:10
    println("F($i) = ", fibonacci(i))
end

# GCD
function gcd_rec(a, b)
    if b == 0
        return abs(a)
    end
    return gcd_rec(b, a % b)
end

println("\nGreatest Common Divisor:")
println("GCD(48, 18) = ", gcd_rec(48, 18))
println("GCD(100, 35) = ", gcd_rec(100, 35))
println("GCD(17, 19) = ", gcd_rec(17, 19))

# Binary search
function binary_search(vec, target, low=1, high=length(vec))
    if low > high
        return -1
    end
    
    mid = div(low + high, 2)
    
    if vec[mid] == target
        return mid
    elseif vec[mid] > target
        return binary_search(vec, target, low, mid - 1)
    else
        return binary_search(vec, target, mid + 1, high)
    end
end

sorted_vec = [2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78]
println("\nBinary search in: ", sorted_vec)
println("Index of 23: ", binary_search(sorted_vec, 23))
println("Index of 67: ", binary_search(sorted_vec, 67))
println("Index of 100 (not found): ", binary_search(sorted_vec, 100))

# ============================================
# MEMOIZATION
# ============================================
println("\n" * "=" ^ 50)
println("MEMOIZATION")
println("=" ^ 50)

# Fibonacci with memoization
const fib_cache = Dict{Int, Int}()

function fibonacci_memo(n)
    if haskey(fib_cache, n)
        return fib_cache[n]
    end
    
    if n == 0
        return 0
    end
    if n == 1
        return 1
    end
    
    result = fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
    fib_cache[n] = result
    return result
end

println("\nFibonacci with memoization (much faster for large n):")

# Time comparison
println("\nWithout memoization (n=30):")
start_time = time()
result1 = fibonacci(30)
end_time = time()
println("Result: ", result1)
println("Time: ", round(end_time - start_time, digits=4), " seconds")

println("\nWith memoization (n=30):")
start_time = time()
result2 = fibonacci_memo(30)
end_time = time()
println("Result: ", result2)
println("Time: ", round(end_time - start_time, digits=4), " seconds")

println("\nFibonacci(100) with memoization: ", fibonacci_memo(100))

# ============================================
# SUMMARY
# ============================================
println("\n" * "=" ^ 50)
println("SUMMARY")
println("=" ^ 50)
println("1. Functions bundle related commands together into objects:")
println("   - Easier to re-run, re-use, combine, and modify")
println("   - Less risk of error, easier to think about")
println()
println("2. Interfaces control what the function can see and change:")
println("   - Arguments and scope (what it can see)")
println("   - Its internals and return value (what it can change)")
println()
println("3. Calling functions we define works like built-in functions:")
println("   - Named arguments, defaults")
println()
println("4. Multiple functions let us do multiple related jobs:")
println("   - Either on the same object or on similar ones")
println()
println("5. Sub-functions let us break big problems into smaller ones:")
println("   - Re-use the solutions to the smaller ones")
println()
println("6. Recursion is a powerful way of making hard problems simpler")

println("\n" * "=" ^ 50)
println("Writing Functions Tutorial Complete")
println("=" ^ 50)
