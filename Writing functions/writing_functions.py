"""
============================================
WRITING FUNCTIONS IN PYTHON
============================================

AGENDA:
- Defining functions: Tying related commands into bundles
- Interfaces: Controlling what the function can see and do
- Example: Parameter estimation code
- Multiple functions
- Recursion: Making hard problems simpler
============================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import time

# ============================================
# WHY FUNCTIONS?
# ============================================
print("=" * 50)
print("WHY FUNCTIONS?")
print("=" * 50)
print("Data structures tie related values into one object")
print("Functions tie related commands into one object")
print()
print("In both cases: easier to understand, easier to work with,")
print("easier to build into larger things")
print()
print("Benefits of functions:")
print("1. Organize code into reusable chunks")
print("2. Avoid repetition (DRY principle: Don't Repeat Yourself)")
print("3. Make code more readable and maintainable")
print("4. Abstract complex operations into simple interfaces")
print("5. Test and debug code more easily")
print("6. Build complex programs from simple, well-tested components")

# ============================================
# DEFINING FUNCTIONS
# ============================================
print("\n" + "=" * 50)
print("DEFINING FUNCTIONS")
print("=" * 50)

# Example: Cubic function
def cube(x):
    return x ** 3

# Display the function
print("\nFunction definition:")
print(f"cube = {cube}")

# Test with a single value
print(f"\ncube(3) = {cube(3)}")

# Test with an array
print(f"cube(np.arange(1, 11)) = {cube(np.arange(1, 11))}")

# Test with a matrix
matrix = np.arange(1, 9).reshape(2, 4)
print(f"\ncube(matrix(1:8, 2, 4)):")
print(cube(matrix))

print(f"\ntype(cube) = {type(cube)}")

# ============================================
# ROBUST LOSS FUNCTION EXAMPLE
# ============================================
print("\n" + "=" * 50)
print("ROBUST LOSS FUNCTION")
print("=" * 50)

# "Robust" loss function, for outlier-resistant regression
# Inputs: array of numbers (x)
# Outputs: array with x^2 for small entries, 2|x|-1 for large ones
def psi_1(x):
    x = np.asarray(x)
    psi = np.where(x**2 > 1, 2*np.abs(x)-1, x**2)
    return psi

# Our functions get used just like the built-in ones:
z = np.array([-0.5, -5, 0.9, 9])
print(f"z = {z}")
print(f"psi_1(z) = {psi_1(z)}")

# ============================================
# FUNCTION ANATOMY
# ============================================
print("\n" + "=" * 50)
print("FUNCTION ANATOMY")
print("=" * 50)
print("INTERFACES: the inputs (arguments) and outputs (return value)")
print("- Calls other functions: np.where(), np.abs(), operators ** and >")
print("- return statement says what the output is")
print("- Comments: Not required by Python, but a good idea")

# ============================================
# WHAT SHOULD BE A FUNCTION?
# ============================================
print("\n" + "=" * 50)
print("WHAT SHOULD BE A FUNCTION?")
print("=" * 50)
print("- Things you're going to re-run, especially with changes")
print("- Chunks of code you keep highlighting and running")
print("- Chunks of code which are small parts of bigger analyses")
print("- Chunks which are very similar to other chunks")

# ============================================
# NAMED AND DEFAULT ARGUMENTS
# ============================================
print("\n" + "=" * 50)
print("NAMED AND DEFAULT ARGUMENTS")
print("=" * 50)

# "Robust" loss function with adjustable crossover scale
def psi_2(x, c=1):
    x = np.asarray(x)
    psi = np.where(x**2 > c**2, 2*c*np.abs(x)-c**2, x**2)
    return psi

# Default value c=1 makes psi_2 equivalent to psi_1
print(f"np.array_equal(psi_1(z), psi_2(z, c=1)): {np.array_equal(psi_1(z), psi_2(z, c=1))}")

# Default values get used if names are missing:
print(f"np.array_equal(psi_2(z, c=1), psi_2(z)): {np.array_equal(psi_2(z, c=1), psi_2(z))}")

# Named arguments can go in any order when explicitly tagged:
print(f"np.array_equal(psi_2(x=z, c=2), psi_2(c=2, x=z)): {np.array_equal(psi_2(x=z, c=2), psi_2(c=2, x=z))}")

# ============================================
# CHECKING ARGUMENTS
# ============================================
print("\n" + "=" * 50)
print("CHECKING ARGUMENTS")
print("=" * 50)

# Problem: Odd behavior when arguments aren't as we expect
print("\nProblem - array c causes element-wise comparison:")
print(f"psi_2(x=z, c=np.array([1,1,1,10])) = {psi_2(x=z, c=np.array([1,1,1,10]))}")

print("\nProblem - negative c gives nonsensical results:")
print(f"psi_2(x=z, c=-1) = {psi_2(x=z, c=-1)}")

# Solution: Put little sanity checks into the code
def psi_3(x, c=1):
    # Scale should be a single positive number
    if not isinstance(c, (int, float)) or c <= 0:
        raise ValueError("c must be a single positive number")
    x = np.asarray(x)
    psi = np.where(x**2 > c**2, 2*c*np.abs(x)-c**2, x**2)
    return psi

print("\nSolution - validation in function:")
print(f"psi_3(z, c=2) = {psi_3(z, c=2)}")

print("\nTry uncommenting these to see the errors:")
print("# psi_3(z, c=np.array([1,1,1,10]))  # Error: c must be a single positive number")
print("# psi_3(z, c=-1)                     # Error: c must be a single positive number")

# ============================================
# SIMPLE FUNCTION EXAMPLES
# ============================================
print("\n" + "=" * 50)
print("SIMPLE FUNCTION EXAMPLES")
print("=" * 50)

# Example 1: Simple function with no arguments
def hello():
    print("Hello, World!")

hello()

# Example 2: Function with one argument
def square(x):
    return x ** 2

print(f"\nSquare of 5: {square(5)}")
print(f"Square of 10: {square(10)}")

# Example 3: Function with multiple arguments
def add(a, b):
    result = a + b
    return result

print(f"\n3 + 7 = {add(3, 7)}")

# Example 4: Function with default arguments
def greet(name="User", greeting="Hello"):
    message = f"{greeting} {name}!"
    return message

print(f"\n{greet()}")
print(greet("Alice"))
print(greet("Bob", "Good morning"))

# ============================================
# FUNCTION SCOPE
# ============================================
print("\n" + "=" * 50)
print("WHAT THE FUNCTION CAN SEE AND DO")
print("=" * 50)
print("Key principles:")
print("- Each function has its own scope (local namespace)")
print("- Names here over-ride names in the global scope")
print("- Local scope starts with the named arguments")
print("- Assignments inside the function only change the local scope")
print("- Names undefined in the function are looked for in enclosing scopes")

# Example: Variable scope
x = 10

def test_scope():
    x = 5  # Local variable
    print(f"Inside function: x = {x}")
    return x

print(f"\nBefore function call: x = {x}")
result = test_scope()
print(f"After function call: x = {x}")
print(f"Function returned: {result}")

# ============================================
# INTERNAL ENVIRONMENT EXAMPLES
# ============================================
print("\n" + "=" * 50)
print("INTERNAL ENVIRONMENT EXAMPLES")
print("=" * 50)

# Example 1: Function creates its own local x
x = 7
y = ["A", "C", "G", "T", "U"]

def adder(y):
    x_local = x + y  # Uses global x, creates local x_local
    return x_local

print(f"\nBefore adder: x = {x}")
print(f"adder(1) = {adder(1)}")
print(f"After adder: x = {x}")
print(f"y = {y}")

# Example 2: Function uses global constant
def circle_area(r):
    return np.pi * r**2

print(f"\nWith built-in pi:")
print(f"circle_area(np.array([1,2,3])) = {circle_area(np.array([1,2,3]))}")

# Override pi in global scope (not recommended!)
true_pi = np.pi
np.pi = 3

print(f"\nWith pi = 3:")
print(f"circle_area(np.array([1,2,3])) = {circle_area(np.array([1,2,3]))}")

# Restore sanity
np.pi = true_pi

print(f"\nWith pi restored:")
print(f"circle_area(np.array([1,2,3])) = {circle_area(np.array([1,2,3]))}")

# ============================================
# RESPECT THE INTERFACES
# ============================================
print("\n" + "=" * 50)
print("RESPECT THE INTERFACES")
print("=" * 50)
print("Interfaces mark out a controlled inner environment for our code")
print("Interact with the rest of the system only at the interface")
print()
print("Advice: arguments explicitly give the function all the information")
print("  - Reduces risk of confusion and error")
print("  - Exception: true universals like π")
print()
print("Likewise, output should only be through the return value")

# ============================================
# FITTING A MODEL
# ============================================
print("\n" + "=" * 50)
print("FITTING A MODEL")
print("=" * 50)
print("Fact: bigger cities tend to produce more economically per capita")
print()
print("A proposed statistical model (Geoffrey West et al.):")
print("Y = y0 * N^a + noise")
print("where Y is the per-capita gross metropolitan product of a city,")
print("N is its population, and y0 and a are parameters")

# Create sample GMP data
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

print(f"\nData created: gmp dataset with {len(gmp)} cities")
print("(Simulated data based on the power-law model)")

# Plot the data with initial guess
plt.figure(figsize=(10, 6))
plt.scatter(gmp['pop'], gmp['pcgmp'], alpha=0.5)
plt.xscale('log')
x_range = np.logspace(4.5, 7.5, 100)
plt.plot(x_range, 6611 * x_range**(1/8), 'b-', linewidth=2, label='Initial guess (a=1/8)')
plt.xlabel('Population')
plt.ylabel('Per-Capita Economic Output ($/person-year)')
plt.title('US Metropolitan Areas (Simulated)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gmp_initial.png', dpi=100, bbox_inches='tight')
plt.close()

print("Plot created with initial curve: y = 6611 * pop^(1/8)")

# ============================================
# OPTIMIZATION APPROACH
# ============================================
print("\n" + "=" * 50)
print("OPTIMIZATION APPROACH")
print("=" * 50)
print("Take y0 = 6611 for now and estimate a by minimizing mean squared error")
print("Approximate the derivative of error w.r.t a and move against it")
print()
print("MSE(a) = (1/n) * sum((Y_i - y0*N_i^a)^2)")
print("MSE'(a) ≈ (MSE(a+h) - MSE(a)) / h")
print("a_{t+1} = a_t - step_scale * MSE'(a)")

# ============================================
# VERSION 1: BASIC FUNCTION
# ============================================
print("\n" + "=" * 50)
print("VERSION 1: Basic Function")
print("=" * 50)

def estimate_scaling_exponent_1(a, gmp_data):
    maximum_iterations = 100
    deriv_step = 1/1000
    step_scale = 1e-12
    stopping_deriv = 1/100
    iteration = 0
    deriv = np.inf
    
    while (iteration < maximum_iterations) and (abs(deriv) > stopping_deriv):
        iteration += 1
        mse_1 = np.mean((gmp_data['pcgmp'] - 6611*gmp_data['pop']**a)**2)
        mse_2 = np.mean((gmp_data['pcgmp'] - 6611*gmp_data['pop']**(a+deriv_step))**2)
        deriv = (mse_2 - mse_1) / deriv_step
        a = a - step_scale * deriv
    
    fit = {
        'a': a,
        'iterations': iteration,
        'converged': iteration < maximum_iterations
    }
    return fit

result1 = estimate_scaling_exponent_1(0.15, gmp)
print(f"estimate_scaling_exponent_1(0.15):")
print(f"  a = {result1['a']}")
print(f"  iterations = {result1['iterations']}")
print(f"  converged = {result1['converged']}")

# ============================================
# VERSION 2: MAGIC NUMBERS → DEFAULTS
# ============================================
print("\n" + "=" * 50)
print("VERSION 2: Magic Numbers → Defaults")
print("=" * 50)

def estimate_scaling_exponent_2(a, y0=6611, gmp_data=None,
                                maximum_iterations=100, deriv_step=0.001,
                                step_scale=1e-12, stopping_deriv=0.01):
    if gmp_data is None:
        gmp_data = gmp
    
    iteration = 0
    deriv = np.inf
    
    while (iteration < maximum_iterations) and (abs(deriv) > stopping_deriv):
        iteration += 1
        mse_1 = np.mean((gmp_data['pcgmp'] - y0*gmp_data['pop']**a)**2)
        mse_2 = np.mean((gmp_data['pcgmp'] - y0*gmp_data['pop']**(a+deriv_step))**2)
        deriv = (mse_2 - mse_1) / deriv_step
        a = a - step_scale * deriv
    
    fit = {
        'a': a,
        'y0': y0,
        'iterations': iteration,
        'converged': iteration < maximum_iterations
    }
    return fit

result2 = estimate_scaling_exponent_2(0.15)
print(f"estimate_scaling_exponent_2(0.15):")
print(f"  a = {result2['a']}")
print(f"  iterations = {result2['iterations']}")

# ============================================
# VERSION 3: DECLARE HELPER FUNCTION
# ============================================
print("\n" + "=" * 50)
print("VERSION 3: Declare Helper Function")
print("=" * 50)

def estimate_scaling_exponent_3(a, y0=6611, gmp_data=None,
                                maximum_iterations=100, deriv_step=0.001,
                                step_scale=1e-12, stopping_deriv=0.01):
    if gmp_data is None:
        gmp_data = gmp
    
    iteration = 0
    deriv = np.inf
    
    def mse(a_val):
        return np.mean((gmp_data['pcgmp'] - y0*gmp_data['pop']**a_val)**2)
    
    while (iteration < maximum_iterations) and (abs(deriv) > stopping_deriv):
        iteration += 1
        deriv = (mse(a+deriv_step) - mse(a)) / deriv_step
        a = a - step_scale * deriv
    
    fit = {
        'a': a,
        'y0': y0,
        'iterations': iteration,
        'converged': iteration < maximum_iterations
    }
    return fit

print("mse() declared inside the function:")
print("  - Can see y0 and gmp_data")
print("  - Not added to global namespace")

result3 = estimate_scaling_exponent_3(0.15)
print(f"\nestimate_scaling_exponent_3(0.15):")
print(f"  a = {result3['a']}")

# ============================================
# VERSION 4: FLEXIBLE DATA INPUTS
# ============================================
print("\n" + "=" * 50)
print("VERSION 4: Flexible Data Inputs")
print("=" * 50)

def estimate_scaling_exponent_4(a, y0=6611, response=None, predictor=None,
                                maximum_iterations=100, deriv_step=0.001,
                                step_scale=1e-12, stopping_deriv=0.01):
    if response is None:
        response = gmp['pcgmp'].values
    if predictor is None:
        predictor = gmp['pop'].values
    
    iteration = 0
    deriv = np.inf
    
    def mse(a_val):
        return np.mean((response - y0*predictor**a_val)**2)
    
    while (iteration < maximum_iterations) and (abs(deriv) > stopping_deriv):
        iteration += 1
        deriv = (mse(a+deriv_step) - mse(a)) / deriv_step
        a = a - step_scale * deriv
    
    fit = {
        'a': a,
        'y0': y0,
        'iterations': iteration,
        'converged': iteration < maximum_iterations
    }
    return fit

result4 = estimate_scaling_exponent_4(0.15)
print(f"estimate_scaling_exponent_4(0.15):")
print(f"  a = {result4['a']}")

# ============================================
# VERSION 5: while() → for() Loop
# ============================================
print("\n" + "=" * 50)
print("VERSION 5: while() → for() Loop")
print("=" * 50)

def estimate_scaling_exponent_5(a, y0=6611, response=None, predictor=None,
                                maximum_iterations=100, deriv_step=0.001,
                                step_scale=1e-12, stopping_deriv=0.01):
    if response is None:
        response = gmp['pcgmp'].values
    if predictor is None:
        predictor = gmp['pop'].values
    
    def mse(a_val):
        return np.mean((response - y0*predictor**a_val)**2)
    
    for iteration in range(1, maximum_iterations + 1):
        deriv = (mse(a+deriv_step) - mse(a)) / deriv_step
        a = a - step_scale * deriv
        if abs(deriv) <= stopping_deriv:
            break
    
    fit = {
        'a': a,
        'y0': y0,
        'iterations': iteration,
        'converged': iteration < maximum_iterations
    }
    return fit

result5 = estimate_scaling_exponent_5(0.15)
print(f"estimate_scaling_exponent_5(0.15):")
print(f"  a = {result5['a']}")
print(f"  iterations = {result5['iterations']}")

# ============================================
# HOW WE EXTEND FUNCTIONS
# ============================================
print("\n" + "=" * 50)
print("HOW WE EXTEND FUNCTIONS")
print("=" * 50)
print("Two main approaches:")
print("1. Multiple functions: Doing different things to the same object")
print("2. Sub-functions: Breaking up big jobs into small ones")

# ============================================
# PREDICTING FROM A FITTED MODEL
# ============================================
print("\n" + "=" * 50)
print("PREDICTING FROM A FITTED MODEL")
print("=" * 50)

def predict_plm(model, newdata):
    """
    Predict response values from a power-law scaling model
    
    Parameters:
    - model: dict with 'a' and 'y0' keys
    - newdata: array of predictor values
    
    Returns:
    - array of predicted values
    """
    if 'a' not in model or 'y0' not in model:
        raise ValueError("model must contain 'a' and 'y0' keys")
    
    a = model['a']
    y0 = model['y0']
    
    newdata = np.asarray(newdata)
    return y0 * newdata**a

# Test prediction function
test_cities = np.array([1e5, 5e5, 1e6, 5e6])
predictions = predict_plm(result5, test_cities)
print(f"\nPredictions for cities of size {test_cities}:")
print(f"Per-capita output: {predictions}")

# ============================================
# PLOTTING FUNCTIONS
# ============================================
print("\n" + "=" * 50)
print("PLOTTING FUNCTIONS")
print("=" * 50)

def plot_plm(model, from_val, to_val, n=101, **kwargs):
    """Plot fitted curve from power law model"""
    x = np.linspace(from_val, to_val, n)
    y = predict_plm(model, x)
    plt.plot(x, y, **kwargs)

# Create demonstration plot
plt.figure(figsize=(10, 6))
plt.scatter(gmp['pop'], gmp['pcgmp'], alpha=0.5, s=20, c='red', label='Data')
plt.xscale('log')
x_range = np.logspace(4.5, 7.5, 100)
plt.plot(x_range, 6611 * x_range**(1/8), 'b--', linewidth=2, label='Initial (a=1/8)')
plot_plm(result5, 10**4.5, 10**7.5, linewidth=2, color='green', label=f"Estimated (a={result5['a']:.4f})")
plt.xlabel('Population')
plt.ylabel('Per-Capita Output')
plt.title('Power-Law Model Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gmp_fitted.png', dpi=100, bbox_inches='tight')
plt.close()

print("Created plot using plot_plm()")

# ============================================
# RECURSION
# ============================================
print("\n" + "=" * 50)
print("RECURSION")
print("=" * 50)

# Factorial
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print("\nFactorial:")
for i in range(7):
    print(f"{i}! = {factorial(i)}")

# Fibonacci
def fibonacci(n):
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

print("\nFibonacci sequence:")
for i in range(11):
    print(f"F({i}) = {fibonacci(i)}")

# GCD
def gcd(a, b):
    if b == 0:
        return abs(a)
    return gcd(b, a % b)

print("\nGreatest Common Divisor:")
print(f"GCD(48, 18) = {gcd(48, 18)}")
print(f"GCD(100, 35) = {gcd(100, 35)}")
print(f"GCD(17, 19) = {gcd(17, 19)}")

# Binary search
def binary_search(vec, target, low=0, high=None):
    if high is None:
        high = len(vec) - 1
    
    if low > high:
        return -1
    
    mid = (low + high) // 2
    
    if vec[mid] == target:
        return mid
    elif vec[mid] > target:
        return binary_search(vec, target, low, mid - 1)
    else:
        return binary_search(vec, target, mid + 1, high)

sorted_vec = [2, 5, 8, 12, 16, 23, 38, 45, 56, 67, 78]
print(f"\nBinary search in: {sorted_vec}")
print(f"Index of 23: {binary_search(sorted_vec, 23)}")
print(f"Index of 67: {binary_search(sorted_vec, 67)}")
print(f"Index of 100 (not found): {binary_search(sorted_vec, 100)}")

# ============================================
# MEMOIZATION
# ============================================
print("\n" + "=" * 50)
print("MEMOIZATION")
print("=" * 50)

# Fibonacci with memoization
_fib_cache = {}

def fibonacci_memo(n):
    if n in _fib_cache:
        return _fib_cache[n]
    
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    result = fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
    _fib_cache[n] = result
    return result

print("\nFibonacci with memoization (much faster for large n):")

# Time comparison
print("\nWithout memoization (n=30):")
start = time.time()
result1 = fibonacci(30)
end = time.time()
print(f"Result: {result1}")
print(f"Time: {end - start:.4f} seconds")

print("\nWith memoization (n=30):")
start = time.time()
result2 = fibonacci_memo(30)
end = time.time()
print(f"Result: {result2}")
print(f"Time: {end - start:.4f} seconds")

print(f"\nFibonacci(100) with memoization: {fibonacci_memo(100)}")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print("1. Functions bundle related commands together into objects:")
print("   - Easier to re-run, re-use, combine, and modify")
print("   - Less risk of error, easier to think about")
print()
print("2. Interfaces control what the function can see and change:")
print("   - Arguments and scope (what it can see)")
print("   - Its internals and return value (what it can change)")
print()
print("3. Calling functions we define works like built-in functions:")
print("   - Named arguments, defaults")
print()
print("4. Multiple functions let us do multiple related jobs:")
print("   - Either on the same object or on similar ones")
print()
print("5. Sub-functions let us break big problems into smaller ones:")
print("   - Re-use the solutions to the smaller ones")
print()
print("6. Recursion is a powerful way of making hard problems simpler")

print("\n" + "=" * 50)
print("Writing Functions Tutorial Complete")
print("=" * 50)
