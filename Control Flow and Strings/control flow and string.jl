# ============================================
# CONTROL FLOW AND STRINGS
# ============================================
# 
# AGENDA:
# - Control flow (or alternatively, flow of control)
# - if(), for(), and while()
# - Avoiding iteration
# - Introduction to strings and string operations
# ============================================

using Statistics
using Plots
using Printf

# ============================================
# CONDITIONALS
# ============================================
# Have the computer decide what to do next
#
# Mathematically:
# |x| = { x    if x >= 0
#       { -x   if x < 0
#
# ψ(x) = { x^2      if |x| <= 1
#        { 2|x| - 1 if |x| > 1
#
# Exercise: plot ψ in Julia

# Define the piecewise function ψ(x)
function psi(x)
    if abs(x) <= 1
        return x^2
    else
        return 2*abs(x) - 1
    end
end

# Create a sequence of x values
x_values = -3:0.01:3

# Apply the function to each x value
y_values = [psi(x) for x in x_values]

# Create plots directory if it doesn't exist
if !isdir("plots")
    mkdir("plots")
end

# Plot the function
p = plot(x_values, y_values, linewidth=2, label="ψ(x)", color=:blue,
         title="Plot of psi(x)", xlabel="x", ylabel="psi(x)",
         grid=true, legend=:top)

# Add vertical lines at x = -1 and x = 1 to show where the function changes
vline!([-1, 1], linestyle=:dash, color=:red, label="")

# Save the plot
savefig(p, "plots/psi_function_plot.png")

println("Plot saved to plots/psi_function_plot.png")

# Computationally:
# if the country code is not "US", multiply prices by current exchange rate

# Example implementation
function adjust_price(price, country_code, exchange_rate=1.0)
    if country_code != "US"
        return price * exchange_rate
    else
        return price
    end
end

# Test the function
println(adjust_price(100, "US", 1.2))       # Returns 100 (no change)
println(adjust_price(100, "CA", 1.35))      # Returns 135 (converted)
println(adjust_price(100, "UK", 0.79))      # Returns 79 (converted)

# ============================================
# if() EXAMPLES
# ============================================
# Example: Absolute value using if()
x = -5  # Define x for demonstration
if x >= 0
    println(x)
else
    println(-x)
end

# Example: Piecewise function using nested if()
x = 1.5  # Define x for demonstration
if x^2 < 1
    println(x^2)
else
    if x >= 0
        println(2*x-1)
    else
        println(-2*x-1)
    end
end

# ============================================
# COMBINING BOOLEANS
# ============================================
# & and | work element-wise on arrays
# && and || provide short-circuit evaluation for scalars

# Example: && stops evaluation if first condition is false
println((0 > 0) && (42%6 == 169%13))
# Since (0 > 0) is false, the second part is never evaluated
# This is "lazy" or "short-circuit" evaluation

# ============================================
# ITERATION
# ============================================
# Repeat similar actions multiple times

# Example: Create a table of logarithms using a for loop
table_of_logarithms = zeros(7)
println(table_of_logarithms)

for i in 1:length(table_of_logarithms)
    table_of_logarithms[i] = log(i)
end
println(table_of_logarithms)

# ============================================
# for() LOOPS
# ============================================
# for executes a block of code repeatedly for each element in a sequence
# Syntax: for variable in sequence code block end
# The loop variable takes on each value in the sequence, one at a time
#
# for increments a counter (here i) along a range
# and loops through the **body** until it runs through the range
#
# "iterates over the range"
#
# Note, there is a better way to do this job!
#
# Can contain just about anything, including:
# - if clauses
# - other for loops (nested iteration)

# Example: Basic for loop filling a vector
for i in 1:length(table_of_logarithms)
    table_of_logarithms[i] = log(i)
end

# Example: Nested iteration - Matrix multiplication
# First, create two matrices for demonstration
a = [1 3 5; 2 4 6]  # 2x3 matrix
b = [7 10; 8 11; 9 12]  # 3x2 matrix

c = zeros(size(a, 1), size(b, 2))
if size(a, 2) == size(b, 1)
    for i in 1:size(c, 1)
        for j in 1:size(c, 2)
            for k in 1:size(a, 2)
                c[i,j] += a[i,k] * b[k,j]
            end
        end
    end
else
    error("matrices a and b non-conformable")
end
println("Matrix multiplication result:")
println(c)

# ============================================
# while() LOOPS
# ============================================
# Condition in the argument to while must be a single Boolean value (like if)
#
# Body is looped over until the condition is false so can loop forever
#
# Loop never begins unless the condition starts true

# Example: Repeatedly take square root until convergence
x = [5.0, 10.0, 15.0]  # Initialize with some values
println("Initial x:")
println(x)

while maximum(x) - 1 > 1e-06
    global x = sqrt.(x)
end

println("Final x after convergence:")
println(x)

# ============================================
# VECTORIZED ARITHMETIC
# ============================================
# How many languages add 2 vectors:

# Example: Adding two vectors element by element using a for loop
a = [1, 2, 3, 4, 5]
b = [10, 20, 30, 40, 50]

c = zeros(Int, length(a))
for i in 1:length(a)
    c[i] = a[i] + b[i]
end

println("Vector addition using for loop:")
println(c)

# ============================================
# VECTORIZED CALCULATIONS
# ============================================
# Vectorized calculations
# Many functions are set up to vectorize automatically with broadcast (.)

# Example: abs() function with broadcast
println(abs.(-3:3))

# Example: log() function also vectorizes with broadcast
println(log.(1:7))

# ============================================
# VECTORIZED CONDITIONS: ifelse equivalent
# ============================================
# Julia uses ternary operator or array comprehension for vectorized conditionals
# Similar to R's ifelse(): condition ? value_if_true : value_if_false

# Example: Piecewise function using array comprehension
x = -3:0.5:3
println("x values:")
println(x)

result = [xi^2 > 1 ? 2*abs(xi)-1 : xi^2 for xi in x]
println("Vectorized conditional (x^2 > 1 ? 2*abs(x)-1 : x^2):")
println(result)

# ============================================
# WHAT IS TRUTH?
# ============================================
# 0 does NOT count as false in Julia; only the boolean false counts as false
# The booleans true and false work as expected
# Julia is stricter than R about types in conditionals
#
# Advice: Make sure control expressions are Boolean values
#
# Conversely, in arithmetic, false is 0 and true is 1

# Example: Using true/false as numbers
# Load state data (simplified version matching R's state.x77)
states_murder = [13.2, 10.0, 8.1, 8.8, 9.0, 7.9, 3.3, 5.9, 15.4, 
                 17.4, 5.3, 2.6, 10.4, 7.2, 6.0, 6.0, 9.7, 15.4,
                 2.1, 11.3, 4.4, 12.2, 2.7, 16.1, 9.0, 6.0, 4.3,
                 12.2, 2.1, 7.4, 11.4, 11.1, 13.0, 0.8, 7.3, 6.6,
                 4.9, 6.3, 3.4, 14.4, 3.8, 13.2, 12.7, 3.2, 2.2,
                 8.5, 4.0, 5.7, 2.6, 6.8]

# Calculate the proportion of states with murder rate > 7
# true becomes 1, false becomes 0, so mean gives the proportion
proportion_high_murder = mean(states_murder .> 7)
println("Proportion of states with Murder rate > 7:")
println(proportion_high_murder)

# ============================================
# switch() - Julia uses if-elseif or Dict for switch-like behavior
# ============================================
# Simplify nested if with dictionary or if-elseif chain

# Example: Different summary statistics based on type
type_of_summary = "mean"  # Try changing this to "median", "histogram", or something else

# Using if-elseif chain to simulate switch
result = if type_of_summary == "mean"
    mean(states_murder)
elseif type_of_summary == "median"
    median(states_murder)
elseif type_of_summary == "histogram"
    "histogram"
else
    "I don't understand"
end

println("Switch result:")
println(result)

# ============================================
# EXERCISE IMPLEMENTATION
# ============================================
# Set type_of_summary to, successively, "mean", "median", "histogram", and "mode", and explain what happens

# Test 1: "mean"
println("\n--- Test 1: type_of_summary = 'mean' ---")
type_of_summary = "mean"
result = if type_of_summary == "mean"
    mean(states_murder)
elseif type_of_summary == "median"
    median(states_murder)
elseif type_of_summary == "histogram"
    "histogram"
else
    "I don't understand"
end
println(result)
println("Explanation: Returns the mean of Murder rates")

# Test 2: "median"
println("\n--- Test 2: type_of_summary = 'median' ---")
type_of_summary = "median"
result = if type_of_summary == "mean"
    mean(states_murder)
elseif type_of_summary == "median"
    median(states_murder)
elseif type_of_summary == "histogram"
    "histogram"
else
    "I don't understand"
end
println(result)
println("Explanation: Returns the median of Murder rates")

# Test 3: "histogram"
println("\n--- Test 3: type_of_summary = 'histogram' ---")
type_of_summary = "histogram"
if type_of_summary == "histogram"
    # Create plots directory if it doesn't exist
    if !isdir("plots")
        mkdir("plots")
    end
    p_hist = histogram(states_murder, bins=10, xlabel="Murder Rate", 
                       ylabel="Frequency", title="Histogram of Murder Rates",
                       legend=false)
    savefig(p_hist, "plots/murder_histogram.png")
end
println("Histogram saved to plots/murder_histogram.png")
println("Explanation: Creates and displays a histogram of Murder rates")

# Test 4: "mode"
println("\n--- Test 4: type_of_summary = 'mode' ---")
type_of_summary = "mode"
result = if type_of_summary == "mean"
    mean(states_murder)
elseif type_of_summary == "median"
    median(states_murder)
elseif type_of_summary == "histogram"
    "histogram"
else
    "I don't understand"
end
println(result)
println("Explanation: 'mode' is not in the switch cases, so it returns the default message 'I don't understand'")

# ============================================
# UNCONDITIONAL ITERATION
# ============================================
# while true creates an infinite loop that continues forever unless explicitly stopped with break

# Example: Infinite loop (commented out to prevent actual infinite execution)
# WARNING: This will run forever if uncommented!
# while true
#     println("Help! I am Dr. Morris Culpepper, trapped in an endless loop!")
# end

# Practical example: Using while true with a break condition
println("\n--- Example: while true with break condition ---")
counter = 1
while true
    global counter
    println("Iteration: $counter")
    counter += 1
    if counter > 3
        println("Breaking out of the loop!")
        break  # Exit the loop when counter > 3
    end
end

# ============================================
# "MANUAL" CONTROL OVER ITERATION
# ============================================
# break exits the loop; continue skips the rest of the body and goes back into the loop
# Both work with for and while loops

# Example: Using break and continue in while true loop
println("\n--- Example: Manual control with break and continue ---")

watched = false
rescued = false
iteration_count = 0

while true
    global iteration_count += 1
    global watched, rescued
    
    # Simulate different conditions
    if iteration_count == 2
        watched = true
    end
    if iteration_count == 3
        watched = false
    end
    if iteration_count == 5
        rescued = true
    end
    
    if watched
        println("Iteration $iteration_count: Skipping (watched)")
        continue  # Skip the rest and go back to start of loop
    end
    
    println("Iteration $iteration_count: Help! I am Dr. Morris Culpepper, trapped in an endless loop!")
    
    if rescued
        println("Dr. Culpepper has been rescued!")
        break  # Exit the loop
    end
end

# ============================================
# break and continue with for and while
# ============================================

# Example: continue in a for loop - skip even numbers
println("\n--- Example: continue in for loop - skip even numbers ---")
for i in 1:5
    if i % 2 == 0
        continue  # Skip even numbers
    end
    println("Processing odd number: $i")
end

# Example: break in a while loop - stop at threshold
println("\n--- Example: break in while loop ---")
total = 0
i = 1
while i <= 10
    global total, i
    total += i
    println("i = $i, total = $total")
    if total > 15
        println("Total exceeded 15, breaking!")
        break
    end
    i += 1
end

# ============================================
# EXERCISE
# ============================================
# Exercise: how would you replace while() with while true?
#
# Answer: while condition body end
# becomes:
# while true if !condition break end; body end

# ============================================
# CHARACTERS VS. STRINGS
# ============================================
# Character: a symbol in a written language, specifically what you can enter at a keyboard:
#   letters, numerals, punctuation, space, newlines, etc.
#   Examples: 'L', 'i', 'n', 'c', 'o', 'l', 'n'
#
# String: a sequence of characters bound together
#   Example: "Lincoln"
#
# Note: Julia has a Char type for single characters and String type for sequences

println(typeof('L'))      # Char
println(typeof("L"))      # String
println(typeof("Lincoln")) # String

# ============================================
# MAKING STRINGS
# ============================================
# Use double quotes for strings, single quotes for characters
# Use length() to get the number of characters in a string
# Why do we prefer double quotes for strings?

println("Lincoln")
println(length("Lincoln"))
println("Abraham Lincoln")
println(length("Abraham Lincoln"))
println("As Lincoln never said, \"Four score and seven beers ago\"")

# ============================================
# CHARACTER DATA TYPE
# ============================================
# String is one of the built-in data types in Julia
#
# Can go into variables, arrays, or be values in dictionaries

# Example: length() for strings
# length() counts the number of characters in a string
println(length("Abraham Lincoln's beard"))
println(length("Abraham Lincoln's beard"))

# For arrays of strings
string_array = ["Abraham", "Lincoln's", "beard"]
println(length(string_array))  # Number of elements
println([length(s) for s in string_array])  # Length of each string

# ============================================
# CHARACTER-VALUED VARIABLES
# ============================================
# They work just like others, e.g., with arrays:

president = "Lincoln"
println(length(president))

presidents = ["Fillmore", "Pierce", "Buchanan", "Davis", "Johnson"]
println(presidents[3])  # Julia is 1-indexed
println(presidents[4:end])  # Skip first 3 elements

# ============================================
# DISPLAYING CHARACTERS
# ============================================
# We know println(), of course

# Example: Different display methods
println("Abraham Lincoln")
# In Julia, println() is the primary way to display output
println("Abraham Lincoln")

# Printing an array
println(presidents)

# Printing with custom separator
println(join(presidents, " "))

# ============================================
# SUBSTRING OPERATIONS
# ============================================
# Substring: a smaller string from the big string, but still a string in its own right.
# In Julia, we use slicing [start:stop] to extract substrings

# Example: Extracting a substring
phrase = "Christmas Bonus"
println("Original phrase:")
println(phrase)

result = phrase[8:12]  # Julia uses 1-indexing
println("phrase[8:12]:")
println(result)
# "as Bo"

# We can create a new string with replacement:
phrase = phrase[1:12] * "g" * phrase[14:end]
println("After replacing character 13 with 'g':")
println(phrase)
# "Christmas Bogus"

# ============================================
# SUBSTRING FOR STRING ARRAYS
# ============================================
# Julia array comprehension for substring operations

println("Presidents array:")
println(presidents)

# First two characters
println("\nFirst two characters:")
println([p[1:min(2, length(p))] for p in presidents])

# Last two characters
println("\nLast two characters:")
println([p[max(1, length(p)-1):length(p)] for p in presidents])

# Characters 20-21 (beyond string length returns empty)
println("\nCharacters 20-21 (beyond string length):")
println([length(p) >= 21 ? p[20:21] : "" for p in presidents])

# ============================================
# DIVIDING STRINGS INTO ARRAYS
# ============================================
# split() divides a string according to delimiter characters

scarborough_fair = "parsley, sage, rosemary, thyme"

# Split on comma only (leaves spaces at beginning of words)
println("\nSplitting on ',' - split(scarborough_fair, ','):")
result1 = split(scarborough_fair, ",")
println(result1)

# Split on comma followed by space (cleaner result)
println("\nSplitting on ', ' - split(scarborough_fair, ', '):")
result2 = split(scarborough_fair, ", ")
println(result2)

# Pattern works on arrays too:
println("\nSplitting multiple strings:")
for s in [scarborough_fair, "Garfunkel, Oates", "Clement, McKenzie"]
    println(split(s, ", "))
end

# ============================================
# COMBINING VALUES INTO STRINGS
# ============================================
# Converting one variable type to another is called casting:

# Conversion to string
println("\nstring(7.2):")
println(string(7.2))

# Scientific notation
println("\nstring(7.2e12):")
println(string(7.2e12))

# Array of numbers
println("\nstring.(nums) where nums = [7.2, 7.2e12]:")
nums = [7.2, 7.2e12]
println(string.(nums))

# Large number
println("\nstring(7.2e5):")
println(string(7.2e5))

# ============================================
# BUILDING STRINGS FROM MULTIPLE PARTS
# ============================================
# Julia uses join() and string interpolation

# With one array argument:
println("\njoin(string.(41:45), ' '):")
println(join(string.(41:45), " "))

# With 2 arrays, zip them together:
println("\nJoining presidents with numbers:")
result = [string(p, " ", n) for (p, n) in zip(presidents, 41:45)]
println(result)

# Not historically accurate!
println("\nJoining presidents with party:")
parties = repeat(["R", "D"], 3)  # Repeat to match length
result = [string(p, " ", parties[mod1(i, 2)]) for (i, p) in enumerate(presidents)]
println(result)

# Multiple arguments with formatting
println("\nFormatted strings:")
result = [string(p, " ( ", parties[mod1(i, 2)], " ", n, " )") for (i, (p, n)) in enumerate(zip(presidents, 41:45))]
println(result)

# Changing the separator:
println("\nWith underscore separator:")
result = [string(p, "_(", n, ")") for (p, n) in zip(presidents, 41:45)]
println(result)

println("\nWith no separator:")
result = [string(p, " (", n, ")") for (p, n) in zip(presidents, 41:45)]
println(result)

# ============================================
# MORE COMPLICATED EXAMPLE
# ============================================
# Exercise: Convince yourself of why this works as it does

println("\nComplicated recycling example:")
hw_lab = ["HW", "Lab"]
numbers = Int[]
for i in 1:11
    push!(numbers, i, i)  # Add each number twice
end

result = [string(hw_lab[mod1(i, 2)], " ", numbers[i]) for i in 1:length(numbers)]
println(result)

# Explanation:
# - ["HW","Lab"] is recycled to match length of numbers
# - numbers contains: 1, 1, 2, 2, 3, 3, ..., 11, 11
# - Result pairs HW/Lab with each number twice

# ============================================
# CONDENSING MULTIPLE STRINGS
# ============================================
# Producing one big string:

println("\nJoining with semicolon separator:")
result_joined = join([string(p, " (", n, ")") for (p, n) in zip(presidents, 41:45)], "; ")
println(result_joined)

# Without join (returns an array)
println("\nWithout join (returns array):")
result_array = [string(p, " (", n, ")") for (p, n) in zip(presidents, 41:45)]
println(result_array)

# ============================================
# FUNCTION FOR WRITING REGRESSION FORMULAS
# ============================================
# Julia function to build formula strings

# Simulate state.x77 column names
state_columns = ["Population", "Income", "Illiteracy", "Life Exp", "Murder", 
                 "HS Grad", "Frost", "Area"]

function my_formula(dep, indeps, columns)
    """Build a regression formula string"""
    rhs = join([columns[i] for i in indeps], "+")
    return string(columns[dep], " ~ ", rhs)
end

# Example: Create formula using column indices
println("\nmy_formula(2, [3, 5, 7], columns=state_columns):")
formula_result = my_formula(2, [3, 5, 7], state_columns)
println(formula_result)

# This function builds a formula string by:
# - Taking the dependent variable column index (dep)
# - Taking independent variable column indices (indeps)
# - Joining column names together with "+" and "~"

# ============================================
# GENERAL SEARCH
# ============================================
# Use occursin() and findall() to find matching strings

# Sample text data
sample_text = ["The quick brown fox", 
               "jumps over the lazy dog",
               "The fox was very clever",
               "Dogs and foxes are animals",
               "Cats are also animals"]

println("\nSample text:")
println(sample_text)

# Find which strings contain "fox"
println("\nStrings containing 'fox':")
fox_indices = findall(s -> occursin("fox", s), sample_text)
println(fox_indices)
println([sample_text[i] for i in fox_indices])

# Find which strings contain "dog" (case sensitive)
println("\nStrings containing 'dog':")
dog_indices = findall(s -> occursin("dog", s), sample_text)
println(dog_indices)
println([sample_text[i] for i in dog_indices])

# Case-insensitive search
println("\nStrings containing 'DOG' (case-insensitive):")
dog_insensitive = findall(s -> occursin(r"dog"i, s), sample_text)
println(dog_insensitive)

# ============================================
# RECONSTITUTING AND WORD COUNTING
# ============================================
# Make one long string, then split the words

# Combine all text into one string
println("\nReconstituting text:")
long_string = join(sample_text, " ")
println(long_string)

# Split into words
println("\nSplitting into words:")
words = split(long_string, " ")
println(words)

# Counting words with counter dictionary
println("\nCounting words with counter:")
word_counts = Dict{String, Int}()
for word in words
    word_counts[word] = get(word_counts, word, 0) + 1
end
println(word_counts)

# Most frequent words
println("\nMost frequent words:")
sorted_counts = sort(collect(word_counts), by=x->x[2], rev=true)
println(sorted_counts[1:min(5, length(sorted_counts))])

# ============================================
# TEXT PATTERNS AND REGULAR EXPRESSIONS
# ============================================
# Need to learn how to work with text patterns and not just constants
# Searching for text patterns using regular expressions

# Find strings starting with "The"
println("\nStrings starting with 'The':")
starts_with_the = findall(s -> occursin(r"^The", s), sample_text)
println(starts_with_the)
println([sample_text[i] for i in starts_with_the])

# Find strings ending with "animals"
println("\nStrings ending with 'animals':")
ends_with_animals = findall(s -> occursin(r"animals$", s), sample_text)
println(ends_with_animals)
println([sample_text[i] for i in ends_with_animals])

# Find strings containing any digit
numbers_text = ["There are 3 foxes", "No numbers here", "Year 2023", "Another text"]
println("\nStrings with digits:")
has_digit = findall(s -> occursin(r"[0-9]", s), numbers_text)
println(has_digit)
println([numbers_text[i] for i in has_digit])

# Find strings with "fox" or "dog" using pattern
println("\nStrings with 'fox' or 'dog':")
fox_or_dog = findall(s -> occursin(r"fox|dog", s), sample_text)
println(fox_or_dog)
println([sample_text[i] for i in fox_or_dog])

println("\n=== Regular Expression Patterns ===")
println("^ - Start of string")
println("$ - End of string")
println("[0-9] - Any digit")
println("| - OR operator")
println(". - Any character")
println("* - Zero or more repetitions")
println("+ - One or more repetitions")
