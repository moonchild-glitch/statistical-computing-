#!/usr/bin/env python3
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

import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import Counter

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
# Exercise: plot ψ in Python

# Define the piecewise function ψ(x)
def psi(x):
    if abs(x) <= 1:
        return x**2
    else:
        return 2*abs(x) - 1

# Create a sequence of x values
x_values = np.arange(-3, 3, 0.01)

# Apply the function to each x value
y_values = np.array([psi(x) for x in x_values])

# Create plots directory if it doesn't exist
if not os.path.exists("plots"):
    os.makedirs("plots")

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, 'b-', linewidth=2)
plt.title("Plot of psi(x)")
plt.xlabel("x")
plt.ylabel("psi(x)")
plt.grid(True)
plt.axvline(x=-1, color='red', linestyle='--')
plt.axvline(x=1, color='red', linestyle='--')
plt.savefig("plots/psi_function_plot.png", dpi=100)
plt.close()

print("Plot saved to plots/psi_function_plot.png")

# Computationally:
# if the country code is not "US", multiply prices by current exchange rate

# Example implementation
def adjust_price(price, country_code, exchange_rate=1.0):
    if country_code != "US":
        return price * exchange_rate
    else:
        return price

# Test the function
print(adjust_price(100, "US", 1.2))       # Returns 100 (no change)
print(adjust_price(100, "CA", 1.35))      # Returns 135 (converted)
print(adjust_price(100, "UK", 0.79))      # Returns 79 (converted)

# ============================================
# if() EXAMPLES
# ============================================
# Example: Absolute value using if()
x = -5  # Define x for demonstration
if x >= 0:
    print(x)
else:
    print(-x)

# Example: Piecewise function using nested if()
x = 1.5  # Define x for demonstration
if x**2 < 1:
    print(x**2)
else:
    if x >= 0:
        print(2*x-1)
    else:
        print(-2*x-1)

# ============================================
# COMBINING BOOLEANS
# ============================================
# & and | work element-wise in NumPy arrays
# For scalars, use 'and' and 'or' which provide short-circuit evaluation

# Example: 'and' stops evaluation if first condition is FALSE
print((0 > 0) and (42%6 == 169%13))
# Since (0 > 0) is FALSE, the second part is never evaluated
# This is "lazy" or "short-circuit" evaluation

# ============================================
# ITERATION
# ============================================
# Repeat similar actions multiple times

# Example: Create a table of logarithms using a for loop
table_of_logarithms = np.zeros(7)
print(table_of_logarithms)

for i in range(len(table_of_logarithms)):
    table_of_logarithms[i] = np.log(i+1)  # i+1 because Python is 0-indexed
print(table_of_logarithms)

# ============================================
# for() LOOPS
# ============================================
# for executes a block of code repeatedly for each element in a sequence
# Syntax: for variable in sequence: code block
# The loop variable takes on each value in the sequence, one at a time
#
# for increments a counter (here i) along a sequence
# and loops through the **body** until it runs through the sequence
#
# "iterates over the sequence"
#
# Note, there is a better way to do this job!
#
# Can contain just about anything, including:
# - if clauses
# - other for loops (nested iteration)

# Example: Basic for loop filling a vector
for i in range(len(table_of_logarithms)):
    table_of_logarithms[i] = np.log(i+1)

# Example: Nested iteration - Matrix multiplication
# First, create two matrices for demonstration
a = np.array([[1, 3, 5], [2, 4, 6]])  # 2x3 matrix
b = np.array([[7, 10], [8, 11], [9, 12]])  # 3x2 matrix

c = np.zeros((a.shape[0], b.shape[1]))
if a.shape[1] == b.shape[0]:
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(a.shape[1]):
                c[i,j] += a[i,k] * b[k,j]
else:
    raise ValueError("matrices a and b non-conformable")
print("Matrix multiplication result:")
print(c)

# ============================================
# while() LOOPS
# ============================================
# Condition in the argument to while must be a single Boolean value (like if)
#
# Body is looped over until the condition is FALSE so can loop forever
#
# Loop never begins unless the condition starts TRUE

# Example: Repeatedly take square root until convergence
x = np.array([5.0, 10.0, 15.0])  # Initialize with some values
print("Initial x:")
print(x)

while np.max(x) - 1 > 1e-06:
    x = np.sqrt(x)

print("Final x after convergence:")
print(x)

# ============================================
# VECTORIZED ARITHMETIC
# ============================================
# How many languages add 2 vectors:

# Example: Adding two vectors element by element using a for loop
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

c = np.zeros(len(a))
for i in range(len(a)):
    c[i] = a[i] + b[i]

print("Vector addition using for loop:")
print(c)

# ============================================
# VECTORIZED CALCULATIONS
# ============================================
# Vectorized calculations
# Many functions are set up to vectorize automatically

# Example: abs() function automatically works on entire array
print(np.abs(np.arange(-3, 4)))

# Example: log() function also vectorizes automatically
print(np.log(np.arange(1, 8)))

# ============================================
# VECTORIZED CONDITIONS: np.where()
# ============================================
# np.where() allows vectorized conditional operations
# Similar to R's ifelse(): np.where(condition, value_if_true, value_if_false)

# Example: Piecewise function using np.where()
x = np.arange(-3, 3.5, 0.5)
print("x values:")
print(x)

result = np.where(x**2 > 1, 2*np.abs(x)-1, x**2)
print("np.where(x**2 > 1, 2*np.abs(x)-1, x**2):")
print(result)

# ============================================
# WHAT IS TRUTH?
# ============================================
# 0 counts as FALSE; other numeric values count as TRUE
# The booleans True and False work as expected
# Most everything else gives an error
#
# Advice: Don't play games here; try to make sure control expressions are getting Boolean values
#
# Conversely, in arithmetic, False is 0 and True is 1

# Example: Using True/False as numbers
# Load state data (simplified version)
# In Python, we'll create a simple dictionary mimicking state.x77
states_murder = np.array([13.2, 10.0, 8.1, 8.8, 9.0, 7.9, 3.3, 5.9, 15.4, 
                          17.4, 5.3, 2.6, 10.4, 7.2, 6.0, 6.0, 9.7, 15.4,
                          2.1, 11.3, 4.4, 12.2, 2.7, 16.1, 9.0, 6.0, 4.3,
                          12.2, 2.1, 7.4, 11.4, 11.1, 13.0, 0.8, 7.3, 6.6,
                          4.9, 6.3, 3.4, 14.4, 3.8, 13.2, 12.7, 3.2, 2.2,
                          8.5, 4.0, 5.7, 2.6, 6.8])

# Calculate the proportion of states with murder rate > 7
# True becomes 1, False becomes 0, so mean gives the proportion
proportion_high_murder = np.mean(states_murder > 7)
print("Proportion of states with Murder rate > 7:")
print(proportion_high_murder)

# ============================================
# switch() - Python uses dict for switch-like behavior
# ============================================
# Simplify nested if with dictionary: give a key to select on, then a value for each option

# Example: Different summary statistics based on type
type_of_summary = "mean"  # Try changing this to "median", "histogram", or something else

# Using dictionary to simulate switch
switch_dict = {
    "mean": np.mean(states_murder),
    "median": np.median(states_murder),
    "histogram": "histogram"  # Placeholder
}

result = switch_dict.get(type_of_summary, "I don't understand")
print("Switch result:")
print(result)

# ============================================
# EXERCISE IMPLEMENTATION
# ============================================
# Set type_of_summary to, successively, "mean", "median", "histogram", and "mode", and explain what happens

# Test 1: "mean"
print("\n--- Test 1: type_of_summary = 'mean' ---")
type_of_summary = "mean"
result = switch_dict.get(type_of_summary, "I don't understand")
if type_of_summary == "mean":
    result = np.mean(states_murder)
print(result)
print("Explanation: Returns the mean of Murder rates")

# Test 2: "median"
print("\n--- Test 2: type_of_summary = 'median' ---")
type_of_summary = "median"
if type_of_summary == "median":
    result = np.median(states_murder)
print(result)
print("Explanation: Returns the median of Murder rates")

# Test 3: "histogram"
print("\n--- Test 3: type_of_summary = 'histogram' ---")
type_of_summary = "histogram"
if type_of_summary == "histogram":
    plt.figure(figsize=(10, 6))
    plt.hist(states_murder, bins=10, edgecolor='black')
    plt.title("Histogram of Murder Rates")
    plt.xlabel("Murder Rate")
    plt.ylabel("Frequency")
    plt.savefig("plots/murder_histogram.png", dpi=100)
    plt.close()
print("Histogram saved to plots/murder_histogram.png")
print("Explanation: Creates and displays a histogram of Murder rates")

# Test 4: "mode"
print("\n--- Test 4: type_of_summary = 'mode' ---")
type_of_summary = "mode"
result = switch_dict.get(type_of_summary, "I don't understand")
print(result)
print("Explanation: 'mode' is not in the switch cases, so it returns the default message 'I don't understand'")

# ============================================
# UNCONDITIONAL ITERATION
# ============================================
# while True creates an infinite loop that continues forever unless explicitly stopped with break

# Example: Infinite loop (commented out to prevent actual infinite execution)
# WARNING: This will run forever if uncommented!
# while True:
#     print("Help! I am Dr. Morris Culpepper, trapped in an endless loop!")

# Practical example: Using while True with a break condition
print("\n--- Example: while True with break condition ---")
counter = 1
while True:
    print(f"Iteration: {counter}")
    counter += 1
    if counter > 3:
        print("Breaking out of the loop!")
        break  # Exit the loop when counter > 3

# ============================================
# "MANUAL" CONTROL OVER ITERATION
# ============================================
# break exits the loop; continue skips the rest of the body and goes back into the loop
# Both work with for and while loops

# Example: Using break and continue in while True loop
print("\n--- Example: Manual control with break() and continue ---")

watched = False
rescued = False
iteration_count = 0

while True:
    iteration_count += 1
    
    # Simulate different conditions
    if iteration_count == 2:
        watched = True
    if iteration_count == 3:
        watched = False
    if iteration_count == 5:
        rescued = True
    
    if watched:
        print(f"Iteration {iteration_count}: Skipping (watched)")
        continue  # Skip the rest and go back to start of loop
    
    print(f"Iteration {iteration_count}: Help! I am Dr. Morris Culpepper, trapped in an endless loop!")
    
    if rescued:
        print("Dr. Culpepper has been rescued!")
        break  # Exit the loop

# ============================================
# break and continue with for and while
# ============================================

# Example: continue in a for loop - skip even numbers
print("\n--- Example: continue in for loop - skip even numbers ---")
for i in range(1, 6):
    if i % 2 == 0:
        continue  # Skip even numbers
    print(f"Processing odd number: {i}")

# Example: break in a while loop - stop at threshold
print("\n--- Example: break in while loop ---")
total = 0
i = 1
while i <= 10:
    total += i
    print(f"i = {i}, total = {total}")
    if total > 15:
        print("Total exceeded 15, breaking!")
        break
    i += 1

# ============================================
# EXERCISE
# ============================================
# Exercise: how would you replace while() with while True?
#
# Answer: while condition: body 
# becomes:
# while True: if not condition: break; body

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
# Note: Python has a string type for sequences of characters

print(type("L"))
print(type("Lincoln"))
print(type("Lincoln"))

# ============================================
# MAKING STRINGS
# ============================================
# Use single or double quotes to construct a string
# Use len() to get the length of a string
# Why do we prefer double quotes?

print("Lincoln")
print(len("Lincoln"))
print("Abraham Lincoln")
print(len("Abraham Lincoln"))
print('As Lincoln never said, "Four score and seven beers ago"')

# ============================================
# CHARACTER DATA TYPE
# ============================================
# String is one of the built-in data types in Python
#
# Can go into variables, lists, arrays, tuples, or be values in dictionaries

# Example: len() for strings
# len() counts the number of characters in a string
print(len("Abraham Lincoln's beard"))
print(len("Abraham Lincoln's beard"))

# For lists of strings
string_list = ["Abraham", "Lincoln's", "beard"]
print(len(string_list))  # Number of elements
print([len(s) for s in string_list])  # Length of each string

# ============================================
# CHARACTER-VALUED VARIABLES
# ============================================
# They work just like others, e.g., with lists:

president = "Lincoln"
print(len(president))

presidents = ["Fillmore", "Pierce", "Buchanan", "Davis", "Johnson"]
print(presidents[2])  # Python is 0-indexed, so this is the 3rd element
print(presidents[3:])  # Skip first 3 elements

# ============================================
# DISPLAYING CHARACTERS
# ============================================
# We know print(), of course

# Example: Different display methods
print("Abraham Lincoln")
# In Python, print() is the primary way to display output
print("Abraham Lincoln")

# Printing a list
print(presidents)

# Printing with custom separator
print(" ".join(presidents))

# ============================================
# SUBSTRING OPERATIONS
# ============================================
# Substring: a smaller string from the big string, but still a string in its own right.
# In Python, we use slicing [start:stop] to extract substrings

# Example: Extracting a substring
phrase = "Christmas Bonus"
print("Original phrase:")
print(phrase)

result = phrase[7:12]  # Python uses 0-indexing and stop is exclusive
print("phrase[7:12]:")
print(result)
# [1] "as Bo"

# We can also use slicing to replace characters:
phrase_list = list(phrase)
phrase_list[12] = "g"
phrase = "".join(phrase_list)
print("After replacing character 13 with 'g':")
print(phrase)
# [1] "Christmas Bogus"

# ============================================
# SUBSTRING FOR STRING LISTS
# ============================================
# Python list comprehension for substring operations

print("Presidents list:")
print(presidents)

# First two characters
print("\nFirst two characters:")
print([p[:2] for p in presidents])

# Last two characters
print("\nLast two characters:")
print([p[-2:] for p in presidents])

# Characters 20-21 (beyond string length returns empty)
print("\nCharacters 20-21 (beyond string length):")
print([p[19:21] if len(p) >= 21 else "" for p in presidents])

# ============================================
# DIVIDING STRINGS INTO LISTS
# ============================================
# str.split() divides a string according to key characters

scarborough_fair = "parsley, sage, rosemary, thyme"

# Split on comma only (leaves spaces at beginning of words)
print("\nSplitting on ',' - scarborough_fair.split(','):")
result1 = scarborough_fair.split(",")
print(result1)

# Split on comma followed by space (cleaner result)
print("\nSplitting on ', ' - scarborough_fair.split(', '):")
result2 = scarborough_fair.split(", ")
print(result2)

# Pattern works on lists too:
print("\nSplitting multiple strings:")
for s in [scarborough_fair, "Garfunkel, Oates", "Clement, McKenzie"]:
    print(s.split(", "))

# ============================================
# COMBINING VALUES INTO STRINGS
# ============================================
# Converting one variable type to another is called casting:

# Conversion to string
print("\nstr(7.2):")
print(str(7.2))

# Scientific notation preserved
print("\nstr(7.2e12):")
print(str(7.2e12))

# List of numbers
print("\nstr([7.2, 7.2e12]):")
nums = [7.2, 7.2e12]
print([str(n) for n in nums])

# Scientific notation expanded
print("\nstr(7.2e5):")
print(str(7.2e5))

# ============================================
# BUILDING STRINGS FROM MULTIPLE PARTS
# ============================================
# Python uses join() and formatting

# With one list argument:
print("\n' '.join(map(str, range(41, 46))):")
print(" ".join(map(str, range(41, 46))))

# With 2 lists, zip them together:
print("\nZipping presidents with numbers:")
result = [f"{p} {n}" for p, n in zip(presidents, range(41, 46))]
print(result)

# Not historically accurate!
print("\nZipping presidents with party:")
parties = ["R", "D"] * 3  # Repeat to match length
result = [f"{p} {parties[i%2]}" for i, p in enumerate(presidents)]
print(result)

# Multiple arguments with formatting
print("\nFormatted strings:")
result = [f"{p} ( {parties[i%2]} {n} )" for i, (p, n) in enumerate(zip(presidents, range(41, 46)))]
print(result)

# Changing the separator:
print("\nWith underscore separator:")
result = [f"{p}_({n})" for p, n in zip(presidents, range(41, 46))]
print(result)

print("\nWith no separator:")
result = [f"{p} ({n})" for p, n in zip(presidents, range(41, 46))]
print(result)

# ============================================
# MORE COMPLICATED EXAMPLE
# ============================================
# Exercise: Convince yourself of why this works as it does

print("\nComplicated recycling example:")
hw_lab = ["HW", "Lab"]
numbers = []
for i in range(1, 12):
    numbers.extend([i, i])  # Add each number twice

result = [f"{hw_lab[i%2]} {numbers[i]}" for i in range(len(numbers))]
print(result)

# Explanation:
# - ["HW","Lab"] is recycled to match length of numbers
# - numbers contains: 1, 1, 2, 2, 3, 3, ..., 11, 11
# - Result pairs HW/Lab with each number twice

# ============================================
# CONDENSING MULTIPLE STRINGS
# ============================================
# Producing one big string:

print("\nJoining with semicolon separator:")
result_joined = "; ".join([f"{p} ({n})" for p, n in zip(presidents, range(41, 46))])
print(result_joined)

# Without join (returns a list)
print("\nWithout join (returns list):")
result_list = [f"{p} ({n})" for p, n in zip(presidents, range(41, 46))]
print(result_list)

# ============================================
# FUNCTION FOR WRITING REGRESSION FORMULAS
# ============================================
# Python function to build formula strings

# Simulate state.x77 column names
state_columns = ["Population", "Income", "Illiteracy", "Life Exp", "Murder", 
                 "HS Grad", "Frost", "Area"]

def my_formula(dep, indeps, columns):
    """Build a regression formula string"""
    rhs = "+".join([columns[i] for i in indeps])
    return f"{columns[dep]} ~ {rhs}"

# Example: Create formula using column indices
print("\nmy_formula(1, [2, 4, 6], columns=state_columns):")
formula_result = my_formula(1, [2, 4, 6], columns=state_columns)
print(formula_result)

# This function builds a formula string by:
# - Taking the dependent variable column index (dep)
# - Taking independent variable column indices (indeps)
# - Joining column names together with "+" and "~"

# ============================================
# GENERAL SEARCH
# ============================================
# Use list comprehension and string methods to find matching strings

# Sample text data
sample_text = ["The quick brown fox", 
               "jumps over the lazy dog",
               "The fox was very clever",
               "Dogs and foxes are animals",
               "Cats are also animals"]

print("\nSample text:")
print(sample_text)

# Find which strings contain "fox"
print("\nStrings containing 'fox':")
fox_indices = [i for i, s in enumerate(sample_text) if "fox" in s]
print(fox_indices)
print([sample_text[i] for i in fox_indices])

# Find which strings contain "dog" (case sensitive)
print("\nStrings containing 'dog':")
dog_indices = [i for i, s in enumerate(sample_text) if "dog" in s]
print(dog_indices)
print([sample_text[i] for i in dog_indices])

# Case-insensitive search
print("\nStrings containing 'DOG' (case-insensitive):")
dog_insensitive = [i for i, s in enumerate(sample_text) if "dog" in s.lower()]
print(dog_insensitive)

# ============================================
# RECONSTITUTING AND WORD COUNTING
# ============================================
# Make one long string, then split the words

# Combine all text into one string
print("\nReconstituting text:")
long_string = " ".join(sample_text)
print(long_string)

# Split into words
print("\nSplitting into words:")
words = long_string.split(" ")
print(words)

# Counting words with Counter
print("\nCounting words with Counter():")
word_counts = Counter(words)
print(dict(word_counts))

# Most frequent words
print("\nMost frequent words:")
sorted_counts = word_counts.most_common(5)
print(sorted_counts)

# ============================================
# TEXT PATTERNS AND REGULAR EXPRESSIONS
# ============================================
# Need to learn how to work with text patterns and not just constants
# Searching for text patterns using regular expressions

# Find strings starting with "The"
print("\nStrings starting with 'The':")
starts_with_the = [i for i, s in enumerate(sample_text) if re.match(r'^The', s)]
print(starts_with_the)
print([sample_text[i] for i in starts_with_the])

# Find strings ending with "animals"
print("\nStrings ending with 'animals':")
ends_with_animals = [i for i, s in enumerate(sample_text) if re.search(r'animals$', s)]
print(ends_with_animals)
print([sample_text[i] for i in ends_with_animals])

# Find strings containing any digit
numbers_text = ["There are 3 foxes", "No numbers here", "Year 2023", "Another text"]
print("\nStrings with digits:")
has_digit = [i for i, s in enumerate(numbers_text) if re.search(r'[0-9]', s)]
print(has_digit)
print([numbers_text[i] for i in has_digit])

# Find strings with "fox" or "dog" using pattern
print("\nStrings with 'fox' or 'dog':")
fox_or_dog = [i for i, s in enumerate(sample_text) if re.search(r'fox|dog', s)]
print(fox_or_dog)
print([sample_text[i] for i in fox_or_dog])

print("\n=== Regular Expression Patterns ===")
print("^ - Start of string")
print("$ - End of string")
print("[0-9] - Any digit")
print("| - OR operator")
print(". - Any character")
print("* - Zero or more repetitions")
print("+ - One or more repetitions")
