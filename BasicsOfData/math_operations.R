# Operators
#In R, operators are special symbols or keywords that perform operations on values (like numbers, vectors, or data frames).
# 1. Arithmetic Operators

7+5
## [1] 12
7-5
## [1] 2
7*5
## [1] 35
7^5
## [1] 16807

#2. Relational (Comparison) Operators
# ==  equal to
# !=  not equal to
# >   greater than 
# <   less than
# >=  greater than or equal to
# <=  less than or equal to

# 3. Logical Operators
# &   element-wise AND
# |   element-wise OR
# !   NOT
# &&  first-element AND
# ||  first-element OR
(5 > 7) & (6*7 == 42)
## [1] FALSE
(5 > 7) | (6*7 == 42)
## [1] TRUE

# 4. Assignment Operators
# <-   # most common (x <- 5)
# ->   # assign to the right (5 -> x)
# =    # alternative assignment (x = 5)

typeof(7)
## [1] "double" In R, all numbers are stored as doubles (floating-point numbers) by default, unless you explicitly mark them as integers (7L).
is.numeric(7)
## [1] TRUE
is.na(7)
# NA in R stands for “Not Available”
## [1] FALSE 7 is just a valid number, so it’s not NA (missing).

is.na(7/0)
## [1] FALSE this actually leads to infinity where inf in Valid numeric constant, not misssing
is.na(0/0)
## [1] TRUE
# Why is 7/0 not NA, but 0/0 is? In mathematics: 
# 0 / 0 is indeterminate (not infinity, not zero, no single value).
# In IEEE 754, this gives NaN = “Not a Number”.

is.character(7)
## [1] FALSE 7 is a numeric (double) value, not text
is.character("7")
## [1] TRUE
is.character("seven")
## [1] TRUE
is.na("seven")
## [1] FALSE

as.character(5/6)
## [1] "0.833333333333333"
# The real fraction 5/6 = 0.8333... repeating forever.
# Computers can’t store infinite decimals they approximate it.
# R prints 15 decimal places by default when converting to a string.
# But still, it’s only an approximation of the true fraction.
as.numeric(as.character(5/6))
## [1] 0.8333333
# When converting back, R stores a slightly different approximation (rounded at a certain precision).
# It’s extremely close, but not bit-for-bit identical to the original 5/6.
6*as.numeric(as.character(5/6))
## [1] 5
5/6 == as.numeric(as.character(5/6))
## [1] FALSE
# == checks for exact equality at the binary level.

# The two values are almost equal, but due to tiny rounding differences in floating-point representation, they don’t match exactly.
# (why is that last FALSE?) Floating-point numbers are approximations. Never rely on == for decimals — use all.equal() or tolerances.

# a few are built in variables
pi
## [1] 3.141593

pi*10
## [1] 31.41593
cos(pi)
## [1] -1

#Most variables are created with the assignment operator <-

approx.pi <- 22/7
approx.pi
## [1] 3.142857
diameter.in.cubits = 10
#Once defined, variables can be used in calculations just like numbers.
approx.pi*diameter.in.cubits 
## [1] 31.42857

# A variable in R is just a name pointing to a value.
# If you assign it again, the old value is lost (unless you saved it somewhere else).

circumference.in.cubits <- approx.pi*diameter.in.cubits
circumference.in.cubits
## [1] 31.42857
circumference.in.cubits <- 30
circumference.in.cubits
## [1] 30


# ls() and objects() both show you all the variable names currently stored in memory.
ls() 
## [1] "approx.pi"               "circumference.in.cubits"
## [3] "diameter.in.cubits"
objects()
## [1] "approx.pi"               "circumference.in.cubits"
## [3] "diameter.in.cubits"
# Removing a variable (delete)
rm("circumference.in.cubits") 
ls()
## [1] "approx.pi"          "diameter.in.cubits"

# First data structure: vectors
# What is a vector?
# A vector is simply an ordered collection of values.
# All values must be of the same type (all numbers, all characters, all logical values, etc.).
# You used the c() function, which stands for combine or concatenate.
x <- c(7, 8, 10, 45)
# This creates a numeric vector with 4 elements.
# Checking if it’s a vector
is.vector(x)
# [1] TRUE
# This confirms that x really is a vector.

# Types of vectors
# Vectors can hold different types of values (but only one type at a time):
# Numeric vector
nums <- c(3.5, 4.2, 9.1)
# Character vector
names <- c("Alice", "Bob", "Clifford")
# Logical vector
flags <- c(TRUE, FALSE, TRUE)

# Indexing with [ ]
# In R, you use square brackets [] to get elements of a vector.
# Indexing starts at 1 (unlike some languages like Python or C, which start at 0).
x <- c(7, 8, 10, 45)

x[1]   # first element
# [1] 7

x[4]   # fourth element
# [1] 45

# 2. Negative indices
# If you use a negative number, R will return the vector with that position excluded.
x[-4]
# [1] 7  8 10
# This means “everything except the 4th element.”

# 3. Multiple indices
# You can also get multiple elements at once by giving a vector of indices:
x[c(1, 3)]
# [1]  7 10
# This selects the 1st and 3rd elements.

# 4. Index ranges
# You can use : to get a sequence of indices:
x[2:4]
# [1]  8 10 45
# This means “elements 2 through 4.

# Creating an empty vector
# weekly.hours <- vector(length = 5)
# This makes a vector with 5 slots.
# By default, since you didn’t specify a type, it creates a logical vector (filled with FALSE).
# weekly.hours
# [1] FALSE FALSE FALSE FALSE FALSE

# 2. Assigning a value
weekly.hours[5] <- 8
# Now you’ve put the value 8 in the 5th position.
# Because vectors must hold one type of data, R converts the whole vector from logical → numeric.
# FALSE becomes 0 when coerced to numeric.
weekly.hours
# [1] 0 0 0 0 8

# 3. Specifying the type up front
# You can also tell R what kind of vector you want:
# Numeric vector
nums <- vector("numeric", length = 5)
# [1] 0 0 0 0 0

# Character vector
words <- vector("character", length = 5)
# [1] "" "" "" "" ""

# Logical vector
flags <- vector("logical", length = 5)
# [1] FALSE FALSE FALSE FALSE FALSE

# Elementwise operations
# In R, when you apply an operator (+, -, *, /, etc.) to vectors of the same length, the operation is applied element by element.
x <- c(7, 8, 10, 45)
y <- c(-7, -8, -10, -45)

2. Addition
x + y
# [1] 0 0 0 0
First elements: 7 + (-7) = 0
Second elements: 8 + (-8) = 0
Third elements: 10 + (-10) = 0
Fourth elements: 45 + (-45) = 0
So the result is a vector of zeros.

# 3. Multiplication
x * y
# [1]   -49   -64  -100 -2025
First: 7 * (-7) = -49
Second: 8 * (-8) = -64
Third: 10 * (-10) = -100
Fourth: 45 * (-45) = -2025
Result: c(-49, -64, -100, -2025)

# 4. Other elementwise operators
# This works the same for subtraction and division:
x - y   # elementwise subtraction
x / y   # elementwise division

# 5. Vector recycling (extra tip)
# If the vectors are different lengths, R will “recycle” the shorter one until it matches the length of the longer one (but gives a warning if lengths don’t divide evenly).
# Example:
x + c(1, 2)
# [1]  8 10 11 47
Explanation:
First: 7 + 1 = 8
Second: 8 + 2 = 10
Third: 10 + 1 = 11 (recycled 1)
Fourth: 45 + 2 = 47 (recycled 2)

# Recycling
# Recycling with a shorter vector
x <- c(7, 8, 10, 45)
x + c(-7, -8)
# [1]  0  0  3 37
# R lines up elements of both vectors.
# Since the shorter vector has only 2 elements, R recycles it:
# Calculation:
7 + (-7) = 0
8 + (-8) = 0
10 + (-7) = 3 ← recycled -7
45 + (-8) = 37 ← recycled -8
So the result is c(0, 0, 3, 37)

2. Recycling with exponents
x ^ c(1, 0, -1, 0.5)
# [1] 7.000000 1.000000 0.100000 6.708204
# Here, the second vector has exactly 4 elements — same length as x, so no recycling is needed.
7^1 = 7
8^0 = 1 (anything to power 0 is 1)
10^-1 = 0.1 (1/10)
45^0.5 ≈ 6.708204 (square root of 45)

3. Single numbers as length-1 vectors
2 * x
# [1] 14 16 20 90
# In R, a single number like 2 is treated as a vector of length 1.
# When used with a longer vector, it’s recycled across every element:
2 * c(7, 8, 10, 45)
= c(2*7, 2*8, 2*10, 2*45)
= c(14, 16, 20, 90)

✅ Key rules of recycling in R:
# If vectors have the same length, elements are paired one-to-one.
# If one vector is shorter, its elements repeat until the lengths match.
# A single number is just a vector of length 1, so it’s repeated for all elements.
# If lengths don’t divide evenly, R gives a warning.

Example:
x + c(1, 2, 3)
# Warning: longer object length is not a multiple of shorter object length

Pairwise comparisons
x <- c(7, 8, 10, 45)
x > 9
# [1] FALSE FALSE  TRUE  TRUE

# R compares each element of x with 9, one by one.
# Results are stored in a logical vector (TRUE/FALSE).
# Step by step:
# 7 > 9 → FALSE
# 8 > 9 → FALSE
# 10 > 9 → TRUE
# 45 > 9 → TRUE
So you get: c(FALSE, FALSE, TRUE, TRUE)

# 2. Boolean operators elementwise
(x > 9) & (x < 20)
# [1] FALSE FALSE  TRUE FALSE


# Here:
# x > 9 → FALSE FALSE TRUE TRUE
# x < 20 → TRUE TRUE TRUE FALSE

# Now apply & (logical AND), element by element:
# FALSE & TRUE → FALSE
# FALSE & TRUE → FALSE
# TRUE & TRUE → TRUE
# TRUE & FALSE → FALSE
Result: c(FALSE, FALSE, TRUE, FALSE)

# 3. Other Boolean operators
# & → AND (both must be true)
# | → OR (at least one must be true)
# ! → NOT (flips TRUE/FALSE)
# Example:
x < 10 | x > 40
# [1]  TRUE  TRUE FALSE  TRUE

# TRUE if element is less than 10 OR greater than 40.
# 4. Using logical vectors for filtering
# This is where it gets powerful: you can use a logical vector inside [] to select elements.

x[x > 9]
# [1] 10 45
# Here, x > 9 is FALSE FALSE TRUE TRUE, so R picks only the elements where the condition is TRUE.

Elementwise comparison (==)
x == -y
# [1] TRUE TRUE TRUE TRUE

# This checks each element of x against the corresponding element of -y.
Since x = c(7, 8, 10, 45) and y = c(-7, -8, -10, -45), we have:
7 == 7 → TRUE
8 == 8 → TRUE
10 == 10 → TRUE
45 == 45 → TRUE
# So all comparisons are TRUE.

2. identical()
identical(x, -y)
# [1] TRUE
identical() checks if two objects are exactly the same (same length, same type, same values, no tolerance).
# Here, x and -y are identical vectors, so the result is TRUE.

3. Floating-point precision problem
identical(c(0.5-0.3, 0.3-0.1),
          c(0.3-0.1, 0.5-0.3))
# [1] FALSE
# You might expect 0.5 - 0.3 = 0.2 and 0.3 - 0.1 = 0.2, so they should match.
# But computers represent decimals in binary floating-point, so results aren’t always exact:

0.5 - 0.3
# [1] 0.2
0.3 - 0.1
# [1] 0.2
# Looks like 0.2, but internally one might be 0.20000000000000001 and the other 0.19999999999999998.
# That’s why identical() returns FALSE.

4. all.equal()
all.equal(c(0.5-0.3, 0.3-0.1),
          c(0.3-0.1, 0.5-0.3))
# [1] TRUE

# all.equal() checks if two objects are “nearly equal,” allowing for tiny numerical differences.
# This is the recommended way to compare numeric vectors.
# Since the difference is within machine precision, all.equal() says TRUE.


# Addressing vectors
# Positive indices → select elements
x <- c(7, 8, 10, 45)
x[c(2,4)]
# [1]  8 45

# Here, R returns the 2nd and 4th elements of x:
2nd element = 8
4th element = 45
So the result is c(8, 45) ✅

# 2. Negative indices → drop elements
x[c(-1, -3)]
# [1]  8 45
# This means: “give me all elements of x, except the 1st and 3rd.”
# Removing 1st (7) and 3rd (10) leaves: c(8, 45)

# 3. Why not 8 10?
# Because negative indices don’t mean “select these positions” — they mean “exclude these positions.”
x[c(2,4)] → include elements at positions 2 and 4
x[c(-1,-3)] → exclude elements at positions 1 and 3
# That’s why the answer is the same (8 45) in your example, not 8 10.

# 4. Key rule
# You can’t mix positive and negative indices in the same call:
x[c(2, -3)]
# Error in x[c(2, -3)] : only 0's may be mixed with negative subscripts


# Logical indexing
x <- c(7, 8, 10, 45)
y <- c(-7, -8, -10, -45)

x[x > 9]
# [1] 10 45


x > 9 produces a Boolean (logical) vector:
[1] FALSE FALSE TRUE TRUE
# Using that inside x[...] keeps only the elements where the condition is TRUE.
# So we get 10 and 45.

# Similarly:
y[x > 9]
# [1] -10 -45
# Same logical filter applied to y, so we pull the 3rd and 4th elements of y.

2. which()
places <- which(x > 9)
places
# [1] 3 4

# which() converts a logical vector into the positions of the TRUE values.
# So instead of [FALSE FALSE TRUE TRUE], you get [3 4].
3. Using which() for indexing
y[places]
# [1] -10 -45
# This is equivalent to y[x > 9], but now the filter is expressed as numeric indices rather than logical ones.

4. When to use which()?
Logical indexing is concise and direct (x[x > 9]).
which() is useful when:

# You actually need the positions of the matches (not just the values).
# You want to use those positions in multiple vectors (like y[places]).
# You’re debugging and want to “see where the TRUEs are.”