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