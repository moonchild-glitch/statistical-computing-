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
# Exercise: plot ψ in R

# Define the piecewise function ψ(x)
psi <- function(x) {
  if (abs(x) <= 1) {
    return(x^2)
  } else {
    return(2*abs(x) - 1)
  }
}

# Create a sequence of x values
x_values <- seq(-3, 3, by = 0.01)

# Apply the function to each x value
y_values <- sapply(x_values, psi)

# Create plots directory if it doesn't exist
if (!dir.exists("plots")) {
  dir.create("plots")
}

# Save plot to file
png("plots/psi_function_plot.png", width = 800, height = 600)

# Plot the function
plot(x_values, y_values, type = "l", col = "blue", lwd = 2,
     main = "Plot of psi(x)",
     xlab = "x", ylab = "psi(x)",
     las = 1)

# Add gridlines for clarity
grid()

# Add a vertical line at x = -1 and x = 1 to show where the function changes
abline(v = c(-1, 1), col = "red", lty = 2)

# Close the plotting device to save the file
dev.off()

cat("Plot saved to plots/psi_function_plot.png\n")


# Computationally:
# if the country code is not "US", multiply prices by current exchange rate

# Example implementation
adjust_price <- function(price, country_code, exchange_rate = 1.0) {
  if (country_code != "US") {
    return(price * exchange_rate)
  } else {
    return(price)
  }
}

# Test the function
print(adjust_price(100, "US", 1.2))       # Returns 100 (no change)
print(adjust_price(100, "CA", 1.35))      # Returns 135 (converted)
print(adjust_price(100, "UK", 0.79))      # Returns 79 (converted)

# ============================================
# if() EXAMPLES
# ============================================
# Example: Absolute value using if()
x <- -5  # Define x for demonstration
if (x >= 0) {
  print(x)
} else {
  print(-x)
}

# Example: Piecewise function using nested if()
x <- 1.5  # Define x for demonstration
if (x^2 < 1) {
  print(x^2)
} else {
  if (x >= 0) {
    print(2*x-1)
  } else {
    print(-2*x-1)
  }
}

# ============================================
# COMBINING BOOLEANS
# ============================================
# & and | work like + or *: combine terms element-wise
#
# Flow control wants one Boolean value, and to skip calculating what's not needed
#
# && and || give one Boolean, lazily

# Example: && stops evaluation if first condition is FALSE
(0 > 0) && (all.equal(42%%6, 169%%13))
# Since (0 > 0) is FALSE, the second part is never evaluated
# This is "lazy" or "short-circuit" evaluation

# ============================================
# ITERATION
# ============================================
# Repeat similar actions multiple times

# Example: Create a table of logarithms using a for loop
table.of.logarithms <- vector(length=7,mode="numeric")
table.of.logarithms

for (i in 1:length(table.of.logarithms)) {
  table.of.logarithms[i] <- log(i)
}
table.of.logarithms

# ============================================
# for() LOOPS
# ============================================
# for() executes a block of code repeatedly for each element in a sequence
# Syntax: for (variable in sequence) { code block }
# The loop variable takes on each value in the sequence, one at a time
#
# for increments a counter (here i) along a vector (here 1:length(table.of.logarithms))
# and loops through the **body** until it runs through the vector
#
# "iterates over the vector"
#
# Note, there is a better way to do this job!
#
# Can contain just about anything, including:
# - if() clauses
# - other for() loops (nested iteration)

# Example: Basic for loop filling a vector
for (i in 1:length(table.of.logarithms)) {
  table.of.logarithms[i] <- log(i)
}

# Example: Nested iteration - Matrix multiplication
# First, create two matrices for demonstration
a <- matrix(c(1, 2, 3, 4, 5, 6), nrow=2, ncol=3)
b <- matrix(c(7, 8, 9, 10, 11, 12), nrow=3, ncol=2)

c <- matrix(0, nrow=nrow(a), ncol=ncol(b))
if (ncol(a) == nrow(b)) {
  for (i in 1:nrow(c)) {
    for (j in 1:ncol(c)) {
      for (k in 1:ncol(a)) {
        c[i,j] <- c[i,j] + a[i,k]*b[k,j]
      }
    }
  }
} else {
  stop("matrices a and b non-conformable")
}
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
x <- c(5, 10, 15)  # Initialize with some values
print("Initial x:")
print(x)

while (max(x) - 1 > 1e-06) {
  x <- sqrt(x)
}

print("Final x after convergence:")
print(x)

# ============================================
# VECTORIZED ARITHMETIC
# ============================================
# How many languages add 2 vectors:

# Example: Adding two vectors element by element using a for loop
a <- c(1, 2, 3, 4, 5)
b <- c(10, 20, 30, 40, 50)

c <- vector(length=length(a))
for (i in 1:length(a)) {  c[i] <- a[i] + b[i]  }

print("Vector addition using for loop:")
print(c)

# ============================================
# VECTORIZED CALCULATIONS
# ============================================
# Vectorized calculations
# Many functions are set up to vectorize automatically

# Example: abs() function automatically works on entire vector
abs(-3:3)

# Example: log() function also vectorizes automatically
log(1:7)

# ============================================
# VECTORIZED CONDITIONS: ifelse()
# ============================================
# ifelse() allows vectorized conditional operations
# 1st argument is a Boolean vector, then pick from the 2nd or 3rd vector arguments as TRUE or FALSE

# Example: Piecewise function using ifelse()
x <- seq(-3, 3, by = 0.5)
print("x values:")
print(x)

result <- ifelse(x^2 > 1, 2*abs(x)-1, x^2)
print("ifelse(x^2 > 1, 2*abs(x)-1, x^2):")
print(result)

# ============================================
# WHAT IS TRUTH?
# ============================================
# 0 counts as FALSE; other numeric values count as TRUE
# The strings "TRUE" and "FALSE" count as you'd hope
# Most everything else gives an error
#
# Advice: Don't play games here; try to make sure control expressions are getting Boolean values
#
# Conversely, in arithmetic, FALSE is 0 and TRUE is 1

# Example: Using TRUE/FALSE as numbers
library(datasets)
states <- data.frame(state.x77, abb=state.abb, region=state.region, division=state.division)

# Calculate the proportion of states with murder rate > 7
# TRUE becomes 1, FALSE becomes 0, so mean gives the proportion
proportion_high_murder <- mean(states$Murder > 7)
print("Proportion of states with Murder rate > 7:")
print(proportion_high_murder)

# ============================================
# switch()
# ============================================
# Simplify nested if with switch(): give a variable to select on, then a value for each option

# Example: Different summary statistics based on type
type.of.summary <- "mean"  # Try changing this to "median", "histogram", or something else

result <- switch(type.of.summary,
       mean=mean(states$Murder),
       median=median(states$Murder),
       histogram=hist(states$Murder),
       "I don't understand")

print("Switch result:")
print(result)

# ============================================
# EXERCISE IMPLEMENTATION
# ============================================
# Set type.of.summary to, successively, "mean", "median", "histogram", and "mode", and explain what happens

# Test 1: "mean"
cat("\n--- Test 1: type.of.summary = 'mean' ---\n")
type.of.summary <- "mean"
result <- switch(type.of.summary,
       mean=mean(states$Murder),
       median=median(states$Murder),
       histogram=hist(states$Murder),
       "I don't understand")
print(result)
cat("Explanation: Returns the mean of Murder rates\n")

# Test 2: "median"
cat("\n--- Test 2: type.of.summary = 'median' ---\n")
type.of.summary <- "median"
result <- switch(type.of.summary,
       mean=mean(states$Murder),
       median=median(states$Murder),
       histogram=hist(states$Murder),
       "I don't understand")
print(result)
cat("Explanation: Returns the median of Murder rates\n")

# Test 3: "histogram"
cat("\n--- Test 3: type.of.summary = 'histogram' ---\n")
type.of.summary <- "histogram"
# Create plots directory if it doesn't exist
if (!dir.exists("plots")) {
  dir.create("plots")
}
png("plots/murder_histogram.png", width = 800, height = 600)
result <- switch(type.of.summary,
       mean=mean(states$Murder),
       median=median(states$Murder),
       histogram=hist(states$Murder, main="Histogram of Murder Rates", xlab="Murder Rate"),
       "I don't understand")
dev.off()
cat("Histogram saved to plots/murder_histogram.png\n")
cat("Explanation: Creates and displays a histogram of Murder rates\n")

# Test 4: "mode"
cat("\n--- Test 4: type.of.summary = 'mode' ---\n")
type.of.summary <- "mode"
result <- switch(type.of.summary,
       mean=mean(states$Murder),
       median=median(states$Murder),
       histogram=hist(states$Murder),
       "I don't understand")
print(result)
cat("Explanation: 'mode' is not in the switch cases, so it returns the default message 'I don't understand'\n")

# ============================================
# UNCONDITIONAL ITERATION
# ============================================
# repeat creates an infinite loop that continues forever unless explicitly stopped with break

# Example: Infinite loop (commented out to prevent actual infinite execution)
# WARNING: This will run forever if uncommented!
# repeat {
#   print("Help! I am Dr. Morris Culpepper, trapped in an endless loop!")
# }

# Practical example: Using repeat with a break condition
cat("\n--- Example: repeat with break condition ---\n")
counter <- 1
repeat {
  print(paste("Iteration:", counter))
  counter <- counter + 1
  if (counter > 3) {
    cat("Breaking out of the loop!\n")
    break  # Exit the loop when counter > 3
  }
}

# ============================================
# "MANUAL" CONTROL OVER ITERATION
# ============================================
# break() exits the loop; next() skips the rest of the body and goes back into the loop
# Both work with for(), while(), and repeat()

# Example: Using break() and next() in repeat loop
cat("\n--- Example: Manual control with break() and next() ---\n")

watched <- FALSE
rescued <- FALSE
iteration_count <- 0

repeat {
  iteration_count <- iteration_count + 1
  
  # Simulate different conditions
  if (iteration_count == 2) watched <- TRUE
  if (iteration_count == 3) watched <- FALSE
  if (iteration_count == 5) rescued <- TRUE
  
  if (watched) { 
    cat(paste("Iteration", iteration_count, ": Skipping (watched)\n"))
    next  # Skip the rest and go back to start of loop
  }
  
  print(paste("Iteration", iteration_count, ": Help! I am Dr. Morris Culpepper, trapped in an endless loop!"))
  
  if (rescued) { 
    cat("Dr. Culpepper has been rescued!\n")
    break  # Exit the loop
  }
}

# ============================================
# break() and next() with for() and while()
# ============================================

# Example: next() in a for loop - skip even numbers
cat("\n--- Example: next() in for loop - skip even numbers ---\n")
for (i in 1:5) {
  if (i %% 2 == 0) {
    next  # Skip even numbers
  }
  print(paste("Processing odd number:", i))
}

# Example: break() in a while loop - stop at threshold
cat("\n--- Example: break() in while loop ---\n")
total <- 0
i <- 1
while (i <= 10) {
  total <- total + i
  print(paste("i =", i, ", total =", total))
  if (total > 15) {
    cat("Total exceeded 15, breaking!\n")
    break
  }
  i <- i + 1
}

# ============================================
# EXERCISE
# ============================================
# EXERCISE
# ============================================
# Exercise: how would you replace while() with repeat()?
#
# Answer: while(condition) { body } 
# becomes:
# repeat { if (!condition) break; body }

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
# Note: R does not have a separate type for characters and strings
#   Both are represented as character vectors

mode("L")

mode("Lincoln")


class("Lincoln")

# ============================================
# MAKING STRINGS
# ============================================
# Use single or double quotes to construct a string
# Use nchar() to get the length of a single string
# Why do we prefer double quotes?

"Lincoln"


nchar("Lincoln")

"Abraham Lincoln"

nchar("Abraham Lincoln")

"As Lincoln never said, \"Four score and seven beers ago\""

# ============================================
# CHARACTER DATA TYPE
# ============================================
# One of the atomic data types, like numeric or logical
#
# Can go into scalars, vectors, arrays, lists, or be the type of a column in a data frame.

# Example: length() vs nchar()
# length() counts the number of elements in a vector
length("Abraham Lincoln's beard")


length(c("Abraham", "Lincoln's", "beard"))


# nchar() counts the number of characters in each string
nchar("Abraham Lincoln's beard")


nchar(c("Abraham", "Lincoln's", "beard"))

# ============================================
# CHARACTER-VALUED VARIABLES
# ============================================
# They work just like others, e.g., with vectors:

president <- "Lincoln"
nchar(president)  # NOT 9


presidents <- c("Fillmore","Pierce","Buchanan","Davis","Johnson")
presidents[3]

presidents[-(1:3)]

# ============================================
# DISPLAYING CHARACTERS
# ============================================
# We know print(), of course; cat() writes the string directly to the console.
# If you're debugging, message() is R's preferred syntax.

# Example: Different display functions
print("Abraham Lincoln")


cat("Abraham Lincoln")


cat(presidents)
cat("\n")  # Add newline for readability

message(presidents)

# ============================================
# SUBSTRING OPERATIONS
# ============================================
# Substring: a smaller string from the big string, but still a string in its own right.
# A string is not a vector or a list, so we cannot use subscripts like [[ ]] or []
# to extract substrings; we use substr() instead.

# Example: Extracting a substring
phrase <- "Christmas Bonus"
print("Original phrase:")
print(phrase)

result <- substr(phrase, start=8, stop=12)
print("substr(phrase, start=8, stop=12):")
print(result)
## [1] "as Bo"

# We can also use substr to replace elements:
substr(phrase, 13, 13) <- "g"
print("After replacing character 13 with 'g':")
print(phrase)
## [1] "Christmas Bogus"

# ============================================
# substr() FOR STRING VECTORS
# ============================================
# substr() vectorizes over all its arguments:

print("Presidents vector:")
print(presidents)
## [1] "Fillmore" "Pierce" "Buchanan" "Davis" "Johnson"

# First two characters
print("\nFirst two characters - substr(presidents, 1, 2):")
print(substr(presidents, 1, 2))
## [1] "Fi" "Pi" "Bu" "Da" "Jo"

# Last two characters
print("\nLast two characters - substr(presidents, nchar(presidents)-1, nchar(presidents)):")
print(substr(presidents, nchar(presidents)-1, nchar(presidents)))
## [1] "re" "ce" "an" "is" "on"

# No such substrings so return the null string
print("\nCharacters 20-21 (beyond string length) - substr(presidents, 20, 21):")
print(substr(presidents, 20, 21))
## [1] "" "" "" "" ""

# ============================================
# DIVIDING STRINGS INTO VECTORS
# ============================================
# strsplit() divides a string according to key characters, by splitting each element
# of the character vector x at appearances of the pattern split.

scarborough.fair <- "parsley, sage, rosemary, thyme"

# Split on comma only (leaves spaces at beginning of words)
print("\nSplitting on ',' - strsplit(scarborough.fair, ','):")
result1 <- strsplit(scarborough.fair, ",")
print(result1)
## [[1]]
## [1] "parsley"  " sage"    " rosemary" " thyme"

# Split on comma followed by space (cleaner result)
print("\nSplitting on ', ' - strsplit(scarborough.fair, ', '):")
result2 <- strsplit(scarborough.fair, ", ")
print(result2)
## [[1]]
## [1] "parsley"  "sage"     "rosemary" "thyme"

# Pattern is recycled over elements of the input vector:
print("\nSplitting multiple strings - strsplit(c(...), ', '):")
result3 <- strsplit(c(scarborough.fair, "Garfunkel, Oates", "Clement, McKenzie"), ", ")
print(result3)
## [[1]]
## [1] "parsley"  "sage"     "rosemary" "thyme"
## 
## [[2]]
## [1] "Garfunkel" "Oates"
## 
## [[3]]
## [1] "Clement"  "McKenzie"

# ============================================
# COMBINING VECTORS INTO STRINGS
# ============================================
# Converting one variable type to another is called casting:

# Obvious conversion
print("\nas.character(7.2):")
print(as.character(7.2))
## [1] "7.2"

# Obvious - scientific notation preserved
print("\nas.character(7.2e12):")
print(as.character(7.2e12))
## [1] "7.2e+12"

# Obvious - vector of numbers
print("\nas.character(c(7.2, 7.2e12)):")
print(as.character(c(7.2, 7.2e12)))
## [1] "7.2"     "7.2e+12"

# Not quite so obvious - scientific notation expanded
print("\nas.character(7.2e5):")
print(as.character(7.2e5))
## [1] "720000"

# ============================================
# BUILDING STRINGS FROM MULTIPLE PARTS
# ============================================
# The paste() function is very flexible!

# With one vector argument, works like as.character():
print("\npaste(41:45):")
print(paste(41:45))
## [1] "41" "42" "43" "44" "45"

# With 2 or more vector arguments, combines them with recycling:
print("\npaste(presidents, 41:45):")
print(paste(presidents, 41:45))
## [1] "Fillmore 41" "Pierce 42"   "Buchanan 43" "Davis 44"    "Johnson 45"

# Not historically accurate!
print("\npaste(presidents, c('R', 'D')):")
print(paste(presidents, c("R", "D")))
## [1] "Fillmore R" "Pierce D"   "Buchanan R" "Davis D"    "Johnson R"

# Multiple arguments with recycling
print("\npaste(presidents, '(', c('R','D'), 41:45, ')'):")
print(paste(presidents, "(", c("R","D"), 41:45, ")"))
## [1] "Fillmore ( R 41 )" "Pierce ( D 42 )"   "Buchanan ( R 43 )"
## [4] "Davis ( D 44 )"    "Johnson ( R 45 )"

# Changing the separator between pasted-together terms:
print("\npaste(presidents, ' (', 41:45, ')', sep='_'):")
print(paste(presidents, " (", 41:45, ")", sep="_"))
## [1] "Fillmore_ (_41_)" "Pierce_ (_42_)"   "Buchanan_ (_43_)"
## [4] "Davis_ (_44_)"    "Johnson_ (_45_)"

print("\npaste(presidents, ' (', 41:45, ')', sep=''):")
print(paste(presidents, " (", 41:45, ")", sep=""))
## [1] "Fillmore (41)" "Pierce (42)"   "Buchanan (43)" "Davis (44)"
## [5] "Johnson (45)"

# Exercise: what happens if you give sep a vector?
print("\nExercise - paste(presidents, 41:45, sep=c('-', '=')):")
print(paste(presidents, 41:45, sep=c("-", "=")))
## Only the first element of sep is used (with a warning)

# ============================================
# MORE COMPLICATED EXAMPLE OF RECYCLING
# ============================================
# Exercise: Convince yourself of why this works as it does

print("\npaste(c('HW','Lab'), rep(1:11, times=rep(2,11))):")
result <- paste(c("HW","Lab"), rep(1:11, times=rep(2,11)))
print(result)
## [1] "HW 1"   "Lab 1"  "HW 2"   "Lab 2"  "HW 3"   "Lab 3"  "HW 4"
## [8] "Lab 4"  "HW 5"   "Lab 5"  "HW 6"   "Lab 6"  "HW 7"   "Lab 7"
## [15] "HW 8"   "Lab 8"  "HW 9"   "Lab 9"  "HW 10"  "Lab 10" "HW 11"
## [22] "Lab 11"

# Explanation:
# - c("HW","Lab") is recycled 11 times to match length of second vector
# - rep(1:11, times=rep(2,11)) creates: 1, 1, 2, 2, 3, 3, ..., 11, 11
# - Result pairs HW/Lab with each number twice

# ============================================
# CONDENSING MULTIPLE STRINGS
# ============================================
# Producing one big string:

print("\npaste(presidents, ' (', 41:45, ')', sep='', collapse='; '):")
result_collapsed <- paste(presidents, " (", 41:45, ")", sep="", collapse="; ")
print(result_collapsed)
## [1] "Fillmore (41); Pierce (42); Buchanan (43); Davis (44); Johnson (45)"

# Default value of collapse is NULL - that is, it won't use it
print("\nWithout collapse (default):")
result_no_collapse <- paste(presidents, " (", 41:45, ")", sep="")
print(result_no_collapse)
## Returns a vector of 5 strings

# ============================================
# FUNCTION FOR WRITING REGRESSION FORMULAS
# ============================================
# R has a standard syntax for models: outcome ~ predictors.

my.formula <- function(dep, indeps, df) {
  rhs <- paste(colnames(df)[indeps], collapse="+")
  return(paste(colnames(df)[dep], "~", rhs, collapse=""))
}

# Example: Create formula using column indices
print("\nmy.formula(2, c(3,5,7), df=state.x77):")
formula_result <- my.formula(2, c(3,5,7), df=state.x77)
print(formula_result)
## [1] "Income ~ Illiteracy+Murder+Frost"

# This function builds a formula string by:
# - Taking the dependent variable column index (dep)
# - Taking independent variable column indices (indeps)
# - Pasting column names together with "+" and "~"

# ============================================
# GENERAL SEARCH
# ============================================
# Use grep() to find which strings have a matching search term

# Sample text data
sample_text <- c("The quick brown fox", 
                 "jumps over the lazy dog",
                 "The fox was very clever",
                 "Dogs and foxes are animals",
                 "Cats are also animals")

print("\nSample text:")
print(sample_text)

# Find which strings contain "fox"
print("\ngrep('fox', sample_text):")
fox_indices <- grep("fox", sample_text)
print(fox_indices)
print("Strings containing 'fox':")
print(sample_text[fox_indices])

# Find which strings contain "dog" (case sensitive)
print("\ngrep('dog', sample_text):")
dog_indices <- grep("dog", sample_text)
print(dog_indices)
print("Strings containing 'dog':")
print(sample_text[dog_indices])

# Case-insensitive search
print("\ngrep('DOG', sample_text, ignore.case=TRUE):")
dog_insensitive <- grep("DOG", sample_text, ignore.case=TRUE)
print(dog_insensitive)

# ============================================
# RECONSTITUTING AND WORD COUNTING
# ============================================
# Make one long string, then split the words

# Combine all text into one string
print("\nReconstituting text:")
long_string <- paste(sample_text, collapse=" ")
print(long_string)

# Split into words
print("\nSplitting into words:")
words <- unlist(strsplit(long_string, " "))
print(words)

# Counting words with table()
print("\nCounting words with table():")
word_counts <- table(words)
print(word_counts)

# Most frequent words
print("\nMost frequent words:")
sorted_counts <- sort(word_counts, decreasing=TRUE)
print(head(sorted_counts, 5))

# ============================================
# TEXT PATTERNS AND REGULAR EXPRESSIONS
# ============================================
# Need to learn how to work with text patterns and not just constants
# Searching for text patterns using regular expressions

# Find strings starting with "The"
print("\ngrep('^The', sample_text):")
starts_with_the <- grep("^The", sample_text)
print(starts_with_the)
print(sample_text[starts_with_the])

# Find strings ending with "animals"
print("\ngrep('animals$', sample_text):")
ends_with_animals <- grep("animals$", sample_text)
print(ends_with_animals)
print(sample_text[ends_with_animals])

# Find strings containing any digit
numbers_text <- c("There are 3 foxes", "No numbers here", "Year 2023", "Another text")
print("\ngrep('[0-9]', numbers_text):")
has_digit <- grep("[0-9]", numbers_text)
print(has_digit)
print(numbers_text[has_digit])

# Find strings with "fox" or "dog" using pattern
print("\ngrep('fox|dog', sample_text):")
fox_or_dog <- grep("fox|dog", sample_text)
print(fox_or_dog)
print(sample_text[fox_or_dog])

cat("\n=== Regular Expression Patterns ===\n")
cat("^ - Start of string\n")
cat("$ - End of string\n")
cat("[0-9] - Any digit\n")
cat("| - OR operator\n")
cat(". - Any character\n")
cat("* - Zero or more repetitions\n")
cat("+ - One or more repetitions\n")



 











