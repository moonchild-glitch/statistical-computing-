# Operators in Julia
# In Julia, operators are also special symbols or keywords that perform operations on values.
# Numbers in Julia are Int64 by default (or Float64 when decimals are used).

# ----------------------------------------
# 1. Arithmetic Operators
println("Arithmetic Operators:")
println(7 + 5)   # 12
println(7 - 5)   # 
println(7 * 5)   # 35
println(7 ^ 5)   # 16807
println()

# ----------------------------------------
# 2. Relational (Comparison) Operators
println("Relational Operators:")
println(7 == 5)   # false
println(7 != 5)   # true
println(7 > 5)    # true
println(7 < 5)    # false
println(7 >= 5)   # true
println(7 <= 5)   # false
println()

# ----------------------------------------
# 3. Logical Operators
println("Logical Operators:")
println((5 > 7) & (6*7 == 42))   # false  (elementwise AND for scalars = same as &&)
println((5 > 7) | (6*7 == 42))   # true   (elementwise OR for scalars = same as ||)
println(!(5 > 7))                # true   (NOT)
println()

# ----------------------------------------
# 4. Assignment Operators
x = 5
println("Assignment Example: x = ", x)
println()

# ----------------------------------------
# 5. Type and NA checks
println("Type checks:")
println(typeof(7))          # Int64 (not double by default like R)
println(isa(7, Number))     # true
println()

println("Missing / NaN checks:")
println(ismissing(7))       # false (Julia uses `missing` not NA)
println(ismissing(7/0))     # false → 7/0 is Inf in Julia
println(ismissing(0/0))     # false (this gives NaN, not missing)
println(isnan(0/0))         # true
println(isinf(7/0))         # true
println()

# ----------------------------------------
# 6. Character checks
println("Character checks:")
println(isa(7, String))        # false
println(isa("7", String))      # true
println(isa("seven", String))  # true
println()

# ----------------------------------------
# 7. Conversions
val = 5/6
as_char = string(val)
println(as_char)                      # "0.8333333333333334"
back_to_num = parse(Float64, as_char)
println(back_to_num)                  # 0.8333333333333334
println(6 * back_to_num)              # ≈ 5.0
println(val == back_to_num)           # false (precision issue)
println(isapprox(val, back_to_num))   # true (like all.equal in R)
println()

# ----------------------------------------
# 8. Built-in constants
println("Constants:")
println(pi)               # 3.141592653589793
println(pi * 10)          # 31.41592653589793
println(cos(pi))          # -1.0
println()

# ----------------------------------------
# 9. Variables
approx_pi = 22/7
diameter_in_cubits = 10
circumference_in_cubits = approx_pi * diameter_in_cubits
println(circumference_in_cubits)   # 31.42857
circumference_in_cubits = 30
println(circumference_in_cubits)   # 30
println()

# ----------------------------------------
# 10. Listing and removing variables
println("Listing variables in scope:")
println(names(Main))   # like ls() in R

# To remove a variable:
circumference_in_cubits = nothing   # assign to nothing
println()

# ----------------------------------------
# 11. Vectors
println("Vectors:")
x = [7, 8, 10, 45]   # numeric vector
println(x)
println(isa(x, Vector))   # true
println()

# Indexing
println(x[1])       # 7 (Julia uses 1-based indexing like R)
println(x[4])       # 45
println(x[1:3])     # [7,8,10]
println(deleteat!(copy(x), 4))  # drop 4th element
println()

# Other vector types
nums = [3.5, 4.2, 9.1]                 # numeric
name_list = ["Alice", "Bob", "Clifford"]   # character (avoid conflict with Base.names)
flags = [true, false, true]            # logical
println()

# ----------------------------------------
# 12. Elementwise operations
x = [7, 8, 10, 45]
y = [-7, -8, -10, -45]
println(x .+ y)    # [0,0,0,0]
println(x .* y)    # [-49, -64, -100, -2025]
println(x .- y)    # [14,16,20,90]
println(x ./ y)    # [-1.0, -1.0, -1.0, -1.0]
println()

# Broadcasting note: Julia does NOT auto-recycle shorter vectors like R.
# To mimic recycling, make lengths match explicitly:
println(x .+ [1,2,1,2])    # [8,10,11,47]
println(x .^ [1,0,-1,0.5]) # [7.0, 1.0, 0.1, 6.7082...]
println(2 .* x)            # [14,16,20,90]
println()

# ----------------------------------------
# 13. Boolean comparisons
x = [7, 8, 10, 45]
y = [-7, -8, -10, -45]
println(x .> 9)             # [false,false,true,true]
println((x .> 9) .& (x .< 20))   # [false,false,true,false]
println((x .< 10) .| (x .> 40))  # [true,true,false,true]
println(x[x .> 9])          # filtering → [10,45]
println(y[x .> 9])          # filter y with condition on x → [-10,-45]
println(findall(x .> 9))    # [3,4] like which() in R
println()

# ----------------------------------------
# 14. Floating point precision
println("Precision checks:")
println((0.5 - 0.3) == (0.3 - 0.1))        # false
println(isapprox(0.5 - 0.3, 0.3 - 0.1))    # true
println()