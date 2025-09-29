# Operators in Julia

# 1. Arithmetic Operators
println(7 + 5)   # 12
println(7 - 5)   # 2
println(7 * 5)   # 35
println(7 ^ 5)   # 16807

# 2. Relational (Comparison) Operators
# ==  equal to
# !=  not equal to
# >   greater than 
# <   less than
# >=  greater than or equal to
# <=  less than or equal to

# 3. Logical Operators
# &   bitwise AND
# |   bitwise OR
# !   NOT
# &&  short-circuit AND
# ||  short-circuit OR
println((5 > 7) & (6 * 7 == 42))   # false
println((5 > 7) | (6 * 7 == 42))   # true

# 4. Assignment Operators
x = 5   # Julia only uses "=" for assignment
println(x)

# Type checks
println(typeof(7))         # Int64 (or Int32 depending on system)
println(isa(7, Number))    # true

# Missing and NaN handling
println(ismissing(7))      # false
println(ismissing(missing))# true

println(isnan(7/0))        # false (gives Inf)
println(7/0)               # Inf
println(isnan(0/0))        # true (NaN)

# Character checks
println(isa(7, String))        # false
println(isa("7", String))      # true
println(isa("seven", String))  # true
println(ismissing("seven"))    # false

# Conversions
println(string(5/6))       # "0.8333333333333334"
println(parse(Float64, string(5/6)))   # 0.8333333333333334
println(6 * parse(Float64, string(5/6)))  # 5.0
println(5/6 == parse(Float64, string(5/6)))  # false
println(isapprox(5/6, parse(Float64, string(5/6)))) # true