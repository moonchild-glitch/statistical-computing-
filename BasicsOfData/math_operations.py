import math

# 1. Arithmetic Operators
print("Arithmetic Operators:")
print(7 + 5)   # 12
print(7 - 5)   # 2
print(7 * 5)   # 35
print(7 ** 5)  # 16807
print()

# 2. Relational Operators
print("Relational Operators:")
print(7 == 5)   # False
print(7 != 5)   # True
print(7 > 5)    # True
print(7 < 5)    # False
print(7 >= 5)   # True
print(7 <= 5)   # False
print()

# 3. Logical Operators
print("Logical Operators:")
print((5 > 7) and (6 * 7 == 42))  # False
print((5 > 7) or (6 * 7 == 42))   # True
print()

# 4. Assignment
x = 5
print("Assignment Example: x =", x)
print()

# Type checks
print("Type checks:")
print(type(7))                       # <class 'int'>
print(isinstance(7, (int, float)))   # True
print()

# Handling special values
print("Special values:")
try:
    print(7/0)   # will raise ZeroDivisionError
except ZeroDivisionError:
    print("7/0 -> Infinity in R, but ZeroDivisionError in Python. Use float('inf') instead.")
print(float('inf'))   # inf

try:
    print(0/0)   # will raise ZeroDivisionError
except ZeroDivisionError:
    print("0/0 -> NaN in R, but ZeroDivisionError in Python. Use float('nan') instead.")
print(math.isnan(float('nan')))  # True
print()

# Character checks
print("Character checks:")
print(isinstance(7, str))        # False
print(isinstance("7", str))      # True
print(isinstance("seven", str))  # True
print()

# Conversions
print("Conversions:")
print(str(5/6))                  # string
print(float(str(5/6)))           # back to float
print(6 * float(str(5/6)))       # 5.0
print(5/6 == float(str(5/6)))    # False
print(math.isclose(5/6, float(str(5/6)), rel_tol=1e-9))  # True