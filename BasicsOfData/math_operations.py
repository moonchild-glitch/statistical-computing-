import math
import numpy as np

# --------------------------------
# 1. Arithmetic Operators
print("Arithmetic Operators:")
print(7 + 5)   # 12
print(7 - 5)   # 2
print(7 * 5)   # 35
print(7 ** 5)  # 16807
print() 

# --------------------------------
# 2. Relational Operators
print("Relational Operators:")
print(7 == 5)   # False
print(7 != 5)   # True
print(7 > 5)    # True
print(7 < 5)    # False
print(7 >= 5)   # True
print(7 <= 5)   # False
print()

# --------------------------------
# 3. Logical Operators
print("Logical Operators:")
print((5 > 7) and (6 * 7 == 42))  # False
print((5 > 7) or (6 * 7 == 42))   # True
print()

# --------------------------------
# 4. Assignment
x = 5
print("Assignment Example: x =", x)
print()

# --------------------------------
# 5. Type checks
print("Type checks:")
print(type(7))                       # <class 'int'>
print(isinstance(7, (int, float)))   # True
print()

# --------------------------------
# 6. Handling special values
print("Special values:")
try:
    print(7/0)   # raises ZeroDivisionError
except ZeroDivisionError:
    print("7/0 -> Infinity in R, but ZeroDivisionError in Python. Use float('inf') instead.")
print(float('inf'))   # inf

try:
    print(0/0)   # raises ZeroDivisionError
except ZeroDivisionError:
    print("0/0 -> NaN in R, but ZeroDivisionError in Python. Use float('nan') instead.")
print(math.isnan(float('nan')))  # True
print()

# --------------------------------
# 7. Character checks
print("Character checks:")
print(isinstance(7, str))        # False
print(isinstance("7", str))      # True
print(isinstance("seven", str))  # True
print()

# --------------------------------
# 8. Conversions
print("Conversions:")
val = 5/6
as_char = str(val)
print(as_char)                    # string "0.8333..."
back_to_num = float(as_char)
print(back_to_num)                # 0.8333...
print(6 * back_to_num)            # ~5.0
print(val == back_to_num)         # False (precision issue)
print(math.isclose(val, back_to_num, rel_tol=1e-9))  # True
print()

# --------------------------------
# 9. Built-in constants
print("Constants:")
print(math.pi)                     # 3.14159...
print(math.pi * 10)                # 31.4159...
print(math.cos(math.pi))           # -1.0
print()

# --------------------------------
# 10. Variables
print("Variables:")
approx_pi = 22/7
diameter_in_cubits = 10
print(approx_pi * diameter_in_cubits)  # 31.4285...
circumference_in_cubits = approx_pi * diameter_in_cubits
print(circumference_in_cubits)
circumference_in_cubits = 30
print(circumference_in_cubits)
print()

# --------------------------------
# 11. NumPy Arrays (Vectors in R)
print("Vectors with NumPy:")
x = np.array([7, 8, 10, 45])
print(x)
print(x[0])       # first element (7) - R uses 1-based indexing
print(x[2])       # 10
print(x[1:4])     # slice -> [ 8 10 45 ]
print(np.delete(x, 3))  # remove last element
print()

# --------------------------------
# 12. Vector Operations
print("Vectorized operations:")
y = np.array([-7, -8, -10, -45])
print(x + y)   # [0 0 0 0]
print(x * y)   # [-49 -64 -100 -2025]
print(x - y)   # [14 16 20 90]
print(x / y)   # [-1. -1. -1. -1.]
print()

# Broadcasting note: NumPy does NOT auto-recycle shorter arrays like R.
# To mimic recycling, ensure shapes are compatible (match lengths or reshape):
print(x + np.array([1, 2, 1, 2]))  # [ 8 10 11 47]
print(x ** np.array([1, 0, -1, 0.5]))  # [7. 1. 0.1 6.708]
print()

# --------------------------------
# 13. Boolean comparisons
print("Boolean comparisons:")
print(x > 9)                 # [False False  True  True]
print((x > 9) & (x < 20))    # [False False  True False]
print((x < 10) | (x > 40))   # [ True  True False  True]
print(x[x > 9])              # [10 45]
print(y[x > 9])              # [-10 -45]
print(np.where(x > 9))       # (array([2, 3]),)
print()

# --------------------------------
# 14. Floating-point precision
print("Precision checks:")
print(0.5 - 0.3 == 0.3 - 0.1)     # False
print(math.isclose(0.5 - 0.3, 0.3 - 0.1))  # True