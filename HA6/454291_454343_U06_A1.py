import sympy
from sympy import symbols, integrate, Piecewise, And, simplify, oo

# Define symbolic variables
x, w, y = symbols('x w y')

# Define the base PDF for a single uniform variable U(0, 1) using sympy's Piecewise
# f(x) = 1 for 0 <= x <= 1, and 0 otherwise
f_sym = Piecewise((1, And(x >= 0, x <= 1)), (0, True))

print("Base PDF f(x):")
print(f_sym)
print("-" * 20)

# --- Case k = 1 ---
# The density for k=1 is just the original density, but with w as the variable
fw_k1 = f_sym.subs(x, w)

print("PDF for k=1 (fw_k1):")
print(fw_k1)
print("-" * 20)


# --- Case k = 2 ---
# fw_k2(w) = integral(f(x) * f(w - x), dx)
# f(w - x) is the base PDF shifted and reflected: 1 for 0 <= w - x <= 1, i.e., w-1 <= x <= w
f_wx = f_sym.subs(x, w - x)

print("f(w - x):")
print(f_wx)
print("-" * 20)

# Perform the convolution integral symbolically
# The integration range is technically -oo to +oo, but sympy handles the Piecewise correctly
fw_k2 = integrate(f_sym * f_wx, x)

# Simplify the result if possible
fw_k2 = simplify(fw_k2)

print("PDF for k=2 (fw_k2, analytical check result):")
print(fw_k2) # Expected: Piecewise((w, w >= 0 & w <= 1), (-w + 2, w > 1 & w <= 2), (0, True))
print("-" * 20)


# --- Case k = 3 ---
# fw_k3(w) = integral(fw_k2(y) * f(w - y), dy)
# Replace w with y in fw_k2 to get fw_k2(y)
fw_k2_y = fw_k2.subs(w, y)

# f(w - y) is the base PDF with x replaced by w-y
f_wy = f_sym.subs(x, w - y)

print("fw_k2(y):")
print(fw_k2_y)
print("f(w - y):")
print(f_wy)
print("-" * 20)

# Perform the convolution integral symbolically
# The integration variable is y
fw_k3 = integrate(fw_k2_y * f_wy, y)

# Simplify the result
fw_k3 = simplify(fw_k3)

print("PDF for k=3 (fw_k3):")
print(fw_k3) # Expected: A piecewise polynomial on [0, 3]
print("-" * 20)

# --- Preparation for Plotting ---
# We need to convert the sympy expressions into numerical functions
# using lambdify so we can plot them with matplotlib.
import numpy as np
import matplotlib.pyplot as plt

# Lambdify the sympy expressions.
# The first argument is a list of the variables in the expression.
# The third argument specifies the library to use for evaluation (numpy).
fw_k1_np = sympy.lambdify([w], fw_k1, 'numpy')
fw_k2_np = sympy.lambdify([w], fw_k2, 'numpy')
fw_k3_np = sympy.lambdify([w], fw_k3, 'numpy')

# Define the range for the plot. The sum of k uniforms on [0,1] is supported on [0, k].
# We can plot from 0 up to slightly beyond 3 to show all three functions.
w_values = np.linspace(0, 3.1, 400) # Generate 400 points between 0 and 3.1

# Evaluate the numerical functions over the range
fw_k1_values = fw_k1_np(w_values)
fw_k2_values = fw_k2_np(w_values)
fw_k3_values = fw_k3_np(w_values)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(w_values, fw_k1_values, label='k=1 (Uniform)', color='blue')
plt.plot(w_values, fw_k2_values, label='k=2 (Triangular)', color='red', linestyle='--')
plt.plot(w_values, fw_k3_values, label='k=3', color='green', linestyle='-.')

plt.title('Probability Density Functions of the Sum of k Uniform U(0,1) Variables')
plt.xlabel('w')
plt.ylabel('fw(w)')
plt.ylim(-0.1, 1.1) # Set y-limits to include 0 and the max value (1 for k=1, 1 for k=2, 0.5 for k=3)
plt.xlim(-0.1, 3.1)
plt.grid(True)
plt.legend()
plt.show()