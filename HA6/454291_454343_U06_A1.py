import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, w, y = sp.symbols('x w y')

f_sym = sp.Piecewise((1, sp.And(x >= 0, x <= 1)), (0, True))

print("Base PDF f(x):")
print(f_sym)
print("-" * 20)

fw_k1 = sp.Piecewise((1, sp.And(w >= 0, w <= 1)), (0, True))

print("PDF for k=1 (fw_k1):")
print("Zusammenfassend gilt für k=1:")
print("fw(w) = Piecewise(")
for i, (expr, cond) in enumerate(fw_k1.args):
    print(f"    ({expr}, {cond})" + ("," if i < len(fw_k1.args) - 1 else ""))
print(")")
print("-" * 20)

fw_k2 = sp.Piecewise(
    (sp.integrate(1, (x, 0, w)), sp.And(w >= 0, w <= 1)),
    (sp.integrate(1, (x, w - 1, 1)), sp.And(w > 1, w <= 2)),
    (0, True)
)

fw_k2 = sp.simplify(fw_k2)

print("PDF for k=2 (fw_k2, calculated piecewise):")
print("Zusammenfassend gilt für k=2:")
print("fw(w) = Piecewise(")
for i, (expr, cond) in enumerate(fw_k2.args):
    print(f"    ({expr}, {cond})" + ("," if i < len(fw_k2.args) - 1 else ""))
print(")")
print("-" * 20)

fw_k2_piece1 = fw_k2.args[0][0]
integral_case1 = sp.integrate(fw_k2_piece1.subs(w, y), (y, 0, w))

fw_k2_piece1 = fw_k2.args[0][0]
fw_k2_piece2 = fw_k2.args[1][0]

integral_case2 = sp.integrate(fw_k2_piece1.subs(w, y), (y, w - 1, 1)) + sp.integrate(fw_k2_piece2.subs(w, y), (y, 1, w))

fw_k2_piece2 = fw_k2.args[1][0]
integral_case3 = sp.integrate(fw_k2_piece2.subs(w, y), (y, w - 1, 2))

fw_k3 = sp.Piecewise(
    (integral_case1, sp.And(w >= 0, w <= 1)),
    (integral_case2, sp.And(w > 1, w <= 2)),
    (integral_case3, sp.And(w > 2, w <= 3)),
    (0, True)
)

fw_k3 = sp.simplify(fw_k3)

print("PDF for k=3 (fw_k3, calculated piecewise):")
print("Zusammenfassend gilt für k=3:")
print("fw(w) = Piecewise(")
for i, (expr, cond) in enumerate(fw_k3.args):
    print(f"    ({expr}, {cond})" + ("," if i < len(fw_k3.args) - 1 else ""))
print(")")
print("-" * 20)

fw_k1_np = sp.lambdify([w], fw_k1, 'numpy')
fw_k2_np = sp.lambdify([w], fw_k2, 'numpy')
fw_k3_np = sp.lambdify([w], fw_k3, 'numpy')

w_values = np.linspace(0, 3.1, 400)

fw_k1_values = fw_k1_np(w_values)
fw_k2_values = fw_k2_np(w_values)
fw_k3_values = fw_k3_np(w_values)

plt.figure(figsize=(10, 6))
plt.plot(w_values, fw_k1_values, label='k=1 (Uniform)', color='blue')
plt.plot(w_values, fw_k2_values, label='k=2 (Triangular)', color='red', linestyle='--')
plt.plot(w_values, fw_k3_values, label='k=3', color='green', linestyle='-.')

plt.title('Probability Density Functions of the Sum of k Uniform U(0,1) Variables')
plt.xlabel('w')
plt.ylabel('fw(w)')
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 3.1)
plt.grid(True)
plt.legend()
plt.show()