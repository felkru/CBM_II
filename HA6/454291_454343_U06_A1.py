# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d  # Added for 1b optimization
from scipy.stats import norm  # For comment in 1b

print('--- Ex. 1a) ---')

x, w, y = sp.symbols('x w y')

# Base PDF f(x) uniform on [0,1]
f_sym = sp.Piecewise((1, sp.And(x >= 0, x <= 1)), (0, True))

print("Base PDF f(x):")
print(f_sym)
print("-" * 20)

# PDF for k=1
fw_k1 = sp.Piecewise((1, sp.And(w >= 0, w <= 1)), (0, True))

print("PDF for k=1 (fw_k1):")
print("Zusammenfassend gilt für k=1:")
print("fw(w) = Piecewise(")
for i, (expr, cond) in enumerate(fw_k1.args):
    print(f"    ({expr}, {cond})" + ("," if i < len(fw_k1.args) - 1 else ""))
print(")")
print("-" * 20)

# PDF for k=2 (convolution of f with f)
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

# PDF for k=3 (convolution of fw_k2 with f)
fw_k2_piece1_symexpr = fw_k2.args[0][0]
fw_k2_piece2_symexpr = fw_k2.args[1][0]

integral_k3_case1 = sp.integrate(fw_k2_piece1_symexpr.subs(w, y), (y, 0, w))
integral_k3_case2 = sp.integrate(fw_k2_piece1_symexpr.subs(w, y), (y, w - 1, 1)) + \
                    sp.integrate(fw_k2_piece2_symexpr.subs(w, y), (y, 1, w))
integral_k3_case3 = sp.integrate(fw_k2_piece2_symexpr.subs(w, y), (y, w - 1, 2))

fw_k3 = sp.Piecewise(
    (integral_k3_case1, sp.And(w >= 0, w <= 1)),
    (integral_k3_case2, sp.And(w > 1, w <= 2)),
    (integral_k3_case3, sp.And(w > 2, w <= 3)),
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

w_values_1a = np.linspace(-0.1, 3.1, 400)
fw_k1_plot_values = fw_k1_np(w_values_1a)
fw_k2_plot_values = fw_k2_np(w_values_1a)
fw_k3_plot_values = fw_k3_np(w_values_1a)

plt.figure(figsize=(10, 6))
plt.plot(w_values_1a, fw_k1_plot_values, label='k=1 (Sympy)', color='blue')
plt.plot(w_values_1a, fw_k2_plot_values, label='k=2 (Sympy)', color='red', linestyle='--')
plt.plot(w_values_1a, fw_k3_plot_values, label='k=3 (Sympy)', color='green', linestyle='-.')

plt.title('PDFs (Sympy) - Sum of k U(0,1) Variables')
plt.xlabel('w')
plt.ylabel('fw(w)')
plt.ylim(-0.1, 1.1)
plt.xlim(-0.1, 3.1)
plt.grid(True)
plt.legend()
plt.show()

print('--- Ex. 1b) ---')


# Base PDF f(x) = 1 for 0 <= x <= 1, else 0 (numerical version)
def f_base_numeric(x_val):
    if 0 <= x_val <= 1:
        return 1.0
    else:
        return 0.0


pdf_integrands_for_next_step = []
pdf_integrands_for_next_step.append(f_base_numeric)

pdf_plot_values_list = []

num_grid_points = 600
w_grid = np.linspace(-0.5, 10.5, num_grid_points)

f1_grid_values = np.array([f_base_numeric(w_val) for w_val in w_grid])
pdf_plot_values_list.append(f1_grid_values)

for k_build in range(2, 11):
    print(f"Numerically constructing PDF for k={k_build}...")

    prev_pdf_as_integrand = pdf_integrands_for_next_step[k_build - 2]


    def current_fk_point_evaluator(w_arg_local):
        lower_x_limit = max(0, w_arg_local - 1)
        upper_x_limit = min(k_build - 1, w_arg_local)

        if lower_x_limit >= upper_x_limit:
            return 0.0

        integral_val, err = quad(prev_pdf_as_integrand, lower_x_limit, upper_x_limit,
                                 limit=100,  # Increased limit for quad
                                 epsabs=1.49e-8, epsrel=1.49e-8)

        if abs(integral_val) < 1e-9:
            return 0.0
        return integral_val


    current_fk_grid_values = np.array([current_fk_point_evaluator(w_val) for w_val in w_grid])
    pdf_plot_values_list.append(current_fk_grid_values)

    if k_build < 10:
        current_fk_interp = interp1d(w_grid, current_fk_grid_values,
                                     kind='cubic',  # Using cubic interpolation for smoother integrand
                                     bounds_error=False, fill_value=0.0)
        pdf_integrands_for_next_step.append(current_fk_interp)

plt.figure(figsize=(12, 8))
plot_colors = plt.get_cmap('tab10').colors

for idx, y_values_k_num in enumerate(pdf_plot_values_list):
    k_current = idx + 1
    plt.plot(w_grid, y_values_k_num, label=f'k={k_current} (Numerical)', color=plot_colors[idx % len(plot_colors)])

plt.title('PDFs of the Sum of k U(0,1) Variables (Numerical Integration with Interpolation)')
plt.xlabel('w')
plt.ylabel('fw(w)')
plt.grid(True)
plt.legend(loc='upper right', fontsize='small')
plt.ylim(bottom=-0.05)
plt.xlim(w_grid[0], w_grid[-1])
plt.show()

print("\n--- Limit Distribution for large k (Central Limit Theorem) ---")
print("As k increases, the probability density function fw(w) of the sum of k independent and")
print("identically distributed U(0,1) random variables approaches a Normal (Gaussian) distribution.")
print("This is a consequence of the Central Limit Theorem.")
print("For a sum W_k = X_1 + ... + X_k, where X_i ~ U(0,1):")
print("  Mean of X_i: E[X_i] = 0.5")
print("  Variance of X_i: Var[X_i] = (1-0)^2 / 12 = 1/12")
print("Therefore, for W_k:")
print("  Mean of W_k: E[W_k] = k * E[X_i] = k * 0.5 = k/2")
print("  Variance of W_k: Var[W_k] = k * Var[X_i] = k * (1/12) = k/12")
print("So, fw(w) approximates N(mean = k/2, variance = k/12).")