# 1)
# Matrikelnummer: 454291
# Name: Julia Els
# Email: julia.els@rwth-aachen.de
#
# 2)
# Matrikelnummer: 454343
# Name: Felix Krückel
# Email: felix.krueckel@rwth-aachen.de

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import pathlib

def plot_histogram_data(bin_edges, data_list, labels_list, title, xlabel="Value", ylabel="Counts", log_y=False, ylimit=0):
    """
    Parameters:
    - bin_edges (array-like): The edges of the bins (length N_bins + 1).
    - data_list (list of array-like): A list, where each element is an array of bin counts (length N_bins).
    - labels_list (list of str): A list of labels for each data set in data_list.
    - log_y (bool): Whether to use a logarithmic scale for the y-axis.
    """
    if not isinstance(data_list, list):
        data_list = [data_list]
        labels_list = [labels_list]
    plt.figure(figsize=(10, 6))
    for i, data_counts in enumerate(data_list):
        plt.stairs(data_counts, bin_edges, label=labels_list[i], linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log_y:
        plt.yscale('log')
        plt.ylim(bottom=0.1)  # Avoid issues with log(0)
    else:
        plt.ylim(bottom=ylimit)
    plt.grid(True, which="both", linestyle='--')
    plt.legend()
    plt.show()
# Define base path for data files
# Assumes the script is in HA6/ and data is in HA6/data/
base_path = pathlib.Path(__file__).parent
data_path = base_path / "data"

print("--- a) Reading data and plotting measured signal ---")
bins_file = data_path / "bins.txt"
bin_edges = np.loadtxt(bins_file)
print(f"Loaded {len(bin_edges) - 1} bins from {bins_file}")

folded_data_file = data_path / "FoldedData.txt" #gemessene Daten, 20 Zeilen für jedes Bin
d_measured = np.loadtxt(folded_data_file)
matrix_file = data_path / "convolution_matrix.csv"#20x20
R_matrix = np.loadtxt(matrix_file, delimiter=',')
print(f"Loaded {len(d_measured)} measured data points from {folded_data_file}")
print(f"Loaded convolution matrix of shape {R_matrix.shape} from {matrix_file}")
N_bins = len(d_measured)

plot_histogram_data(bin_edges, d_measured, "Measured Data (d)",
                    "Measured Signal", "Bin Index (implies energy or similar)", "Counts")

print("\n--- b) Unfolding data by direct matrix inversion ---")
try:
    R_inv = np.linalg.inv(R_matrix)
    c_inverted = R_inv @ d_measured #@ = Operator für Matrix-Multiplikation in Python. inverse Matrix wird mit Vektor/Matrix d_measured multipliziert. Ergebnis in c_inverted gespeichert.

    plot_histogram_data(bin_edges, [d_measured, c_inverted],
                        ["Measured Data (d)", "Unfolded (Matrix Inversion)"],
                        "Unfolding via Matrix Inversion",
                        "Bin Index", "Counts", False, -800)
except np.linalg.LinAlgError:
    print("Error: Convolution matrix is singular, cannot invert directly.")
    c_inverted = np.full_like(d_measured, np.nan)  # Placeholder if inversion fails

print("\n--- c) Unfolding data using Likelihood method ---")
# Initial guess for c: flat distribution scaled to total measured counts
# c_likelihood = np.full(N_bins, np.mean(d_measured)) # Ensures positivity
c_likelihood = np.ones(N_bins) * (np.sum(d_measured) / N_bins) #Mittelwert der geg. Daten, Starverteilung gleichverteilt
if np.sum(c_likelihood) <= 0:  # wenn d_measured Null ist oder negative Werte enthält, wird c_likelihood =Vektor aus 1-en
    c_likelihood = np.ones(N_bins)

num_iterations_likelihood = 100
for iteration in range(num_iterations_likelihood):
    Rc = R_matrix @ c_likelihood #  This is lambda = R @ c (Erwartungswert Poission)-> im Skript =y_i
    ratio = d_measured / (Rc) # This is n_i / lambda_i
    correction_factor = R_matrix.T @ ratio # This calculates sum_i R_ik * (n_i / lambda_i) for each k, wenn y=lambda, K=R
    c_likelihood_new = c_likelihood * correction_factor # This is the multiplicative update x_k^(t+1) = x_k^(t) * (sum_i R_ik * n_i/lambda_i)

    # Optional: Check for convergence (e.g., if change is small)
    if np.allclose(c_likelihood, c_likelihood_new, rtol=1e-5):
        print(f"Likelihood method converged after {iteration + 1} iterations.")
        break
    c_likelihood = c_likelihood_new
else:  # Executed if loop finishes without break
    print(f"Likelihood method finished {num_iterations_likelihood} iterations.")

plot_histogram_data(bin_edges, [d_measured, c_inverted, c_likelihood],
                    ["Measured Data (d)", "Unfolded (Matrix Inversion)", "Unfolded (Likelihood Method)"],
                    f"Unfolding via Likelihood Method ({num_iterations_likelihood} iter.)",
                    "Bin Index", "Counts", False, -800)

print("\n--- d) Unfolding data using Tikhonov Regularization ---")
# Construct L matrix for second derivative damping
# L_0 is (N-2) x N
L0 = np.zeros((N_bins - 2, N_bins))
for i in range(N_bins - 2):
    L0[i, i] = 1
    L0[i, i + 1] = -2
    L0[i, i + 2] = 1

S_reg_matrix = L0.T @ L0  # This is L^T L in formulas, (N x N)
# Weights for Poisson data: W_ii = 1/d_i (or 1/max(d_i,1) to avoid division by zero)
W = np.diag(1.0 / np.maximum(d_measured, 1.0))

# Experiment with tau values
tau_values_to_try = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
c_tikhonov_results = []

print("Trying Tikhonov regularization with different tau values:")
for tau in tau_values_to_try:
    try:
        # (R^T W R + tau^2 S)^-1 R^T W d
        term1 = R_matrix.T @ W @ R_matrix
        term2 = tau ** 2 * S_reg_matrix
        matrix_to_invert = term1 + term2

        c_tikhonov = np.linalg.inv(matrix_to_invert) @ R_matrix.T @ W @ d_measured
        c_tikhonov_results.append(c_tikhonov)
        print(f"  Tau = {tau}: Success.")
    except np.linalg.LinAlgError:
        print(f"  Tau = {tau}: Matrix inversion failed (singular).")
        c_tikhonov_results.append(np.full_like(d_measured, np.nan))

# Plot results for different taus
plot_labels_tikhonov = [f"Tikhonov (tau={t:.3f})" for t in tau_values_to_try]
plot_histogram_data(bin_edges, c_tikhonov_results, plot_labels_tikhonov,
                    "Tikhonov Unfolding with Various Tau Values",
                    "Bin Index", "Counts", log_y=False)  # Log y can be helpful here

# Choose a "good" tau for the final comparison. This is subjective.
# Let's pick one that seems to balance smoothness and feature preservation.
# For example, the one corresponding to tau=0.05 or 0.1 from the list.
chosen_tau_index = 2  # Corresponds to tau = 0.05 if list is unchanged
# Or find index of a specific tau, e.g. tau=0.1
try:
    chosen_tau_index = tau_values_to_try.index(0.05)
    print(f"Selected tau = {tau_values_to_try[chosen_tau_index]} for final Tikhonov result.")
except ValueError:
    print(f"Could not find chosen tau, using index {chosen_tau_index} by default.")
    chosen_tau_index = min(chosen_tau_index, len(c_tikhonov_results) - 1)

c_tikhonov_final = c_tikhonov_results[chosen_tau_index]
if np.isnan(c_tikhonov_final).any():  # Fallback if chosen tau failed
    print("Warning: Chosen tau led to failed inversion. Trying to find a successful one.")
    for res in c_tikhonov_results:
        if not np.isnan(res).any():
            c_tikhonov_final = res
            break
    else:  # If all failed (unlikely but possible)
        c_tikhonov_final = np.zeros_like(d_measured)


print("\n--- e) Comparison with true (Gaussian) distribution ---")

mu_true = 0.0
sigma_true = 1.1

# Bin the true Gaussian PDF
# Integrate Gauss(x; mu, sigma) over each bin
true_binned_probs = np.zeros(N_bins)
for i in range(N_bins):
    prob, _ = quad(lambda x: norm.pdf(x, loc=mu_true, scale=sigma_true),
                   bin_edges[i], bin_edges[i + 1]) #Berechnung bestimmtes Integral, returns (integral_value, error_estimate)
    true_binned_probs[i] = prob

# Scale binned probabilities to total number of events.
# Use sum of measured counts as an estimate for total true events.
# Alternatively, sum of Tikhonov unfolded counts could be used if it's more stable.
total_measured_counts = np.sum(d_measured)
# total_tikhonov_counts = np.sum(c_tikhonov_final) # Could use this as well

c_true_gaussian_scaled = true_binned_probs * total_measured_counts

plot_histogram_data(bin_edges,
                    [c_true_gaussian_scaled, d_measured, c_tikhonov_final],
                    ["True Gaussian (binned & scaled)",
                     "Measured Data (d)",
                     f"Unfolded (Tikhonov, tau={tau_values_to_try[chosen_tau_index]:.3f})"],
                    "Comparison of Signals with True Distribution",
                    "Bin Value (from bins.txt)", "Counts", log_y=False)