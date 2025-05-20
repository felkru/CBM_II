print("\n--- c) Unfolding data using Likelihood method (MLEM) ---")

c_likelihood = np.ones(N_bins) * (np.sum(d_measured) / N_bins)
c_likelihood[c_likelihood <= 0] = 1.0

num_iterations_likelihood = 100
epsilon = 1e-12

print(f"Starting MLEM iterations ({num_iterations_likelihood} iterations)...")

for iteration in range(num_iterations_likelihood):
    Rc = R_matrix @ c_likelihood
    ratio = d_measured / (Rc + epsilon)
    correction_factor = R_matrix.T @ ratio
    c_likelihood_new = c_likelihood * correction_factor

    if np.allclose(c_likelihood, c_likelihood_new, rtol=1e-5, atol=1e-8):
        print(f"MLEM method converged after {iteration + 1} iterations.")
        c_likelihood = c_likelihood_new
        break

    c_likelihood = c_likelihood_new

else:
    print(f"MLEM method finished {num_iterations_likelihood} iterations without converging.")

c_likelihood[c_likelihood < 0] = 0

plot_histogram_data(bin_edges, [d_measured, c_likelihood],
                    ["Measured Data (d)", "Unfolded (Likelihood/MLEM)"],
                    f"Unfolding via Likelihood Method (MLEM, {iteration + 1} iter.)",
                    "Bin Index", "Counts")