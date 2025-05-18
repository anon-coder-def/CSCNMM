from scipy.stats import poisson
from scipy.optimize import minimize
import numpy as np

def fit_poisson_mixture_em(data, n_components=2, max_iter=100, tol=1e-4):
    """
    Fit a Poisson Mixture Model using Expectation-Maximization (EM).
    
    Args:
        data (np.ndarray): Input data (can include negative integers).
        n_components (int): Number of Poisson components.
        max_iter (int): Maximum number of EM iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        dict: Fitted Poisson mixture parameters.
    """
    # Step 1: Shift data to non-negative
    shift = abs(np.min(data)) + 1 if np.min(data) < 0 else 0
    shifted_data = data + shift

    # Initialize parameters
    n = len(shifted_data)
    pis = np.ones(n_components) / n_components  # Mixing coefficients
    lambdas = np.random.uniform(1, 10, n_components)  # Poisson rates

    # EM Algorithm
    for iteration in range(max_iter):
        # E-Step: Compute responsibilities
        responsibilities = np.zeros((n, n_components))
        for k in range(n_components):
            responsibilities[:, k] = pis[k] * poisson.pmf(shifted_data, lambdas[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)

        # M-Step: Update parameters
        N_k = responsibilities.sum(axis=0)
        pis_new = N_k / n
        lambdas_new = (responsibilities.T @ shifted_data) / N_k

        # Check convergence
        if np.max(np.abs(pis_new - pis)) < tol and np.max(np.abs(lambdas_new - lambdas)) < tol:
            break

        pis, lambdas = pis_new, lambdas_new

    # Return results
    return {
        "weights": pis,
        "rates": lambdas - shift,  # Adjust for shift
        "shift": shift
    }

# Example usage
data = np.random.randint(-10, 10, size=100)  # Example data with negative integers
result = fit_poisson_mixture_em(data)

print("Mixing Weights:", result["weights"])
print("Poisson Rates (Original Scale):", result["rates"])