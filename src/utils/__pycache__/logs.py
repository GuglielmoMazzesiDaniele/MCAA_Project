from datetime import datetime


def log_successful_params(log_file, params, beta, scheduler_params, energy, iteration, N, K):
    """Log parameters that successfully converged to energy 0."""
    with open(log_file, 'a') as f:
        params_str = ", ".join([f"param_{i}: {p:.6f}" for i, p in enumerate(scheduler_params)])
        f.write(f"[{datetime.now().isoformat()}] Iteration {iteration} | N={N}, K={K} | "
                f"beta: {beta:.6f}, {params_str} | Final energy: {energy}\n")
        
def log_config(log_file, config):
    """Log config that successfully converged to energy 0."""
    with open(log_file, 'a') as f:
        f.write(f"[{datetime.now().isoformat()}] | {config}\n")