import torch
import matplotlib.pyplot as plt
import N3Queens2DGrid
from utils.scheduler import ExponentialScheduler
from optim.bo_optim import optimize_with_bo


def exponential_parser(v):
    """
    Parse optimization vector for ExponentialScheduler.
    v[0]: beta (start_beta)
    v[1]: end_beta
    Returns: beta, start_beta, end_beta (max_iters is set separately)
    """
    beta = v[0]
    end_beta = v[1]
    return beta, beta, end_beta  # start_beta = beta


def run_optimization(args):
    """Run Bayesian Optimization to find optimal hyperparameters."""
    device = torch.device(args.device)
    
    # Bounds for [beta (start_beta), end_beta]
    bounds = torch.tensor([
        [0.01, 1.0],    # beta_min    | start_beta_min bounds
        [10.0, 100.0],  # beta_max    | end_beta_max bounds
    ], dtype=torch.double).T  # Transpose to get shape (2, num_params)
    
    def exponential_scheduler_factory(start_beta, end_beta):
        return ExponentialScheduler(start_beta=start_beta, end_beta=end_beta, max_iters=args.max_iters)
    
    best_params, best_energy, energies = optimize_with_bo(
        N=args.N,
        device=device,
        bounds=bounds,
        scheduler_class=exponential_scheduler_factory,
        output_parser=exponential_parser,
        max_iters_model=args.max_iters,
        max_iters_bo=args.max_iters_bo,
    )
    
    print(f"\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Best energy: {best_energy}")
    
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Energy evolution (best run) - Final: {best_energy}")
    plt.savefig("./bo_optimization_result.png")
    plt.close()


def run_pipeline(args):
    """Run the N3Queens solver with specified parameters."""
    scheduler = ExponentialScheduler(
        start_beta=args.beta,
        end_beta=args.end_beta,
        max_iters=args.max_iters,
    )
    
    q_problem = N3Queens2DGrid.N3Queens(
        N=args.N,
        max_iters=args.max_iters,
        scheduler=scheduler,
        beta=args.beta,
        reheating=args.reheating,
        patience=args.patience
    )

    assignments, energies = q_problem.solve()
    
    print(f"\nPipeline complete!")
    print(f"Final energy: {energies[-1]}")
    
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title(f"Energy evolution - Final: {energies[-1]}")
    plt.savefig("./pipeline_result.png")
    plt.close()
    
    return assignments, energies
