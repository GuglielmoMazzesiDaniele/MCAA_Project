import torch
import matplotlib.pyplot as plt
import N3Queens2DGrid
from utils.scheduler import ExponentialScheduler

from optim.configs import BOConfig
from optim.bo_optim import optimize_with_bo

def run_optimization(config: BOConfig, args):
    """Run Bayesian Optimization to find optimal hyperparameters."""
    device = torch.device(args.device)
    
    best_params, best_energy, energies = optimize_with_bo(
        N=args.N,
        device=device,
        bounds=config.bounds,
        scheduler_class=config.scheduler,
        output_parser=config.parser,
        max_iters_model=args.max_iters,
        max_iters_bo=args.max_iters_bo,
        log_file=f"successful_params_N={args.N}_K={args.K}",
        K=args.K,
        reheating=args.reheating,
        patience=args.patience,
        gibbs=args.gibbs
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


def run_pipeline(args, scheduler):
    """Run the N3Queens solver with specified parameters.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    scheduler : Scheduler
        Scheduler instance to use for the run
    """
    q_problem = N3Queens2DGrid.N3Queens(
        N=args.N,
        max_iters=args.max_iters,
        scheduler=scheduler,
        beta=args.beta,
        reheating=args.reheating,
        patience=args.patience,
        K=args.K,
        gibbs=args.gibbs
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
