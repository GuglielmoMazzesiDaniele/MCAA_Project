import torch
import matplotlib.pyplot as plt
import N3Queens2DGrid
from utils.scheduler import ExponentialScheduler
from utils.plot import plot_energy_evolution

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
        log_file=f"params_N={args.N}_K={args.K}_rh={args.reheating}_pat={args.patience}_sch={config.scheduler.name()}_m_iter={args.max_iters}_m_iter_bo={args.max_iters_bo}_move={args.proposal_move}.log",
        K=args.K,
        reheating=args.reheating,
        patience=args.patience,
        name_proposal_move=args.proposal_move
    )    

    print(f"\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Minimal energy: {min(energies)}")
    
    plot_energy_evolution(energies, args, args.proposal_move, filename="pipeline_result.png")


def run_pipeline(args, scheduler, name_proposal_move):
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
        name_proposal_move=name_proposal_move
    )

    assignments, energies = q_problem.solve()
    
    print(f"\nPipeline complete!")
    print(f"Minimal energy: {min(energies)}")
    
    plot_energy_evolution(energies, args, name_proposal_move, filename="pipeline_result.png")
    
    return assignments, energies
