import torch
import matplotlib.pyplot as plt
import N3Queens2DGrid
from utils.scheduler import *
from utils.plot import plot_energy_evolution, plot_all_schedulers

from optim.configs import BOConfig
from optim.bo_optim import optimize_with_bo
import numpy as np

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

def multiple_simple_runs(args, scheduler, name_proposal_move, n_runs=5):
    """Run multiple independent runs of the N3Queens solver."""
    all_energies = []
    number_of_successes = 0
    
    for run_idx in range(n_runs):
        print(f"\nStarting run {run_idx + 1}/{n_runs}")
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

        _, energies = q_problem.solve()
        
        # Record success
        if energies[-1] == 0:
            number_of_successes += 1
        
        # Pad energies to max_iters if needed, if we get 0 at some point and we exit early
        if len(energies) < args.max_iters:
            energies += [energies[-1]] * (args.max_iters - len(energies))
            
        all_energies.append(energies)
    
    # Average the energies over runs
    max_len = max(len(e) for e in all_energies)
    arr = np.full((len(all_energies), max_len), np.nan)
    for i, energies in enumerate(all_energies):
        arr[i, :len(energies)] = energies

    avg_energies = np.nanmean(arr, axis=0).tolist()
    
    plot_energy_evolution(avg_energies, args, name_proposal_move, filename="average_energy.png")
    
    return avg_energies, number_of_successes

def run_all_schedulers(args, name_proposal_move, n_runs=5):
    """Run multiple schedulers and compare their performance."""
    
    schedulers = [
        ExponentialScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters),
        ConstantScheduler(beta=args.beta),
        # GeometricScheduler(alpha=1.5),
        # LinearScheduler(a=2.0, b=0.0),
        LogScheduler(alpha=0.2)
    ]
    
    energies = {}
    
    for scheduler in schedulers:
        print(f"\nRunning scheduler: {scheduler.name()}")
        avg_energies, number_of_successes = multiple_simple_runs(args, scheduler=scheduler, name_proposal_move=name_proposal_move, n_runs=n_runs)
        energies[scheduler.name()] = {
            'energy': avg_energies,
            'successes': number_of_successes
        }
        
    plot_all_schedulers(energies, args, name_proposal_move, n_runs, filename="all_schedulers_comparison.png")





