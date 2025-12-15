import torch
import matplotlib.pyplot as plt
import N3Queens2DGrid
from utils.scheduler import *
from utils.plot import plot_energy_evolution, plot_all_schedulers, plot_vary_n

from optim.configs import BOConfig
from optim.bo_optim import optimize_with_bo
import numpy as np

def run_optimization(config: BOConfig, args):
    """Run Bayesian Optimization to find optimal hyperparameters."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
        gibbs=args.gibbs,
        name_proposal_move=args.proposal_move,
    )    

    print(f"\nOptimization complete!")
    print(f"Best parameters: {best_params}")
    print(f"Minimal energy: {min(energies)}")
    
    plot_energy_evolution(energies, args, args.proposal_move, filename=f"{config.scheduler.name()}_pipeline_result.png")


def run_multi_scheduler_optimization(scheduler_configs, args, stop_on_success=True):
    """
    Run Bayesian Optimization for multiple schedulers sequentially.
    
    Optimizes each scheduler configuration one after the other, logging results
    and optionally stopping as soon as a solution with energy=0 is found.
    
    Parameters:
    -----------
    scheduler_configs : dict
        Dictionary mapping scheduler names to BOConfig objects.
        Example: {
            "Exponential": BOConfig(...),
            "Linear": BOConfig(...),
            "Log": BOConfig(...)
        }
    args : argparse.Namespace
        Command line arguments containing N, max_iters, etc.
    stop_on_success : bool, optional
        If True, stops optimization across all schedulers as soon as one finds 
        a solution with energy=0. Default is True.
        
    Returns:
    --------
    dict : Dictionary containing results for each scheduler tested.
        Keys are scheduler names, values are dicts with:
        - 'best_params': optimized parameters
        - 'best_energy': minimal energy achieved
        - 'converged': whether energy=0 was reached
        - 'status': 'completed' or 'stopped_early'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    
    print(f"\n{'='*80}")
    print(f"Starting Multi-Scheduler Optimization Pipeline")
    print(f"N={args.N}, max_iters={args.max_iters}, max_iters_bo={args.max_iters_bo}")
    print(f"Stop on success: {stop_on_success}")
    print(f"{'='*80}\n")
    
    for scheduler_name, config in scheduler_configs.items():
        print(f"\n{'='*80}")
        print(f"Optimizing scheduler: {scheduler_name}")
        print(f"{'='*80}")
        
        log_file = f"params_N={args.N}_K={args.K}_rh={args.reheating}_pat={args.patience}_sch={scheduler_name}_m_iter={args.max_iters}_m_iter_bo={args.max_iters_bo}_move={args.proposal_move}.log"
        
        try:
            best_params, best_energy, energies = optimize_with_bo(
                N=args.N,
                device=device,
                bounds=config.bounds,
                scheduler_class=config.scheduler,
                output_parser=config.parser,
                max_iters_model=args.max_iters,
                max_iters_bo=args.max_iters_bo,
                log_file=log_file,
                K=args.K,
                reheating=args.reheating,
                patience=args.patience,
                gibbs=args.gibbs,
                name_proposal_move=args.proposal_move,
            )
            
            converged = (best_energy == 0)
            
            results[scheduler_name] = {
                'best_params': best_params,
                'best_energy': best_energy,
                'all_energies': energies,
                'converged': converged,
                'status': 'completed'
            }
            
            print(f"\n{'-'*80}")
            print(f"Scheduler '{scheduler_name}' optimization complete!")
            print(f"Best parameters: {best_params}")
            print(f"Best energy: {best_energy}")
            print(f"Converged to 0: {'YES ✓' if converged else 'NO ✗'}")
            print(f"{'-'*80}\n")
            
            if converged and stop_on_success:
                print(f"\n{'='*80}")
                print(f"SUCCESS! Scheduler '{scheduler_name}' found a solution with energy=0")
                print(f"Stopping optimization pipeline as requested.")
                print(f"Best parameters logged to: {log_file}")
                print(f"{'='*80}\n")
                
                remaining_schedulers = [s for s in scheduler_configs.keys() 
                                       if s not in results]
                for remaining in remaining_schedulers:
                    results[remaining] = {
                        'status': 'not_tested',
                        'reason': f'Stopped after {scheduler_name} found solution'
                    }
                
                break
                
        except Exception as e:
            print(f"\n⚠️  Error optimizing scheduler '{scheduler_name}': {e}")
            results[scheduler_name] = {
                'status': 'error',
                'error': str(e)
            }
            continue
    
    print(f"\n{'='*80}")
    print(f"Multi-Scheduler Optimization Pipeline Complete")
    print(f"{'='*80}")
    print(f"\nSummary:")
    for name, result in results.items():
        if result['status'] == 'completed':
            status_icon = '✓' if result['converged'] else '✗'
            print(f"  {status_icon} {name}: energy={result['best_energy']:.2f}, "
                  f"converged={'Yes' if result['converged'] else 'No'}")
        elif result['status'] == 'not_tested':
            print(f"  - {name}: {result['reason']}")
        elif result['status'] == 'error':
            print(f"  ✗ {name}: Error - {result['error']}")
    print(f"{'='*80}\n")
    
    summary_file = f"multi_scheduler_summary_N={args.N}_K={args.K}_rh={args.reheating}_pat={args.patience}_m_iter={args.max_iters}_m_iter_bo={args.max_iters_bo}_move={args.proposal_move}.log"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("Multi-Scheduler Optimization Results Summary\n")
        f.write("="*80 + "\n")
        f.write(f"N={args.N}, max_iters={args.max_iters}, max_iters_bo={args.max_iters_bo}\n")
        f.write(f"K={args.K}, reheating={args.reheating}, patience={args.patience}\n")
        f.write(f"gibbs={args.gibbs}, proposal_move={args.proposal_move}\n")
        f.write("="*80 + "\n\n")
        
        for name, result in results.items():
            f.write(f"\nScheduler: {name}\n")
            f.write("-"*80 + "\n")
            if result['status'] == 'completed':
                f.write(f"Status: {'CONVERGED ✓' if result['converged'] else 'NOT CONVERGED ✗'}\n")
                f.write(f"Best Energy: {result['best_energy']:.6f}\n")
                f.write(f"Best Parameters: {result['best_params']}\n")
                f.write(f"Individual log file: params_N={args.N}_K={args.K}_rh={args.reheating}_pat={args.patience}_sch={name}_m_iter={args.max_iters}_m_iter_bo={args.max_iters_bo}_move={args.proposal_move}.log\n")
            elif result['status'] == 'not_tested':
                f.write(f"Status: NOT TESTED\n")
                f.write(f"Reason: {result['reason']}\n")
            elif result['status'] == 'error':
                f.write(f"Status: ERROR\n")
                f.write(f"Error: {result['error']}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Consolidated summary saved to: {summary_file}\n")
    
    return results


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
        gibbs=args.gibbs,
        name_proposal_move=name_proposal_move
    )

    assignments, energies = q_problem.solve()
    
    if(energies[-1] == 0):
        print(f"Final configuration that worked:\n{assignments}")
    
    print(f"\nPipeline complete!")
    print(f"Minimal energy: {min(energies)}")
    
    plot_energy_evolution(energies, args, name_proposal_move, filename="pipeline_result.png")
    
    return assignments, energies

def multiple_simple_runs(args, scheduler, name_proposal_move, n_runs=5):
    """Run multiple independent runs of the N3Queens solver."""
    all_energies = []
    remaining_conflicting_queens = []
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
            name_proposal_move=name_proposal_move,
            gibbs=args.gibbs
        )

        config, energies, n_queens_conflict = q_problem.solve()
        
        if energies[-1] == 0:
            number_of_successes += 1
            print(f"Final configuration that worked:\n{config}")
        
        if len(energies) < args.max_iters:
            energies += [energies[-1]] * (args.max_iters - len(energies))
        
        remaining_conflicting_queens.append(n_queens_conflict)
        all_energies.append(energies)

    max_len = max(len(e) for e in all_energies)
    arr = np.full((len(all_energies), max_len), np.nan)
    for i, energies in enumerate(all_energies):
        arr[i, :len(energies)] = energies

    avg_energies = np.nanmean(arr, axis=0).tolist()
    std_errors = (np.nanstd(arr, axis=0) / np.sqrt(n_runs)).tolist()
    
    plot_energy_evolution(avg_energies, number_of_successes, args, name_proposal_move, filename="average_energy.png")
    
    return avg_energies, std_errors, number_of_successes, np.mean(np.array(remaining_conflicting_queens))

def run_all_schedulers(args, name_proposal_move, n_runs=10):
    """Run multiple schedulers and compare their performance."""
    
    schedulers = [
        LogScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters),
        ExponentialScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters),
        ConstantScheduler(beta=2.5),
        PowerScheduler(beta_max=10.0, total_iters=args.max_iters),
        LogisticScheduler(beta_max=args.end_beta, k=10, total_iters=args.max_iters),
    ]
    
    energies = {}
    queens_conflicts = {}
    
    for scheduler in schedulers:
        print(f"\nRunning scheduler: {scheduler.name()}")
        avg_energies, std_errors, number_of_successes, avg_queens_conflict = multiple_simple_runs(args, scheduler=scheduler, name_proposal_move=name_proposal_move, n_runs=n_runs)
        energies[scheduler.name()] = {
            'energy': avg_energies,
            'std_errors': std_errors,
            'successes': number_of_successes
        }
        queens_conflicts[scheduler.name()] = avg_queens_conflict

    plot_all_schedulers(energies, args, name_proposal_move, n_runs, filename=f"{args.N}_{args.max_iters}_gibbs={args.gibbs}_all_schedulers_comparison.png")
    plot_queens_conflict_comparison(queens_conflicts, args, name_proposal_move, n_runs, filename=f"{args.N}_{args.max_iters}_gibbs={args.gibbs}_queens_conflict_comparison.png")

def plot_queens_conflict_comparison(queens_conflicts, args, name_proposal_move, n_runs, filename="queens_conflict_comparison.png"):
    """Plot bar chart comparing average queens conflict across schedulers."""
    
    scheduler_names = list(queens_conflicts.keys())
    conflict_values = list(queens_conflicts.values())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(scheduler_names)))
    bars = ax.bar(scheduler_names, conflict_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    for bar, value in zip(bars, conflict_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Scheduler', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Conflicting Queens', fontsize=12, fontweight='bold')
    ax.set_title(f'Average Conflicting Queens per Scheduler\n(N={args.N}, {n_runs} runs, Move: {name_proposal_move})', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    
    param_text = (
        f"N = {args.N}\n"
        f"runs = {n_runs}\n"
        f"K = {args.K}\n"
        f"beta = {args.beta}\n"
        f"max_iters = {args.max_iters}\n"
        f"reheating = {args.reheating}\n"
        f"patience = {args.patience if args.reheating else 0}\n"
        f"move = {name_proposal_move}"
    )
    
    ax.text(0.98, 0.98, param_text,
            transform=ax.transAxes,
            va='top', ha='right',
            fontsize=9,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Queens conflict comparison plot saved to: {filename}")


def vary_n_values(args, scheduler, name_proposal_move, n_min, n_max, n_runs=5, name_complement=''):
    all_minimal_energies = []
    number_of_successes = np.zeros(n_max - n_min + 1, dtype=int)
    i = 0
    
    for N in range(n_min, n_max + 1):
        temp = []
        for run_idx in range(n_runs):
            print(f"\nStarting run with N={N}, {run_idx + 1}/{n_runs}")
            
            q_problem = N3Queens2DGrid.N3Queens(
                N=N,
                max_iters=args.max_iters,
                scheduler=scheduler,
                beta=args.beta,
                reheating=args.reheating,
                patience=args.patience,
                K=args.K,
                name_proposal_move=name_proposal_move
            )

            config, energies = q_problem.solve()
            
            if energies[-1] == 0:
                number_of_successes[i] += 1
                print(f"Final configuration that worked:\n{config}")
                
            temp.append(min(energies))

        avg_energies = np.mean(temp).tolist()
        all_minimal_energies.append(avg_energies)
        i += 1
        
        
    plot_vary_n(all_minimal_energies, number_of_successes, args, name_proposal_move, n_min, n_max, n_runs, filename=f"{name_complement}_vary_n_comparison.png")
    return None, None


def vary_n_values_multiple_schedulers(args, scheduler_dict, name_proposal_move, n_min, n_max, n_runs=5, name_complement=''):
    """
    Run multiple schedulers across varying N values and compare their performance.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command line arguments
    scheduler_dict : dict
        Dictionary mapping scheduler names to callables that create scheduler instances
        Example: {"Exponential": lambda: ExponentialScheduler(...), "Constant": lambda: ConstantScheduler(...)}
    name_proposal_move : str
        Name of the proposal move strategy
    n_min : int
        Minimum N value to test
    n_max : int
        Maximum N value to test
    n_runs : int
        Number of runs per configuration
    name_complement : str
        Additional string for the output filename
        
    Returns:
    --------
    dict : Dictionary with scheduler names as keys and their results as values
    """
    all_results = {}
    
    for scheduler_name, schedule in scheduler_dict.items():
        print(f"\n{'='*60}")
        print(f"Testing scheduler: {scheduler_name}")
        print(f"{'='*60}")
        
        all_minimal_energies = []
        all_std_errors = []
        number_of_successes = np.zeros(n_max - n_min + 1, dtype=int)
        i = 0
        
        for N in range(n_min, n_max + 1):
            temp = []
            for run_idx in range(n_runs):
                print(f"\nStarting run with N={N}, {run_idx + 1}/{n_runs} (Scheduler: {scheduler_name})")
                
                scheduler = schedule
                
                q_problem = N3Queens2DGrid.N3Queens(
                    N=N,
                    max_iters=args.max_iters,
                    scheduler=scheduler,
                    beta=args.beta,
                    reheating=args.reheating,
                    patience=args.patience,
                    K=args.K,
                    name_proposal_move=name_proposal_move
                )

                config, energies, _ = q_problem.solve()
                
                if energies[-1] == 0:
                    number_of_successes[i] += 1
                    print(f"Final configuration that worked:\n{config}")
                    
                temp.append(min(energies))

            avg_energies = np.mean(temp)
            std_error = np.std(temp) / np.sqrt(n_runs) 
            all_minimal_energies.append(avg_energies)
            all_std_errors.append(std_error)
            i += 1
        
        all_results[scheduler_name] = {
            'minimal_energies': all_minimal_energies,
            'std_errors': all_std_errors,
            'successes': number_of_successes
        }
    
    _plot_vary_n_multiple_schedulers(all_results, args, name_proposal_move, n_min, n_max, n_runs, 
                                     filename=f"{name_complement}_vary_n_schedulers_comparison.png")
    
    return all_results


def _plot_vary_n_multiple_schedulers(results, args, name_proposal_move, n_min, n_max, n_runs, filename):
    """Helper function to plot multiple schedulers on the same graph."""
    Ns = np.arange(n_min, n_max + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for (scheduler_name, data), color in zip(results.items(), colors):
        means = np.array(data['minimal_energies'])
        std_errors = np.array(data['std_errors'])

        ax.plot(Ns, means, marker='o', label=scheduler_name, 
                linewidth=2, markersize=8, color=color)
        
        ax.fill_between(Ns, means - std_errors, means + std_errors,
                        alpha=0.2, color=color)
    
    ax.set_xlabel("N")
    ax.set_ylabel("Minimal Energy")
    ax.set_title(f"Minimal Energy per N ({n_runs} runs each)")
    ax.legend()
    
    ymin, ymax = ax.get_ylim()
    
    num_schedulers = len(results)
    base_offset = 0.05
    spacing = 0.04
    
    for scheduler_idx, ((scheduler_name, data), color) in enumerate(zip(results.items(), colors)):
        y_text_level = ymin - (base_offset + scheduler_idx * spacing) * (ymax - ymin)
        
        for x, s in zip(Ns, data['successes']):
            success_color = "green" if s > 0 else "red"
            ax.text(
                x, y_text_level,
                f"{s}/{n_runs}",
                ha='center',
                va='top',
                fontsize=8,
                color=success_color,
                bbox=dict(facecolor=color, edgecolor=success_color, boxstyle="round,pad=0.2", alpha=0.3)
            )
    
    total_offset = base_offset + (num_schedulers - 1) * spacing + 0.05
    ax.set_ylim(ymin - total_offset * (ymax - ymin), ymax)
    
    plt.grid(True)
    plt.subplots_adjust(left=0.28)
    
    patience_display = args.patience if args.reheating else 0
    
    param_text = (
        f"n_min = {n_min}\n"
        f"n_max = {n_max}\n"
        f"runs = {n_runs}\n"
        f"K = {args.K}\n"
        f"beta = {args.beta}\n"
        f"max_iters = {args.max_iters}\n"
        f"reheating = {args.reheating}\n"
        f"patience = {patience_display}\n"
        f"move = {name_proposal_move}"
    )
    
    fig.text(
        0.02, 0.5, param_text,
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.85)
    )
    
    plt.savefig(filename, dpi=300)
    plt.close()
    
    print(f"\nMulti-scheduler comparison plot saved to: {filename}")




