import argparse
from utils.runners import run_optimization, multiple_simple_runs, run_all_schedulers, vary_n_values_multiple_schedulers
import utils.scheduler as schedule
import utils.parsers as parse
from optim.configs import BOConfig
import torch
from utils.runners import run_multi_scheduler_optimization

"""Main entry point for N3 Queens Problem Solver."""

def main():
    parser = argparse.ArgumentParser(description='N3 Queens Problem Solver')
    parser.add_argument('--mode', type=str, choices=['optimize', 'all_schedulers', 'vary_n', 'run', 'optimize_all'], default='run',
                        help='Mode: "optimize" for BO optimization, "run" for single pipeline run')
    parser.add_argument('--N', type=int, default=8, help='Size of the board (N x N x N)')
    parser.add_argument('--max_iters', type=int, default=20000, help='Maximum number of iterations for the model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    
    # Optimization-specific arguments
    parser.add_argument('--K', type=int, default=None, help='Number of non-conflicting queens for smart initialization')
    parser.add_argument('--max_iters_bo', type=int, default=200, help='Maximum number of BO iterations (optimize mode only)')
    parser.add_argument('--reheating', action='store_true', help='Allow reheating')
    parser.add_argument('--patience', type=int, default=10000, help='Number of steps before reheating the model')
    
    # Pipeline-specific arguments
    parser.add_argument('--beta', type=float, default=0.1, help='Initial beta parameter (run mode only)')
    parser.add_argument('--end_beta', type=float, default=50.0, help='End beta parameter (run mode only)')

    parser.add_argument('--gibbs', action='store_true', help='Use gibbs sampling instead of Metropolis-Hastings')
    # Proposal Move
    parser.add_argument('--proposal_move', type=str, choices=['random', 'delta_move'], default='random',
                        help='Type of proposal move to use in the solver')
    
    args = parser.parse_args()

    if args.mode == 'optimize':
        _logistic_scheduler = schedule.LogisticScheduler
        _logistic_bounds = torch.tensor(
            [[10.0, 150.0],
             [15.0, 200.0]]
        )
        _logistic_parser = parse.logistic_parser

        config_logistic = BOConfig(
            parser=_logistic_parser,
            bounds=_logistic_bounds,
            scheduler=_logistic_scheduler
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        def step_parser(v):
            gamma = v[0].item()
            step_sz = int(v[1].item())
            return gamma, step_sz

        config_step = BOConfig(
            parser=step_parser,
            bounds=torch.tensor([
                [0.001,  25000.0],# Min: Hot start, slow cooling, fast steps
                [0.5,    100000.0]# Max: Warm start, fast cooling, slow steps
            ], dtype=torch.double, device=device),
            scheduler=schedule.StepScheduler
        )

        _scheduler = schedule.ExponentialScheduler
        _parser = parse.exponential_parser
        
        _bounds = torch.tensor(
            [[0.001, 1.0],
            [0.5, 100.0]]
        , dtype=torch.double, device="cuda")

        config = BOConfig(
            parser=_parser,
            bounds=_bounds,
            scheduler=_scheduler,
        )
        run_optimization(config, args)

    elif args.mode == 'optimize_all':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        config_exponential = BOConfig(
            parser=parse.exponential_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# start_beta
                [0.5, 10.0]# end_beta
            ], dtype=torch.double, device=device),
            scheduler=schedule.ExponentialScheduler
        )
        
        config_logistic = BOConfig(
            parser=parse.logistic_parser,
            bounds=torch.tensor([
                [5.0, 10.0],# beta_max
                [15.0, 200.0]# k (steepness)
            ], dtype=torch.double, device=device),
            scheduler=schedule.LogisticScheduler
        )
        
        config_step = BOConfig(
            parser=parse.step_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# initial beta
                [1.001, 25000.0],# gamma (multiplicative factor)
                [100.0, 100000.0]# step_size (iterations between increases)
            ], dtype=torch.double, device=device),
            scheduler=schedule.StepScheduler
        )
        
        config_log = BOConfig(
            parser=parse.log_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# start_beta
                [0.5, 10.0]# end_beta
            ], dtype=torch.double, device=device),
            scheduler=schedule.LogScheduler
        )
        
        config_power = BOConfig(
            parser=parse.power_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# initial beta
                [1.0, 10.0],# beta_max
                [0.5, 3.0]# p (power exponent)
            ], dtype=torch.double, device=device),
            scheduler=schedule.PowerScheduler
        )
        
        config_constant = BOConfig(
            parser=parse.constant_parser,
            bounds=torch.tensor([
                [0.001, 50.0]# beta
            ], dtype=torch.double, device=device),
            scheduler=schedule.ConstantScheduler
        )
        
        config_geometric = BOConfig(
            parser=parse.geometric_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# initial beta
                [1.001, 2.0]# alpha (multiplicative factor)
            ], dtype=torch.double, device=device),
            scheduler=schedule.GeometricScheduler
        )
        
        # Linear Scheduler Config
        config_linear = BOConfig(
            parser=parse.linear_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# initial beta
                [0.5, 1.5],# a (multiplicative factor)
                [0.0, 1.0]# b (additive factor)
            ], dtype=torch.double, device=device),
            scheduler=schedule.LinearScheduler
        )
        
        # Adaptive Step Scheduler Config
        config_adaptive_step = BOConfig(
            parser=parse.adaptive_step_parser,
            bounds=torch.tensor([
                [0.001, 1.0],# start_beta
                [0.5, 10.0],# end_beta
                [2.0, 10.0]# num_steps
            ], dtype=torch.double, device=device),
            scheduler=schedule.AdaptiveStepScheduler
        )
        
        all_scheduler_configs = {
            "Exponential": config_exponential,
            "Logistic": config_logistic,
            "Step": config_step,
            "Log": config_log,
            "Power": config_power,
            "Constant": config_constant,
            "Geometric": config_geometric,
            "Linear": config_linear,
            "AdaptiveStep": config_adaptive_step
        }
        
        run_multi_scheduler_optimization(all_scheduler_configs, args, stop_on_success=False)
    
    elif args.mode == 'all_schedulers':
        
        run_all_schedulers(args, name_proposal_move=args.proposal_move, n_runs=10)
        
    elif args.mode == 'vary_n':
        
        schedulers = {
            'logistic' : schedule.LogisticScheduler(beta_max=args.end_beta, k=10.0, total_iters=args.max_iters),
            'log' : schedule.LogScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters),
            'exp' : schedule.ExponentialScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters),
            'power' : schedule.PowerScheduler(beta_max=args.end_beta, p=1.2, total_iters=args.max_iters),
            'constant' : schedule.ConstantScheduler(beta=2.5)
        }

        vary_n_values_multiple_schedulers(args, scheduler_dict=schedulers, name_proposal_move=args.proposal_move, n_min=3, n_max=17, n_runs=5, name_complement="multi_scheduler_no_gibbs")
    
    else:

        scheduler = schedule.ExponentialScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters)
        _, _ = multiple_simple_runs(args, scheduler=scheduler, name_proposal_move=args.proposal_move, n_runs=5)



if __name__ == '__main__':
    main()