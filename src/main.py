import argparse
from utils.runners import run_optimization, run_pipeline, multiple_simple_runs, run_all_schedulers, vary_n_values
import utils.scheduler as schedule
import utils.proposal_move as pmove
import utils.parsers as parse
from optim.configs import BOConfig
import torch

def main():
    parser = argparse.ArgumentParser(description='N3 Queens Problem Solver')
    parser.add_argument('--mode', type=str, choices=['optimize', 'all_schedulers', 'vary_n', 'run'], default='run',
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
    
    # Proposal Move
    parser.add_argument('--proposal_move', type=str, choices=['random', 'delta_move'], default='random',
                        help='Type of proposal move to use in the solver')
    
    args = parser.parse_args()

    if args.mode == 'optimize':

        _scheduler = schedule.ExponentialScheduler
        _parser = parse.exponential_parser
        
        _bounds = torch.tensor(
            [[0.01, 10.0],
            [1.0, 100.0]]
        , dtype=torch.double)

        config = BOConfig(
            parser=_parser,
            bounds=_bounds,
            scheduler=_scheduler,
        )

        run_optimization(config, args)
    
    elif args.mode == 'all_schedulers':
        
        run_all_schedulers(args, name_proposal_move=args.proposal_move, n_runs=5)
        
    elif args.mode == 'vary_n':
        
        scheduler = schedule.LogScheduler(alpha=0.2)
        _, _ = vary_n_values(args, scheduler=scheduler, name_proposal_move=args.proposal_move, n_min=3, n_max=17, n_runs=5)
    
    else:

        scheduler = schedule.ExponentialScheduler(start_beta=args.beta, end_beta=args.end_beta, max_iters=args.max_iters)
        #scheduler = schedule.LogScheduler(alpha=0.2)
        #scheduler = schedule.LogisticScheduler(beta_max=args.end_beta, k=10.0, total_iters=args.max_iters)
        #scheduler = schedule.PowerScheduler(beta_max=args.end_beta, p=1.2, total_iters=args.max_iters)
        _, _ = multiple_simple_runs(args, scheduler=scheduler, name_proposal_move=args.proposal_move, n_runs=5)



if __name__ == '__main__':
    main()