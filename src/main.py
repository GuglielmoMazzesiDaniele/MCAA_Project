import argparse
from utils.runners import run_optimization, run_pipeline


def main():
    parser = argparse.ArgumentParser(description='N3 Queens Problem Solver')
    parser.add_argument('--mode', type=str, choices=['optimize', 'run'], default='run',
                        help='Mode: "optimize" for BO optimization, "run" for single pipeline run')
    parser.add_argument('--N', type=int, default=8, help='Size of the board (N x N x N)')
    parser.add_argument('--max_iters', type=int, default=20000, help='Maximum number of iterations for the model')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cpu or cuda)')
    parser.add_argument('--reheating', action='store_true', help='Allow reheating')
    parser.add_argument('--patience', type=int, default=10000, help='Number of steps before reheating the model')
    
    parser.add_argument('--K', type=int, default=None, help='Number of non-conflicting queens for smart initialization')
    # Optimization-specific arguments
    parser.add_argument('--max_iters_bo', type=int, default=200, help='Maximum number of BO iterations (optimize mode only)')
    
    # Pipeline-specific arguments
    parser.add_argument('--beta', type=float, default=0.1, help='Initial beta parameter (run mode only)')
    parser.add_argument('--end_beta', type=float, default=50.0, help='End beta parameter (run mode only)')
    
    args = parser.parse_args()
    
    if args.mode == 'optimize':
        run_optimization(args)
    else:
        run_pipeline(args)


if __name__ == '__main__':
    main()