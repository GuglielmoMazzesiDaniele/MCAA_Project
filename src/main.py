import argparse
from N3Queen import N3Queens
import N3Queens2DGrid
from utils.scheduler import LinearScheduler, LogScheduler, ExponentialScheduler
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='N3 Queens Problem Solver')
    parser.add_argument('--N', type=int, required=True, help='Size of the board (N x N x N)')
    parser.add_argument('--max_iters', type=int, required=True, help='Maximum number of iterations')
    parser.add_argument('--beta', type=float, required=True, help='Beta parameter for annealing in the algorithm')
    parser.add_argument('--k', type=int, required=False, default=None, help='Number of previous non conflicting queens for the initialization')
    
    args = parser.parse_args()
    
    scheduler = ExponentialScheduler(end_beta=20, max_iters=args.max_iters, start_beta=args.beta)
    q_problem = N3Queens2DGrid.N3Queens(
        N=args.N,
        max_iters=args.max_iters,
        scheduler=scheduler,
        beta=args.beta,
        k=args.k
    )

    assignements, energies = q_problem.solve()
    
    plt.plot(energies)
    plt.savefig("./test.png")
    plt.close()

    return



if __name__ == '__main__':
    main()