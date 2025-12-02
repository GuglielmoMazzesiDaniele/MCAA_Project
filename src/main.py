import argparse
from N3Queen import N3Queens
from utils.scheduler import LinearScheduler, LogScheduler
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='N3 Queens Problem Solver')
    parser.add_argument('--N', type=int, required=True, help='Size of the board (N x N x N)')
    parser.add_argument('--max_iters', type=int, required=True, help='Maximum number of iterations')
    parser.add_argument('--beta', type=float, required=True, help='Beta parameter for annealing in the algorithm')
    
    args = parser.parse_args()
    
    scheduler = LinearScheduler(a = 1.0, b = 1/20000)

    q_problem = N3Queens(N=args.N, 
                         max_iters=args.max_iters, 
                         beta=args.beta,
                         scheduler=scheduler)
    
    assignements, energies = q_problem.solve()

    print(energies[-1])
    plt.plot(energies)
    plt.savefig("./test.png")
    plt.close()

    return



if __name__ == '__main__':
    main()