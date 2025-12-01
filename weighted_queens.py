import numpy as np
import random
import matplotlib.pyplot as plt
import time

# =====================================================
#  GEOMETRY: 3D QUEEN ATTACK CHECK
# =====================================================

def conflicts_for_queen(config, idx):
    """Return number of queens attacking queen idx"""
    Q = config.shape[0]
    px, py, pz, _ = config[idx]

    DX = config[:,0] - px
    DY = config[:,1] - py
    DZ = config[:,2] - pz

    DX[idx] = DY[idx] = DZ[idx] = 999999

    # axis
    axis = (
        ((DX == 0) & (DY == 0) & (DZ != 0)) |
        ((DX == 0) & (DY != 0) & (DZ == 0)) |
        ((DX != 0) & (DY == 0) & (DZ == 0))
    )

    # 2D diagonals
    xy = (np.abs(DX) == np.abs(DY)) & (DZ == 0)
    xz = (np.abs(DX) == np.abs(DZ)) & (DY == 0)
    yz = (np.abs(DY) == np.abs(DZ)) & (DX == 0)

    # 3D diagonal
    diag3 = (
        (np.abs(DX) == np.abs(DY)) &
        (np.abs(DY) == np.abs(DZ)) &
        (DX != 0)
    )

    weight_attack = np.sum(axis | xy | xz | yz | diag3)
    config[idx, 3] = weight_attack
    return weight_attack

def weight_queen(config, idx):
    return config[idx, 3]

def total_weight(config):
    return np.sum(config[:, 3])

def queens_attack(p, q):
    """Return True if queens at p and q attack each other in 3D."""
    x1, y1, z1, _ = p
    x2, y2, z2, _ = q

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Same coordinate axis
    if dx == 0 and dy == 0 and dz != 0: return True
    if dx == 0 and dy != 0 and dz == 0: return True
    if dx != 0 and dy == 0 and dz == 0: return True

    # 2D diagonals
    if abs(dx) == abs(dy) and dz == 0: return True
    if abs(dx) == abs(dz) and dy == 0: return True
    if abs(dy) == abs(dz) and dx == 0: return True

    # 3D space diagonal
    if abs(dx) == abs(dy) == abs(dz) and dx != 0:
        return True

    return False

def sample_weighted_queen(config):
    """Sample a queen index with probability proportional to its weight."""
    weights = config[:, 3]
    total_weight = np.sum(weights)
    
    if total_weight == 0:
        # If all weights are zero, sample uniformly
        idx = np.random.choice(len(config))
        return idx, config[idx, 3], 0
    
    probabilities = weights / total_weight
    idx = np.random.choice(len(config), p=probabilities)
    
    return idx, config[idx, 3], int(total_weight)

# =====================================================
#  ENERGY FUNCTION: number of attacking pairs
# =====================================================

def energy(config):
    """Compute number of attacking queen pairs."""
    E = 0
    Q = len(config)
    for i in range(Q):
        for j in range(i+1, Q):
            if queens_attack(config[i], config[j]):
                config[i, 3] += 1  # increase weight of attacking queen i
                config[j, 3] += 1  # increase weight of attacking queen j
                E += 1
    return E


# =====================================================
#  METROPOLIS–HASTINGS STEP
# =====================================================

def metropolis_fast(config, N, E_old, beta):
    
    # Select queen index based on weights, return also its weight and the total weight before move
    idx, c_old, total_weight_before = sample_weighted_queen(config)
    #print(f"Selected queen {idx} with weight {c_old}")

    # pick new empty cell and check it is empty
    occupied = set(map(tuple, config[:, :3]))
    while True:
        new_pos = (np.random.randint(N), np.random.randint(N), np.random.randint(N))
        if new_pos not in occupied:
            break

    old_pos = tuple(config[idx])
    config[idx] = np.array((*new_pos, 0)) # insert the new position with a 0 weight initially

    # new conflicts
    c_new = conflicts_for_queen(config, idx)
    total_weight_after = total_weight(config)

    E_new = E_old + (c_new - c_old)
    #print(E_new)

    if E_new <= E_old or np.random.rand() < min(1, (np.exp(-beta * (E_new - E_old)) * (c_new * total_weight_before) / (c_old * total_weight_after + 1e-10))): # small epsilon to avoid dividing by 0
        
        # Update all weights for the other queens
        E = energy(config)
        
        return config, E
    else:
        # reject — revert
        config[idx] = old_pos
        return config, E_old

def solve_3d_queens(N, steps=20000, beta0=0.1, schedule=False):
    """
    N: board size => N^2 queens
    steps: MCMC steps
    beta0: initial inverse temperature
    """

    Q = N*N

    # Initial random configuration
    config = np.zeros((Q, 4)) 
    occupied = set()

    idx = 0
    while idx < Q:
        pos = (random.randrange(N), random.randrange(N), random.randrange(N))
        if pos not in occupied:
            occupied.add(pos)
            config[idx, :3] = pos
            config[idx, 3] = 0
            idx += 1

    E = energy(config)
    energies = [E]

    for t in range(1, steps+1):

        # Simulated annealing schedule
        if schedule:
            beta = beta0 * np.log(1+t)
        else:
            beta = beta0

        config, E = metropolis_fast(config, N, energies[-1], beta)
        energies.append(E)

        # Found perfect solution
        if E == 0:
            print(f"Solution found at step {t}")
            return config, energies

    print("Reached max steps")
    return config, energies

'''
This function runs the solver for increasing N and plots the time taken.
'''
def run_time_vs_N(beta, schedule):
    Ns = list(range(3, 25))
    times = []

    for N in Ns:
        start = time.time()
        solve_3d_queens(N, steps=300000, beta0=beta, schedule=schedule)
        end = time.time()
        times.append(end - start)
        print(f"N={N}, time={end - start:.2f} seconds")

    plt.plot(Ns, times, marker='o')
    plt.xlabel("Board size N")
    plt.ylabel("Time (seconds)")
    plt.title("Time vs Board Size for 3D N-Queens Solver")
    plt.show()


if __name__ == "__main__":
    N = 5
    beta0 = 0.2
    for i in range(1):
        
        print("Running solver...")
        config, energies = solve_3d_queens(N, steps=300000, beta0=beta0, schedule=True)

        print("Final energy:", energies[-1])
        N += 1

    # # Example output
    print("Final config:", config)

    #If you want: save energies for plotting
    energies = energies[::100]  # Sample every 100th value
    plt.plot(energies)
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.show()