import numpy as np
import random
import matplotlib.pyplot as plt

# =====================================================
#  GEOMETRY: 3D QUEEN ATTACK CHECK
# =====================================================

def plot_energy(energies):
    energies = energies[::100]  # Sample every 100th value
    plt.plot(energies)
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.show()

def conflicts_for_queen(config, idx):
    """Return number of queens attacking queen idx using NumPy vectorization."""
    Q = config.shape[0]
    px, py, pz = config[idx]

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

    return np.sum(axis | xy | xz | yz | diag3)

def queens_attack(p, q):
    """Return True if queens at p and q attack each other in 3D."""
    x1, y1, z1 = p
    x2, y2, z2 = q

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
                E += 1
    return E


# =====================================================
#  RANDOM MOVE: pick a queen, move to empty cell
# =====================================================

def random_move(config, N):
    """Return a new configuration with one queen moved."""
    Q = len(config)
    new_config = config.copy()

    # Pick queen index
    idx = random.randrange(Q)

    # Available empty cells
    occupied = set(config)
    while True:
        x = random.randrange(N)
        y = random.randrange(N)
        z = random.randrange(N)
        if (x, y, z) not in occupied:
            break

    new_config[idx] = (x, y, z)
    return new_config


# =====================================================
#  METROPOLIS–HASTINGS STEP
# =====================================================

def metropolis_fast(config, N, E_old, beta):
    Q = config.shape[0]

    idx = np.random.randint(Q)

    # old conflicts for this queen
    c_old = conflicts_for_queen(config, idx)

    # pick new empty cell
    occupied = set(map(tuple, config))
    while True:
        new_pos = (np.random.randint(N), np.random.randint(N), np.random.randint(N))
        if new_pos not in occupied:
            break

    # apply move temporarily
    old_pos = tuple(config[idx])
    config[idx] = new_pos

    # new conflicts
    c_new = conflicts_for_queen(config, idx)

    dE = c_new - c_old

    if dE <= 0 or np.random.rand() < np.exp(-beta * dE):
        return config, E_old + dE
    else:
        # reject — revert
        config[idx] = old_pos
        return config, E_old

def metropolis_step(config, N, beta):
    """Perform a Metropolis step with inverse temperature beta."""
    E_old = energy(config)
    candidate = random_move(config, N)
    E_new = energy(candidate)

    dE = E_new - E_old

    # Accept if energy decreases or with prob e^{-beta ΔE}
    if dE <= 0:
        return candidate, E_new
    else:
        if random.random() < np.exp(-beta * dE):
            return candidate, E_new
        else:
            return config, E_old

def solve_3d_queens(N, steps=20000, beta0=0.1, schedule=False):
    """
    N: board size => N^2 queens
    steps: MCMC steps
    beta0: initial inverse temperature
    """

    Q = N*N

    # Initial random configuration
    config = []
    occupied = set()
    while len(config) < Q:
        pos = (random.randrange(N), random.randrange(N), random.randrange(N))
        if pos not in occupied:
            occupied.add(pos)
            config.append(pos)

    occupied = set(config)
    E = energy(config)
    energies = [E]

    for t in range(1, steps+1):

        # Simulated annealing schedule
        if schedule:
            beta = beta0 * np.log(1+t)
        else:
            beta = beta0

        config, E = metropolis_fast(np.array(config), N, energies[-1], beta)
        energies.append(E)

        # Found perfect solution
        if E == 0:
            print(f"Solution found at step {t}")
            print(f"(N={N}, beta={beta}) : {config}")
            return config, energies

    print("Reached max steps")
    return config, energies


def vary_beta(N, steps=20000, schedule=True):
    beta_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    results = []

    for beta0 in beta_values:
        print(f"Running solver with beta0={beta0}...")
        _, energies = solve_3d_queens(N, steps=steps, beta0=beta0, schedule=schedule)
        results.append(energies[-1])

    plt.plot(beta_values, results, marker='o')
    plt.xlabel("beta")
    plt.ylabel("Energy")
    plt.title(f"3D N-Queens Energy vs Beta, basic algo (N={N}, schedule={schedule})")
    plt.show()
    
def vary_N(beta0, max_N, steps=20000, schedule=True):
    results = []

    for N in range(3, max_N+1):
        print(f"Running solver with N={N}...")
        _, energies = solve_3d_queens(N, steps=steps, beta0=beta0, schedule=schedule)
        results.append(energies[-1])

    plt.plot(range(3, max_N+1), results, marker='o')
    plt.xlabel("N")
    plt.ylabel("Energy")
    plt.title(f"3D N-Queens Energy vs Beta, basic algo (beta0={beta0}, schedule={schedule})")
    plt.show()


def main():
    vary_beta(N=14, steps=20000, schedule=True)
    
main()