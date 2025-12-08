import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
#  GEOMETRY: 3D QUEEN ATTACK CHECK
# =====================================================

def plot_3d_queens(config, N, title="3D Queens Configuration"):
    """
    Draw a 3D NxNxN chessboard and the N^2 queens as red spheres/dots.
    
    config: array of shape (Q,3)
    N: board size
    """

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    xs = config[:, 0]
    ys = config[:, 1]
    zs = config[:, 2]

    # --- Draw queens ---
    ax.scatter(xs, ys, zs, c='red', s=60, depthshade=True, label="Queens")

    # --- Draw board edges ---
    r = range(N)
    for s in [0, N-1]:
        # Faces perpendicular to x
        xx, yy = np.meshgrid([s], r)
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05)
        ax.plot_surface(xx, yy, np.ones_like(xx)*(N-1), alpha=0.05)

        # Faces perpendicular to y
        xx, yy = np.meshgrid(r, [s])
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.05)
        ax.plot_surface(xx, yy, np.ones_like(xx)*(N-1), alpha=0.05)

        # Faces perpendicular to z
        xx, yy = np.meshgrid(r, r)
        ax.plot_surface(xx, yy, np.ones_like(xx)*s, alpha=0.03)

    # --- Style ---
    ax.set_xlim(0, N-1)
    ax.set_ylim(0, N-1)
    ax.set_zlim(0, N-1)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    
def plot_energy(energies):
    energies = energies[::100]  # Sample every 100th value
    
    plt.plot(energies)
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.title("Energy vs MCMC Steps (min E = {})".format(min(energies)))
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

def number_position_delta_motion(config, occupied_set, idx, N):
    x, y, z = config[idx]
    #occupied = set(map(tuple, config))
    count = 0
    results = []

    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                nx = x + dx
                ny = y + dy
                nz = z + dz

                if nx < 0 or nx >= N: 
                    continue
                if ny < 0 or ny >= N:
                    continue
                if nz < 0 or nz >= N:
                    continue

                new_pos = (nx, ny, nz)

                if new_pos not in occupied_set:
                    count += 1
                    results.append(new_pos)

    return count, results

# =====================================================
#  METROPOLIS–HASTINGS STEP
# =====================================================

def metropolis_fast(config, occupied_set, N, E_old, beta):

    idx = np.random.randint(N*N)

    old_pos = tuple(config[idx])

    # old conflicts for this queen
    c_old = conflicts_for_queen(config, idx)
    allowed_cells_before, allowed = number_position_delta_motion(config, occupied_set, idx, N)

    # pick new empty cell
    while True:
        delta = (np.random.randint(-1, 2),
                 np.random.randint(-1, 2),
                 np.random.randint(-1, 2))
        new_pos = (old_pos[0] + delta[0], old_pos[1] + delta[1], old_pos[2] + delta[2])
        if new_pos in allowed:
            break

    # ---------- TEMPORARY APPLY MOVE ----------
    occupied_set.remove(old_pos)
    occupied_set.add(new_pos)
    config[idx] = new_pos

    allowed_cells_after, _ = number_position_delta_motion(config, occupied_set, idx, N)

    # new conflicts
    c_new = conflicts_for_queen(config, idx)

    dE = c_new - c_old

    # MH acceptance ratio
    accept = (dE <= 0 or
              np.random.rand() < min(1, np.exp(-beta * dE) * (allowed_cells_before / allowed_cells_after)))

    if accept:
        return config, E_old + dE

    # ---------- REJECT: ROLLBACK ----------
    occupied_set.remove(new_pos)
    occupied_set.add(old_pos)
    config[idx] = old_pos

    return config, E_old
        
def pairwise_distance_stats(config):
    """
    Compute the mean and variance of pairwise Euclidean distances
    between all queens in the configuration.
    
    config: (Q, 3) integer positions.
    
    Returns: (mean_distance, variance_distance)
    """
    pts = config.astype(float)
    Q = pts.shape[0]

    # Compute all pairwise differences using broadcasting
    diff = pts[:, None, :] - pts[None, :, :]   # shape (Q, Q, 3)

    # Norm of each difference vector
    dist_matrix = np.linalg.norm(diff, axis=2)  # shape (Q, Q)

    # We want upper triangle without diagonal
    iu = np.triu_indices(Q, k=1)
    distances = dist_matrix[iu]    # flatten all unique pairwise distances

    mean_d = np.mean(distances)
    var_d  = np.var(distances)

    return mean_d, var_d
        
def latin_cube_initial(N):
    config = []
    for i in range(N):
        for j in range(N):
            z = (i + j) % N
            config.append((i, j, z))
    return np.array(config, dtype=int)

def random_initialization(N):
    Q = N*N
    config = []
    occupied = set()
    while len(config) < Q:
        pos = (random.randrange(N), random.randrange(N), random.randrange(N))
        if pos not in occupied:
            occupied.add(pos)
            config.append(pos)

def solve_3d_queens(N, steps=20000, beta0=0.1, schedule=False):
    """
    N: board size => N^2 queens
    steps: MCMC steps
    beta0: initial inverse temperature
    """

    #config = random_initialization(N)
    config = latin_cube_initial(N)
    occupied_set = set(map(tuple, config)) # to search easily occupied positions
    E = energy(config)
    
    # Variables
    energies = [E]
    means = []
    variances = []
    beta = beta0

    for t in range(1, steps+1):

        # Simulated annealing schedule
        if schedule and beta <= 3.0:
            beta = 0.01 + (3.0 - 0.01) * (t / steps)
        else:
            beta = beta0
            schedule = False
            
        #print(f"Step {t}, Energy: {energies[-1]}, Beta: {beta}")

        config, E = metropolis_fast(np.array(config), occupied_set, N, energies[-1], beta)
        energies.append(E)
        
        mean, var = pairwise_distance_stats(config)
        means.append(mean)
        variances.append(var)

        # Found perfect solution
        if E == 0:
            print(f"Solution found at step {t}")
            print(f"(N={N}, beta={beta}) : {config}")
            return config, energies

    print("Reached max steps")
    print(f"(N={N}, beta={beta})")
    print(f"mean distance: {np.mean(means)}, variance: {np.mean(variances)}")
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
    plt.title(f"3D N-Queens Energy vs Beta, delta algo (N={N}, schedule={schedule})")
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
    plt.title(f"3D N-Queens Energy vs Beta, delta algo (beta0={beta0}, schedule={schedule})")
    plt.show()
    
    plot_3d_queens(_, N-1, title=f"Final Configuration for N={N-1}")


def main():
    outdir = "results"
    import os
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    ###########################
    # Sweep 1: vary N
    ###########################
    beta0 = 0.1
    max_N = 12
    steps = 40000

    energies_final = []
    Ns = list(range(11, max_N + 1))

    for N in Ns:
        config, energies = solve_3d_queens(
            N=N,
            steps=steps,
            beta0=beta0,
            schedule=False
        )
        energies_final.append(min(energies))

    plt.figure()
    plt.plot(Ns, energies_final, marker='o')
    for i,N in enumerate(Ns):
        plt.text(N, energies_final[i], f"{energies_final[i]}", ha='center', va='bottom')
    plt.xlabel("N")
    plt.ylabel("Min Energy")
    plt.title("Min Energy vs N")
    plt.savefig(f"{outdir}/vary_N.png")
    plt.close()


    ###########################
    # Sweep 2: vary beta
    ###########################
    N = 11
    betas = [0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
    energies_beta = []

    for b in betas:
        config, energies = solve_3d_queens(
            N=N,
            steps=steps,
            beta0=b,
            schedule=True
        )
        energies_beta.append(min(energies))

    plt.figure()
    plt.plot(betas, energies_beta, marker='o')
    for i,b in enumerate(betas):
        plt.text(b, energies_beta[i], f"{energies_beta[i]}", ha='center', va='bottom')
    plt.xlabel("beta0")
    plt.ylabel("Min Energy")
    plt.title(f"Min Energy vs beta0 (N={N})")
    plt.savefig(f"{outdir}/vary_beta.png")
    plt.close()

def main2():
    config, energies = solve_3d_queens(
            N=11,
            steps=2000000,
            beta0=0.1,
            schedule=True
        )
    
    plot_energy(energies)

if __name__ == "__main__":
    main2()