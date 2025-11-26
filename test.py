import numpy as np
import random
import matplotlib.pyplot as plt

# =====================================================
#  BASIC GEOMETRY: 3D QUEEN ATTACK CHECK
# =====================================================

def queens_attack_pos(p, q):
    """Check if two queens at positions p, q (x,y,z) attack each other."""
    x1, y1, z1 = p
    x2, y2, z2 = q

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Same axis (exactly one nonzero difference)
    if dx == 0 and dy == 0 and dz != 0: return True
    if dx == 0 and dy != 0 and dz == 0: return True
    if dx != 0 and dy == 0 and dz == 0: return True

    # 2D diagonals
    if abs(dx) == abs(dy) and dz == 0 and dx != 0: return True
    if abs(dx) == abs(dz) and dy == 0 and dx != 0: return True
    if abs(dy) == abs(dz) and dx == 0 and dy != 0: return True

    # 3D space diagonal
    if abs(dx) == abs(dy) == abs(dz) and dx != 0:
        return True

    return False


# =====================================================
#  ENERGY AND CONFLICT COUNTS
# =====================================================

def initialize_energy_and_weights(config):
    """
    Given config of shape (Q,3) with unique (x,y,z), compute:
    - E: number of attacking pairs
    - weights: conflicts per queen (length Q)
    """
    Q = config.shape[0]
    weights = np.zeros(Q, dtype=int)
    E = 0

    for i in range(Q):
        for j in range(i + 1, Q):
            if queens_attack_pos(config[i], config[j]):
                E += 1
                weights[i] += 1
                weights[j] += 1

    return E, weights


def conflicts_vector(config, idx):
    """
    For queen idx in config (Q,3), return:
    - c: number of conflicts of queen idx
    - mask: int array of length Q with 1 where there is a conflict, 0 otherwise
    """
    Q = config.shape[0]
    px, py, pz = config[idx]

    DX = config[:, 0] - px
    DY = config[:, 1] - py
    DZ = config[:, 2] - pz

    # Axis conflicts
    axis = (
        ((DX == 0) & (DY == 0) & (DZ != 0)) |
        ((DX == 0) & (DY != 0) & (DZ == 0)) |
        ((DX != 0) & (DY == 0) & (DZ == 0))
    )

    # 2D diagonals
    xy = (np.abs(DX) == np.abs(DY)) & (DZ == 0) & (DX != 0)
    xz = (np.abs(DX) == np.abs(DZ)) & (DY == 0) & (DX != 0)
    yz = (np.abs(DY) == np.abs(DZ)) & (DX == 0) & (DY != 0)

    # 3D diagonals
    diag3 = (
        (np.abs(DX) == np.abs(DY)) &
        (np.abs(DY) == np.abs(DZ)) &
        (DX != 0)
    )

    mask = axis | xy | xz | yz | diag3
    mask[idx] = False  # ignore self

    mask_int = mask.astype(int)
    c = int(mask_int.sum())
    return c, mask_int


def energy_slow(config):
    """
    Pure energy check (for debugging / validation).
    config shape (Q,3).
    """
    E = 0
    Q = config.shape[0]
    for i in range(Q):
        for j in range(i + 1, Q):
            if queens_attack_pos(config[i], config[j]):
                E += 1
    return E


# =====================================================
#  METROPOLIS–HASTINGS STEP (WEIGHTED PROPOSAL)
# =====================================================

def metropolis_weighted_step(config, weights, E_old, beta, N):
    """
    One Metropolis–Hastings step with:
      - config: (Q,3) int, unique positions
      - weights: conflicts per queen (Q,)
      - E_old: current energy
      - beta: inverse temperature
      - N: board size

    Weighted proposal (Alternative 3):
      w_m = weights[m] + 1   (always > 0)
      P(select m) = w_m / W
    Acceptance:
      alpha = min(1, e^{-beta ΔE} * (w_m(C')/W(C')) * (W(C)/w_m(C)) )
    """

    Q = config.shape[0]

    # Selection weights BEFORE move
    w = weights + 1.0
    W = float(w.sum())
    probs = w / W

    # Sample queen index according to weights
    idx = np.random.choice(Q, p=probs)
    w_before = w[idx]        # w_m(C)
    W_before = W             # W(C)

    # Make candidate copies
    cand_config = config.copy()
    cand_weights = weights.copy()

    # Old conflicts for this queen (in original config)
    c_old, mask_old = conflicts_vector(config, idx)

    # Remove old conflicts from candidate weights
    cand_weights[idx] -= c_old
    cand_weights -= mask_old  # subtract 1 from all neighbors that attacked idx

    # Sample new empty position (x,y,z) not occupied by other queens
    occupied = set(map(tuple, config))
    old_pos = tuple(config[idx])
    occupied.remove(old_pos)  # we allow moving back to old position only if resampled

    while True:
        x = random.randrange(N)
        y = random.randrange(N)
        z = random.randrange(N)
        new_pos = (x, y, z)
        if new_pos not in occupied:
            break

    cand_config[idx] = new_pos

    # New conflicts for this queen in the candidate config
    c_new, mask_new = conflicts_vector(cand_config, idx)

    # Add new conflicts into candidate weights
    cand_weights[idx] += c_new
    cand_weights += mask_new  # add 1 to neighbors that now attack idx

    # Energy update
    dE = c_new - c_old
    E_new = E_old + dE

    # Selection weights AFTER move
    w_after = cand_weights + 1.0
    W_after = float(w_after.sum())
    w_m_after = w_after[idx]  # w_m(C')

    # Metropolis–Hastings acceptance ratio
    ratio = np.exp(-beta * dE) * (w_m_after / W_after) * (W_before / w_before)
    alpha = min(1.0, ratio)

    if np.random.rand() < alpha:
        # Accept
        return cand_config, cand_weights, E_new
    else:
        # Reject
        return config, weights, E_old


# =====================================================
#  SOLVER
# =====================================================

def solve_3d_queens(N, steps=200000, beta0=0.1, schedule=True, seed=None):
    """
    N: board size => Q = N^2 queens
    steps: number of MCMC steps
    beta0: initial inverse temperature
    schedule: if True, use beta_t = beta0 * log(1+t) (simulated annealing)
    seed: optional random seed
    """

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    Q = N * N

    # Initial random configuration with unique positions
    config = np.zeros((Q, 3), dtype=int)
    occupied = set()
    idx = 0
    while idx < Q:
        pos = (random.randrange(N), random.randrange(N), random.randrange(N))
        if pos not in occupied:
            occupied.add(pos)
            config[idx] = pos
            idx += 1

    # Initial energy and weights
    E, weights = initialize_energy_and_weights(config)
    energies = [E]

    for t in range(1, steps + 1):
        if schedule:
            beta = beta0 * np.log(1 + t)
        else:
            beta = beta0

        config, weights, E = metropolis_weighted_step(config, weights, E, beta, N)
        energies.append(E)

        if E == 0:
            print(f"Solution found at step {t}")
            # Optional: verify with slow energy
            E_check = energy_slow(config)
            print(f"Verified energy (slow check): {E_check}")
            return config, energies

    print("Reached max steps without perfect solution.")
    # Optional: verify with slow energy
    E_check = energy_slow(config)
    print(f"Final energy: {E}, verified: {E_check}")
    return config, energies

def run_multiple_N(N_values, beta0=0.3, steps=300000, seed=None):
    """
    Run solver for multiple board sizes N.
    Returns dict with results for each N.
    """
    results = {}
    
    for N in N_values:
        print(f"\n{'='*50}")
        print(f"Running N={N} (Q={N*N} queens)")
        print(f"{'='*50}")
        
        config, energies = solve_3d_queens(N, steps=steps, beta0=beta0, schedule=True, seed=seed)
        min_energy = min(energies)
        results[N] = {
            'config': config,
            'energies': energies,
            'min_energy': min_energy
        }
        print(f"Min energy achieved: {min_energy}")
    
    # Plot minimum energy vs N
    N_list = sorted(results.keys())
    min_energies = [results[N]['min_energy'] for N in N_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(N_list, min_energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Board Size (N)")
    plt.ylabel("Minimum Energy Achieved")
    plt.title("3D N²-Queens: Best Energy vs Board Size")
    plt.grid(True)
    plt.show()
    
    return results

def run_single_time(N, steps, beta0):
    config, energies = solve_3d_queens(N, steps=steps, beta0=beta0, schedule=True, seed=42)

    print("Final energy:", energies[-1])
    print("Final config (first 10 queens):")
    print(config[:10])

    # Plot energy curve (down-sampled)
    ds = max(1, len(energies) // 1000)
    plt.plot(energies[::ds])
    plt.xlabel("Iteration (downsampled)")
    plt.ylabel("Energy")
    plt.title(f"3D N^2-Queens MCMC (N={N})")
    plt.grid(True)
    plt.show()


# =====================================================
#  MAIN / DEMO
# =====================================================

if __name__ == "__main__":
    N = 10
    beta0 = 0.3
    steps = 300000

    print("Running weighted MCMC solver...")
    run_multiple_N(N_values=range(2, 21), beta0=beta0, steps=steps, seed=42)
