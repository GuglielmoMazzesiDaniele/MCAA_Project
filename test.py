import numpy as np
import random
import matplotlib.pyplot as plt
import time


# =====================================================
#  GEOMETRY: 3D QUEEN ATTACK CHECK
# =====================================================

def queens_attack(p, q):
    """Return True if queens at p and q attack each other in 3D."""
    x1,y1,z1 = p
    x2,y2,z2 = q

    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1

    # Axis
    if dx == 0 and dy == 0 and dz != 0: return True
    if dx == 0 and dy != 0 and dz == 0: return True
    if dx != 0 and dy == 0 and dz == 0: return True

    # 2D diagonals
    if abs(dx) == abs(dy) and dz == 0 and dx != 0: return True
    if abs(dx) == abs(dz) and dy == 0 and dx != 0: return True
    if abs(dy) == abs(dz) and dx == 0 and dy != 0: return True

    # 3D diagonal
    if abs(dx) == abs(dy) == abs(dz) and dx != 0:
        return True

    return False


# =====================================================
#  ENERGY + CONFLICT WEIGHTS
# =====================================================

def compute_conflicts_for_queen(config, idx):
    """Count attackers of queen idx and return mask of attackers."""
    Q = config.shape[0]
    p = config[idx]
    dx = config[:,0] - p[0]
    dy = config[:,1] - p[1]
    dz = config[:,2] - p[2]

    # Same axis
    axis = (
        ((dx == 0) & (dy == 0) & (dz != 0)) |
        ((dx == 0) & (dy != 0) & (dz == 0)) |
        ((dx != 0) & (dy == 0) & (dz == 0))
    )

    # 2D diagonals
    xy = (np.abs(dx) == np.abs(dy)) & (dz == 0) & (dx != 0)
    xz = (np.abs(dx) == np.abs(dz)) & (dy == 0) & (dx != 0)
    yz = (np.abs(dy) == np.abs(dz)) & (dx == 0) & (dy != 0)

    # 3D diagonals
    diag3 = (
        (np.abs(dx) == np.abs(dy)) &
        (np.abs(dy) == np.abs(dz)) &
        (dx != 0)
    )

    mask = axis | xy | xz | yz | diag3
    mask[idx] = False

    return mask.astype(int).sum(), mask.astype(int)


def compute_all_weights(config):
    """Compute conflicts per queen and return weights."""
    Q = config.shape[0]
    weights = np.zeros(Q, dtype=int)

    for i in range(Q):
        for j in range(i+1, Q):
            if queens_attack(config[i], config[j]):
                weights[i] += 1
                weights[j] += 1
    return weights


def energy(config):
    """Pure energy: number of attacking pairs."""
    E = 0
    Q = config.shape[0]
    for i in range(Q):
        for j in range(i+1, Q):
            if queens_attack(config[i], config[j]):
                E += 1
    return E


# =====================================================
#  METROPOLIS–HASTINGS STEP (WEIGHTED PROPOSAL)
# =====================================================

def metropolis_fast(config, weights, E_old, beta, N):
    Q = config.shape[0]

    # selection weights w_m = weights + 1
    w_before = weights + 1
    W_before = float(w_before.sum())

    probs = w_before / W_before
    idx = np.random.choice(Q, p=probs)
    c_old = weights[idx]

    # prepare candidate state
    cand_config = config.copy()
    cand_weights = weights.copy()

    # old conflict mask for this queen
    c_old2, mask_old = compute_conflicts_for_queen(config, idx)

    # remove old conflicts from cand_weights
    cand_weights[idx] -= c_old2
    cand_weights -= mask_old

    # pick new empty position
    occupied = set(map(tuple, cand_config))
    old_pos = tuple(cand_config[idx])
    occupied.remove(old_pos)

    while True:
        x = random.randrange(N)
        y = random.randrange(N)
        z = random.randrange(N)
        new_pos = (x,y,z)
        if new_pos not in occupied:
            break

    cand_config[idx] = new_pos

    # new conflicts
    c_new2, mask_new = compute_conflicts_for_queen(cand_config, idx)

    # add new conflicts into cand_weights
    cand_weights[idx] += c_new2
    cand_weights += mask_new

    dE = c_new2 - c_old2
    E_new = E_old + dE

    # compute new selection weights
    w_after = cand_weights + 1
    W_after = float(w_after.sum())

    # Metropolis–Hastings ratio
    ratio = np.exp(-beta * dE) * (w_after[idx] / W_after) * (W_before / w_before[idx])
    alpha = min(1.0, ratio)

    if random.random() < alpha:
        return cand_config, cand_weights, E_new
    else:
        return config, weights, E_old


# =====================================================
#  SOLVER
# =====================================================

def solve_3d_queens(N, steps=200000, beta0=0.1, schedule=False, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    Q = N*N

    # Initial random unique positions
    config = np.zeros((Q, 3), dtype=int)
    occupied = set()

    idx = 0
    while idx < Q:
        pos = (random.randrange(N), random.randrange(N), random.randrange(N))
        if pos not in occupied:
            occupied.add(pos)
            config[idx] = pos
            idx += 1

    # Initial energy + weights
    E = energy(config)
    weights = compute_all_weights(config)
    energies = [E]

    for t in range(1, steps+1):

        beta = beta0 * np.log(1+t) if schedule else beta0
        config, weights, E = metropolis_fast(config, weights, E, beta, N)

        energies.append(E)

        if E == 0:
            print(f"\nSolution found at step {t}")
            print("Verified energy:", energy(config))
            return config, energies

    print("\nReached max steps. Final energy =", E)
    print("Verified energy:", energy(config))
    return config, energies


# =====================================================
#  MAIN
# =====================================================

if __name__ == "__main__":
    N = 11         # smallest N where a 0-energy solution is possible
    beta0 = 0.3
    steps = 300000

    print("Running weighted 3D N^2-queens solver…")
    config, energies = solve_3d_queens(N, steps=steps, beta0=beta0, schedule=True, seed=42)

    print("\nFinal config (first 10 positions):")
    print(config[:10])

    # Plot energies (downsample)
    ds = max(1, len(energies) // 1000)
    plt.plot(energies[::ds])
    plt.xlabel("Iteration (downsampled)")
    plt.ylabel("Energy")
    plt.title(f"3D Queens MCMC (N={N})")
    plt.grid(True)
    plt.show()
