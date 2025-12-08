import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# =====================================================
# 3D QUEEN ATTACK DIRECTIONS (26 directions)
# =====================================================

DIRS = []
for dx in [-1,0,1]:
    for dy in [-1,0,1]:
        for dz in [-1,0,1]:
            if dx==0 and dy==0 and dz==0:
                continue
            # simplify direction using gcd (primitive vector)
            g = np.gcd(np.gcd(abs(dx),abs(dy)),abs(dz))
            DIRS.append((dx//g, dy//g, dz//g))
DIRS = list(set(DIRS))  # unique 26 dirs


# =====================================================
#  ENCODING / DECODING POSITIONS
# =====================================================

def encode(x,y,z,N):
    return x*N*N + y*N + z

def decode(id,N):
    z = id % N
    y = (id//N) % N
    x = id//(N*N)
    return x,y,z


# =====================================================
#  FAST CONFLICT CHECK
# =====================================================

def fast_conflicts(id, occ, N):
    """Counts how many queens attack this queen using integer stepping."""
    x,y,z = decode(id,N)
    c = 0
    for dx,dy,dz in DIRS:
        nx,ny,nz = x+dx, y+dy, z+dz
        while 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
            if occ[encode(nx,ny,nz,N)]:
                c += 1
                break
            nx += dx; ny += dy; nz += dz
    return c


# =====================================================
# NEIGHBOR MOVES
# =====================================================

def number_position_delta_motion(config, occ, idx, N):
    """Return available moves for queen idx within +-1 delta."""
    id0 = config[idx]
    x,y,z = decode(id0, N)
    moves = []
    count = 0

    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            for dz in [-1,0,1]:
                if dx==0 and dy==0 and dz==0:
                    continue
                nx,ny,nz = x+dx, y+dy, z+dz
                if 0<=nx<N and 0<=ny<N and 0<=nz<N:
                    new_id = encode(nx,ny,nz,N)
                    if not occ[new_id]:
                        moves.append(new_id)
                        count += 1

    return count, moves


# =====================================================
# ENERGY FUNCTION (unchanged)
# =====================================================

def energy(config):
    """Compute number of attacking pairs (slow baseline)."""
    Q = len(config)
    E = 0
    for i in range(Q):
        for j in range(i+1, Q):
            if queens_attack(config[i], config[j]):
                E += 1
    return E


# =====================================================
# ORIGINAL queens_attack (kept for compatibility)
# =====================================================

def queens_attack(p, q):
    x1,y1,z1 = p
    x2,y2,z2 = q
    dx = x2-x1; dy=y2-y1; dz=z2-z1

    if dx==0 and dy==0 and dz!=0: return True
    if dx==0 and dy!=0 and dz==0: return True
    if dx!=0 and dy==0 and dz==0: return True
    if abs(dx)==abs(dy) and dz==0: return True
    if abs(dx)==abs(dz) and dy==0: return True
    if abs(dy)==abs(dz) and dx==0: return True
    if abs(dx)==abs(dy)==abs(dz) and dx!=0: return True
    return False


# =====================================================
# FAST METROPOLIS–HASTINGS
# =====================================================

def metropolis_fast(config, conflicts, occ, N, E_old, beta):
    Q = len(config)

    idx = random.randrange(Q)
    old_id = config[idx]

    # allowed moves (fast)
    allowed_count, allowed_moves = number_position_delta_motion(config, occ, idx, N)
    if allowed_count == 0:
        return config, E_old

    new_id = random.choice(allowed_moves)

    # old conflict count for this queen
    old_conf = conflicts[idx]

    # temporary apply
    occ[old_id] = False
    occ[new_id] = True
    config[idx] = new_id

    # compute new conflicts for this queen (fast)
    new_conf = fast_conflicts(new_id, occ, N)

    dE = new_conf - old_conf

    # MH acceptance rule
    if (dE <= 0) or (random.random() < np.exp(-beta*dE)):
        conflicts[idx] = new_conf
        return config, E_old + dE

    # reject → rollback
    config[idx] = old_id
    occ[new_id] = False
    occ[old_id] = True
    return config, E_old


# =====================================================
# INITIALIZATION
# =====================================================

def latin_cube_initial(N):
    config = []
    for i in range(N):
        for j in range(N):
            z = (i + j) % N
            config.append((i, j, z))
    return np.array(config, dtype=int)


# =====================================================
# MAIN SOLVER
# =====================================================

def solve_3d_queens(N, steps=20000, beta0=0.1, schedule=False):
    # Starting config
    config_xyz = latin_cube_initial(N)
    Q = N*N

    # convert to 1D encoded positions
    config = np.array([encode(x,y,z,N) for (x,y,z) in config_xyz], dtype=int)

    # occupancy grid
    occ = np.zeros(N*N*N, dtype=bool)
    for v in config:
        occ[v] = True

    # initial conflicts
    conflicts = np.array([fast_conflicts(v, occ, N) for v in config])
    E = conflicts.sum()

    energies = [E]
    beta = beta0

    for t in range(1, steps+1):
        # schedule
        if schedule:
            beta = beta0 + (3.0 - beta0)*(t/steps)

        config, E = metropolis_fast(config, conflicts, occ, N, E, beta)
        energies.append(E)

        if E == 0:
            print(f"Solution found at step {t}")
            final = np.array([decode(v,N) for v in config])
            return final, energies

    print("Reached max steps. Final E =", E)
    final = np.array([decode(v,N) for v in config])
    return final, energies


# =====================================================
# PLOTTING FUNCTIONS (unchanged)
# =====================================================

def plot_3d_queens(config, N, title="3D Queens"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    xs = config[:,0]; ys = config[:,1]; zs = config[:,2]
    ax.scatter(xs, ys, zs, c='red', s=60, depthshade=True)
    ax.set_xlim(0,N-1); ax.set_ylim(0,N-1); ax.set_zlim(0,N-1)
    plt.title(title)
    plt.show()

def plot_energy(energies):
    plt.plot(energies)
    plt.title("Energy vs MCMC Steps (min E = {})".format(min(energies)))
    plt.xlabel("Step")
    plt.ylabel("Energy")
    plt.show()


# =====================================================
# MAIN ENTRY POINTS (same as yours)
# =====================================================

def main2():
    N = 11
    config, energies = solve_3d_queens(
        N=N,
        steps=2000000,
        beta0=0.1,
        schedule=True
    )
    print(config)
    plot_energy(energies)
    plot_3d_queens(config, N)

def main():
    config, energies = solve_3d_queens(
        N=10,
        steps=50000,
        beta0=0.15,
        schedule=False
    )
    plot_energy(energies)
    plot_3d_queens(config, 10)

if __name__ == "__main__":
    main2()
