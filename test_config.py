import numpy as np

test_array = np.array([
    [1, 3, 2, 3],
    [4, 3, 3, 2],
    [0, 4, 1, 5],
    [4, 0, 4, 3],
    [3, 0, 0, 3],
    [0, 0, 4, 4],
    [3, 2, 4, 4],
    [3, 1, 2, 2],
    [1, 4, 0, 3],
    [4, 2, 0, 2],
    [2, 1, 0, 4],
    [4, 4, 3, 2],
    [2, 4, 4, 1],
    [0, 1, 0, 2],
    [3, 4, 1, 3],
    [2, 2, 0, 5],
    [2, 0, 3, 3],
    [4, 4, 1, 3],
    [0, 0, 2, 1],
    [2, 0, 1, 5],
    [1, 0, 3, 5],
    [1, 2, 4, 3],
    [4, 1, 2, 3],
    [0, 2, 0, 4],
    [0, 3, 4, 1]
])

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

if __name__ == "__main__":
    E = energy(test_array)
    print("Energy (number of attacking pairs):", E)