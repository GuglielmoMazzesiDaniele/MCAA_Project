import numpy as np

arr = np.array([
    [ 0,  0,  5],
    [ 0,  1,  2],
    [ 0,  2, 10],
    [ 0,  3,  7],
    [ 0,  4,  4],
    [ 0,  5,  1],
    [ 0,  6,  9],
    [ 0,  7,  6],
    [ 0,  8,  3],
    [ 0,  9,  0],
    [ 0, 10,  8],

    [ 1,  0, 10],
    [ 1,  1,  7],
    [ 1,  2,  4],
    [ 1,  3,  1],
    [ 1,  4,  9],
    [ 1,  5,  6],
    [ 1,  6,  3],
    [ 1,  7,  0],
    [ 1,  8,  8],
    [ 1,  9,  5],
    [ 1, 10,  2],

    [ 2,  0,  4],
    [ 2,  1,  1],
    [ 2,  2,  9],
    [ 2,  3,  6],
    [ 2,  4,  3],
    [ 2,  5,  0],
    [ 2,  6,  8],
    [ 2,  7,  5],
    [ 2,  8,  2],
    [ 2,  9, 10],
    [ 2, 10,  7],

    [ 3,  0,  9],
    [ 3,  1,  6],
    [ 3,  2,  3],
    [ 3,  3,  0],
    [ 3,  4,  8],
    [ 3,  5,  5],
    [ 3,  6,  2],
    [ 3,  7, 10],
    [ 3,  8,  7],
    [ 3,  9,  4],
    [ 3, 10,  1],

    [ 4,  0,  3],
    [ 4,  1,  0],
    [ 4,  2,  8],
    [ 4,  3,  5],
    [ 4,  4,  2],
    [ 4,  5, 10],
    [ 4,  6,  7],
    [ 4,  7,  4],
    [ 4,  8,  1],
    [ 4,  9,  9],
    [ 4, 10,  6],

    [ 5,  0,  8],
    [ 5,  1,  5],
    [ 5,  2,  2],
    [ 5,  3, 10],
    [ 5,  4,  7],
    [ 5,  5,  4],
    [ 5,  6,  1],
    [ 5,  7,  9],
    [ 5,  8,  6],
    [ 5,  9,  3],
    [ 5, 10,  0],

    [ 6,  0,  2],
    [ 6,  1, 10],
    [ 6,  2,  7],
    [ 6,  3,  4],
    [ 6,  4,  1],
    [ 6,  5,  9],
    [ 6,  6,  6],
    [ 6,  7,  3],
    [ 6,  8,  0],
    [ 6,  9,  8],
    [ 6, 10,  5],

    [ 7,  0,  7],
    [ 7,  1,  4],
    [ 7,  2,  1],
    [ 7,  3,  9],
    [ 7,  4,  6],
    [ 7,  5,  3],
    [ 7,  6,  0],
    [ 7,  7,  8],
    [ 7,  8,  5],
    [ 7,  9,  2],
    [ 7, 10, 10],

    [ 8,  0,  1],
    [ 8,  1,  9],
    [ 8,  2,  6],
    [ 8,  3,  3],
    [ 8,  4,  0],
    [ 8,  5,  8],
    [ 8,  6,  5],
    [ 8,  7,  2],
    [ 8,  8, 10],
    [ 8,  9,  7],
    [ 8, 10,  4],

    [ 9,  0,  6],
    [ 9,  1,  3],
    [ 9,  2,  0],
    [ 9,  3,  8],
    [ 9,  4,  5],
    [ 9,  5,  2],
    [ 9,  6, 10],
    [ 9,  7,  7],
    [ 9,  8,  4],
    [ 9,  9,  1],
    [ 9, 10,  9],

    [10,  0,  0],
    [10,  1,  8],
    [10,  2,  5],
    [10,  3,  2],
    [10,  4, 10],
    [10,  5,  7],
    [10,  6,  4],
    [10,  7,  1],
    [10,  8,  9],
    [10,  9,  6],
    [10, 10,  3],
], dtype=int)


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

def energy(config):
    """Compute number of attacking queen pairs."""
    E = 0
    Q = len(config)
    for i in range(Q):
        for j in range(i+1, Q):
            if queens_attack(config[i], config[j]):
                E += 1
    return E

if __name__ == "__main__":
    E = energy(arr)
    print("Energy (number of attacking pairs):", E)