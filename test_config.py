import numpy as np

arr = np.array([
    [0, 0, 5],
    [0, 1, 0],
    [0, 2, 6],
    [0, 3, 1],
    [0, 4, 7],
    [0, 5, 2],
    [0, 6, 8],
    [0, 7, 3],
    [0, 8, 9],
    [0, 9, 4],
    [0, 10, 10],
    [1, 0, 7],
    [1, 1, 2],
    [1, 2, 8],
    [1, 3, 3],
    [1, 4, 9],
    [1, 5, 4],
    [1, 6, 10],
    [1, 7, 5],
    [1, 8, 0],
    [1, 9, 6],
    [1, 10, 1],
    [2, 0, 9],
    [2, 1, 4],
    [2, 2, 10],
    [2, 3, 5],
    [2, 4, 0],
    [2, 5, 6],
    [2, 6, 1],
    [2, 7, 7],
    [2, 8, 2],
    [2, 9, 8],
    [2, 10, 3],
    [3, 0, 0],
    [3, 1, 6],
    [3, 2, 1],
    [3, 3, 7],
    [3, 4, 2],
    [3, 5, 8],
    [3, 6, 3],
    [3, 7, 9],
    [3, 8, 4],
    [3, 9, 10],
    [3, 10, 5],
    [4, 0, 2],
    [4, 1, 8],
    [4, 2, 3],
    [4, 3, 9],
    [4, 4, 4],
    [4, 5, 10],
    [4, 6, 5],
    [4, 7, 0],
    [4, 8, 6],
    [4, 9, 1],
    [4, 10, 7],
    [5, 0, 4],
    [5, 1, 10],
    [5, 2, 5],
    [5, 3, 0],
    [5, 4, 6],
    [5, 5, 1],
    [5, 6, 7],
    [5, 7, 2],
    [5, 8, 8],
    [5, 9, 3],
    [5, 10, 9],
    [6, 0, 6],
    [6, 1, 1],
    [6, 2, 7],
    [6, 3, 2],
    [6, 4, 8],
    [6, 5, 3],
    [6, 6, 9],
    [6, 7, 4],
    [6, 8, 10],
    [6, 9, 5],
    [6, 10, 0],
    [7, 0, 8],
    [7, 1, 3],
    [7, 2, 9],
    [7, 3, 4],
    [7, 4, 10],
    [7, 5, 5],
    [7, 6, 0],
    [7, 7, 6],
    [7, 8, 1],
    [7, 9, 7],
    [7, 10, 2],
    [8, 0, 10],
    [8, 1, 5],
    [8, 2, 0],
    [8, 3, 6],
    [8, 4, 1],
    [8, 5, 7],
    [8, 6, 2],
    [8, 7, 8],
    [8, 8, 3],
    [8, 9, 9],
    [8, 10, 4],
    [9, 0, 1],
    [9, 1, 7],
    [9, 2, 2],
    [9, 3, 8],
    [9, 4, 3],
    [9, 5, 9],
    [9, 6, 4],
    [9, 7, 10],
    [9, 8, 5],
    [9, 9, 0],
    [9, 10, 6],
    [10, 0, 3],
    [10, 1, 9],
    [10, 2, 4],
    [10, 3, 10],
    [10, 4, 5],
    [10, 5, 0],
    [10, 6, 6],
    [10, 7, 1],
    [10, 8, 7],
    [10, 9, 2],
    [10, 10, 8]
])

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