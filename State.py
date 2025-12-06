from dataclasses import dataclass
from typing import List,Tuple, Optional
import random

Coord = Tuple[int, int, int]


@dataclass
class State:
    """
    A configuration of N^2 queens on an N x N x N board.

    Attributes
    ----------
    N : int
        Board size (each coordinate in {1, ..., N}).
    queens : List[Coord]
        List of queen positions (distinct coordinates).
    """
    N: int
    queens: List[Coord]
    empties: List[Coord]


# ---------- Board utilities ----------

def generate_all_cells(N: int) -> List[Coord]:
    """
    Generate all coordinates of an N x N x N board.
    """
    return [(x, y, z)
            for x in range(1, N + 1)
            for y in range(1, N + 1)
            for z in range(1, N + 1)]


# We'll reuse this per board size to avoid recomputing every time.
_ALL_CELLS_CACHE = {}

def get_all_cells_cached(N: int) -> List[Coord]:
    if N not in _ALL_CELLS_CACHE:
        _ALL_CELLS_CACHE[N] = generate_all_cells(N)
    return _ALL_CELLS_CACHE[N]


# ---------- State initialization ----------

def generate_initial_state(N: int,
                           rng: Optional[random.Random] = None) -> State:
    """
    Create a random initial state with N^2 queens on distinct cells.

    Parameters
    ----------
    N : int
        Board size.
    rng : random.Random, optional
        RNG for reproducibility. If None, uses global random.

    Returns
    -------
    State
        Random configuration of N^2 queens.
    """
    if rng is None:
        rng = random

    cells = get_all_cells_cached(N)[:]
    rng.shuffle(cells)

    num_queens = N * N
    queens = cells[:num_queens]
    empties = cells[num_queens:]
    return State(N=N, queens=queens, empties=empties)


# ---------- Attack relation and energy ----------

def are_queens_attacking(q1: Coord, q2: Coord) -> bool:
    """
    Check whether two queens attack each other in 3D.

    A queen at (x,y,z) attacks any other square that:
      - shares two coordinates (same x and y or x and z or y and z), or
      - lies on a 2D diagonal in one of the coordinate planes:
            - xy-plane: same z and |dx| = |dy|
            - xz-plane: same y and |dx| = |dz|
            - yz-plane: same x and |dy| = |dz|
      - lies on a 3D diagonal: |dx| = |dy| = |dz|.
    """
    x1, y1, z1 = q1
    x2, y2, z2 = q2

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dz = abs(z1 - z2)

    # Same coordinate (axis-aligned lines)
    if x1 == x2 and y1 == y2 or x1 == x2 and z1 == z2 or y1 == y2 and z1 == z2:
        return True

    # 3D diagonal
    if dx == dy == dz and dx != 0:
        return True

    # 2D diagonals in coordinate planes
    # xy-plane diagonal: same z, |dx| = |dy|
    if z1 == z2 and dx == dy and dx != 0:
        return True

    # xz-plane diagonal: same y, |dx| = |dz|
    if y1 == y2 and dx == dz and dx != 0:
        return True

    # yz-plane diagonal: same x, |dy| = |dz|
    if x1 == x2 and dy == dz and dy != 0:
        return True

    return False

def compute_conflicts(state: State) -> List[int]:
    """
    Compute the number of conflicts for each queen in the current state.

    Parameters
    ----------
    state : State
        Current configuration.

    Returns
    -------
    List[int]
        conflicts[i] = number of other queens that attack queens[i].
    """
    Q = state.queens
    n = len(Q)
    conflicts = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if are_queens_attacking(Q[i], Q[j]):
                conflicts[i] += 1
                conflicts[j] += 1
    return conflicts


def compute_energy(state: State) -> int:
    """
    Compute the energy H(X) = number of attacking queen pairs.

    Parameters
    ----------
    state : State
        Current configuration.

    Returns
    -------
    int
        Number of unordered pairs (i,j) of queens that attack each other.
    """
    H = 0
    Q = state.queens
    n = len(Q)
    for i in range(n):
        for j in range(i + 1, n):
            if are_queens_attacking(Q[i], Q[j]):
                H += 1
    return H
