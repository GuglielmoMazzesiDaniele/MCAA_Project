from __future__ import annotations

import random
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from numpy.ma.extras import average

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

def all_cells(N: int) -> List[Coord]:
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
        _ALL_CELLS_CACHE[N] = all_cells(N)
    return _ALL_CELLS_CACHE[N]


# ---------- State initialization ----------

def random_initial_state(N: int,
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

def queens_attack(q1: Coord, q2: Coord) -> bool:
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


def energy(state: State) -> int:
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
            if queens_attack(Q[i], Q[j]):
                H += 1
    return H


# ---------- Proposal kernel (symmetric) ----------

def compute_empties(state: State) -> List[Coord]:
    """
    Compute the list of empty cells as the complement of queen positions.

    Parameters
    ----------
    state : State

    Returns
    -------
    List[Coord]
        All empty positions on the board.
    """
    N = state.N
    all_cells_list = get_all_cells_cached(N)
    queen_set = set(state.queens)
    return [c for c in all_cells_list if c not in queen_set]


def propose_swap(state: State,
                 rng: Optional[random.Random] = None) -> State:
    """
    Propose a new state by swapping a random queen with a random empty cell.

    This defines a symmetric proposal kernel Q(X,Y):
      - pick one queen uniformly among N^2 queens,
      - pick one empty cell uniformly among N^3 - N^2 empties,
      - move the queen to that empty cell.

    Parameters
    ----------
    state : State
        Current state X.
    rng : random.Random, optional
        RNG for reproducibility.

    Returns
    -------
    State
        Proposed new state Y.
    """
    if rng is None:
        rng = random

    queens = state.queens[:]
    empties = state.empties[:]


    # Choose a random queen and a random empty cell indices
    i_q = rng.randrange(len(queens))
    i_e = rng.randrange(len(empties))

    # Extracting the Coords
    q = queens[i_q]
    e = empties[i_e]

    # Move queen from q -> e
    queens[i_q] = e
    empties[i_e] = q

    # Regenerating state
    return State(N=N, queens=queens, empties=empties)


# ---------- Metropolis / simulated annealing ----------

def metropolis_step(state: State,
                    T: float,
                    rng: Optional[random.Random] = None) -> State:
    """
    Perform a single Metropolis(-Hastings) update with symmetric proposal.

    Acceptance rule:
        alpha(X,Y) = min(1, exp(-(H(Y) - H(X)) / T))

    Parameters
    ----------
    state : State
        Current state X.
    T : float
        Temperature (T > 0). Smaller T => more selective, fewer uphill moves.
    rng : random.Random, optional
        RNG.

    Returns
    -------
    State
        New state (either the proposal Y or the original X).
    """
    if rng is None:
        rng = random

    current_E = energy(state)
    proposal = propose_swap(state, rng)
    proposal_E = energy(proposal)

    dH = proposal_E - current_E

    # Accept uphill moves with probability exp(-dH/T)
    accept_prob = math.exp(-dH / T) if T > 0 else 0.0

    if rng.random() < accept_prob:
        return proposal
    else:
        return state


def default_temperature_schedule(t: int,
                                 T0: float = 1.0,
                                 schedule: str = "log") -> float:
    """
    Simple temperature schedule T_t.

    Parameters
    ----------
    t : int
        Current step (0-based).
    T0 : float
        Initial temperature scale.
    schedule : {"log", "exp", "const"}
        Type of cooling schedule.

    Returns
    -------
    float
        Temperature T_t.
    """
    if schedule == "const":
        return T0
    elif schedule == "exp":
        # Exponential decay
        return T0 * (0.995 ** t)
    elif schedule == "log":
        # Logarithmic cooling T_t = T0 / log(2 + t)
        # (slow but theoretically nice)
        return T0 / math.log(2.0 + t) if t > 0 else T0
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def run_annealing(N: int,
                  steps: int = 50_000,
                  T0: float = 2.0,
                  schedule: str = "log",
                  rng: Optional[random.Random] = None,
                  verbose: bool = True) -> State:
    """
    Run simulated annealing to search for a configuration with low energy.

    Parameters
    ----------
    N : int
        Board size.
    steps : int
        Number of Metropolis steps to run.
    T0 : float
        Initial temperature for the schedule.
    schedule : {"log", "exp", "const"}
        Cooling schedule type.
    rng : random.Random, optional
        RNG.
    verbose : bool
        Whether to print progress.

    Returns
    -------
    State
        Best state found (lowest energy observed).
    """
    if rng is None:
        rng = random.Random()

    state = random_initial_state(N, rng)
    best_state = state
    best_E = energy(state)

    if verbose:
        print(f"Initial energy: {best_E}")

    for t in range(steps):
        T = default_temperature_schedule(t, T0=T0, schedule=schedule)
        state = metropolis_step(state, T, rng)
        E = energy(state)

        if E < best_E:
            best_E = E
            best_state = state
            if verbose:
                print(f"[step {t}] New best energy: {best_E}")
            if best_E == 0:
                if verbose:
                    print(f"Found solution (energy 0) at step {t}")
                break

    if verbose:
        print(f"Finished after {t+1} steps. Best energy: {best_E}")

    return best_state

# ---------- CLI entry point ----------
if __name__ == "__main__":
    # Number of simulations
    simulations = 5

    bests = []

    # Board size
    N = 6
    steps = 10000
    T0 = 2.5
    # "log", "exp", or "const"
    schedule = "log"

    print(f"\nRunning {simulations} simulations of 3D N^2-Queens MCMC with N={N}, steps={steps}, schedule={schedule}\n")

    # Executing simulations
    for i in range (simulations):
        print(f"Starting simulation {i + 1} of {simulations}")

        best_state = run_annealing(N=N, steps=steps, T0=T0, schedule=schedule, verbose=False)
        best_energy = energy(best_state)

        print(f"Iteration {i + 1}, best energy {best_energy}\n")
        bests.append(best_energy)

    print(f"Final averaged best energy: {average(bests)}")

