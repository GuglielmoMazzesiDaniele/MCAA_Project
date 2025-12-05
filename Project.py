from __future__ import annotations
from dataclasses import dataclass
import random
import math
from typing import List, Tuple

Coordinate = Tuple[int, int, int]


@dataclass
class State:
    """Configuration X = (Q, E) with N^2 queens and N^2 tracked empties."""
    N: int
    queens: List[Coordinate]   # Q
    empties: List[Coordinate]  # E


def all_coordinates(N: int) -> List[Coordinate]:
    return [(x, y, z) for x in range(1, N + 1)
                      for y in range(1, N + 1)
                      for z in range(1, N + 1)]


def random_initial_state(N: int, rng: random.Random | None = None) -> State:
    if rng is None:
        rng = random

    # Generating all coordinates
    coordinates = all_coordinates(N)
    rng.shuffle(coordinates)

    # Splitting the coordinates in occupied and unoccupied
    queens = coordinates[ : N^2]
    empties = coordinates[N^2 : N^3]

    return State(N = N, queens = queens, empties = empties)

def attack(q1: Coordinate, q2: Coordinate) -> bool:
    x1, y1, z1 = q1
    x2, y2, z2 = q2

    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    dz = abs(z1 - z2)

    # same coordinate
    if x1 == x2 or y1 == y2 or z1 == z2:
        return True

    # 2D diagonals
    if dx == dy or dx == dz or dy == dz:
        return True

    # 3D diagonal
    if dx == dy == dz:
        return True

    return False

def energy(state: State) -> int:
    H = 0
    Q = state.queens
    n = len(Q)
    for i in range(n):
        for j in range(i + 1, n):
            if attack(Q[i], Q[j]):
                H += 1
    return H

def propose_swap(state: State, rng: random.Random | None = None) -> State:
    """
    Propose a new state by swapping a random queen with a random empty cell.

    Parameters
    ----------
    state : State
        Current configuration, containing N^2 queen positions.
    rng : random.Random, optional
        Random generator used for reproducibility. If None, the default
        global RNG is used.

    Returns
    -------
    State
        A new state with one queen moved to a previously empty location.
    """

    if rng is None:
        rng = random

    N = state.N
    Q = state.queens[:]
    E = state.empties[:]

    # choose random indices
    i_q = rng.randrange(len(Q))
    i_e = rng.randrange(len(E))

    q = Q[i_q]
    e = E[i_e]

    # swap them
    Q[i_q] = e
    E[i_e] = q

    return State(N=N, queens=Q, empties=E)

def metropolis_step(state: State,
                    T: float,
                    rng: random.Random | None = None) -> State:
    if rng is None:
        rng = random

    current_E = energy(state)
    proposal = propose_swap(state, rng)
    proposal_E = energy(proposal)

    dH = proposal_E - current_E

    if dH <= 0:
        return proposal  # always accept if energy does not increase

    # accept uphill move with probability exp(-dH / T)
    if rng.random() < math.exp(-dH / T):
        return proposal
    else:
        return state