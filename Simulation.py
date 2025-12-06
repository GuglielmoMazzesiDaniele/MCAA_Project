import State
import Scheduling
import random
import math

from tqdm import tqdm
from typing import Optional
from Configuration import N, steps_amount

# ---------- Proposal kernel (symmetric) ----------

def propose(state: State,
            eps: float = 1.0,
            rng: Optional[random.Random] = None):
    """
    Propose a move:
      - pick a queen with probability proportional to (conflicts + eps),
      - pick an empty cell uniformly,
      - move the chosen queen there.

    Parameters
    ----------
    state : State
        Current configuration X.
    eps : float
        Small positive constant to ensure all queens have nonzero weight.
    rng : random.Random, optional
        RNG for reproducibility.

    Returns
    -------
    proposal_state : State
        Proposed new state Y.
    i_q : int
        Index of the moved queen in the original state's queen list.
    old_pos : Coord
        Old position of that queen in X.
    new_pos : Coord
        New position of that queen in Y (an empty cell in X).
    conflicts_X : List[int]
        Conflict counts in X (for proposal ratio).
    Z_Q_X : float
        Sum of weights w_i = conflicts_X[i] + eps in X.
    """
    if rng is None:
        rng = random

    N = state.N
    queens = state.queens[:]
    empties = state.empties[:]
    conflicts_X = State.compute_conflicts(state)

    # weights w_i = conflicts_i + eps
    weights = [c + eps for c in conflicts_X]
    Z_Q_X = sum(weights)

    # choose queen index with probability proportional to weights
    r = rng.random() * Z_Q_X
    cumulative = 0.0
    i_q = 0
    for idx, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            i_q = idx
            break

    old_pos = queens[i_q]

    # choose random empty
    i_e = rng.randrange(len(empties))
    new_pos = empties[i_e]

    # move queen
    queens[i_q] = new_pos
    proposal_state = State.State(N=N, queens=queens, empties=empties)

    return proposal_state, i_q, old_pos, new_pos, conflicts_X, Z_Q_X

# ---------- Metropolis / simulated annealing ----------

def metropolis_step(state: State,
                    T: float,
                    eps: float = 1.0,
                    rng: Optional[random.Random] = None) -> State:
    """
    One Metropolis-Hastings step with conflict-biased proposal.

    Proposal Q(X,·):
      - choose queen i with prob ~ conflicts_i(X) + eps,
      - choose empty uniformly,
      - move queen i there.

    Acceptance probability:
      alpha = min(1,
                  exp(-ΔH / T) * [Q(Y->X) / Q(X->Y)] )

    Parameters
    ----------
    state : State
        Current state X.
    T : float
        Temperature.
    eps : float
        Small positive constant added to conflicts to ensure all queens
        have nonzero weight.
    rng : random.Random, optional
        RNG.

    Returns
    -------
    State
        New state (either accepted proposal Y or original X).
    """
    if rng is None:
        rng = random

    current_E = State.compute_energy(state)

    # Propose Y and get information about Q(X->Y)
    (proposal,
     i_q,
     old_pos,
     new_pos,
     conflicts_X,
     Z_Q_X) = propose(state, eps=eps, rng=rng)

    proposal_E = State.compute_energy(proposal)
    dH = proposal_E - current_E

    # Compute data for Q(Y->X)
    conflicts_Y = State.compute_conflicts(proposal)
    queens_Y = proposal.queens

    # In Y, the queen we moved is at new_pos; find its index
    idx_back = queens_Y.index(new_pos)

    w_i_X = conflicts_X[i_q] + eps
    w_k_Y = conflicts_Y[idx_back] + eps
    Z_Q_Y = sum(c + eps for c in conflicts_Y)

    # |empties| cancels in the ratio, so:
    # Q(Y->X) / Q(X->Y) = [w_k(Y)/Z_Q(Y)] / [w_i(X)/Z_Q(X)]
    q_ratio = (w_k_Y * Z_Q_X) / (w_i_X * Z_Q_Y)

    if T <= 0:
        # Degenerate case: purely greedy with Hastings correction
        accept_prob = 1.0 if (dH < 0 and q_ratio >= 1.0) else 0.0
    else:
        accept_prob = min(1.0, math.exp(-dH / T) * q_ratio)

    if rng.random() < accept_prob:
        return proposal
    else:
        return state


def run_annealing(eps: float = 1.0,
                  rng: Optional[random.Random] = None,
                  verbose: bool = True) -> (State, int):
    """
    Run simulated annealing using conflict-biased Metropolis-Hastings.

    Parameters
    ----------
    N : int
        Board size.
    steps : int
        Number of MH steps.
    T0 : float
        Initial temperature.
    schedule : {"log", "exp", "const"}
        Cooling schedule used by default_temperature_schedule.
    eps : float
        Small positive constant for conflict weights.
    rng : random.Random, optional
        RNG.
    verbose : bool
        If True, print progress.

    Returns
    -------
    State
        Best state found (lowest energy observed).
    """
    if rng is None:
        rng = random.Random()

    current_state = State.generate_initial_state(N, rng)
    best_state = current_state
    best_energy = State.compute_energy(current_state)

    if verbose:
        print(f"Initial energy: {best_energy}")

    for t in tqdm(range(steps_amount), desc=f"Progress: ", colour="white"):
        current_state = metropolis_step(
            current_state,
            Scheduling.temperature_schedule(t=t),
            eps=eps,
            rng=rng)

        current_energy = State.compute_energy(current_state)

        if current_energy < best_energy:
            best_energy = current_energy
            best_state = current_state
            if verbose:
                print(f"[step {t}] New best energy: {best_energy}")
            if best_energy == 0:
                if verbose:
                    print(f"Found solution (energy 0) at step {t}")
                break

    if verbose:
        print(f"Finished after {t+1} steps. Best energy: {best_energy}")

    return best_state, best_energy