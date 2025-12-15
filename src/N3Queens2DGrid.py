import random
import numpy as np
from utils.scheduler import ConstantScheduler, Scheduler
from utils.proposal_move import select_proposal_move
    
class N3Queens:
    def __init__(self, 
                 N=8, 
                 max_iters=20000, 
                 beta=0.5, 
                 K = None, 
                 scheduler : Scheduler = None,
                 name_proposal_move : str = "random",
                 reheating : bool = False,
                 patience : int = 10000,
                 gibbs: bool = False):
        
        """
        N-3 Queens problem in a 2D grid representation
        :param N: Size of the grid (N x N) and number of queens
        :param max_iters: Maximum number of iterations to perform
        :param beta: Initial beta parameter for simulated annealing, else fixed beta
        :param K: Number of non-conflicting queens to initialize with (if None, random initialization)
        :param scheduler: Scheduler object to manage beta updates
        :param name_proposal_move: Name of the proposal move strategy to use
        :param reheating: Whether to use reheating strategy
        :param patience: Number of iterations to wait before reheating
        :param gibbs: Whether to use Gibbs sampling for updates
        """

        self.N = N
        self.max_iters = max_iters
        
        self.reheating = reheating
        self.patience = patience

        self.last_mod = 1
        self.beta = beta
        self.start_beta = self.beta
        self.scheduler = scheduler if scheduler != None else ConstantScheduler(self.beta)
        
        self.proposal_move = select_proposal_move(name_proposal_move, self)

        self.X, self.Y = np.indices((N, N))
        self.gibbs = gibbs
        self.n_queens_shuffle = self.N**2

        self.initialize(K)

    def initialize(self, k = None):
        """Randomize the board for initialization or use smart initialization"""

        if k == None:
            self.grid = np.random.randint(0, self.N, size=(self.N, self.N))
            self.current_energy = self.compute_initial_energy()
        elif k >= 1: 
            print(f"Init with {k} non conflicting queens")
            self.smart_init(k)
        else:
            raise ValueError(f"k must be a None or an integer > 0")
    
    def solve(self):
        
        print(f"Solving problem for N={self.N}")
        energies = []

        for t in range(1, self.max_iters + 1):
            self.t = t

            if self.gibbs : 
                self.gibbs_step()
            else : 
                self.proposal_move.step()

            self.scheduler.step(self)
            energies.append(self.current_energy)

            if self.current_energy == 0:
                print(f"Solved in {t} steps")
                print(f"number of conflict energy : {self.compute_initial_energy()}, number of single conflicting queens : {self.count_queens_with_conflicts()}")
                return self.format_output(), energies, self.count_queens_with_conflicts()
            
            if t % 5000 == 0:
                print(f"Step {t}: Energy = {self.current_energy}, Beta = {self.beta:.2f}")

            if self.reheating and ((self.t - self.last_mod) >= self.patience):
                self.shuffle_and_reheat(self.N)
                self.beta = self.start_beta
                self.last_mod = self.t
                print(f"SHUFFLE shuffle queens at step {t}: Energy = {self.current_energy}, Beta = {self.beta:.2f}")

        print(f"Algorithm did not converge in {self.max_iters} steps, final energy : {energies[-1]}, single conflict are : {self.count_queens_with_conflicts()}")
        return self.format_output(), energies, self.count_queens_with_conflicts()

    def conflicts_at(self, x, y, z):
        """
        Compute the number of conflicts for a queen at position (x, y, z)
        1. Axis (Horizontal only, since Vertical is impossible by loop)
           Same Z, and aligned on X or Y
        2. 2D Diagonals
        3. 3D Diagonal
        4. Return total number of conflicts
        """
        DX = self.X - x
        DY = self.Y - y 
        DZ = self.grid - z
        
        DX[x, y] = 100000
        DY[x, y] = 200000
        DZ[x, y] = 300000

        ADX = np.abs(DX)
        ADY = np.abs(DY)
        ADZ = np.abs(DZ)

        z_axis = (DZ == 0) & ((DX == 0) | (DY == 0))
        xz_diag = (DY == 0) & (ADX == ADZ)
        yz_diag = (DX == 0) & (ADY == ADZ)
        xy_diag = (DZ == 0) & (ADX == ADY)
        diag3 = (ADX == ADY) & (ADY == ADZ)

        tot = np.sum(z_axis | xz_diag | yz_diag | xy_diag | diag3)
        return tot

    def compute_initial_energy(self):
        """
        Compute the total number of conflicts in the current grid configuration
        1. For each queen, compute its conflicts
        2. Sum all conflicts and divide by 2 (since each conflict is counted twice)
        3. Return total energy
        """
        E = 0
        for x in range(self.N):
            for y in range(self.N):
                E += self.conflicts_at(x, y, self.grid[x, y])
        return E // 2
    
    def format_output(self):
        queens = []
        for x in range(self.N):
            for y in range(self.N):
                queens.append((x, y, self.grid[x, y]))
        return np.array(queens)
    
    def count_queens_with_conflicts(self):
        """
        Returns the number of individual queens that have at least one conflict.
        """
        count = 0
        for x in range(self.N):
            for y in range(self.N):
                if self.conflicts_at(x, y, self.grid[x, y]) > 0:
                    count += 1
        return count
    
    def smart_init(self, k):
        """
        Initialize the grid with the least amount of conflicts for the k previous queens
        
        :param k: Number of previous queen to avoid conflict with
        """
        self.grid = np.full((self.N, self.N), -1, dtype=int)

        placed_queens = []

        for x in range(self.N):
            for y in range(self.N):
                best_z = 0 
                min_conflicts = float('inf')

                candidates = list(range(self.N))

                random.shuffle(candidates)

                for z_candidates in candidates:
                    conflicts = self.count_partial_conflicts(x, y, z_candidates, placed_queens, k) 

                    if conflicts == 0:
                        best_z = z_candidates
                        min_conflicts = 0
                        break

                    if conflicts < min_conflicts:
                        min_conflicts = conflicts
                        best_z = z_candidates
                
                self.grid[x, y] = best_z
                placed_queens.append((x, y))

        self.current_energy = self.compute_initial_energy()

    def shuffle_and_reheat(self, Q: int = 25):
        """
        Shuffle the Q queens positions to not have been modified for the longest time 
        and restore the beta parameters to what it was at the last modification
        
        :param Q: Number of queen to select for shuffling 

        Note : Algorithm description

        the patience parameter select how long the model accepts to not do any change, once
        this threshold is passed.

        If the model is still solving at time t, the it must be that it has not converged and 
        if the model has not made any change for some time it could mean that the model cooled down
        and is not able to accept interesting moves anymore. 

        Therefore when the model observe that it has not moved any queen for a while and the energy
        is more than 0, then it will try to pick a few queens and then shuffle them by assigning them
        with a random value, recompute the energy of the model, and then restore a previous value for 
        beta
        """
        #First we want to pick Q queens
        picked = []
        picked_set = set()

        while len(picked) != Q:
            rx, ry = random.randrange(0, self.N), random.randrange(0, self.N)
            
            if (rx, ry) not in picked_set:
                picked_set.add((rx, ry))
                picked.append((rx, ry))

                while True:
                    rz = np.random.randint(0, self.N)

                    if rz != self.grid[rx, ry]:
                        break
                
                self.grid[rx, ry] = rz
        
        new_e = self.compute_initial_energy()
        
        self.current_energy = new_e
        self.last_mod = self.t
        self.scheduler.reset(self)

    
    def count_partial_conflicts(self, x, y, z, placed_queens, k=None):
        """
        Helper to check conflicts only against specific queens.
        """
        conflicts = 0
        
        # Determine which queens to check (last k or all)
        if k is not None:
            targets = placed_queens[-k:]
        else:
            targets = placed_queens

        for (tx, ty) in targets:
            tz = self.grid[tx, ty]
            
            dx = abs(x - tx)
            dy = abs(y - ty)
            dz = abs(z - tz)
            
            # 1. Axis (Horizontal only, since Vertical is impossible by loop)
            # Same Z, and aligned on X or Y
            axis = (dz == 0) and ((dx == 0) or (dy == 0))
            
            # 2. 2D Diagonals
            xy_diag = (dz == 0) and (dx == dy)
            xz_diag = (dy == 0) and (dx == dz)
            yz_diag = (dx == 0) and (dy == dz)
            
            # 3. 3D Diagonal
            diag3 = (dx == dy) and (dy == dz)
            
            if axis or xy_diag or xz_diag or yz_diag or diag3:
                conflicts += 1
                
        return conflicts

    def gibbs_step(self):
            rx, ry = random.randrange(0, self.N), random.randrange(0, self.N)
            current_z = self.grid[rx, ry]
            
            potential_energies = []
            for z in range(self.N):
                if z == current_z:
                    potential_energies.append(self.conflicts_at(rx, ry, current_z))
                else:
                    potential_energies.append(self.conflicts_at(rx, ry, z))
            
            potential_energies = np.array(potential_energies)

            min_E = np.min(potential_energies)
            stable_energies = potential_energies - min_E
            
            beta_val = self.beta.item() if hasattr(self.beta, "item") else self.beta
            
            weights = np.exp(-beta_val * stable_energies)
            probs = weights / np.sum(weights)

            new_z = np.random.choice(np.arange(self.N), p=probs)

            if new_z != current_z:
                old_conflict = potential_energies[current_z]
                new_conflict = potential_energies[new_z]
                
                delta_e = new_conflict - old_conflict

                if delta_e < 0:
                    self.last_mod = self.t

                self.grid[rx, ry] = new_z
                self.current_energy += delta_e

    def write_solution_file(self, filepath: str, grid: np.ndarray | None = None) -> None:
        """
        Export the current configuration to a file with N^2 lines.
        Each line contains 'x,y,z' (0-based) separated by commas.

        Parameters
        ----------
        filepath : str
            Path of the output file to write.
        grid : np.ndarray, optional
            If provided, exports this grid instead of self.grid. Must be shape (N, N)
            with integer entries in {0, ..., N-1}.
        """
        g = self.grid if grid is None else grid

        if np.any(g < 0) or np.any(g >= self.N):
            raise ValueError("grid contains z values outside {0, ..., N-1}")

        with open(filepath, "w", encoding="utf-8") as f:
            for x in range(self.N):
                for y in range(self.N):
                    z = int(g[x, y])
                    f.write(f"{x},{y},{z}\n")