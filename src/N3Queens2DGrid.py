import random
import numpy as np
from utils.scheduler import ConstantScheduler, Scheduler
    
class N3Queens:
    def __init__(self, N=8, 
                 max_iters=20000, 
                 beta=0.5, 
                 K = None, 
                 scheduler : Scheduler = None,
                 reheating : bool = False,
                 patience : int = 10000,
                 mh: bool = True):
        self.N = N
        self.max_iters = max_iters
        
        self.reheating = reheating
        self.patience = patience

        self.last_mod = 1
        self.checkpoint_beta = beta
        self.beta = beta
        self.scheduler = scheduler if scheduler != None else ConstantScheduler(self.beta)

        self.X, self.Y = np.indices((N, N))
        self.initialize(K)
        self.mh = mh

        self.n_queens_shuffle = self.N**2

    def initialize(self, k = None):
        """Randomize the board"""
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
            self.step()
            self.scheduler.step(self)
            energies.append(self.current_energy)

            if self.current_energy == 0:
                print(f"Solved in {t} steps")
                print(f"number of conflict energy : {self.compute_initial_energy()}, number of single conflicting queens : {self.count_queens_with_conflicts()}")
                return self.format_output(), energies
            
            if t % 5000 == 0:
                print(f"Step {t}: Energy = {self.current_energy}, Beta = {self.beta:.2f}")

            if self.reheating and ((self.t - self.last_mod) >= self.patience):
                self.shuffle_and_reheat(self.N)
                print(f"SHUFFLE shuffle queens at step {t}: Energy = {self.current_energy}, Beta = {self.beta:.2f}")

        print(f"Algorithm did not converge in {self.max_iters} steps, final energy : {energies[-1]}")
        return self.format_output(), energies

    def step(self):
        N = self.N
        # Pick a random queen
        rx, ry = random.randrange(0, N), random.randrange(0, N)
        old_z = self.grid[rx, ry]

        while True:
            new_z = random.randrange(0, N)
            if new_z != old_z:
                break

        old_conflicts = self.conflicts_at(rx, ry, old_z)
        new_conlficts = self.conflicts_at(rx, ry, new_z)

        delta_e = new_conlficts - old_conflicts

        if delta_e < 0:
            self.last_mod = self.t # This allow the shuffle method to check if shuffling is needed

        # We move only if we accept, otherwise we do nothing 
        if delta_e <= 0 or random.random() < np.exp(-delta_e * self.beta):
            self.grid[rx, ry] = new_z
            self.current_energy += delta_e

    def conflicts_at(self, x, y, z):
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

        ## TODO add a def reset() to the schedulers (Linear schedulers might struggle with this otherwise)
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
        
        if self.current_energy < new_e:
            self.beta = self.checkpoint_beta

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

        # Vectorized check is hard with a partial list, so we do a quick loop.
        # Since this runs only once at init, it's fine.
        for (tx, ty) in targets:
            tz = self.grid[tx, ty]
            
            dx = abs(x - tx)
            dy = abs(y - ty)
            dz = abs(z - tz)
            
            # Check for conflicts (Same logic as standard check)
            
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
    

