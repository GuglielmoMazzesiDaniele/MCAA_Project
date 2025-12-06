import random
import numpy as np
from utils.scheduler import ConstantScheduler, Scheduler
    
class N3Queens:
    def __init__(self, N=8, 
                 max_iters=20000, 
                 beta=0.5, 
                 k = None, 
                 scheduler : Scheduler = None):
        self.N = N
        self.max_iters = max_iters
        self.beta = beta
        self.scheduler = scheduler if scheduler != None else ConstantScheduler(self.beta)
        
        self.X, self.Y = np.indices((N, N))
        self.initialize(k)

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
                return self.format_output(), energies
            
            if t % 5000 == 0:
                print(f"Step {t}: Energy = {self.current_energy}, Beta = {self.beta:.2f}")

        
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

        return np.sum(z_axis | xz_diag | yz_diag | xy_diag | diag3)

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
    
    def smart_init(self, k):
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