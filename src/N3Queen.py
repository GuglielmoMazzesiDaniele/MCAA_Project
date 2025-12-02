import numpy as np
import random

from utils.scheduler import ConstantScheduler, Scheduler
    
class N3Queens:
    def __init__(self, N=8, max_iters=20000, beta=0.5, init_function=None, scheduler : Scheduler = None):
        self.N = N
        self.max_iters = max_iters
        self.beta = beta
        self.scheduler = scheduler if scheduler != None else ConstantScheduler(self.beta)
        initial_positions = init_function() if init_function else self.random_initialization()
        
        self.queens = np.array(initial_positions, dtype=int)
        self.occupied_set = set(map(tuple, initial_positions))

        self.current_energy = self.compute_initial_energy()

    def solve(self):

        energies = []

        for t in range(1, self.max_iters + 1):
            self.t = t
            self.step()
            self.scheduler.step(self)

            energies.append(self.current_energy)

            if self.current_energy <= 0:
                print(f"Solved the problem in {t} steps")
                return self.queens, self.current_energy
        
        print(f"Algorithm did not converge in {self.max_iters} steps")
        return self.queens, energies

    def step(self):
        """
        Performs one Metropolis-Hastings step with O(1) updates.
        Returns True if the move was accepted, False otherwise.
        """
        N = self.N
        
        idx = random.randrange(len(self.queens))
        old_pos = tuple(self.queens[idx])

        while True:
            proposal = (random.randrange(N), random.randrange(N), random.randrange(N))
            if proposal not in self.occupied_set:
                new_pos = proposal
                break

        old_c = self.conflicts_for_queen(idx)

        self.queens[idx] = new_pos 
        new_c = self.conflicts_for_queen(idx)
        
        delta_E = new_c - old_c
        
        if delta_E <= 0 or random.random() < np.exp(-delta_E * self.beta):
            # ACCEPT: The array is already updated
            # Update the Set (Remove old, Add new)
            self.occupied_set.remove(old_pos)
            self.occupied_set.add(new_pos)
            
            # Update Total Energy
            self.current_energy += delta_E
            
            return True 
            
        else:
            # REJECT: revert the change to the array
            self.queens[idx] = old_pos
            
            return False 

    def conflicts_for_queen(self, idx):
        """Return number of queens attacking queen idx using NumPy vectorization."""
        Q = self.queens.shape[0]
        px, py, pz = self.queens[idx]

        DX = self.queens[:,0] - px
        DY = self.queens[:,1] - py
        DZ = self.queens[:,2] - pz

        DX[idx] = DY[idx] = DZ[idx] = 999999

        # axis
        axis = (
            ((DX == 0) & (DY == 0) & (DZ != 0)) |
            ((DX == 0) & (DY != 0) & (DZ == 0)) |
            ((DX != 0) & (DY == 0) & (DZ == 0))
        )

        # 2D diagonals
        xy = (np.abs(DX) == np.abs(DY)) & (DZ == 0)
        xz = (np.abs(DX) == np.abs(DZ)) & (DY == 0)
        yz = (np.abs(DY) == np.abs(DZ)) & (DX == 0)

        # 3D diagonal
        diag3 = (
            (np.abs(DX) == np.abs(DY)) &
            (np.abs(DY) == np.abs(DZ)) &
            (DX != 0)
        )

        return np.sum(axis | xy | xz | yz | diag3)

    def compute_initial_energy(self):
        E = 0
        num_queens = len(self.queens)
        for i in range(num_queens):
            E += self.conflicts_for_queen(i)
        return E // 2 
    
    def random_initialization(self):
        Q = self.N**2

        # Initial random configuration
        config = []
        occupied = set()
        while len(config) < Q:
            pos = (random.randrange(self.N), random.randrange(self.N), random.randrange(self.N))
            if pos not in occupied:
                occupied.add(pos)
                config.append(pos)

        return config