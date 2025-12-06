import random
import numpy as np
from utils.scheduler import ConstantScheduler, Scheduler
    
class N3Queens:
    def __init__(self, N=8, max_iters=20000, beta=0.5, init_function=None, scheduler : Scheduler = None):
        self.N = N
        self.max_iters = max_iters
        self.beta = beta
        self.scheduler = scheduler if scheduler != None else ConstantScheduler(self.beta)
        
        self.X, self.Y = np.indices((N, N))
        self.initialize()

    def initialize(self):
        """Randomize the board"""
        self.grid = np.random.randint(0, self.N, size=(self.N, self.N))
        self.current_energy = self.compute_initial_energy()

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
        return self.format_output, energies

    def step(self):
        N = self.N
        # Pick a random queen
        rx, ry = random.randrange(0, N), random.randrange(0, N)
        
        old_z = self.grid[rx, ry]
        new_z = random.randrange(0, N)

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