import numpy as np
import random

from utils.scheduler import ConstantScheduler, Scheduler
    
class N3Queens:
    """
    A class implementing the N³-Queens problem solver using Metropolis-Hastings algorithm.
    
    The N³-Queens problem is a 3D generalization of the classic N-Queens problem, where N² queens
    must be placed on an N×N×N chessboard such that no two queens attack each other along axes,
    2D diagonals, or 3D diagonals.
    
    Attributes:
        N (int): The size of the 3D chessboard (N×N×N).
        max_iters (int): Maximum number of iterations for the solving algorithm.
        beta (float): Inverse temperature parameter for the Metropolis-Hastings algorithm.
        scheduler (Scheduler): Scheduler for updating beta during optimization.
        queens (np.ndarray): Array of queen positions, shape (N², 3).
        occupied_set (set): Set of occupied positions for O(1) collision detection.
        current_energy (int): Current number of conflicts in the configuration.
        t (int): Current iteration number.
    """
    
    def __init__(self, N=8, max_iters=20000, beta=0.5, init_function=None, scheduler : Scheduler = None):
        """
        Initialize the N³-Queens solver.
        
        Args:
            N (int, optional): Size of the 3D chessboard. Defaults to 8.
            max_iters (int, optional): Maximum number of iterations. Defaults to 20000.
            beta (float, optional): Initial inverse temperature parameter. Defaults to 0.5.
            init_function (callable, optional): Custom initialization function that returns
                a list of (x, y, z) positions. If None, uses random initialization. Defaults to None.
            scheduler (Scheduler, optional): Scheduler for updating beta during optimization.
                If None, uses ConstantScheduler. Defaults to None.
        """
        self.N = N
        self.max_iters = max_iters
        self.beta = beta
        self.scheduler = scheduler if scheduler != None else ConstantScheduler(self.beta)
        initial_positions = init_function() if init_function else self.random_initialization()
        
        self.queens = np.array(initial_positions, dtype=int)
        self.occupied_set = set(map(tuple, initial_positions))

        self.current_energy = self.compute_initial_energy()

    def solve(self):
        """
        Execute the Metropolis-Hastings algorithm to solve the N³-Queens problem.
        
        Runs the algorithm for up to max_iters iterations, stopping early if a valid
        solution (energy <= 0) is found. Updates beta according to the scheduler at
        each step.
        
        Returns:
            tuple: A tuple containing:
                - queens (np.ndarray): Final configuration of queen positions, shape (N², 3).
                - result (int or list): If solved, returns the final energy (0).
                    If not solved, returns a list of energy values at each iteration.
        
        Notes:
            Prints progress messages when a solution is found or when max_iters is reached.
        """
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
        Perform one Metropolis-Hastings step with O(1) energy updates.
        
        The step involves:
        1. Randomly selecting a queen to move
        2. Proposing a new random position (not currently occupied)
        3. Computing the change in conflicts (delta_E)
        4. Accepting or rejecting the move based on the Metropolis criterion:
           - Always accept if delta_E <= 0 (improvement)
           - Accept with probability exp(-delta_E * beta) if delta_E > 0
        
        Returns:
            bool: True if the move was accepted, False if rejected.
        
        Notes:
            - Uses O(1) energy updates by only recomputing conflicts for the moved queen
            - Updates the occupied_set and current_energy upon acceptance
            - Reverts the move upon rejection
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
        """
        Compute the number of conflicts for a specific queen using NumPy vectorization.
        
        A conflict occurs when another queen can attack the queen at position idx along:
        - Axes: same x, y, or z coordinate (but not all three)
        - 2D diagonals: diagonal in xy, xz, or yz planes
        - 3D diagonal: |dx| = |dy| = |dz| and all non-zero
        
        Args:
            idx (int): Index of the queen in the self.queens array.
        
        Returns:
            int: Number of queens attacking the queen at position idx.
        
        Notes:
            - Uses vectorized NumPy operations for efficiency
            - Sets differences to 999999 for the queen itself to exclude self-conflicts
            - Time complexity: O(N²) where N² is the number of queens
        """
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
        """
        Compute the total initial energy (number of conflicts) for the configuration.
        
        Iterates through all queens and sums their individual conflicts. Since each
        conflict is counted twice (once for each queen in the attacking pair), the
        result is divided by 2.
        
        Returns:
            int: Total number of attacking pairs (conflicts) in the current configuration.
        
        Notes:
            - Time complexity: O(N⁴) where N is the board size
            - Each queen requires O(N²) to compute conflicts, and there are N² queens
        """
        E = 0
        num_queens = len(self.queens)
        for i in range(num_queens):
            E += self.conflicts_for_queen(i)
        return E // 2 
    
    def random_initialization(self):
        """
        Generate a random initial configuration of N² queens on the N×N×N board.
        
        Places N² queens at random, distinct positions on the 3D chessboard. Ensures
        no two queens occupy the same position.
        
        Returns:
            list: List of N² tuples, where each tuple (x, y, z) represents a queen's
                position with coordinates in range [0, N).
        
        Notes:
            - Q = N² is the total number of queens to place
            - Uses rejection sampling to ensure distinct positions
            - Expected time complexity: O(N²) assuming uniform random distribution
        """
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