from abc import ABC, abstractmethod
import numpy as np
import random

offsets = np.array([
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
])

def select_proposal_move(move_type, queen_class):
    if move_type == "random":
        return RandomMove(queen_class)
    if move_type == "delta_move":
        return DeltaMove(queen_class)
    else:
        raise ValueError(f"Unknown proposal move type: {move_type}")

class ProposalMove(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def accept(self, delta_energy, pos_x, pos_y, new_z):
        pass
    
    @abstractmethod
    def step(self):
        pass
    
class RandomMove(ProposalMove):
    def __init__(self, queen_class):
        self.queen_class = queen_class

    def accept(self, delta_energy, pos_x, pos_y, new_z):
        if delta_energy <= 0 or random.random() < np.exp(-delta_energy * self.queen_class.beta):
            self.queen_class.grid[pos_x, pos_y] = new_z
            self.queen_class.current_energy += delta_energy
            
    def name(self):
        return "random"
            
    def step(self):
        
        N = self.queen_class.N
        rx, ry = random.randrange(0, N), random.randrange(0, N)
        old_z = self.queen_class.grid[rx, ry]

        while True:
            new_z = random.randrange(0, N)
            if new_z != old_z:
                break

        old_conflicts = self.queen_class.conflicts_at(rx, ry, old_z)
        new_conlficts = self.queen_class.conflicts_at(rx, ry, new_z)

        delta_e = new_conlficts - old_conflicts

        if delta_e < 0:
            self.queen_class.last_mod = self.queen_class.t # This allow the shuffle method to check if shuffling is needed
            
        self.accept(delta_e, rx, ry, new_z)
        
class DeltaMove(ProposalMove):
    
    def __init__(self, queen_class):
        self.queen_class = queen_class
        
    def name(self):
        return "delta_move"
        
    def neighbors_3d(self, x, y, z):
        
        N = self.queen_class.N
        #all_neighbors = {}
        neighs = []

        for dx, dy, dz in offsets:
            nx, ny, nz = x + dx, y + dy, z + dz

            # keep only neighbors inside the 3D cube
            if 0 <= nx < N and 0 <= ny < N and 0 <= nz < N:
                neighs.append((nx, ny, nz))

        #all_neighbors[(x, y, z)] = neighs

        return len(neighs), neighs
    
    def step(self):
        
        N = self.queen_class.N
        rx, ry = random.randrange(0, N), random.randrange(0, N)
        old_z = self.queen_class.grid[rx, ry]

        # old conflicts for this queen
        old_conflicts = self.queen_class.conflicts_at(rx, ry, old_z)
        allowed_cells_before, allowed = self.neighbors_3d(rx, ry, old_z)
        new_z = allowed[random.randrange(0, len(allowed))][2]
        
        # new conflicts
        new_conlficts = self.queen_class.conflicts_at(rx, ry, new_z)
        allowed_cells_after, _ = self.neighbors_3d(rx, ry, new_z)
        
        delta_e = new_conlficts - old_conflicts

        if delta_e < 0:
            self.queen_class.last_mod = self.queen_class.t # This allow the shuffle method to check if shuffling is needed
            
        self.accept(delta_e, rx, ry, new_z, allowed_cells_before, allowed_cells_after)

    def accept(self, delta_energy, pos_x, pos_y, new_z, allowed_cells_before, allowed_cells_after):
        if delta_energy <= 0 or random.random() < min(1, np.exp(-self.queen_class.beta * delta_energy) * (allowed_cells_before / allowed_cells_after)):
            self.queen_class.grid[pos_x, pos_y] = new_z
            self.queen_class.current_energy += delta_energy
            
# class WeightedMove(ProposalMove):
#     def __init__(self, queen_class):
#         self.queen_class = queen_class
        
#     def name(self):
#         return "weighted_move"
    
#     def sample_worst_queen(self):
#         """Sample the queen with the highest weight."""
#         weights = config[:, 3]
#         total_weight = np.sum(weights)
#         max_weight = np.max(weights)
#         candidates = np.where(weights == max_weight)[0]
#         idx = np.random.choice(candidates)
#         return idx, config[idx, 3], int(total_weight)
        
#     def step(self):
        
#         N = self.queen_class.N
#         idx, c_old, total_weight_before = self.sample_worst_queen()
#         rx, ry = random.randrange(0, N), random.randrange(0, N)
#         old_z = self.queen_class.grid[rx, ry]

#         while True:
#             new_z = random.randrange(0, N)
#             if new_z != old_z:
#                 break

#         old_conflicts = self.queen_class.conflicts_at(rx, ry, old_z)
#         new_conlficts = self.queen_class.conflicts_at(rx, ry, new_z)

#         delta_e = new_conlficts - old_conflicts

#         if delta_e < 0:
#             self.queen_class.last_mod = self.queen_class.t # This allow the shuffle method to check if shuffling is needed
            
#         self.accept(delta_e, rx, ry, new_z)
        
#     def accept(self, delta_energy, pos_x, pos_y, new_z, c_new, c_old, total_weight_before, total_weight_after):
#         if delta_energy <= 0 or random.random() < min(1, (np.exp(-self.queen_class.beta * delta_energy) * (c_new * total_weight_before) / (c_old * total_weight_after))):
#             self.queen_class.grid[pos_x, pos_y] = new_z
#             self.queen_class.current_energy += delta_energy
    
