import random
import numpy as np

def k_no_conflicts_init(N: int, k: int = 1):
  """
  k : the number of non conflicting queens for at each positionning 
  N : side of the cube

  Note : 
    The algorithm works as follow: 
      1) Place the first queen at a position chosen uniformely at random in the cube
      2) Place all the following queens such that there are no conflict with the k previous queens
    
    The choice of k this might lead to a better initialization than chosing N**2 queens uniformely as 
    this avoid having all the queen conflincting each other 
  """

  first_pos = pick_position(N)
  queens = [first_pos]
  occupied = set(first_pos)  
  k_previous_set = set(queens)
  k_previous_list = [first_pos]

  for _ in range(N**2):
    while True:
       pos = pick_position(N)
       if pos not in occupied and count_conflicts(k_previous_set, pos) == 0:
          break
    
    occupied.add(pos)
    queens.append(pos)

    k_previous_list.append(pos)
    k_previous_set.add(pos)

    if len(k_previous_set) > k:
       key = k_previous_list[0]
       k_previous_list.remove(0)
       k_previous_set.remove(key)

  return queens


def pick_position(N: int):
  """
  return a 3D coordinate picked in a N x N x N cube 
  """
  return (random.randrange(N), random.randrange(N), random.randrange(N))

def count_conflicts(queens, pos):
    """
    """
    q = pos
    
    # Calculate absolute differences
    # We perform math on the whole array at once
    delta = np.abs(queens - q)
    
    dx = delta[:, 0]
    dy = delta[:, 1]
    dz = delta[:, 2]

    # 1. Axis conflicts: Any two deltas are 0 (and not self)
    # We sum boolean arrays (True=1, False=0)
    zeros = (dx == 0).astype(int) + (dy == 0).astype(int) + (dz == 0).astype(int)
    
    # Zeros=3 is the queen itself, Zeros=2 is axis conflict
    axis_conflicts = np.sum(zeros == 2)

    # 2. 2D Diagonals: One delta is 0, other two are equal
    # (dx==dy!=0 and dz==0) OR (dx==dz!=0 and dy==0) ...
    # Simplified: Zeros=1 AND (dx==dy or dx==dz or dy==dz)
    # Note: If zeros=1, exactly one is zero. We just check if the other two match.
    face_diag = np.sum((zeros == 1) & ((dx == dy) | (dx == dz) | (dy == dz)))

    # 3. 3D Diagonals: All three equal and non-zero
    space_diag = np.sum((zeros == 0) & (dx == dy) & (dy == dz))

    return axis_conflicts + face_diag + space_diag