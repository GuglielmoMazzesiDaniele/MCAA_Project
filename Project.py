from __future__ import annotations
from numpy.ma.extras import average

from Simulation import run_annealing
from Configuration import simulations_amount, N, steps_amount, schedule_type, T0

# ---------- CLI entry point ----------
if __name__ == "__main__":
    results = []

    print(f"\nRunning {simulations_amount} simulations with N = {N}, Steps = {steps_amount}, Starting"
          f" Temperature = {T0} and Temperature Scheduling = {schedule_type}\n")

    # Executing simulations
    for i in range (simulations_amount):
        print(f"Starting simulation {i + 1} of {simulations_amount}")

        best_state, best_energy = run_annealing(verbose=False)

        print(f"Iteration {i + 1}, best energy {best_energy}\n")
        results.append(best_energy)

    print(f"Final averaged best energy: {average(results)}")