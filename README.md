# MCAA_Project
This repository contains the project's codebase for the EPFL's course COM-516, Markov Chains and Algorithmic Applications

Students: Guglielmo Daniele Mazzesi, Adrien Jean Deschenaux, Giovanni Luigi Ranieri

## Code Structure
The codebase is structured in the following way:

- src/main.py:            Main entry point. 
- src/optim:              Bayesian Optimization functions
- src/utils:              Utiility functions for plots, definition of schedulers and proposal moves for the queens
- src/N3Queens2DGrid.py:  Metropolis-Hastings Algorithm Implementation

## Running Experiments
The following tables show how you can run your experimentations. 

Mode Overview:
| Mode | Description |
|------|-------------|
| `run` | Run 5 independent standard MCMC pipelines with a fixed specified (in the code) scheduler |
| `all_schedulers` | Benchmark all annealing schedulers with 5 independent runs |
| `vary_n` | Evaluate performance for increasing board sizes with a fixed specified (in the code) scheduler |
| `optimize` | Bayesian optimization of scheduler parameters |

Parameters Mandatory:
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | `str` | `run` | Experiment mode: `run`, `optimize`, `all_schedulers`, `vary_n` |
| `--N` | `int` | `8` | Board size (N × N × N) |
| `--max_iters` | `int` | `20000` | Maximum number of MCMC iterations |
| `--device` | `str` | `cpu` | Device to use (`cpu` or `cuda`) |
| `--proposal_move` | `str` | `random` | Proposal move (`random`, `delta_move`) |

| Proposal Move | Description |
|---------------|-------------|
| `random` | Random queen relocation |
| `delta_move` | Local motion of queen |

Bayesian Optimization Parameter Bounds:

| Parameter | Min | Max |
|----------|-----|-----|
| `start_beta` | 0.01 | 10.0 |
| `end_beta` | 1.0 | 100.0 |

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_iters_bo` | `int` | `200` | Number of Bayesian Optimization iterations |

Future Works Features
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--reheating` | `flag` | `False` | Enable reheating |
| `--patience` | `int` | `10000` | Iterations before reheating |
| `--K` | `int` | `None` | Number of non-conflicting queens for smart initialization |

## Command Arguments for Plots in Report
Find below the commands used for generating the different plots in the report.
## Running Experiments
Run `pip install -r requirements.txt` to install dependencies.

To compare the performance of all the scheduling over a range of N from N=3 to N=17, run :
```bash
python main.py --max_iters=3000000 --beta=1.1 --end_beta=10.0 --mode vary_n
```

To try to solve the problem for a cube of side `N = n`, run :
```bash
python main.py --max_iters=your_maxiters --beta=your_start_beta --end_beta=your_end_beta --mode run --N n
```
Note : to make our plots for  `N=11` and `N=13`, we used : 
```bash
python main.py --max_iters=3000000 --beta=1.1 --end_beta=10.0 --mode run --N 11
```
```bash
python main.py --max_iters=3000000 --beta=1.1 --end_beta=10.0 --mode run --N 13
