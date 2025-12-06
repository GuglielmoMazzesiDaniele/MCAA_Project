import torch
import numpy as np
from tqdm import tqdm

import botorch.fit as fit
from botorch.models import SingleTaskGP
from botorch.acquisition.analytic import UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize, Normalize
from botorch.utils.sampling import draw_sobol_samples
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import N3Queens2DGrid
from utils import scheduler as Scheduler

def output_parser(v):
    beta = v[0] 
    a = v[1]
    b = v[2]
    k = v[3]
    return beta, a, b, k
datatype = torch.double

def next_point(x_train, y_train, bounds, device, num_restarts=10):
    """
    Run the Bayesian Optimization's loop with the given parameters
    
    Parameters:
    -----------
    x_train : np.array(n, p)
        Known input, p : dimension of one input, n : number of input
    
    y_train : np.array(n, m)
        Output values of elements of x_train, m : dimension of an output, n : number of elements in the array

    bounds : Tensor([float, float]) 
        Bounds for the dimension to estimate, need one pair of bound for each dimension, 
        e.g : Tensor([a, b]) if the x is 1 dimension and contained in between 'a' and 'b', 
              Tensor([a, b], [c, d]) if the next point is 2 dimension, first dimension bounded by 'a', 'b' second dimension by 'c', 'd'
        
    Returns:
    --------
    optim : float
        best next point to evaluate
    """
    n, dim = x_train.shape

    model = SingleTaskGP(
        x_train,
        y_train,
        outcome_transform=Standardize(m=1),
        input_transform=Normalize(d=dim),
        )
    
    model.to(device=device)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit.fit_gpytorch_mll(mll)

    beta= 0.2 * np.log(2 * n)

    UCB = UpperConfidenceBound(model=model, beta=beta)
    optim, _ = optimize_acqf( 
        acq_function=UCB,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts, 
        raw_samples=256,
        options={"sample_around_best":True}
        )
    
    return optim


def optimize_with_bo(
        N,
        device,
        bounds,
        scheduler_class,
        output_parser=output_parser,
        max_iters_model=20000,
        max_iters_bo=200,
        num_restarts=128,
        base_knowledges=5):
    """
    Run Bayesian Optimization to find optimal hyperparameters.
    
    Parameters:
    -----------
    N : int
        Size of the N-Queens problem
    device : torch.device
        Device to run computations on
    bounds : Tensor
        Bounds for the parameters to optimize
    scheduler_class : class
        Scheduler class to use. Must accept (a, b, k) as constructor arguments.
    output_parser : callable
        Function to parse the optimization vector into parameters
    max_iters_model : int
        Maximum iterations for the N3Queens model
    max_iters_bo : int
        Maximum iterations for Bayesian Optimization
    num_restarts : int
        Number of restarts for acquisition function optimization
    base_knowledges : int
        Number of initial Sobol samples
    """

    x_train = draw_sobol_samples(bounds=bounds, n=base_knowledges, q=1)
    x_train = x_train.to(dtype=datatype)
    x_train = torch.flatten(x_train, start_dim=1, end_dim=2)

    y_train = []

    for x in x_train:
        beta, *scheduler_params = output_parser(x.flatten())
        scheduler = scheduler_class(*scheduler_params)
        model = N3Queens2DGrid.N3Queens(N=N, beta=beta, scheduler=scheduler, max_iters=max_iters_model)
        final_config, energies = model.solve()
        ## Maximizing the negative energies is equivalent to minimizing the positive energy
        y_train.append(-energies[-1])
    
    y_train = torch.tensor(y_train, dtype=datatype).unsqueeze(-1)

    progress_bar = tqdm(range(0, max_iters_bo), desc="Optimizing with BO", unit='iter', leave=True)
    for _ in progress_bar:
        
        next = next_point(x_train, y_train, bounds, device=device, num_restarts=num_restarts)
        next = next.to(dtype=datatype)
        beta, *scheduler_params = output_parser(next.flatten())
        scheduler = scheduler_class(*scheduler_params)
        model = N3Queens2DGrid.N3Queens(beta=beta, scheduler=scheduler, N=N, max_iters=max_iters_model)

        final_config, energies = model.solve()
        
        y_train = torch.cat([y_train, torch.tensor([[-energies[-1]]], dtype=datatype)])
        x_train = torch.cat([x_train, next])
        
        best_id = torch.argmax(y_train)
        best_params = x_train[best_id]
        best_beta, *best_scheduler_params = output_parser(best_params)
        params_str = ", ".join([f"param_{i}: {p:.4f}" for i, p in enumerate(best_scheduler_params)])
        print(f"Best parameters -> beta: {best_beta:.4f}, {params_str}")
        progress_bar.set_postfix({"Best energy": -y_train[best_id].item()}, refresh=True)
    
    return x_train[best_id], -y_train[best_id].item(), energies


