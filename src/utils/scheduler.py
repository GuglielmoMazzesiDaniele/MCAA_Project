from abc import ABC, abstractmethod
import numpy as np

class Scheduler(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def step(self, model):
        pass

class StepScheduler(Scheduler):
    def __init__(self, beta, stepsize = 100):
        pass

    def step(self, model):
        pass

class ExponentialScheduler(Scheduler):
    def __init__(self, start_beta=0.1, end_beta=50.0, max_iters=100000, alpha=None):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.max_iters = max_iters
        # Use provided alpha or compute from start_beta/end_beta/max_iters
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = (end_beta / start_beta) ** (1 / (max_iters))

    def step(self, model):
        model.beta = model.beta * self.alpha

class ConstantScheduler(Scheduler):
    def __init__(self, beta):
        self.beta = beta

    def step(self, model):
        model.beta = self.beta

class GeometricScheduler(Scheduler):
    def __init__(self, alpha=1.5):
        self.alpha = alpha

    def step(self, model):
        model.beta *= self.alpha

class LinearScheduler(Scheduler):
    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b

    def step(self, model):
        model.beta = self.a * model.beta + self.b

class LogScheduler(Scheduler):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def step(self, model):
        model.beta = self.alpha * np.log(2 * model.t)


