from abc import ABC, abstractmethod
import numpy as np

class Scheduler(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def step(self, model):
        pass

    def reset(self, model):
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

    def reset(self, model):
        self.alpha = (self.end_beta / self.start_beta) ** (1 / (self.max_iters - model.t))
        pass
    
    @staticmethod
    def name():
        return "exponential"

class ConstantScheduler(Scheduler):
    def __init__(self, beta):
        self.beta = beta

    def step(self, model):
        model.beta = self.beta
    
    @staticmethod
    def name():
        return "constant"

class GeometricScheduler(Scheduler):
    def __init__(self, alpha=1.5):
        self.alpha = alpha

    def step(self, model):
        model.beta *= self.alpha
    
    @staticmethod
    def name():
        return "geometric"

class LinearScheduler(Scheduler):
    def __init__(self, a=1.0, b=0.0):
        self.a = a
        self.b = b

    def step(self, model):
        model.beta = self.a * model.beta + self.b
        
    @staticmethod
    def name():
        return "linear"

class LogScheduler(Scheduler):
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def step(self, model):
        model.beta = self.alpha * np.log(2 * model.t)
        
    @staticmethod
    def name():
        return "logarithmic"
        
class LogisticScheduler(Scheduler):
    def __init__(self, beta_max, k=10.0, total_iters=500000):
        self.beta_max = beta_max
        self.k = k / total_iters
        self.mid = total_iters / 2

    def step(self, model):
        t = model.t
        model.beta = self.beta_max / (1 + np.exp(-self.k * (t - self.mid)))
        
    @staticmethod
    def name():
        return "logistic"
        
class PowerScheduler(Scheduler):
    def __init__(self, beta_max, p=1.5, total_iters=500000):
        self.beta_max = beta_max
        self.p = p
        self.T = total_iters

    def step(self, model):
        t = min(model.t, self.T)
        model.beta = self.beta_max * (t / self.T) ** self.p
        
    @staticmethod
    def name():
        return "power"

