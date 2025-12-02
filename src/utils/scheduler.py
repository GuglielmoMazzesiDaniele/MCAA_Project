from abc import ABC, abstractmethod
import numpy as np

class Scheduler(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def step(self, model):
        pass

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


