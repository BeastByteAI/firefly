from firefly.nn.parameter import Parameter


class BaseOptimizer:

    def __init__(self, parameters: list[Parameter]):
        self.parameters = parameters

    def zero_grad(self):
        for param in self.parameters:
            param.tensor.zero_grad()

    def step(self):
        raise NotImplementedError("Step method not implemented.")
