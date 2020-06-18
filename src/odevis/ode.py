import numpy as np


class ODE_System(object):

    def __init__(self, equations):
        super(ODE_System, self).__init__()
        self.equations = equations
        self.num_equations = len(self.equations)

    def __call__(self, variables):
        assert variables.shape[0] == self.num_equations
        apply = np.vectorize(lambda fn, *x: fn(*x))
        return apply(self.equations, *variables)

    def __len__(self):
        return self.num_equations

    def __getitem__(self, i):
        return self.equations[i]

