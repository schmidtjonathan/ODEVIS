import torch


class ODE_System(object):

    def __init__(self, equations):
        super(ODE_System, self).__init__()
        self.equations = equations
        self.num_equations = len(self.equations)

    def __call__(self, var):
        assert var.shape[0] == self.num_equations
        return torch.stack(
            [f(*var) for f in self.equations],
        )

    def __len__(self):
        return self.num_equations

    def __getitem__(self, i):
        return self.equations[i]

