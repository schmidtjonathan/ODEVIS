import collections
import numbers
import numpy as np
import tqdm


Coords2D = collections.namedtuple("Coords2D", ["x", "y"])


class Evaluation(Coords2D):
    """ 2-component vector implementing scalar- and element-wise multiplication and addition """

    def __mul__(self, other):
        if isinstance(other, Evaluation):
            return Evaluation(self.x * other.x, self.y * other.y)
        elif isinstance(other, numbers.Number):
            return Evaluation(self.x * other, self.y * other)
        else:
            raise TypeError(f"Multiplication is only defined for scalars and Evaluation objects. Got {type(other)}")

    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, Evaluation):
            return Evaluation(self.x + other.x, self.y + other.y)
        elif isinstance(other, numbers.Number):
            return Evaluation(self.x + other, self.y + other)
        else:
            raise TypeError(f"Addition is only defined for scalars and Evaluation objects. Got {type(other)}")

    def __eq__(self, other):
        return isinstance(other, Evaluation) and self.x == other.x and self.y == other.y


class Solver(object):
    def __init__(self, ode_system, step_size, x_0, y_0, t_min, t_max):
        self.ode_system = ode_system
        self.step_size = step_size
        self.x_0 = x_0
        self.y_0 = y_0
        self.t_min = t_min
        self.t_max = t_max
        self.state = Evaluation(x_0, y_0)

        if self.t_min >= 0.0:
            self.reset(T=self.t_min)

    def step(self):
        raise NotImplementedError

    def reset(self, T=None):
        self.state = Evaluation(self.x_0, self.y_0)
        if T is not None:
            for step in np.arange(0.0, T, self.step_size):
                self.step()

    def __call__(self):
        print(f"Running simulation for t in [{self.t_min}, {self.t_max}] , dt = {self.step_size}")
        for step in tqdm.tqdm(np.arange(self.t_min, self.t_max, self.step_size)):
            self.step()
            yield self.state.x, self.state.y, step

        print("Done!")

class Euler(Solver):
    """ Euler method (corresponds to 1-st order Runge Kutta method) """

    def step(self):
        x, y = self.state
        k1 = Evaluation(
            x=self.ode_system[0](x, y),
            y=self.ode_system[1](x, y)
        )
        self.state += self.step_size * k1

    def __str__(self):
        return "Euler method"


class RK4(Solver):
    """ 4-th order Runge Kutta Method """

    def step(self):
        x, y = self.state
        k1 = Evaluation(
            x=self.ode_system[0](x, y),
            y=self.ode_system[1](x, y)
        )
        k2 = Evaluation(
            x=self.ode_system[0](x + 0.5 * self.step_size * k1.x, y + 0.5 * self.step_size * k1.y),
            y=self.ode_system[1](x + 0.5 * self.step_size * k1.x, y + 0.5 * self.step_size * k1.y)
        )
        k3 = Evaluation(
            x=self.ode_system[0](x + 0.5 * self.step_size * k2.x, y + 0.5 * self.step_size * k2.y),
            y=self.ode_system[1](x + 0.5 * self.step_size * k2.x, y + 0.5 * self.step_size * k2.y)
        )
        k4 = Evaluation(
            x=self.ode_system[0](x + self.step_size * k3.x, y + self.step_size * k3.y),
            y=self.ode_system[1](x + self.step_size * k3.x, y + self.step_size * k3.y)
        )

        self.state += (self.step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def __str__(self):
        return "Runge Kutta (4th order)"