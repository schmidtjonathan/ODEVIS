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
    """ Abstract class that implements a numerical solver for ODE systems.

    For systems of ordinary differential equations (ODE), i.g., an analytic solution does not exist. Therefore
    one employs numerical solvers that evaluate the approximate solution on a discretized grid.
    Depending on the type of numerical ODE solver, the step() function must be overwritten.

    Attributes:
        ode_system: the system of ODEs represented as a list of callables.
        step_size: the step size of the numerical solver. Must be >0. The step size is chosen as a trade-off between
            speed (large stepsize) and accuracy (small stepsize)
        initial_value_condition: for each variable of the ODE system a set of initial conditions at t_min in order to
            achieve the particular solution to the system
        time_domain: tuple (t_min, t_max) on which the simulation shall be executed
    """
    def __init__(self, ode_system, step_size, initial_value_condition, time_domain):
        self.ode_system = ode_system
        self.step_size = step_size
        self.initial_value_condition = initial_value_condition
        self.t_min, self.t_max = time_domain
        self.state = initial_value_condition

        if self.t_min > 0.0:
            self.reset(T=self.t_min)

    def step(self):
        """ Depending on the numerical solver, takes on step on the discretized grid to approximate the solution """
        raise NotImplementedError

    def reset(self, T=None):
        """ Sets the state of the solver to a specific point in time

        Args:
            T: a time step T can be provided, then the solver is set to the state after T time steps.
                If T is None (default) the solver state is reset to the initial conditions.
        """
        self.state = self.initial_value_condition
        if T is not None:
            for step in np.arange(0.0, T, self.step_size):
                self.step()

    def __call__(self):
        """ For the given time domain, execute the simulation by successively calling the step function """

        print(f"Running simulation for t in [{self.t_min}, {self.t_max}] , dt = {self.step_size}")
        for step in tqdm.tqdm(np.arange(self.t_min, self.t_max, self.step_size)):
            yield self.state.x, self.state.y, step
            self.step()

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


class Heun(Solver):
    """ Heun Method """

    def step(self):
        x, y = self.state
        k1 = Evaluation(
            x=self.ode_system[0](x, y),
            y=self.ode_system[1](x, y)
        )
        proposed_state = self.state + self.step_size * k1

        k2 = k1 + Evaluation(
            x=self.ode_system[0](proposed_state.x, proposed_state.y),
            y=self.ode_system[1](proposed_state.x, proposed_state.y)
        )

        self.state += 0.5 * self.step_size * k2

    def __str__(self):
        return "Heun's method"


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