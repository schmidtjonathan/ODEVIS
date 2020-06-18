import numpy as np
import tqdm


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
        super(Solver, self).__init__()
        self.ode_system = ode_system
        self.num_equations = len(ode_system)
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

    def solve(self):
        num_steps = int((self.t_max - self.t_min) / self.step_size)
        result = np.zeros(shape=(num_steps, self.num_equations))
        for t, step in enumerate(np.arange(self.t_min, self.t_max, self.step_size)):
            result[t, :] = self.state
            self.step()
        return result

    def __call__(self):
        """ For the given time domain, execute the simulation by successively calling the step function """

        print(f"Running simulation for t in [{self.t_min}, {self.t_max}] , dt = {self.step_size}")
        for step in tqdm.tqdm(np.arange(self.t_min, self.t_max, self.step_size)):
            yield step, self.state
            self.step()

        print("Done!")


class Euler(Solver):
    """ Euler method (corresponds to 1-st order Runge Kutta method) """

    def step(self):
        k1 = self.ode_system(self.state)
        self.state += self.step_size * k1

    def __str__(self):
        return "Euler method"


class Heun(Solver):
    """ Heun Method """

    def step(self):
        k1 = self.ode_system(self.state)
        proposed_state = self.state + self.step_size * k1
        k2 = k1 + self.ode_system(proposed_state)

        self.state += 0.5 * self.step_size * k2

    def __str__(self):
        return "Heun's method"


class RK4(Solver):
    """ 4-th order Runge Kutta Method """

    def step(self):
        k1 = self.ode_system(self.state)
        k2 = self.ode_system(self.state + 0.5 * self.step_size * k1)
        k3 = self.ode_system(self.state + 0.5 * self.step_size * k2)
        k4 = self.ode_system(self.state + self.step_size * k3)

        self.state += (self.step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def __str__(self):
        return "Runge Kutta (4th order)"