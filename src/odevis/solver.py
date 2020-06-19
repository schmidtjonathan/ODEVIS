import numpy as np
import tqdm

import odevis


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
    def __init__(self, step_size):
        super(Solver, self).__init__()
        self.step_size = step_size
        self.ode_system = None

    def _step(self, state):
        """ Depending on the numerical solver, takes on step on the discretized grid to approximate the solution """
        raise NotImplementedError

    def solve(self, ode_system, initial_value_condition, time_domain):
        num_equations = len(ode_system)
        t_min, t_max = time_domain
        num_steps = int((t_max - t_min) / self.step_size)
        result = np.zeros(shape=(num_steps, num_equations))
        for step, (t, state) in enumerate(self(ode_system, initial_value_condition, time_domain, verbose=False)):
            result[step, :] = state
        return result

    def __call__(self, ode_system, initial_value_condition, time_domain, verbose=True):
        """ For the given time domain, execute the simulation by successively calling the step function """

        self.ode_system = odevis.ODE_System([eq for eq in ode_system])
        t_min, t_max = time_domain
        print(f"Running simulation for t in [{t_min}, {t_max}] , dt = {self.step_size}")
        state = np.array(initial_value_condition, copy=True)
        for step in tqdm.tqdm(np.arange(t_min, t_max, self.step_size), disable=not verbose):
            yield step, state
            state = self._step(state)

        print("Done!")


class Euler(Solver):
    """ Euler method (corresponds to 1-st order Runge Kutta method) """

    def _step(self, state):
        if self.ode_system is None:
            raise RuntimeError("Seems like you tried to invoke step outside of solve() or __call__().")

        k1 = self.ode_system(state)
        return state + self.step_size * k1

    def __str__(self):
        return "Euler method"


class Heun(Solver):
    """ Heun Method """

    def _step(self, state):
        if self.ode_system is None:
            raise RuntimeError("Seems like you tried to invoke step outside of solve() or __call__().")

        k1 = self.ode_system(state)
        proposed_state = state + self.step_size * k1
        k2 = k1 + self.ode_system(proposed_state)

        return state + 0.5 * self.step_size * k2

    def __str__(self):
        return "Heun's method"


class RK4(Solver):
    """ 4-th order Runge Kutta Method """

    def _step(self, state):
        if self.ode_system is None:
            raise RuntimeError("Seems like you tried to invoke step outside of solve() or __call__().")

        k1 = self.ode_system(state)
        k2 = self.ode_system(state + 0.5 * self.step_size * k1)
        k3 = self.ode_system(state + 0.5 * self.step_size * k2)
        k4 = self.ode_system(state + self.step_size * k3)

        return state + (self.step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def __str__(self):
        return "Runge Kutta (4th order)"