import argparse

import numpy as np

import odevis


MODES = ["lotka_volterra", "pendulum"]
SOLVERS = ["euler", "heun", "rk4"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        help=f"Which ode system to simulate. One of {MODES}",
        type=str,
        choices=MODES
    )

    parser.add_argument(
        "--solver",
        "-s",
        help=f"Which solver to use for simulation. One of {SOLVERS}",
        type=str,
        required=True,
        choices=SOLVERS
    )

    parser.add_argument(
        "--stepsize",
        "-d",
        help="Stepsize for the solver. Must be > 0.",
        type=float,
        default=2e-2
    )

    args = parser.parse_args()
    arg_mode = args.mode
    arg_solver = args.solver
    arg_stepsize = args.stepsize

    if arg_mode == "lotka_volterra":
        """
            POPULATION OF HUNTERS AND PREY
        """
        # Procreation rate of prey
        alpha = 2. / 3.
        # hunting efficiency of hunters
        beta = 4. / 3.
        # death rate of hunters
        gamma = 1.
        # proportionality factor of how hunters procreate given the amount of food (prey)
        delta = 1.
        def x_dot(x, y):
            ''' d/dt (f_prey(x, y)) -- Models the population of prey over time'''
            return alpha * x - beta * x * y

        def y_dot(x, y):
            ''' d/dt (f_hunters(x, y)) -- Models the population of hunters over time'''
            return delta * x * y - gamma * y

        x_extent = (0.0, 6.0)
        y_extent = (0.0, 6.0)
        time_domain = (0.0, 20.0)

        ode_system = odevis.ODE_System([x_dot, y_dot])
        initial_value_condition = np.array([3.0, 1.0])

    elif arg_mode == "pendulum":
        """
            PENDULUM (x = angle, y = angular velocity)
        """
        def x_dot(x, y):
            return y

        def y_dot(x, y):
            mu = 0.4    # Air resistance / friction / ... together in one resistance parameter
            g = 9.8     # gravity
            L = 3.0     # lenght of pendulum

            return -mu * y - (g / L) * np.sin(x)

        x_extent = (-2.0, 15)
        y_extent = (-6.0, 6.0)
        time_domain = (0.0, 20.0)

        ode_system = odevis.ODE_System([x_dot, y_dot])
        initial_value_condition = np.array([0.0, 5.0])

    else:
        print(f"Unknown mode {arg_mode}. (<mode> is one of {MODES})")
        exit(0)

    euler_solver = odevis.solver.Euler(
        ode_system=ode_system,
        step_size=arg_stepsize,
        initial_value_condition=initial_value_condition,
        time_domain=time_domain,
    )

    heun_solver = odevis.solver.Heun(
        ode_system=ode_system,
        step_size=arg_stepsize,
        initial_value_condition=initial_value_condition,
        time_domain=time_domain,
    )

    rk4_solver = odevis.solver.RK4(
        ode_system=ode_system,
        step_size=arg_stepsize,
        initial_value_condition=initial_value_condition,
        time_domain=time_domain,
    )

    possible_solvers = dict(
        euler=euler_solver,
        heun=heun_solver,
        rk4=rk4_solver
    )

    direction_field = odevis.DirectionField2D(
        ode_system=ode_system,
        x_extent=x_extent,
        y_extent=y_extent,
    )

    direction_field.animate_solution(possible_solvers[arg_solver])
