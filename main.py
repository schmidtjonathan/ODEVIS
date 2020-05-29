import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import direction_field
import solver

# colors:
# (Credits to Prof. Philipp Hennig, University of Tuebingen)
COLORS = dict(
    dark = np.array([51.0, 51.0, 51.0]) / 255.0,
    red = np.array([141.0, 45.0, 57.0]) / 255.0,
    gold = np.array([174.0, 159.0, 109.0]) / 255.0,
    gray = np.array([175.0, 179.0, 183.0]) / 255.0,
    lred = np.array([1, 1, 1]) - 0.5 * (np.array([1, 1, 1]) - np.array([141.0, 45.0, 57.0]) / 255.0),
    lgold = np.array([1, 1, 1]) - 0.5 * (np.array([1, 1, 1]) - np.array([174.0, 159.0, 109.0]) / 255.0),
)

# matplotlib settings
plt.style.use("seaborn-whitegrid")
plt.rcParams['axes.axisbelow'] = True
plt.rcParams["figure.figsize"] = (14, 7)


QUIVER_OPTS = dict(
    cmap=matplotlib.cm.jet,
    pivot="middle",
    units="xy",
    alpha=0.6
)


MODES = ["simple", "lotka_volterra", "pendulum", "sir"]
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

    if arg_mode == "simple":
        def x_dot(x, y):
            return 1

        def y_dot(x, y):
            return 2*x
            # return (1. - y**2) / (1. + x**2)

        x_extent = (-2.0, 2.0)
        y_extent = (-1.0, 5.0)
        initial_value_condition = solver.Evaluation(x=-2.0, y=4.0)
        time_domain = (0.0, 4.0)
        axis_labels = {
            "x": "x",
            "y": "y"
        }

    elif arg_mode == "lotka_volterra":
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
        initial_value_condition = solver.Evaluation(x=3.0, y=1.0)
        time_domain = (0.0, 20.0)
        axis_labels = {
            "x": "prey",
            "y": "hunter"
        }

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
        initial_value_condition = solver.Evaluation(x=0.0, y=5.0)
        time_domain = (0.0, 20.0)
        axis_labels = {
            "x": r"angle $\theta$",
            "y": r"angular velocity $\dot \theta$"
        }

    elif arg_mode == "sir":
        N = 10**6
        alpha = 0.4
        k = 3.0

        def x_dot(x, y):
            """ S """
            return -k * y * x / N

        def y_dot(x, y):
            """ I """
            return k * y * x / N - alpha * y

        x_extent = (-50000, N * 1.3)
        y_extent = (-50000, N * 1.3)
        initial_value_condition = solver.Evaluation(x=N-1000.0, y=1000.0)
        time_domain = (0.0, 20.0)
        axis_labels = {
            "x": "Susceptible",
            "y": "Infectious"
        }

    else:
        print(f"Unknown mode {arg_mode}. (<mode> is one of {MODES})")
        exit(0)

    # Start simulation

    print("=" * (len(arg_mode) + 12))
    print(f" .... {arg_mode} .... ")
    print("=" * (len(arg_mode) + 12))

    field = direction_field.DirectionField2D(
        ode_system=[x_dot, y_dot],
        x_extent=x_extent,
        y_extent=y_extent,
        quiver_density=30,
        axis_labels=axis_labels,
    )


    euler_solver = solver.Euler(
        [x_dot, y_dot],
        step_size=arg_stepsize,
        initial_value_condition=initial_value_condition,
        time_domain=time_domain,
    )

    heun_solver = solver.Heun(
        [x_dot, y_dot],
        step_size=arg_stepsize,
        initial_value_condition=initial_value_condition,
        time_domain=time_domain,
    )

    rk4_solver = solver.RK4(
        [x_dot, y_dot],
        step_size=arg_stepsize,
        initial_value_condition=initial_value_condition,
        time_domain=time_domain,
    )

    possible_solvers = dict(
        euler=euler_solver,
        heun=heun_solver,
        rk4=rk4_solver
    )

    field.simulate(solver=possible_solvers[arg_solver])
