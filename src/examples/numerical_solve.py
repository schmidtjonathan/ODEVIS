import argparse

import torch
import matplotlib.pyplot as plt

import odevis


MODES = ["lotka_volterra", "pendulum", "sir"]
SOLVERS = ["euler", "heun", "rk4", "rk45"]


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

    parser.add_argument(
        "--animate",
        help="Whether or not to simulate the numerical solver by means of an animation",
        action="store_true",
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
        initial_value_condition = torch.tensor([3.0, 1.0])
        curve_labels = ["predator", "prey"]

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

            return -mu * y - (g / L) * torch.sin(x)

        x_extent = (-2.0, 15)
        y_extent = (-6.0, 6.0)
        time_domain = (0.0, 20.0)

        ode_system = odevis.ODE_System([x_dot, y_dot])
        initial_value_condition = torch.tensor([0.0, 5.0])
        curve_labels = [r"$\theta$", r"$\dot{\theta}$"]

    elif arg_mode == "sir":
        N = 10**6
        alpha = 1.0 / 14.0
        k = 0.53

        def x_dot(x, y, z):
            """ S """
            return -k * y * x / N

        def y_dot(x, y, z):
            """ I """
            return k * y * x / N - alpha * y

        def z_dot(x, y, z):
            return alpha * y

        x_extent = (-50000, N * 1.3)
        y_extent = (-50000, N * 1.3)
        time_domain = (0.0, 60.0)

        ode_system = odevis.ODE_System([x_dot, y_dot, z_dot])
        initial_value_condition = torch.tensor([N-1000.0, 1000.0, 0.0])
        curve_labels = ["S", "I", "R"]

    else:
        print(f"Unknown mode {arg_mode}. (<mode> is one of {MODES})")
        exit(0)

    euler_solver = odevis.solver.Euler(
        step_size=arg_stepsize,
    )

    heun_solver = odevis.solver.Heun(
        step_size=arg_stepsize,
    )

    rk4_solver = odevis.solver.RK4(
        step_size=arg_stepsize,
    )

    rk45_solver = odevis.solver.RK45(
        step_size=arg_stepsize
    )

    possible_solvers = dict(
        euler=euler_solver,
        heun=heun_solver,
        rk4=rk4_solver,
        rk45=rk45_solver
    )

    odevis.visualization.set_style()

    if args.animate:
        ani = odevis.animate_solution(
            possible_solvers[arg_solver],
            ode_system,
            initial_value_condition,
            time_domain,
            curve_labels=curve_labels,
            verbose=True
        )
        plt.show()
    else:
        odevis.plot_solution(
            possible_solvers[arg_solver],
            ode_system,
            initial_value_condition,
            time_domain,
            curve_labels=curve_labels
        )
