import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import functools


# colors:
# (Credits to Prof. Philipp Hennig, University of Tuebingen)
COLORS = dict(
    dark = np.array([51.0, 51.0, 51.0]) / 255.0,
    red = np.array([141.0, 45.0, 57.0]) / 255.0,
    gold = np.array([174.0, 159.0, 109.0]) / 255.0,
    gray = np.array([175.0, 179.0, 183.0]) / 255.0,
    lred = np.array([1., 1., 1.]) - 0.5 * (np.array([1., 1., 1.]) - np.array([141.0, 45.0, 57.0]) / 255.0),
    lgold = np.array([1., 1., 1.]) - 0.5 * (np.array([1., 1., 1.]) - np.array([174.0, 159.0, 109.0]) / 255.0),
    b1 = np.array([65.0, 90.0, 140.0]) / 255.0,
    g1 = np.array([130.0, 185.0, 160.0]) / 255.0,
    r1 = np.array([200.0, 80.0, 60.0]) / 255.0,
    y1 = np.array([215.0, 180.0, 105.0]) / 255.0,
    b2 = np.array([0.0, 105.0, 170.0]) / 255.0,
    g2 = np.array([125.0, 165.0, 75.0]) / 255.0,
    r2 = np.array([175.0, 110.0, 150.0]) / 255.0,
    y2 = np.array([210.0, 150.0, 0.0]) / 255.0,
    b3 = np.array([80.0, 170.0, 200.0]) / 255.0,
    g3 = np.array([50.0, 110.0, 30.0]) / 255.0,
    r3 = np.array([180.0, 160.0, 150.0]) / 255.0,
    y3 = np.array([145.0, 150.0, 70.0]) / 255.0,
)


def set_style(rcParams=None):
    # matplotlib settings
    plt.style.use("seaborn-whitegrid")
    plt.rcParams["axes.axisbelow"] = True
    plt.rcParams["axes.prop_cycle"] = matplotlib.cycler(color=list(COLORS.values()))

    if rcParams is not None:
        for k, v in rcParams.items():
            plt.rcParams[k] = v


def plot_solution(
    solver,
    ode_system,
    initial_value_condition,
    time_domain,
    curve_labels=None,
    num_xticks=10
):
    fig, ax = plt.subplots()
    fig.suptitle(str(solver))
    t_min, t_max = time_domain
    if num_xticks > (t_max - t_min):
        raise ValueError("You have to choose the number of x ticks that is less than the total time steps")
    dt = int((t_max - t_min) / num_xticks)
    ax.set_xticks(np.arange(t_min, t_max, dt) / solver.step_size)
    ax.set_xticklabels(np.arange(t_min, t_max, dt).astype(int))
    ax.set_xlabel("t")

    evaluations = solver.solve(
        ode_system, initial_value_condition, time_domain
    ).numpy()
    for eq in range(len(ode_system)):
        ax.plot(
            evaluations[:, eq],
            color=f"C{eq}",
            label=None if curve_labels is None else curve_labels[eq],
        )
    if curve_labels is not None:
        ax.legend()
    plt.show()


def animate_solution(
    solver,
    ode_system,
    initial_value_condition,
    time_domain,
    curve_labels=None,
    num_xticks=10,
    verbose=True,
):

    def init():
        """ Initialize the animation
        Plot the vector field as background and initialize the line of the solution function
        """

        t_min, t_max = time_domain
        dt = int((t_max - t_min) / num_xticks)
        ax.set_xticks(np.arange(t_min, t_max + dt, dt) / solver.step_size)
        ax.set_xticklabels(np.arange(t_min, t_max, dt).astype(int))
        ax.set_xlabel("t")

        for i, line in enumerate(over_time_lines):
            line.set_data(t_range, variable_data[i])

    def run(data):
        """ Update the data of the animation """
        t, evaluation = data
        evaluation = evaluation.numpy()
        for i, v in enumerate(evaluation):
            variable_data[i].append(v)
        t_range.append(t / solver.step_size)

        # Adjust y axis of the plot
        ax_y_min, ax_y_max = ax.get_ylim()
        eval_min, eval_max = evaluation.min(), evaluation.max()
        adjust_ylim = False
        if eval_min < ax_y_min:
            ax_y_min = eval_min - 0.2 * np.abs(eval_min)
            adjust_ylim = True
        if eval_max > ax_y_max:
            ax_y_max = eval_max + 0.2 * np.abs(eval_max)
            adjust_ylim = True

        if adjust_ylim:
            ax.set_ylim([ax_y_min, ax_y_max])

        for i, line in enumerate(over_time_lines):
            line.set_data(t_range, variable_data[i])

    fig, ax = plt.subplots()
    fig.suptitle(str(solver))

    over_time_lines = [
        line for line, in [ax.plot([], [], color=f"C{i}") for i in range(len(ode_system))]
    ]

    t_range = []
    variable_data = [[] for _ in range(len(ode_system))]

    gen = functools.partial(solver, ode_system, initial_value_condition, time_domain, verbose)

    ani = animation.FuncAnimation(fig, run, gen, blit=False, interval=10, repeat=False, init_func=init)
    return ani


class DirectionField2D(object):
    """ Representation of a 2 dimensional direction- (or slope-) field.

    This class is mainly designed to carry out the visualization of the simulation.

    Attributes:
        ode_system: the system of ODEs represented as a list of callables
        x_extent: the boundaries of the x-axis for the visulaization as tuple (x_min, x_max)
        y_extent: the boundaries of the y-axis for the visulaization as tuple (y_min, y_max)
        quiver_density: integer, defining how dense the slope field is to be visualized. The number
            determines how many arrows representing the gradient shall be displayed per dimension
    """

    def __init__(self, ode_system, x_extent, y_extent, quiver_density=25):

        min_x, max_x = x_extent
        min_y, max_y = y_extent
        dx = max((max_x - min_x, max_y - min_y)) / quiver_density

        self.ode_system = ode_system
        if len(self.ode_system) != 2:
            raise ValueError(f"You must provide exactly two ODEs, received {len(self.ode_system)}")
        self.x_extent, self.y_extent = x_extent, y_extent
        self.X, self.Y = np.meshgrid(
            np.arange(min_x, max_x, dx),
            np.arange(min_y, max_y, dx)
        )

    def visualize_direction_field(self, axis, quiver_opts=None):
        """ Plots the slope field on the provided axis object """

        if quiver_opts is None:
            quiver_opts = dict(
                cmap=matplotlib.cm.jet,
                pivot="middle",
                units="xy",
                alpha=0.6
            )

        U, V = self.ode_system[0](self.X, self.Y), self.ode_system[1](self.X, self.Y)
        norm = np.sqrt(U**2 + V**2)
        mask_norm_nonzero = norm > 0.0
        U = np.divide(U, norm, where=mask_norm_nonzero)
        V = np.divide(V, norm, where=mask_norm_nonzero)
        axis.quiver(self.X, self.Y, U, V, norm, **quiver_opts)

        axis.set_xlim([self.x_extent[0] - 0.1, self.x_extent[1] + 0.1])
        axis.set_ylim([self.y_extent[0] - 0.1, self.y_extent[1] + 0.1])

        # Coordinate system
        axis.axhline(y=0, color='k', lw=.8)
        axis.axvline(x=0, color='k', lw=.8)

        return axis

    def animate_solution(self, solver, ode_system, initial_value_condition, time_domain, curve_labels=None, verbose=True):
        """ Method that carries out the visualization (animation) of the numerical ODE solver """

        def init():
            """ Initialize the animation
            Plot the vector field as background and initialize the line of the solution function
            """

            ax[0].set_title("Phase space")
            ax[0].set_xlabel(curve_labels[0])
            ax[0].set_ylabel(curve_labels[1])
            ax[1].set_title("Time/Value space")

            t_min, t_max = time_domain

            ax[1].set_xlim(
                (t_min, t_max)
            )
            ax[1].set_ylim(
                (min(self.x_extent[0], self.y_extent[0]), max(self.x_extent[1], self.y_extent[1]))
            )
            ax[1].set_xticks(np.arange(t_min, t_max))
            ax[1].set_xlabel("t")
            ax[1].axhline(y=0, color='k', lw=.8)
            ax[1].axvline(x=0, color='k', lw=.8)
            ax[1].legend()

            self.visualize_direction_field(axis=ax[0])
            numerical_solution.set_data(xdata, ydata)
            x_t_line.set_data(t_range, xdata)
            y_t_line.set_data(t_range, ydata)
            return numerical_solution

        def run(data):
            """ Update the data of the animation """
            t, (x, y) = data
            xdata.append(x)
            ydata.append(y)
            t_range.append(t)

            numerical_solution.set_data(xdata, ydata)
            x_t_line.set_data(t_range, xdata)
            y_t_line.set_data(t_range, ydata)

            return numerical_solution

        fig, ax = plt.subplots(1, 2)
        fig.suptitle(str(solver))

        numerical_solution, = ax[0].plot([], [], lw=2, color=COLORS["gold"])
        x_t_line, = ax[1].plot([], [], lw=2, color=COLORS["red"], label=curve_labels[0])
        y_t_line, = ax[1].plot([], [], lw=2, color=COLORS["dark"], label=curve_labels[1])
        t_range = []
        xdata, ydata = [], []

        gen = functools.partial(solver, ode_system, initial_value_condition, time_domain, verbose)

        ani = animation.FuncAnimation(fig, run, gen, blit=False, interval=10, repeat=False, init_func=init)
        return ani
