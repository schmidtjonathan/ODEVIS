import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from main import QUIVER_OPTS, COLORS


class DirectionField2D(object):

    def __init__(self, ode_system, x_extent, y_extent, quiver_density=25, axis_labels=None):

        min_x, max_x = x_extent
        min_y, max_y = y_extent
        dx = max((max_x - min_x, max_y - min_y)) / quiver_density

        if axis_labels is None:
            self.axis_labels = {
                "x": "x",
                "y": "y"
            }
        else:
            self.axis_labels = axis_labels

        self.ode_system = ode_system
        self.x_extent, self.y_extent = x_extent, y_extent
        self.X, self.Y = np.meshgrid(
            np.arange(min_x, max_x, dx),
            np.arange(min_y, max_y, dx)
        )

    def visualize_direction_field(self, axis):

        U, V = self.ode_system[0](self.X, self.Y), self.ode_system[1](self.X, self.Y)
        norm = np.sqrt(U**2 + V**2)
        mask_norm_nonzero = norm > 0.0
        U = np.divide(U, norm, where=mask_norm_nonzero)
        V = np.divide(V, norm, where=mask_norm_nonzero)
        axis.quiver(self.X, self.Y, U, V, norm, **QUIVER_OPTS)

        # plt.scatter(self.X, self.Y, s=5, edgecolors="red", facecolors="white")

        axis.set_xlim([self.x_extent[0] - 0.1, self.x_extent[1] + 0.1])
        axis.set_ylim([self.y_extent[0] - 0.1, self.y_extent[1] + 0.1])

        axis.set_xlabel(self.axis_labels["x"])
        axis.set_ylabel(self.axis_labels["y"])

        # Coordinate system
        axis.axhline(y=0, color='k', lw=.8)
        axis.axvline(x=0, color='k', lw=.8)

        return axis

    def simulate(self, solver):

        """ <Animation stuff> """

        def init():
            """ Initialize the animation
            Plot the vector field as background and initialize the line of the solution function
            """
            del xdata[:]
            del ydata[:]
            del t_range[:]

            ax[0].set_title("Phase space")
            ax[1].set_title("Time/Value space")

            ax[1].set_xlim(
                (0, (solver.t_max - solver.t_min))
            )
            ax[1].set_ylim(
                (min(self.x_extent[0], self.y_extent[0]), max(self.x_extent[1], self.y_extent[1]))
            )
            ax[1].set_xticks(np.arange(solver.t_min, solver.t_max))
            ax[1].legend()
            ax[1].set_xlabel("t")
            ax[1].axhline(y=0, color='k', lw=.8)
            ax[1].axvline(x=0, color='k', lw=.8)

            self.visualize_direction_field(axis=ax[0])
            numerical_solution.set_data(xdata, ydata)
            x_t_line.set_data(t_range, xdata)
            y_t_line.set_data(t_range, ydata)
            return numerical_solution

        def run(data):
            """ Update the data of the animation """
            x, y, t = data
            xdata.append(x)
            ydata.append(y)
            t_range.append(t)

            numerical_solution.set_data(xdata, ydata)
            x_t_line.set_data(t_range, xdata)
            y_t_line.set_data(t_range, ydata)

            if round(t, 6).is_integer():
                ax[0].scatter(x, y, s=15, edgecolors=COLORS["dark"], facecolors=COLORS["gold"], zorder=10)
                ax[1].scatter(t, x, s=15, edgecolors=COLORS["gold"], facecolors=COLORS["red"], zorder=10)
                ax[1].scatter(t, y, s=15, edgecolors=COLORS["gold"], facecolors=COLORS["dark"], zorder=10)
                fig.suptitle(f"t = {int(t)}")

            return numerical_solution

        """ </Animation stuff> """

        fig, ax = plt.subplots(1, 2, num=str(solver))
        numerical_solution, = ax[0].plot([], [], lw=2, color=COLORS["gold"])
        x_t_line, = ax[1].plot([], [], lw=2, color=COLORS["red"], label=self.axis_labels["x"])
        y_t_line, = ax[1].plot([], [], lw=2, color=COLORS["dark"], label=self.axis_labels["y"])
        t_range = []
        xdata, ydata = [], []

        ani = animation.FuncAnimation(fig, run, solver, blit=False, interval=10, repeat=False, init_func=init)
        plt.show()
