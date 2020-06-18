# ODEVIS - Numerical Solvers for ODEs
A simple playground to visualize the numerical solution to a two-dimensional system of ordinary differential equations (ODE).

## Install

1. clone this repository
2. in your terminal, navigate into the repository folder
3. install by executing the following line in the terminal:

```
pip install -e .
```

---

There are currently three solvers implemented:
1. Euler-method ( `--solver euler` )
2. Heun's method ( `--solver heun` )
3. 4th order Runge-Kutta method ( `--solver rk4` )

The solvers can be simulated on different problems:

1. The Lotka-Volterra equations to model the intertwined dynamics of two populations of hunter and prey ( `lotka_volterra` )
2. A pendulum, represented in state-space by angle (x-axis) and angular velocity (y-axis) ( `pendulum` )
3. The SIR model (cannot be visualized in `examples.direction_field`, since this example only provides 2D direction fields)

As an example, execute

```
python -m examples.direction_field --solver rk4 --stepsize 0.1 lotka_volterra
```

to run a simulation of the Lotka-Volterra equations (phase-space and time/value space) using a 4th-order Runge-Kutta solver with step size 0.1

or

```
python -m examples.numerical_solve --solver euler --stepsize 0.1 --animate sir
```

to plot or animate (`--animate`) the numerical solution of SIR model equations over some time period.
