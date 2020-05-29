# ODEVIS - Numerical Solvers for ODEs
 A simple playground to visualize the numerical solution to a two-dimensional system of ordinary differential equations (ODE).

 There are currently three solvers implemented:
 1. Euler-method ( `--solver euler` )
 2. Heun's method ( `--solver heun` )
 3. 4th order Runge-Kutta method ( `--solver rk4` )

 The solvers can be simulated on different problems:

 1. A simple one-dimensional function as for example a standard parabola ( `simple` )
 2. The Lotka-Volterra equations to model the intertwined dynamics of two populations of hunter and prey ( `lotka_volterra` )
 3. A pendulum, represented in state-space by angle (x-axis) and angular velocity (y-axis) ( `pendulum` )
 4. The SIR model. In order to keep the visualization in two dimensions, only the `S` and `I` part of the ODE system are visualized, the remaining `R` part follows from `N - S - I` where `N` is the total population.

 As an example, execute
 ```
python main.py --solver rk4 --stepsize 0.1 lotka_volterra
 ```
 to run a simulation of the Lotka-Volterra equations using a 4th-order Runge-Kutta solver with step size 0.1
