import pytest

import numpy as np
import matplotlib.pyplot as plt

import odevis


@pytest.fixture
def step_size():
    return 0.01


@pytest.fixture(params=[10.])
def domain(request, step_size):
    return (0., request.param)


@pytest.fixture
def f_x():
    return lambda x: np.sin(x)


@pytest.fixture
def x_dot():
    return lambda x, y: 1.0

@pytest.fixture
def y_dot():
    return lambda x, y: np.cos(x)


def test_euler(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(np.arange(*domain, step=step_size))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.Euler(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=np.array([0.0, true_f[0]]),
        time_domain=domain
    )

    assert np.allclose(true_f, numerical_f[:, 1], atol=1e-2)



def test_Heun(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(np.arange(*domain, step=step_size))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.Heun(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=np.array([0.0, true_f[0]]),
        time_domain=domain
    )

    assert np.allclose(true_f, numerical_f[:, 1])


def test_rk4(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(np.arange(*domain, step=step_size))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.RK4(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=np.array([0.0, true_f[0]]),
        time_domain=domain
    )

    assert np.allclose(true_f, numerical_f[:, 1])
