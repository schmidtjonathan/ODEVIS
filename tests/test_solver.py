import pytest

import numpy as np
import torch
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
    return lambda x: torch.sin(x).to(torch.float64)


@pytest.fixture
def x_dot():
    return lambda x, y: torch.tensor(1.0).to(torch.float64)

@pytest.fixture
def y_dot():
    return lambda x, y: torch.cos(x).to(torch.float64)


def test_euler(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(torch.arange(*domain, step=step_size, dtype=torch.float64))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.Euler(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=torch.tensor([0.0, true_f[0]]),
        time_domain=domain
    )

    assert torch.allclose(true_f, numerical_f[:, 1], atol=1e-2)



def test_Heun(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(torch.arange(*domain, step=step_size, dtype=torch.float64))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.Heun(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=torch.tensor([0.0, true_f[0]]),
        time_domain=domain
    )

    assert torch.allclose(true_f, numerical_f[:, 1])


def test_rk4(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(torch.arange(*domain, step=step_size, dtype=torch.float64))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.RK4(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=torch.tensor([0.0, true_f[0]]),
        time_domain=domain
    )

    assert torch.allclose(true_f, numerical_f[:, 1], atol=1e-2)


def test_rk45(f_x, x_dot, y_dot, domain, step_size):
    true_f = f_x(torch.arange(*domain, step=step_size, dtype=torch.float64))
    ode_system = odevis.ODE_System([x_dot, y_dot])
    solver = odevis.solver.RK45(step_size)

    numerical_f = solver.solve(
        ode_system=ode_system,
        initial_value_condition=torch.tensor([0.0, true_f[0]]),
        time_domain=domain
    )

    assert torch.allclose(true_f, numerical_f[:, 1])
