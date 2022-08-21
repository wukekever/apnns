"""
Solve the one-dimensional Euler equations for inviscid, compressible flow:

.. math::
    \rho_t + (\rho u)_x & = 0 \\
    (\rho u)_t + (\rho u^2 + p)_x & = 0 \\
    E_t + (u (E + p) )_x & = 0.

The fluid is an ideal gas, with pressure given by :math:`p=\rho (\gamma-1)e` where
e is internal energy.
"""
# from clawpack.riemann.euler_with_efix_1D_constants import *
import logging

from clawpack import pyclaw, riemann

logger = logging.getLogger("pyclaw")
logger.setLevel(logging.CRITICAL)


class ClawSolver(object):
    def __init__(
        self, domain_config, solver_type="classic", kernel_language="Fortran"
    ):
        if kernel_language == "Python":
            rs = riemann.euler_1D_py.euler_hllc_1D
        elif kernel_language == "Fortran":
            rs = riemann.euler_with_efix_1D

        if solver_type == "sharpclaw":
            solver = pyclaw.SharpClawSolver1D(rs)
        elif solver_type == "classic":
            solver = pyclaw.ClawSolver1D(rs)
        solver.kernel_language = kernel_language

        if domain_config.BC_type == "periodic":
            solver.bc_lower[0] = getattr(pyclaw.BC, "periodic")
            solver.bc_upper[0] = solver.bc_lower[0]
        elif domain_config.BC_type == "cauchy":
            solver.bc_lower[0] = getattr(pyclaw.BC, "extrap")
            solver.bc_upper[0] = solver.bc_lower[0]
        else:
            raise NotImplementedError
        # solver.bc_lower[0] = pyclaw.BC.extrap
        # solver.bc_upper[0] = pyclaw.BC.extrap

        x = pyclaw.Dimension(
            domain_config.xmin, domain_config.xmax, domain_config.nx, name="x"
        )
        domain = pyclaw.Domain([x])
        self._domain = domain
        state = pyclaw.State(domain, num_eqn=3)
        state.problem_data["gamma"] = 3  # ideal gas in 1D
        state.problem_data["gamma1"] = 2  # gamma - 1
        self._state = state
        self._x = state.grid.x.centers

        claw = pyclaw.Controller()
        claw.tfinal = 0
        claw.solver = solver
        claw.outdir = None
        claw.keep_copy = True
        claw.output_format = None
        self._claw = claw

    def set_initial(self, rho, m, E):
        self._state.q[0, :] = rho
        self._state.q[1, :] = m
        self._state.q[2, :] = E
        self._claw.solution = pyclaw.Solution(self._state, self._domain)
        self._claw.solution.t = 0  # reset initial time

    def solve(self, nt, tmax):
        self._claw.num_output_times = nt
        self._claw.tfinal = tmax
        # remove old frames
        self._claw.frames = []
        self._claw.run()

    @property
    def claw(self):
        return self._claw

    @property
    def state(self):
        return self._state

    @property
    def x(self):
        return self._x


def pyclaw_setup(solver_type="classic", kernel_language="Fortran"):
    gamma = 3
    num_eqn = 3
    # Conserved quantities
    density = 0
    momentum = 1
    energy = 2

    if kernel_language == "Python":
        rs = riemann.euler_1D_py.euler_hllc_1D
    elif kernel_language == "Fortran":
        rs = riemann.euler_with_efix_1D

    if solver_type == "sharpclaw":
        solver = pyclaw.SharpClawSolver1D(rs)
    elif solver_type == "classic":
        solver = pyclaw.ClawSolver1D(rs)
    solver.kernel_language = kernel_language
    solver.bc_lower[0] = pyclaw.BC.extrap
    solver.bc_upper[0] = pyclaw.BC.extrap
    mx = 400
    x = pyclaw.Dimension(-0.5, 0.5, mx, name="x")
    domain = pyclaw.Domain([x])
    state = pyclaw.State(domain, num_eqn)
    state.problem_data["gamma"] = gamma
    state.problem_data["gamma1"] = gamma - 1.0
    x = state.grid.x.centers

    rho_l = 1.0
    rho_r = 1.0 / 8
    p_l = 1.0
    p_r = 1 / 32
    state.q[density, :] = (x < 0.0) * rho_l + (x >= 0.0) * rho_r
    state.q[momentum, :] = 0.0
    velocity = state.q[momentum, :] / state.q[density, :]
    pressure = (x < 0.0) * p_l + (x >= 0.0) * p_r
    state.q[energy, :] = (
        pressure / (gamma - 1.0) + 0.5 * state.q[density, :] * velocity**2
    )

    claw = pyclaw.Controller()
    claw.tfinal = 0.2
    claw.solution = pyclaw.Solution(state, domain)
    claw.solver = solver
    claw.num_output_times = 10
    claw.outdir = None
    claw.keep_copy = True
    claw.output_format = None
    return claw


if __name__ == "__main__":
    claw = pyclaw_setup()
    claw.run()
    print("")

    claw1 = ClawSolver(nx=400)
    x = claw1._x
    rho_l = 1.0
    rho_r = 1.0 / 8
    p_l = 1.0
    p_r = 1 / 32
    rho = (x < 0.0) * rho_l + (x >= 0.0) * rho_r
    m = 0.0
    velocity = m / rho
    pressure = (x < 0.0) * p_l + (x >= 0.0) * p_r
    E = pressure / 2 + 0.5 * rho * velocity**2
    claw1.set_initial(rho, m, E)
    claw1.solve(nt=10, tmax=0.2)
    # x = claw1.frames[-1].states[0].grid.x.centers
    # print(claw1.frames[-1].states[0].q[0, :])
