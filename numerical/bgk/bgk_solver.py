from math import pi

import matplotlib.pyplot as plt
import numpy as np

from bgk.utils import F_m_2, F_p_2


class Grid(object):
    """A grid class that stores the details and solution of the
    computational grid."""

    def __init__(self, domain_config):
        self.xmin = domain_config.xmin
        self.xmax = domain_config.xmax
        self.vmin = domain_config.vmin
        self.vmax = domain_config.vmax
        self.nx, self.nv = domain_config.nx, domain_config.nv

        self.dx = float(self.xmax - self.xmin) / self.nx
        self.x = np.arange(
            self.xmin + self.dx / 2, self.xmax + self.dx / 2, self.dx
        )
        v, w = np.polynomial.legendre.leggauss(self.nv)
        self.v = 0.5 * (v + 1.0) * (self.vmax - self.vmin) + self.vmin
        self.w = 0.5 * (self.vmax - self.vmin) * w
        self.C = np.vstack((self.w, self.v * self.w, self.v**2 * self.w))
        self.CCT = np.linalg.inv(self.C.dot(self.C.T)).dot(self.C)
        # Store the probability density function.
        self.f = np.zeros(self.x.shape + (self.nv,))
        self.U = np.zeros((3, self.x.shape[0]))

    def set_initial(self, f0):
        self.f = f0.copy()

    def set_bc_func(self, func_l, func_r):
        """Sets the BC given a function of two variables."""
        self.f[0, self.v > 0] = func_l(self.v)[self.v > 0]
        self.f[-1, self.v < 0] = func_r(self.v)[self.v < 0]

    def sum(self, f):
        # return f@self.w
        return np.matmul(f, self.w)

    def density(self):
        """Compute the macro quantity: density."""
        return self.sum(self.f)

    def velocity(self):
        return self.sum(self.f * self.v) / self.density()

    def temperature(self):
        return (
            self.sum(self.f * self.v**2) / self.density()
            - self.velocity() ** 2
        )

    def plot_macro(self, macro="rho", style="-"):
        """Plot the macroscopic quantities: density, velocity and temperature."""
        _, ax = plt.subplots()
        if macro == "rho":
            ax.plot(self.x, self.density(), style)
            ax.set_ylabel(r"$\rho(x)$", fontsize="large")
        elif macro == "T":
            ax.plot(self.x, self.temperature(), style)
            ax.set_ylabel(r"$T(x)$", fontsize="large")
        else:
            ax.plot(self.x, self.velocity(), style)
            ax.set_ylabel(r"$u(x)$", fontsize="large")

        ax.set_xlabel(r"$x$", fontsize="large")
        ax.grid(which="both", linestyle=":")
        plt.show()


# BGKSolver with MUSL flux
class BGKSolver(Grid):
    def __init__(self, domain_config):
        super().__init__(domain_config)
        self.kn = np.nan
        # step
        self.set_time_stepper(domain_config.stepper)
        # v+, v-.
        abs_v = np.abs(self.v)
        self._v_p = 0.5 * (self.v + abs_v)
        self._v_m = 0.5 * (self.v - abs_v)
        self.set_BC(domain_config.BC_type)

    def set_time_stepper(self, stepper="IMEX"):
        if stepper == "IMEX":
            self.time_step = self.step_imex
        if stepper == "explicit":
            self.time_step = self.explicit

    def set_initial(self, kn, f0):
        self.kn = kn
        self.f = f0.copy()
        self.fbc = np.zeros((self.nx + 4, self.nv))

    def set_BC(self, BC):
        if BC == "periodic":
            self.apply_BC = self._apply_periodic_BC
        elif BC == "cauchy":
            self.apply_BC = self._extend_boundary
        else:
            raise NotImplementedError

    def _extend_boundary(self):
        self._get_fbc_from_f()
        self.fbc[0] = self.fbc[2]
        self.fbc[1] = self.fbc[2]
        self.fbc[-1] = self.f[-3]
        self.fbc[-2] = self.f[-3]

    def _get_fbc_from_f(self):
        self.fbc[2:-2] = self.f.copy()

    def get_f_from_fbc(self):
        self.f = self.fbc[2:-2].copy()

    def _apply_periodic_BC(self):
        self._get_fbc_from_f()
        self.fbc[:2] = self.fbc[-4:-2]
        self.fbc[-2:] = self.fbc[2:4]

    def _grad_vf(self, f, dt):
        # v*grad_f
        return self._v_p * F_p_2(f, self._v_p, self.dx, dt) + self._v_m * F_m_2(
            f, self._v_m, self.dx, dt
        )

    def _m0(self, f):
        r"""0-th order momentum of f"""
        return self.sum(f)

    def _m1(self, f):
        r"""1st order momentum of f"""
        return self.sum(f * self.v)

    def _m2(self, f):
        r"""2nd order momentum of f"""
        return self.sum(f * self.v**2)

    def _correction(self):
        self.f = self.f + (self.U.T - self.f.dot(self.C.T)).dot(self.CCT)

    def _update_macro(self, fbc, dt):
        f = fbc[2:-2]
        grad_vf = self._grad_vf(fbc, dt)
        # update step
        rho = self._m0(f) - dt * self._m0(grad_vf)
        u_rho = self._m1(f) - dt * self._m1(grad_vf)
        T_rho = self._m2(f) - dt * self._m2(grad_vf)
        # update U = (m0, m1, m2)
        self.U = np.vstack((rho, u_rho, T_rho))
        # compute u and T
        u = u_rho / rho
        T = T_rho / rho - u**2
        return [rho, u, T]

    def maxwellian(self, arg):
        if isinstance(arg, list):
            rho, u, T = arg
        else:
            rho = self._m0(arg)
            u = self._m1(arg) / rho
            T = self._m2(arg) / rho - u**2
        return (rho / np.sqrt(2 * pi * T))[:, None] * np.exp(
            -((u[:, None] - self.v) ** 2) / (2 * T)[:, None]
        )

    def step_imex(self, dt):
        U = self._update_macro(self.fbc, dt)
        self.fbc[2:-2] = self.kn / (self.kn + dt) * (
            self.fbc[2:-2] - dt * self._grad_vf(self.fbc, dt)
        ) + dt / (self.kn + dt) * self.maxwellian(U)

    # under development
    def explicit(self, dt):
        self.fbc[2:-2] -= dt * self._grad_vf(self.fbc, dt) + dt / self.kn * (
            self.maxwellian(self.fbc[2:-2]) - self.fbc[2:-2]
        )

    def solve(self, dt, n_iter=1):
        for _ in range(n_iter):
            self.apply_BC()
            self.step_imex(dt)
            self.get_f_from_fbc()
            self._correction()

    def get_gas_macro(self):
        rho = self.density()
        u = self.velocity()
        T = self.temperature()
        m = rho * u
        E = 0.5 * (u**2 + T) * rho
        return np.vstack((rho, m, E))

    def get_f(self):
        return self.f.copy()

    def get_df(self, dt):
        self.apply_BC()
        return -self._grad_vf(self.fbc, dt) + 1.0 / self.kn * (
            self.maxwellian(self.f) - self.f
        )

    def get_entropy(self):
        # return the entropy of bgk solution in the form: sum(fln(f))
        return self.sum(self.f * np.log(self.f))
