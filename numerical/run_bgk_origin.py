import matplotlib.pyplot as plt
import numpy as np

from bgk.bgk_solver import BGKSolver
from bgk.utils import get_config, maxwellian


def main(config_path):

    # parameters
    kn = 1e-3
    rho_l, u_l, T_l = 1.5, 0.0, 1.5
    rho_r, u_r, T_r = 0.625, 0.0, 0.75

    def f0_l(v): return maxwellian(v, rho_l, u_l, T_l)
    def f0_r(v): return maxwellian(v, rho_r, u_r, T_r)

    config = get_config(config_path)

    solver = BGKSolver(config.domain_config)
    solver.set_time_stepper("IMEX")

    x, v = solver.x, solver.v
    f0 = f0_l(v) * (x < 0)[:, None] + f0_r(v) * (x >= 0)[:, None]
    solver.set_initial(kn, f0)

    # solve
    t_final, dt = 0.1, 0.001
    # soln = []
    for _ in range(int(t_final // dt)):
        solver.solve(dt, n_iter=1)
        # soln.append(np.sum(solver.density()))

    plt.plot(solver.x, solver.get_gas_macro()[0], "r")
    plt.plot(solver.x, solver.get_gas_macro()[1], "b")
    plt.plot(solver.x, solver.get_gas_macro()[2], "k")
    # plt.plot(soln)
    plt.savefig("test.png")

    # save data
    np.savez("./dirichlet_ref_kn1e-3.npz",
             density=solver.get_gas_macro()[0],
             momentum=solver.get_gas_macro()[1],
             energy=solver.get_gas_macro()[2])


if __name__ == "__main__":
    main("bgk/configs/config_physical_riemann.json")
