import matplotlib.pyplot as plt
import numpy as np

from bgk.bgk_solver import BGKSolver
from bgk.utils import get_config, maxwellian, Maxwellian

# ex 1
# def func0(x, func_l, func_r):

#     return func_l + (func_r - func_l) * ((1.0 + np.sin(np.pi * x)) / 2.0)


# ex 2
# def func0(x, func_l, func_r):
#     def fun(xx): return np.power(np.sin(np.pi*xx), 3 /
#                                  7) if xx > 0 else - np.power(-np.sin(np.pi*xx), 3/7)
#     fun_x = np.array(list(map(fun, x)))

#     return func_l + (func_r - func_l) * ((1.0 + fun_x) / 2.0)


# ex 3 and 4
# def func0(x, func_l, func_r):
#     return func_l + (func_r - func_l) * 0.5 * (np.tanh(10.0*x) + 1.0)

# ex 5
def func0(x, func_l, func_r):
    return func_l + (func_r - func_l) * 0.5 * (np.tanh(20.0 * x) + 1.0)


def main(config_path):

    # parameters
    # kn = 1e-3  # ex 1 - 3
    kn = 1e0  # ex 4

    rho_l, u_l, T_l = 1.5, 0.0, 1.5
    rho_r, u_r, T_r = 0.625, 0.0, 0.75

    config = get_config(config_path)

    solver = BGKSolver(config.domain_config)
    solver.set_time_stepper("IMEX")

    x, v = solver.x, solver.v
    rho0 = func0(x, rho_l, rho_r)
    u0 = func0(x, u_l, u_r)
    T0 = func0(x, T_l, T_r)
    f0 = Maxwellian(v, rho0, u0, T0)
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
    plt.plot(solver.x, rho0, marker="x", markevery=5)
    plt.plot(solver.x, rho0 * u0, marker="+", markevery=5)
    plt.plot(solver.x, 0.5 * rho0 * (u0**2 + T0), marker="*", markevery=5)
    plt.grid()
    # plt.plot(soln)
    plt.savefig("test.png")

    # save data
    # ex 1 - 3
    # np.savez(
    #     "./dirichlet_ref_kn1e-3.npz",
    #     density=solver.get_gas_macro()[0],
    #     momentum=solver.get_gas_macro()[1],
    #     energy=solver.get_gas_macro()[2],
    # )

    # ex 4
    np.savez(
        "./dirichlet_ref_kn1e0.npz",
        density=solver.get_gas_macro()[0],
        momentum=solver.get_gas_macro()[1],
        energy=solver.get_gas_macro()[2],
    )


if __name__ == "__main__":
    main("bgk/configs/config_physical_riemann.json")
