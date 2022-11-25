import numpy as np
import tensorflow as tf
from math import pi


class BGK(object):
    def __init__(self, Config, sol, name="BGK_Eqn", **kwargs):

        self.vmin = Config["dataset_config"]["v_range"][0]
        self.vmax = Config["dataset_config"]["v_range"][1]
        self.xmin = Config["dataset_config"]["x_range"][0]
        self.xmax = Config["dataset_config"]["x_range"][1]
        self.tmax = Config["dataset_config"]["t_range"][1]
        self.nvquad = Config["model_config"]["nvquad"]
        v, w = np.polynomial.legendre.leggauss(self.nvquad)
        v = 0.5 * (v + 1.0) * (self.vmax - self.vmin) + self.vmin
        w = 0.5 * (self.vmax - self.vmin) * w
        # vquads and wquads have shape [nvquad, 1]
        self.vquads = tf.convert_to_tensor(v[:, None], dtype=tf.float32)
        self.wquads = tf.convert_to_tensor(w[:, None], dtype=tf.float32)
        self.kn = Config["model_config"]["kn"]
        self.rho_l, self.u_l, self.T_l = 1.5, 0.0, 1.5
        self.rho_r, self.u_r, self.T_r = 0.625, 0.0, 0.75
        self.nx = 100
        self.dx = float(self.xmax - self.xmin) / self.nx
        self.ref_x = np.arange(
            self.xmin + self.dx / 2, self.xmax + self.dx / 2, self.dx
        ).reshape((-1, 1))
        self.ref_t = self.tmax * np.ones((self.nx, 1))

    # inputs: (t, x, v)
    def residual(self, sol, inputs):
        t, x, v = inputs
        values, derivatives = self.value_and_grad(sol, [t, x, v])
        f = values["f"]
        Maxwellian = values["maxwellian"]
        density, momentum, energy = values["flow"]
        avg_fv0, avg_fv1, avg_fv2 = values["average"]

        df_dt, df_dx = derivatives["f"]
        drho_dt, dm_dt, dE_dt = derivatives["flow"]
        df1_dx, df2_dx, df3_dx = derivatives["moment"]

        res = {}

        # residual for BGK equation
        res_bgk = self.kn * (df_dt + v * df_dx) - (Maxwellian - f)
        res.update({"equation": res_bgk})

        # residual for conservation laws
        res_conservation_1 = drho_dt + df1_dx
        res_conservation_2 = dm_dt + df2_dx
        res_conservation_3 = dE_dt + 0.5 * df3_dx
        res.update(
            {
                "conservation": (
                    res_conservation_1,
                    res_conservation_2,
                    res_conservation_3,
                )
            }
        )

        # residual for relaxation
        res_relax_1 = density - avg_fv0
        res_relax_2 = momentum - avg_fv1
        res_relax_3 = energy - 0.5 * avg_fv2
        res.update({"relaxation": (res_relax_1, res_relax_2, res_relax_3)})

        return res

    # inputs: (t, x, v)
    def value_and_grad(self, sol, inputs):
        t, x, v = inputs
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape1:
            tape1.watch([t, x])
            f = sol.fcall(tf.concat([t, x, v], axis=-1))

            values = {}
            values.update({"f": f})

            # approx rho, u, T
            density, velocity, temperature = sol.macrocall(tf.concat([t, x], axis=-1))

            momentum = density * velocity
            energy = 0.5 * density * (velocity ** 2 + temperature)
            values.update({"flow": (density, momentum, energy)})

            fun = sol.fcall
            avg_fv0 = self.average_op(fun, [t, x], [self.vquads, self.wquads])
            avg_fv1 = self.average_op(
                fun, [t, x], [self.vquads, self.wquads * self.vquads]
            )
            avg_fv2 = self.average_op(
                fun, [t, x], [self.vquads, self.wquads * self.vquads ** 2]
            )
            avg_fv3 = self.average_op(
                fun, [t, x], [self.vquads, self.wquads * self.vquads ** 3]
            )
            # print(avg_fv0.shape, avg_fv1.shape, avg_fv2.shape, avg_fv3.shape)
            values.update({"average": (avg_fv0, avg_fv1, avg_fv2)})

            Maxwellian = self.maxwellian(density, velocity, temperature, v)

            values.update({"maxwellian": Maxwellian})

        derivatives = {}
        df_dt, df_dx = tape1.gradient(f, [t, x])
        derivatives.update({"f": (df_dt, df_dx)})

        drho_dt = tape1.gradient(density, [t])[0]
        dm_dt = tape1.gradient(momentum, [t])[0]
        dE_dt = tape1.gradient(energy, [t])[0]
        derivatives.update({"flow": (drho_dt, dm_dt, dE_dt)})

        df1_dx = tape1.gradient(avg_fv1, [x])[0]
        df2_dx = tape1.gradient(avg_fv2, [x])[0]
        df3_dx = tape1.gradient(avg_fv3, [x])[0]
        derivatives.update({"moment": (df1_dx, df2_dx, df3_dx)})

        del tape1

        return values, derivatives

    def average_op(self, func, x, vwquads):
        # [x_dims, dims] -> [x_dims, 1, dims]
        x = tf.expand_dims(tf.concat(x, axis=-1), axis=-2)
        v, w = vwquads
        xv = tf.concat([x + 0.0 * v[..., 0:1], v + 0.0 * x[..., 0:1]], axis=-1)
        return tf.reduce_sum(func(xv) * w, axis=-2)

    # Maxwellian function
    def maxwellian(self, rho, u, T, v):
        return (rho / tf.sqrt(2.0 * pi * T)) * tf.exp(-((u - v) ** 2.0) / (2.0 * T))

    # # vector function
    # def Maxwellian(self, rho, u, T, v):
    #     return (rho / np.sqrt(2 * pi * T))[:, None] * np.exp(
    #         -((v - u[:, None]) ** 2) / (2 * T[:, None])
    #     )

    def macro0(self, macro_l, macro_r, x):
        y = tf.sin(pi * x)
        return macro_l + (macro_r - macro_l) * ((1.0 + y) / 2.0)
        # return macro_l + (macro_r - macro_l) * 0.5 * (tf.tanh(10.0*x) + 1.0)

    # inputs = (tbc, )
    # fbc ?
    def bc(self, sol, inputs):
        tbc = inputs
        # Left
        rho_l_approx, u_l_approx, T_l_approx = sol.macrocall(
            tf.concat([tbc, -0.5 * tf.ones_like(tbc)], axis=-1)
        )
        res_rho_l = rho_l_approx - self.rho_l
        res_u_l = u_l_approx - self.u_l
        res_T_l = T_l_approx - self.T_l
        # Right
        rho_r, u_r, T_r = sol.macrocall(
            tf.concat([tbc, 0.5 * tf.ones_like(tbc)], axis=-1)
        )

        res_rho_r = rho_r - self.rho_r
        res_u_r = u_r - self.u_r
        res_T_r = T_r - self.T_r

        res_boundary = {}
        res_boundary.update({"bc_left": (res_rho_l, res_u_l, res_T_l)})
        res_boundary.update({"bc_right": (res_rho_r, res_u_r, res_T_r)})

        return res_boundary

    # inputs = (xic, vic)
    def ic(self, sol, inputs):

        xic, vic = inputs

        rho0 = self.macro0(self.rho_l, self.rho_r, xic)
        u0 = self.macro0(self.u_l, self.u_r, xic)
        _T0 = self.macro0(self.T_l, self.T_r, xic)

        f0 = self.maxwellian(rho0, u0, _T0, vic)

        rho_init, u_init, _T_init = sol.macrocall(
            tf.concat([tf.zeros_like(vic), xic], axis=-1)
        )

        f_init = sol.fcall(tf.concat([tf.zeros_like(xic), xic, vic], axis=-1))

        res_initial = {}
        res_rho_init = rho_init - rho0
        res_u_init = u_init - u0
        res_T_init = _T_init - _T0
        res_f_init = f_init - f0
        res_initial.update(
            {"macro": (res_rho_init, res_u_init, res_T_init, res_f_init)}
        )

        return res_initial

    # inputs shape: [None, dims = 2]

    def val(self, sol, ref):

        time_freq = 1
        # shape: (100, 1)
        density_ref = ref["density"][::time_freq].astype("float32").reshape(-1, 1)
        momentum_ref = ref["momentum"][::time_freq].astype("float32").reshape(-1, 1)
        energy_ref = ref["energy"][::time_freq].astype("float32").reshape(-1, 1)

        rho_approx, u_approx, T_approx = sol.macrocall(
            tf.concat([self.ref_t, self.ref_x], axis=-1)
        )
        density_approx = rho_approx
        momentum_approx = rho_approx * u_approx
        energy_approx = 0.5 * rho_approx * (u_approx ** 2 + T_approx)

        err_density = tf.sqrt(
            tf.reduce_mean((density_ref - density_approx) ** 2)
            / tf.reduce_mean(density_ref ** 2)
        )
        # err_momentum = tf.sqrt(tf.reduce_mean(
        #     (momentum_ref - momentum_approx)**2) / tf.reduce_mean(momentum_ref**2))
        err_momentum = tf.reduce_mean(tf.sqrt((momentum_ref - momentum_approx) ** 2))
        err_energy = tf.sqrt(
            tf.reduce_mean((energy_ref - energy_approx) ** 2)
            / tf.reduce_mean(energy_ref ** 2)
        )

        return err_density, err_momentum, err_energy
