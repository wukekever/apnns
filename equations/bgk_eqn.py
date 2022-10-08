from math import pi
import numpy as np
import torch


class BGK(object):
    def __init__(self, config, sol, name="BGK_Eqn", **kwargs):

        # device setting
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(
                device_ids[0]) if torch.cuda.is_available() else "cpu"
        )

        # Knudsen number
        self.kn = config["physical_config"]["kn"]

        # domain
        self.tmin = config["physical_config"]["t_range"][0]
        self.tmax = config["physical_config"]["t_range"][1]
        self.xmin = config["physical_config"]["x_range"][0]
        self.xmax = config["physical_config"]["x_range"][1]
        self.vmin = config["physical_config"]["v_range"][0]
        self.vmax = config["physical_config"]["v_range"][1]

        # quadrature points and weights
        self.num_vquads = config["model_config"]["num_vquads"]
        vquads, wquads = np.polynomial.legendre.leggauss(self.num_vquads)
        vquads = 0.5 * (vquads + 1.0) * (self.vmax - self.vmin) + self.vmin
        wquads = 0.5 * (self.vmax - self.vmin) * wquads
        self.vquads = torch.Tensor(vquads).to(self.device)
        self.wquads = torch.Tensor(wquads).to(self.device)

        # bc
        self.rho_l, self.u_l, self.T_l = 1.5, 0.0, 1.5
        self.rho_r, self.u_r, self.T_r = 0.625, 0.0, 0.75

        # ref
        self.nx = 100
        self.dx = float(self.xmax - self.xmin) / self.nx
        self.ref_x = (
            torch.arange(self.xmin + self.dx / 2,
                         self.xmax + self.dx / 2, self.dx)
            .reshape((-1, 1))
            .to(self.device)
        )
        self.ref_t = self.tmax * torch.ones((self.nx, 1)).to(self.device)

    # inputs: (t, x, v)

    def residual(self, sol, inputs):

        t, x, v = inputs

        values, derivatives = self.value_and_grad(sol, t, x, v)
        f = values["f"]
        Maxwellian = values["maxwellian"]
        density, momentum, energy = values["var_flow"]
        avg_fv0, avg_fv1, avg_fv2 = values["var_moment"]
        df_dt, df_dx = derivatives["f"]
        drho_dt, dm_dt, dE_dt = derivatives["var_flow"]
        df1_dx, df2_dx, df3_dx = derivatives["var_moment"]

        eqn_res = {}

        # residual for BGK equation
        res_bgk = self.kn * (df_dt + v * df_dx) - (Maxwellian - f)
        eqn_res.update({"equation": res_bgk})

        # residual for conservation laws
        conserv1 = drho_dt + df1_dx
        conserv2 = dm_dt + df2_dx
        conserv3 = dE_dt + 0.5 * df3_dx
        eqn_res.update({"conservation": (conserv1, conserv2, conserv3)})

        # residual for relaxation
        relax1 = density - avg_fv0
        relax2 = momentum - avg_fv1
        relax3 = energy - 0.5 * avg_fv2
        eqn_res.update({"relaxation": (relax1, relax2, relax3)})

        return eqn_res

    def value_and_grad(self, sol, t, x, v):

        t.requires_grad = True
        x.requires_grad = True

        model_f, model_rho, model_u, model_T = sol

        values = {}
        f = model_f(torch.cat([t, x, v], -1))
        values.update({"f": f})

        density = model_rho(torch.cat([t, x], -1))
        velocity = model_u(torch.cat([t, x], -1))
        temperature = model_T(torch.cat([t, x], -1))

        momentum = density * velocity
        energy = 0.5 * density * (velocity ** 2 + temperature)
        values.update({"var_flow": (density, momentum, energy)})

        avg_fv0 = self.average_op(model_f, [t, x], [self.vquads, self.wquads])
        avg_fv1 = self.average_op(
            model_f, [t, x], [self.vquads, self.wquads * self.vquads]
        )
        avg_fv2 = self.average_op(
            model_f, [t, x], [self.vquads, self.wquads * self.vquads ** 2]
        )
        avg_fv3 = self.average_op(
            model_f, [t, x], [self.vquads, self.wquads * self.vquads ** 3]
        )
        values.update({"var_moment": (avg_fv0, avg_fv1, avg_fv2)})

        Maxwellian = self.maxwellian(density, velocity, temperature, v)
        values.update({"maxwellian": Maxwellian})

        derivatives = {}
        df_dt = torch.autograd.grad(
            outputs=f,
            inputs=t,
            grad_outputs=torch.ones(f.shape).to(self.device),
            create_graph=True,
        )[0]
        df_dx = torch.autograd.grad(
            outputs=f,
            inputs=x,
            grad_outputs=torch.ones(f.shape).to(self.device),
            create_graph=True,
        )[0]
        derivatives.update({"f": (df_dt, df_dx)})

        drho_dt = torch.autograd.grad(
            outputs=density,
            inputs=t,
            grad_outputs=torch.ones(density.shape).to(self.device),
            create_graph=True,
        )[0]
        dm_dt = torch.autograd.grad(
            outputs=momentum,
            inputs=t,
            grad_outputs=torch.ones(momentum.shape).to(self.device),
            create_graph=True,
        )[0]
        dE_dt = torch.autograd.grad(
            outputs=energy,
            inputs=t,
            grad_outputs=torch.ones(energy.shape).to(self.device),
            create_graph=True,
        )[0]
        derivatives.update({"var_flow": (drho_dt, dm_dt, dE_dt)})

        df1_dx = torch.autograd.grad(
            outputs=avg_fv1,
            inputs=x,
            grad_outputs=torch.ones(avg_fv1.shape).to(self.device),
            create_graph=True,
        )[0]
        df2_dx = torch.autograd.grad(
            outputs=avg_fv2,
            inputs=x,
            grad_outputs=torch.ones(avg_fv2.shape).to(self.device),
            create_graph=True,
        )[0]
        df3_dx = torch.autograd.grad(
            outputs=avg_fv3,
            inputs=x,
            grad_outputs=torch.ones(avg_fv3.shape).to(self.device),
            create_graph=True,
        )[0]
        derivatives.update({"var_moment": (df1_dx, df2_dx, df3_dx)})

        return values, derivatives

    def average_op(self, model, t_x, vwquads):
        tx = torch.cat(t_x, -1)[:, None, :]
        v, w = vwquads
        mult_fact = torch.ones((tx.shape[0], v.shape[0], 1)).to(self.device)
        fn = model(torch.cat([tx * mult_fact, v[..., None] * mult_fact], -1))
        return torch.sum(fn * w[..., None], dim=-2)

    # Maxwellian function
    def maxwellian(self, rho, u, T, v):
        return (rho / torch.sqrt(2.0 * pi * T)) * torch.exp(
            -((u - v) ** 2.0) / (2.0 * T)
        )

    def macro0(self, macro_l, macro_r, x):
        # y = tf.pow(tf.sin(pi*x*tf.sign(x)), 3 / 7) * tf.sign(x)
        # return macro_l + (macro_r - macro_l) * ((1.0 + y) / 2.0)
        # return macro_l + (macro_r - macro_l) * 0.5 * (tf.tanh(10.0*x) + 1.0)
        return macro_l + (macro_r - macro_l) * 0.5 * (torch.tanh(20.0*x) + 1.0)

    # inputs = (tbc, )
    def bc(self, sol, inputs):
        tbc = inputs
        model_rho, model_u, model_T = sol[1:]

        # Left
        bcl = torch.cat([tbc, -0.5 * torch.ones_like(tbc)], -1)
        _rho_l, _u_l, _T_l = model_rho(bcl), model_u(bcl), model_T(bcl)

        # Right
        bcr = torch.cat([tbc, 0.5 * torch.ones_like(tbc)], -1)
        _rho_r, _u_r, _T_r = model_rho(bcr), model_u(bcr), model_T(bcr)

        res_rho_l = _rho_l - self.rho_l
        res_u_l = _u_l - self.u_l
        res_T_l = _T_l - self.T_l

        res_rho_r = _rho_r - self.rho_r
        res_u_r = _u_r - self.u_r
        res_T_r = _T_r - self.T_r

        res_bc = {}
        res_bc.update({"bc_left": (res_rho_l, res_u_l, res_T_l)})
        res_bc.update({"bc_right": (res_rho_r, res_u_r, res_T_r)})

        return res_bc

    # inputs = (xic, vic)
    def ic(self, sol, inputs):

        xic, vic = inputs

        rho0 = self.macro0(self.rho_l, self.rho_r, xic)
        u0 = self.macro0(self.u_l, self.u_r, xic)
        T0 = self.macro0(self.T_l, self.T_r, xic)

        f0 = self.maxwellian(rho0, u0, T0, vic)

        model_f, model_rho, model_u, model_T = sol
        ic = torch.cat([torch.zeros_like(vic), xic], -1)
        rho_init = model_rho(ic)
        u_init = model_u(ic)
        _T_init = model_T(ic)
        f_init = model_f(torch.cat([torch.zeros_like(vic), xic, vic], -1))

        res_ic = {}
        res_rho_init = rho_init - rho0
        res_u_init = u_init - u0
        res_T_init = _T_init - T0
        res_f_init = f_init - f0

        res_ic.update(
            {"initial": (res_rho_init, res_u_init, res_T_init, res_f_init)})

        return res_ic

    # inputs shape: [None, dims = 2]
    def val(self, sol, ref):

        time_freq = 1
        # shape: (100, 1)
        density_ref = torch.Tensor(
            ref["density"][::time_freq].astype("float32").reshape(-1, 1)
        )
        momentum_ref = torch.Tensor(
            ref["momentum"][::time_freq].astype("float32").reshape(-1, 1)
        )
        energy_ref = torch.Tensor(
            ref["energy"][::time_freq].astype("float32").reshape(-1, 1)
        )

        ref_tx = torch.cat([self.ref_t, self.ref_x], -1)
        model_rho, model_u, model_T = sol[1:]
        rho_approx, u_approx, T_approx = (
            model_rho(ref_tx).to("cpu"),
            model_u(ref_tx).to("cpu"),
            model_T(ref_tx).to("cpu"),
        )
        density_approx = rho_approx
        momentum_approx = rho_approx * u_approx
        energy_approx = 0.5 * rho_approx * (u_approx ** 2 + T_approx)

        err_density = torch.sqrt(
            torch.mean((density_ref - density_approx) ** 2)
            / torch.mean(density_ref ** 2)
        )
        err_momentum = torch.sqrt(
            torch.mean(torch.sqrt((momentum_ref - momentum_approx) ** 2))
        )
        err_energy = torch.sqrt(
            torch.mean((energy_ref - energy_approx) ** 2) /
            torch.mean(energy_ref ** 2)
        )

        return err_density, err_momentum, err_energy
