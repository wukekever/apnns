from math import pi

import numpy as np
import torch


class LinearTransport(object):
    def __init__(self, config, sol, name="Linear_Transport_Eqn", **kwargs):

        # device setting
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
        )

        # Knudsen number
        self.kn = config["model_config"]["kn"]

        # quadrature points and weights
        self.num_vquads = config["model_config"]["num_vquads"]
        vquads, wquads = np.polynomial.legendre.leggauss(self.num_vquads)
        self.vquads = torch.Tensor(0.5 * (1.0 + vquads)).to(self.device)
        self.wquads = torch.Tensor(0.5 * wquads).to(self.device)

        # domain
        self.tmin = config["physical_config"]["t_range"][0]
        self.tmax = config["physical_config"]["t_range"][1]
        self.xmin = config["physical_config"]["x_range"][0]
        self.xmax = config["physical_config"]["x_range"][1]

        # ref
        self.nx = 100
        self.ref_x = (torch.linspace(self.xmin, self.xmax, self.nx).reshape(-1, 1)).to(
            self.device
        )
        self.ref_t = self.tmax * torch.ones((self.nx, 1)).to(self.device)

        # bc
        self.f_l = 1.0
        self.f_r = 0.0

        # ic
        self.f_init = 0.0

    # inputs: (t, x, v)
    def residual(self, sol, inputs):

        t, x, v = inputs
        values, derivatives = self.value_and_grad(sol, t, x, v)
        f = values["f"]
        avg_f = values["average"]
        f_t, f_x = derivatives["f"]

        eqn_res = {}
        # residual for linear transport equation
        transport = self.kn * f_t + v * f_x
        collision = (avg_f - f) / self.kn
        res_transport = transport - collision
        eqn_res.update({"equation": res_transport})

        return eqn_res

    def value_and_grad(self, sol, t, x, v):

        t.requires_grad = True
        x.requires_grad = True

        model_f = sol

        values = {}
        f = model_f(torch.cat([t, x, v], -1))
        values.update({"f": f})

        average_f = self.average_op(
            model=model_f, t_x=[t, x], vwquads=[self.vquads, self.wquads]
        )
        values.update({"average": average_f})

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

        return values, derivatives

    def average_op(self, model, t_x, vwquads):
        tx = torch.cat(t_x, -1)[:, None, :]
        v, w = vwquads
        mult_fact = torch.ones((tx.shape[0], v.shape[0], 1)).to(self.device)
        fn = model(torch.cat([tx * mult_fact, v[..., None] * mult_fact], -1))
        average_f = torch.sum(fn * w[..., None], axis=-2) * 0.5
        return average_f

    def rho(self, sol, inputs):
        t, x = inputs
        rho_approx = self.average_op(
            model=sol, t_x=[t, x], vwquads=[self.vquads, self.wquads]
        )
        return rho_approx

    # inputs = (tbc, vbc)
    def bc(self, sol, inputs):
        tbc, vbc = inputs
        vbc_l, vbc_r = vbc, -vbc
        model_f = sol
        # Left
        fbc_l = model_f(torch.cat((tbc, self.xmin * torch.ones_like(tbc), vbc_l), -1))
        # Right
        fbc_r = model_f(torch.cat((tbc, self.xmax * torch.ones_like(tbc), vbc_r), -1))

        res_f_l = fbc_l - self.f_l
        res_f_r = fbc_r - self.f_r

        res_bc = {}
        res_bc.update({"bc_left": (res_f_l)})
        res_bc.update({"bc_right": (res_f_r)})

        return res_bc

    # inputs = (xic, vic)
    def ic(self, sol, inputs):
        xic, vic = inputs
        model_f = sol
        f0 = model_f(torch.cat((self.tmin * torch.ones_like(xic), xic, vic), -1))
        res_ic = {}
        res_init = f0 - self.f_init

        res_ic.update({"initial": (res_init)})

        return res_ic

    # inputs shape: [None, dims = 2]
    def val(self, sol, ref):
        density_approx = self.rho(sol, [self.ref_t, self.ref_x]).to(
            "cpu"
        )  # .detach().numpy()
        density_ref = ref
        err_density = torch.sqrt(
            torch.mean((density_ref - density_approx) ** 2)
            / (torch.mean(density_ref ** 2) + 1e-8)
        )
        return err_density
