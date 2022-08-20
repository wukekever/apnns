from math import pi

import numpy as np
import torch


class Burgers(object):
    def __init__(self, config, sol, name="Burgers_Eqn", **kwargs):

        # device setting
        device_ids = config["model_config"]["device_ids"]
        device = torch.device("cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")
        self.device = device

        self.tmin = config["physical_config"]["t_range"][0]
        self.tmax = config["physical_config"]["t_range"][1]
        self.xmin = config["physical_config"]["x_range"][0]
        self.xmax = config["physical_config"]["x_range"][1]

        self.mu = config["physical_config"]["viscosity"] / pi

        # bc
        self.u_l = 0.0
        self.u_r = 0.0

        # ref
        self.nx = 256
        self.ref_x = (torch.linspace(self.xmin, self.xmax, self.nx).reshape(-1, 1)).to(
            self.device
        )
        self.ref_t = self.tmax * torch.ones((self.nx, 1)).to(self.device)

    # inputs: (t, x, v)

    def residual(self, sol, inputs):

        t, x = inputs
        values, derivatives = self.value_and_grad(sol, t, x)
        u = values["u"]
        u_t, u_x, u_xx = derivatives["u"]

        eqn_res = {}
        # residual for Burgers equation
        res_burgers = u_t + u * u_x - self.mu * u_xx
        eqn_res.update({"equation": res_burgers})

        return eqn_res

    def value_and_grad(self, sol, t, x):

        t.requires_grad = True
        x.requires_grad = True

        model_u = sol

        values = {}
        u = model_u(torch.cat([t, x], -1))
        values.update({"u": u})

        derivatives = {}
        du_dt = torch.autograd.grad(
            outputs=u,
            inputs=t,
            grad_outputs=torch.ones(u.shape).to(self.device),
            create_graph=True,
        )[0]
        du_dx = torch.autograd.grad(
            outputs=u,
            inputs=x,
            grad_outputs=torch.ones(u.shape).to(self.device),
            create_graph=True,
        )[0]
        d2u_dx2 = torch.autograd.grad(
            outputs=du_dx,
            inputs=x,
            grad_outputs=torch.ones(du_dx.shape).to(self.device),
            create_graph=True,
        )[0]
        derivatives.update({"u": (du_dt, du_dx, d2u_dx2)})

        return values, derivatives

    # inputs = (tbc, )
    def bc(self, sol, inputs):
        tbc = inputs
        model_u = sol

        # Left
        bcl = torch.cat([tbc, -1.0 * torch.ones_like(tbc)], -1)
        _u_l = model_u(bcl)
        # Right
        bcr = torch.cat([tbc, 1.0 * torch.ones_like(tbc)], -1)
        _u_r = model_u(bcr)

        res_u_l = _u_l - self.u_l
        res_u_r = _u_r - self.u_r

        res_bc = {}
        res_bc.update({"bc_left": (res_u_l)})
        res_bc.update({"bc_right": (res_u_r)})

        return res_bc

    def init_fn(self, inputs):
        x = inputs
        return -torch.sin(pi * x)

    # inputs = (xic, )
    def ic(self, sol, inputs):

        xic = inputs
        u0 = self.init_fn(xic)

        model_u = sol
        ic = torch.cat([torch.zeros_like(xic), xic], -1)
        u_init = model_u(ic)

        res_ic = {}
        res_u_init = u_init - u0

        res_ic.update({"initial": (res_u_init)})

        return res_ic

    # inputs shape: [None, dims = 2]
    def val(self, sol, ref):

        freq = 1
        # shape: (100, 1)
        density_ref = torch.Tensor(
            ref["density"][::freq].astype("float32").reshape(-1, 1)
        )

        ref_tx = torch.cat([self.ref_t, self.ref_x], -1)
        model_u = sol
        density_approx = model_u(ref_tx).to("cpu")

        err_density = torch.sqrt(
            torch.mean((density_ref - density_approx) ** 2)
            / torch.mean(density_ref**2)
        )

        return err_density
