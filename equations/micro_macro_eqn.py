from math import pi

import numpy as np
import torch


class MicroMacro(object):
    def __init__(self, config, sol, name="Micro_Macro_Eqn", **kwargs):

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
        rho_t, rho_x = derivatives["rho"]
        g_t, g_x, aver_vg_x = derivatives["g"]
        collision = values["collision"]
        # residual for micro and macro equation

        """trick
           In the Micro part, one can change the order of integration and derivative
           \Pi (v g_x) = <vg_x> = <vg>_x <-- which exactly occurs in the Macro part !
        """
        eqn_res = {}
        micro_res = self.kn**2 * g_t + self.kn * \
            (v * g_x - aver_vg_x) + v * rho_x - collision
        macro_res = rho_t + aver_vg_x
        eqn_res.update({"micro": micro_res})
        eqn_res.update({"macro": macro_res})
        return eqn_res

    def value_and_grad(self, sol, t, x, v):

        t.requires_grad = True
        x.requires_grad = True

        model_rho, model_g = sol
        g = model_g(torch.cat([t, x, v], -1)) - self.average_op(
            model=model_g, t_x=[t, x], vwquads=[self.vquads, self.wquads])
        rho = model_rho(torch.cat([t, x], -1))

        # Micro part
        g_t = torch.autograd.grad(outputs=g, inputs=t,
                                  grad_outputs=torch.ones(
                                      g.shape).to(self.device),
                                  create_graph=True)[0]

        g_x = torch.autograd.grad(outputs=g, inputs=x,
                                  grad_outputs=torch.ones(
                                      g.shape).to(self.device),
                                  create_graph=True)[0]

        rho_x = torch.autograd.grad(outputs=rho, inputs=x,
                                    grad_outputs=torch.ones(
                                        rho.shape).to(self.device),
                                    create_graph=True)[0]

        # Macro part
        aver_vg = self.average_op(model=model_g, t_x=[t, x], vwquads=[
                                  self.vquads, self.wquads*self.vquads])

        aver_vg_x = torch.autograd.grad(outputs=aver_vg, inputs=x,
                                        grad_outputs=torch.ones(
                                            aver_vg.shape).to(self.device),
                                        create_graph=True)[0]

        rho_t = torch.autograd.grad(outputs=rho, inputs=t,
                                    grad_outputs=torch.ones(
                                        rho.shape).to(self.device),
                                    create_graph=True)[0]

        # Collision term
        """Property
           g = g_bar - <g_bar>
           Lg = <g> - g = <g_bar - <g_bar>> - g_bar - <g_bar> = 0 - g
        """
        collision = 0.0 - g

        values = {}
        values.update({"collision": collision})

        derivatives = {}
        derivatives.update({"rho": (rho_t, rho_x)})
        derivatives.update({"g": (g_t, g_x, aver_vg_x)})

        return values, derivatives

    def average_op(self, model, t_x, vwquads):
        tx = torch.cat(t_x, -1)[:, None, :]
        v, w = vwquads
        mult_fact = torch.ones((tx.shape[0], v.shape[0], 1)).to(self.device)
        fn = model(torch.cat([tx * mult_fact, v[..., None] * mult_fact], -1))
        average_fn = 0.5 * torch.sum(fn * w[..., None], dim=-2)
        return average_fn

    def rho(self, sol, inputs):
        t, x = inputs
        model_rho = sol[0]
        rho_approx = model_rho(torch.cat([t, x], -1))
        return rho_approx

    def model_f(self, sol, inputs):
        t, x, v = inputs
        model_rho, model_g = sol
        g = model_g(torch.cat([t, x, v], -1)) - self.average_op(
            model=model_g, t_x=[t, x], vwquads=[self.vquads, self.wquads])
        rho = model_rho(torch.cat([t, x], -1))
        f_approx = rho + self.kn * g
        return f_approx

    # inputs = (tbc, vbc)
    def bc(self, sol, inputs):
        tbc, vbc = inputs
        vbc_l, vbc_r = vbc, -vbc
        # Left
        fbc_l = self.model_f(
            sol=sol, inputs=[tbc, self.xmin * torch.ones_like(tbc), vbc_l])
        # Right
        fbc_r = self.model_f(
            sol=sol, inputs=[tbc, self.xmax * torch.ones_like(tbc), vbc_r])

        res_f_l = fbc_l - self.f_l
        res_f_r = fbc_r - self.f_r

        res_bc = {}
        res_bc.update({"bc_left": (res_f_l)})
        res_bc.update({"bc_right": (res_f_r)})

        return res_bc

    # inputs = (xic, vic)
    def ic(self, sol, inputs):
        xic, vic = inputs
        f0 = self.model_f(
            sol=sol, inputs=[self.tmin * torch.ones_like(xic), xic, vic])
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
            / torch.mean(density_ref ** 2)
        )
        return err_density
