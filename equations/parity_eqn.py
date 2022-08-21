import numpy as np
import torch


class Parity(object):
    def __init__(self, config, sol, name="Parity_Eqn", **kwargs):

        # device setting
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
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
        self.rho_init = 0.0
        self.f_init = 0.0

    # inputs: (t, x, v)
    def residual(self, sol, inputs):

        t, x, v = inputs

        values, derivatives = self.value_and_grad(sol=sol, inputs=[t, x, v])
        rho, r, j = values["rho"], values["r"], values["j"]
        drho_dt = derivatives["rho"]
        dr_dt, dr_dx = derivatives["r"]
        dj_dt, dj_dx = derivatives["j"]
        davg_dx = derivatives["average"]
        avg_r = values["average"]

        # residual for parity and conservation equation
        eqn_res = {}

        res_parity_1 = self.kn ** 2 * (dr_dt + v * dj_dx) - (rho - r)
        res_parity_2 = self.kn ** 2 * dj_dt + v * dr_dx + j
        res_claw = drho_dt + davg_dx
        res_constraint = rho - avg_r

        eqn_res.update({"parity": (res_parity_1, res_parity_2)})
        eqn_res.update({"conservation": res_claw})
        eqn_res.update({"soft_constraint": res_constraint})

        return eqn_res

    def value_and_grad(self, sol, inputs):
        t, x, v = inputs
        t.requires_grad = True
        x.requires_grad = True

        net_rho, net_r, net_j = sol       
        rho = self.model_rho(net_rho, [t, x])
        r = self.model_r(net_r, [t, x, v])
        j = self.model_j(net_j, [t, x, v])

        values = {}
        values.update({"rho": rho})
        values.update({"r": r})
        values.update({"j": j})

        dr_dt = torch.autograd.grad(outputs=r, inputs=t,
                                    grad_outputs=torch.ones(
                                        r.shape).to(self.device),
                                    create_graph=True)[0]
        dr_dx = torch.autograd.grad(outputs=r, inputs=x,
                                    grad_outputs=torch.ones(
                                        r.shape).to(self.device),
                                    create_graph=True)[0]

        dj_dt = torch.autograd.grad(outputs=j, inputs=t,
                                    grad_outputs=torch.ones(
                                        j.shape).to(self.device),
                                    create_graph=True)[0]
        dj_dx = torch.autograd.grad(outputs=j, inputs=x,
                                    grad_outputs=torch.ones(
                                        j.shape).to(self.device),
                                    create_graph=True)[0]

        drho_dt = torch.autograd.grad(outputs=rho, inputs=t,
                                      grad_outputs=torch.ones(
                                          rho.shape).to(self.device),
                                      create_graph=True)[0]
    
        derivatives = {}
        derivatives.update({"rho": (drho_dt)})
        derivatives.update({"r": (dr_dt, dr_dx)})
        derivatives.update({"j": (dj_dt, dj_dx)})
        """trick:
            <v j_x> = <vj>_x
        """
        avg_1, avg_2 = self.average_op(net_r=net_r, net_j=net_j, t_x=[t, x], vwquads=[self.vquads, self.wquads])
        
        davg_dx = torch.autograd.grad(outputs=avg_1, inputs=x,
                                      grad_outputs=torch.ones(
                                          avg_1.shape).to(self.device),
                                      create_graph=True)[0]

        derivatives.update({"average": davg_dx})
        values.update({"average": avg_2})

        return values, derivatives

    """remark
        other choice for construct average_op by design r and j in the network, then 
        one may have 
                avg_1 = self.average_op(..., vwquads=[self.vquads, self.wquads*self.vquads])
                avg_2 = self.average_op(..., vwquads=[self.vquads, self.wquads])
    """
    def average_op(self, net_r, net_j, t_x, vwquads):
        tx = torch.cat(t_x, -1)[:, None, :]
        v, w1 = vwquads
        w2 = w1 * v
        mult_fact = torch.ones((tx.shape[0], v.shape[0], 1)).to(self.device)

        j_1 = net_j(torch.cat([tx * mult_fact, v[..., None] * mult_fact], -1))
        j_2 = net_j(torch.cat([tx * mult_fact, - v[..., None] * mult_fact], -1))
        gn = j_1 - j_2
        avg_1 = torch.sum(gn * w2[..., None], axis=-2)

        r_1 = net_r(torch.cat([tx * mult_fact, v[..., None] * mult_fact], -1))
        r_2 = net_r(torch.cat([tx * mult_fact, - v[..., None] * mult_fact], -1))
        fn = torch.exp(-0.5*(r_1 + r_2))
        avg_2 = torch.sum(fn * w1[..., None], axis=-2)
        
        return avg_1, avg_2


    # inputs = (tbc, vbc_l, vbc_r)
    def bc(self, net_r, net_j, inputs):
        tbc, vbc_l, vbc_r = inputs
        # Left
        bc_l =  [tbc, self.xmin * torch.ones_like(tbc), vbc_l]
        fbc_l = self.model_r(net_r, bc_l) +  self.kn * self.model_j(net_j, bc_l)
        # Right
        bc_r = [tbc, self.xmax * torch.ones_like(tbc), vbc_r]
        fbc_r = self.model_r(net_r, bc_r) +  self.kn * self.model_j(net_j, bc_r)

        res_f_l = fbc_l - self.f_l
        res_f_r = fbc_r - self.f_r

        res_bc = {}
        res_bc.update({"bc_left": res_f_l})
        res_bc.update({"bc_right": res_f_r})

        return res_bc

    # inputs = (xic, vic)
    def ic(self, net_rho, net_r, net_j, inputs):
        xic, vic = inputs
        rho0 = self.model_rho(net_rho, [0.0*xic, xic])
        init = [0.0*xic, xic, vic]
        f0 = self.model_r(net_r, init) + self.kn * self.model_j(net_j, init)
        
        res_ic = {}
        res_rho_init = rho0 - self.rho_init
        res_f_init = f0 - self.f_init
        res_ic.update({"initial": (res_rho_init, res_f_init)})

        return res_ic

    # exact initial
    def model_rho(self, net_rho, inputs):
        t, x = inputs
        rho = t * net_rho(torch.cat([t, x], -1))
        return rho

    # inputs: (t, x, v)
    def model_r(self, net_r, inputs):
        t, x, v = inputs
        r_1 = net_r(torch.cat([t, x, v], -1))
        r_2 = net_r(torch.cat([t, x, -v], -1))
        return torch.exp(-0.5*(r_1 + r_2))

    # inputs: (t, x, v)
    def model_j(self, net_j, inputs):
        t, x, v = inputs
        j_1 = net_j(torch.cat([t, x, v], -1))
        j_2 = net_j(torch.cat([t, x, -v], -1))
        j = j_1 - j_2
        return j

    # inputs shape: [None, dims = 2]
    def val(self, net_rho, ref):
        density_approx = self.model_rho(net_rho, [self.ref_t, self.ref_x]).to(
            "cpu"
        )  # .detach().numpy()
        density_ref = ref
        err_density = torch.sqrt(
            torch.mean((density_ref - density_approx) ** 2)
            / torch.mean(density_ref ** 2)
        )
        return err_density
