import torch


def train_step(
    sol, equation, trainloader, regularizers, optimizer, scheduler, ref, **kwargs
):

    trainloader_interior = trainloader[0]
    trainloader_boundary = trainloader[1]
    trainloader_initial = trainloader[2]

    net_rho, net_r, net_j = sol

    optimizer.zero_grad()

    eqn_residual = equation.residual(sol=sol, inputs=trainloader_interior)

    res_parity_1, res_parity_2 = eqn_residual["parity"]
    res_claw = eqn_residual["conservation"]
    res_constraint = eqn_residual["soft_constraint"]

    res_boundary = equation.bc(net_r=net_r, net_j=net_j, inputs=trainloader_boundary)
    res_f_l = res_boundary["bc_left"]
    res_f_r = res_boundary["bc_right"]

    res_init = equation.ic(net_rho=net_rho, net_r=net_r, net_j=net_j, inputs=trainloader_initial)
    res_rho0, res_f0 = res_init["initial"]

    res_parity_1_eqn = torch.mean(res_parity_1**2)
    res_parity_2_eqn = torch.mean(res_parity_2**2)
    res_claw_eqn = torch.mean(res_claw**2)
    res_constraint_eqn = torch.mean(res_constraint**2)
    res_bc = torch.mean(res_f_l**2) + torch.mean(res_f_r**2)
    res_rho_ic = torch.mean(res_rho0**2)
    res_f_ic = torch.mean(res_f0**2)


    loss = (
        regularizers[0] * res_parity_1_eqn + regularizers[1] * res_parity_2_eqn + 
        regularizers[2] * res_claw_eqn + regularizers[3] * res_constraint_eqn + 
        regularizers[4] * res_bc + 
        regularizers[5] * res_rho_ic + regularizers[6] * res_f_ic
    )

    risk = {}
    risk.update({"total_loss": loss.item()})
    risk.update({"parity_1": res_parity_1_eqn.item()})
    risk.update({"parity_2": res_parity_2_eqn.item()})
    risk.update({"conservation": res_claw_eqn.item()})
    risk.update({"soft_constraint": res_constraint_eqn.item()})
    risk.update({"bc_f": res_bc.item()})
    risk.update({"ic_rho": res_rho_ic.item()})
    risk.update({"ic_f": res_f_ic.item()})

    error = {}
    err_density = equation.val(net_rho=net_rho, ref=ref)
    error.update({"error": err_density.item()})

    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()  # clear memory

    return risk, error
