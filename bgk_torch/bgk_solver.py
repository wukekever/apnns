import torch


def train_step(
    sol, equation, trainloader, regularizers, optimizer, scheduler, ref, **kwargs
):

    trainloader_interior = trainloader[0]
    trainloader_boundary = trainloader[1]
    trainloader_initial = trainloader[2]

    optimizer.zero_grad()

    eqn_res = equation.residual(sol=sol, inputs=trainloader_interior)

    res_bgk = eqn_res["equation"]
    res_conserv1, res_conserv2, res_conserv3 = eqn_res["conservation"]
    res_relax1, res_relax2, res_relax3 = eqn_res["relaxation"]

    res_boundary = equation.bc(sol=sol, inputs=trainloader_boundary)
    res_rho_l, res_u_l, res_T_l = res_boundary["bc_left"]
    res_rho_r, res_u_r, res_T_r = res_boundary["bc_right"]

    res_init = equation.ic(sol=sol, inputs=trainloader_initial)
    res_rho0, res_u0, res_T0, res_f0 = res_init["initial"]

    res_eqn_1 = torch.mean(res_bgk**2)

    res_eqn_2_1 = torch.mean(res_conserv1**2)
    res_eqn_2_2 = torch.mean(res_conserv2**2)
    res_eqn_2_3 = torch.mean(res_conserv3**2)

    res_eqn_3_1 = torch.mean(res_relax1**2)
    res_eqn_3_2 = torch.mean(res_relax2**2)
    res_eqn_3_3 = torch.mean(res_relax3**2)

    res_eqn = (
        regularizers[0] * res_eqn_1
        + (
            regularizers[1] * res_eqn_2_1
            + regularizers[2] * res_eqn_2_2
            + regularizers[3] * res_eqn_2_3
        )
        + (
            regularizers[4] * res_eqn_3_1
            + regularizers[5] * res_eqn_3_2
            + regularizers[6] * res_eqn_3_3
        )
    )

    res_bc_1 = torch.mean(res_rho_l**2) + torch.mean(res_rho_r**2)
    res_bc_2 = torch.mean(res_u_l**2) + torch.mean(res_u_r**2)
    res_bc_3 = torch.mean(res_T_l**2) + torch.mean(res_T_r**2)
    res_bc = (
        regularizers[7] * res_bc_1
        + regularizers[8] * res_bc_2
        + regularizers[9] * res_bc_3
    )

    res_ic_1 = torch.mean(res_rho0**2)
    res_ic_2 = torch.mean(res_u0**2)
    res_ic_3 = torch.mean(res_T0**2)
    res_ic_4 = torch.mean(res_f0**2)

    res_ic = (
        regularizers[10] * res_ic_1
        + regularizers[11] * res_ic_2
        + regularizers[12] * res_ic_3
        + regularizers[13] * res_ic_4
    )

    loss = res_eqn + res_bc + res_ic

    risk = {}
    risk.update({"total_loss": loss.item()})
    risk.update({"bgk": res_eqn_1.item()})
    risk.update(
        {"conservation": (res_eqn_2_1.item(), res_eqn_2_2.item(), res_eqn_2_3.item())}
    )
    risk.update(
        {"relaxation": (res_eqn_3_1.item(), res_eqn_3_2.item(), res_eqn_3_3.item())}
    )
    risk.update({"bc_rho": res_bc_1.item()})
    risk.update({"bc_u": res_bc_2.item()})
    risk.update({"bc_T": res_bc_3.item()})
    risk.update({"ic_rho": res_ic_1.item()})
    risk.update({"ic_u": res_ic_2.item()})
    risk.update({"ic_T": res_ic_3.item()})
    risk.update({"ic_f": res_ic_4.item()})

    error = {}
    err_density, err_momentum, err_energy = equation.val(sol=sol, ref=ref)
    error.update(
        {"macro_error": (err_density.item(), err_momentum.item(), err_energy.item())}
    )

    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()  # clear memory

    return risk, error
