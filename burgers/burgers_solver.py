import torch


def train_step(
    sol, equation, trainloader, regularizers, optimizer, scheduler, ref, **kwargs
):

    trainloader_interior = trainloader[0]
    trainloader_boundary = trainloader[1]
    trainloader_initial = trainloader[2]

    optimizer.zero_grad()

    eqn_res = equation.residual(sol=sol, inputs=trainloader_interior)

    res_burgers = eqn_res["equation"]

    res_boundary = equation.bc(sol=sol, inputs=trainloader_boundary)
    res_u_l = res_boundary["bc_left"]
    res_u_r = res_boundary["bc_right"]

    res_init = equation.ic(sol=sol, inputs=trainloader_initial)
    res_u0 = res_init["initial"]

    res_eqn = torch.mean(res_burgers**2)
    res_bc = torch.mean(res_u_l**2) + torch.mean(res_u_r**2)
    res_ic = torch.mean(res_u0**2)

    loss = (
        regularizers[0] * res_eqn + regularizers[1] * res_bc + regularizers[2] * res_ic
    )

    risk = {}
    risk.update({"total_loss": loss.item()})
    risk.update({"burgers": res_eqn.item()})
    risk.update({"bc_u": res_bc.item()})
    risk.update({"ic_u": res_ic.item()})

    error = {}
    err_density = equation.val(sol=sol, ref=ref)
    error.update({"error": err_density.item()})

    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()  # clear memory

    return risk, error
