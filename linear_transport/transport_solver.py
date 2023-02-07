import torch


def train_step(
    sol, equation, trainloader, regularizers, optimizer, scheduler, ref, **kwargs
):

    trainloader_interior = trainloader[0]
    trainloader_boundary = trainloader[1]
    trainloader_initial = trainloader[2]

    optimizer.zero_grad()

    eqn_residual = equation.residual(sol=sol, inputs=trainloader_interior)

    res_transport = eqn_residual["equation"]

    res_boundary = equation.bc(sol=sol, inputs=trainloader_boundary)
    res_f_l = res_boundary["bc_left"]
    res_f_r = res_boundary["bc_right"]

    res_init = equation.ic(sol=sol, inputs=trainloader_initial)
    res_f0 = res_init["initial"]

    res_eqn = torch.mean(res_transport**2)
    res_bc = torch.mean(res_f_l**2) + torch.mean(res_f_r**2)
    res_ic = torch.mean(res_f0**2)

    loss = (
        regularizers[0] * res_eqn + regularizers[1] * res_bc + regularizers[2] * res_ic
    )

    risk = {}
    risk.update({"total_loss": loss.item()})
    risk.update({"transport": res_eqn.item()})
    risk.update({"bc_f": res_bc.item()})
    risk.update({"ic_f": res_ic.item()})

    error = {}
    err_density = equation.val(sol=sol, ref=ref)
    error.update({"error": err_density.item()})

    loss.backward()
    optimizer.step()
    scheduler.step()
    torch.cuda.empty_cache()  # clear memory

    return risk, error
