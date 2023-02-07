import torch


def train_step(
    sol, equation, trainloader, regularizers, optimizer, scheduler, ref, **kwargs
):

    trainloader_interior = trainloader[0]
    trainloader_boundary = trainloader[1]
    trainloader_initial = trainloader[2]

    optimizer.zero_grad()

    eqn_residual = equation.residual(sol=sol, inputs=trainloader_interior)

    res_micro = eqn_residual["micro"]
    res_macro = eqn_residual["macro"]

    res_boundary = equation.bc(sol=sol, inputs=trainloader_boundary)
    res_f_l = res_boundary["bc_left"]
    res_f_r = res_boundary["bc_right"]

    res_init = equation.ic(sol=sol, inputs=trainloader_initial)
    res_f0 = res_init["initial"]

    res_micro_eqn = torch.mean(res_micro**2)
    res_macro_eqn = torch.mean(res_macro**2)
    res_bc = torch.mean(res_f_l**2) + torch.mean(res_f_r**2)
    res_ic = torch.mean(res_f0**2)

    loss = (
        regularizers[0] * res_micro_eqn + regularizers[1] * res_macro_eqn + regularizers[2] * res_bc + regularizers[-1] * res_ic
    )

    risk = {}
    risk.update({"total_loss": loss.item()})
    risk.update({"micro": res_micro_eqn.item()})
    risk.update({"macro": res_macro_eqn.item()})
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
