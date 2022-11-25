#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # ignore INFO\WARNING

import numpy as np

from make_dir import mkdir

from tensorflow.keras import layers, Sequential

from load_yaml import get_yaml

import models.solutions as solutions
import equations.bgk_eqn as equation

from dataset import Sampler
from solver import Solver

from math import pi
pi = tf.cast(pi, dtype = float)

import matplotlib.pyplot as plt
import time

# In[ ]:


print("TensorFlow version: {}".format(tf.__version__))
print("Eager exacution: {}".format(tf.executing_eagerly()))
# set gpu
gpus = tf.config.list_physical_devices("GPU")
logical_gpus = tf.config.experimental.list_logical_devices(device_type="GPU")
print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")


# In[ ]:


# load config
current_path = os.path.abspath(".")
yaml_path = os.path.join(current_path, "kinetic.yaml")
Config = get_yaml(yaml_path)

# load reference data
ref_path = os.path.join(current_path, "data/dirichlet_ref_kn1e0.npz")
ref = np.load(ref_path)
time_freq = 1
ref_rho = (
    ref["density"][::time_freq].astype("float32").reshape(-1, 1)
)  # shape: (100, 1)
ref_momentum = ref["momentum"][::time_freq].astype("float32").reshape(-1, 1)
ref_energy = ref["energy"][::time_freq].astype("float32").reshape(-1, 1)


# In[ ]:


# build neural networks for f, rho, u, T
Sol = "solutions.Sol_" + \
    "{}".format(Config["model_config"]["neural_network_type"])
Solution = eval(Sol)

bgk_sol = Solution(
    units_f=Config["model_config"]["units_f"],
    units_rho=Config["model_config"]["units_rho"],
    units_u=Config["model_config"]["units_u"],
    units_T=Config["model_config"]["units_T"],
    activation=tf.nn.swish,
    kernel_initializer="glorot_normal",
    name="bgk_sol",
)
bgk_sol.build()

# define the BGK equation
bgk_eqn = equation.BGK(Config, bgk_sol, name="BGK_Eqn")


# In[ ]:


# make plot
def plot(iter):
    xmin = Config["dataset_config"]["x_range"][0]
    xmax = Config["dataset_config"]["x_range"][1]
    tmax = Config["dataset_config"]["t_range"][1]

    nx = 100
    dx = float(xmax - xmin) / nx
    ref_x = np.arange(xmin + dx / 2, xmax + dx / 2, dx).reshape((-1, 1))
    ref_t = tmax * np.ones((nx, 1))

    rho_approx, u_approx, T_approx = bgk_sol.macrocall(
        inputs=tf.concat([ref_t, ref_x], axis=-1)
    )
    approx_density = rho_approx
    approx_momentum = rho_approx * u_approx
    approx_energy = 0.5 * rho_approx * (u_approx**2 + T_approx)

    plt.plot(ref_x, ref_rho, "r", label="density")
    plt.plot(ref_x, ref_momentum, "k", label="momentum")
    plt.plot(ref_x, ref_energy, "g", label="energy")

    plt.plot(ref_x, approx_density, "r*", markevery=4, label="approx density")
    plt.plot(ref_x, approx_momentum, "k+",
             markevery=4, label="approx momentum")
    plt.plot(ref_x, approx_energy, "gx", markevery=4, label="approx energy")
    plt.grid()
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("macro")
    plt.title("Approximate density, momentum, energy and reference solutions")
    plt.savefig("./figure/solution_iter_{:d}.pdf".format(iter))
    # plt.show()
    plt.close()


# In[ ]:


# generate dataset
sampler = Sampler(Config=Config, name="sampler")
trainloader = [sampler.interior(), sampler.boundary(), sampler.initial()]

mkdir(file_dir="./record")
mkdir(file_dir="./figure")

# set optimizer, regularizers and iterations
Iter = Config["model_config"]["iterations"]
lr_init = Config["model_config"]["lr"]
steps = Config["model_config"]["stage_num"]
decay_rate = Config["model_config"]["decay_rate"]
lr = [lr_init * decay_rate ** (i // steps) for i in range(Iter)]

regularizers = Config["model_config"]["regularizers"]

loss_record, error_record = np.array([[]]).T, np.array([[]] * 3).T

time_start = time.time()
print("Begin first training.")
print("")
for it in range(Iter):

    sampler = Sampler(Config=Config, name="sampler")
    trainloader = [sampler.interior(), sampler.boundary(), sampler.initial()]

    bgk_solver = Solver(
        sol=bgk_sol,
        equation=bgk_eqn,
        trainloader=trainloader,
        regularizers=regularizers,
        ref=ref,
        name="bgk_solver",
    )

    opt = tf.keras.optimizers.Adam(lr[it])

    bgk_solver.compile(optimizer=opt)

    loss = bgk_solver.train_step()["total_loss"].numpy()
    res_bgk_eqn = bgk_solver.train_step()["bgk"].numpy()

    res_conservation_eqn_1 = bgk_solver.train_step()["conservation"][0].numpy()
    res_conservation_eqn_2 = bgk_solver.train_step()["conservation"][1].numpy()
    res_conservation_eqn_3 = bgk_solver.train_step()["conservation"][2].numpy()

    res_relaxation_eqn_1 = bgk_solver.train_step()["relaxation"][0].numpy()
    res_relaxation_eqn_2 = bgk_solver.train_step()["relaxation"][1].numpy()
    res_relaxation_eqn_3 = bgk_solver.train_step()["relaxation"][2].numpy()

    res_bc_rho = bgk_solver.train_step()["bc_rho"].numpy()
    res_bc_u = bgk_solver.train_step()["bc_u"].numpy()
    res_bc_T = bgk_solver.train_step()["bc_T"].numpy()

    res_ic_rho = bgk_solver.train_step()["ic_rho"].numpy()
    res_ic_u = bgk_solver.train_step()["ic_u"].numpy()
    res_ic_T = bgk_solver.train_step()["ic_T"].numpy()
    res_ic_f = bgk_solver.train_step()["ic_f"].numpy()

    error = np.array(bgk_solver.test_step(), dtype=float).reshape(1, -1)
    loss_record = np.concatenate((loss_record, loss * np.ones((1, 1))), axis=0)
    error_record = np.concatenate((error_record, error), axis=0)

    if it % 1 == 0:
        print(
            "[Iter: {:6d}/{:6d} - lr : {:.2e} and Loss: {:.2e}]".format(
                it + 1, Iter, lr[it], loss
            )
        )
        print(
            "[Error for density: {:.2e} - momentum: {:.2e} - energy: {:.2e}]".format(
                float(error[:, 0]), float(error[:, 1]), float(error[:, 2])
            )
        )
        print(
            "[BGK eqn: {:.2e} and Conservation - density: {:.2e} - momentum: {:.2e} - energy: {:.2e}]".format(
                res_bgk_eqn,
                res_conservation_eqn_1,
                res_conservation_eqn_2,
                res_conservation_eqn_3,
            )
        )
        print(
            "[Relaxation - rho: {:.2e} - u: {:.2e} - T: {:.2e}]".format(
                res_relaxation_eqn_1, res_relaxation_eqn_2, res_relaxation_eqn_3
            )
        )
        # print(
        #     "[Boundary - rho: {:.2e} - u: {:.2e} - T: {:.2e}]".format(
        #         res_bc_rho, res_bc_u, res_bc_T
        #     )
        # )
        print(
            "[Initial - rho: {:.2e} - u: {:.2e} - T: {:.2e} - f: {:.2e}]".format(
                res_ic_rho, res_ic_u, res_ic_T, res_ic_f
            )
        )
    if (it + 1) % 100 == 0:
        plot(iter=it + 1)

    # if np.max(error) < 1e-2:
    #     print("Iteration step: ", it)
    #     break


# np.savez("./record/result.npz",
#          loss=loss_record,
#          error_rho=error_record[:, 0],
#          error_momentum=error_record[:, 1],
#          error_energy=error_record[:, 2])


print("")
print("Finished first training.")
time_end = time.time()
print("Total first time is: {:.2e}".format(time_end - time_start), "seconds")


# In[ ]:


yaml_path = os.path.join(current_path, "kinetic_finetune.yaml")
Config = get_yaml(yaml_path)

# set optimizer, regularizers and iterations
Iter = Config["model_config"]["iterations"]
lr_init = Config["model_config"]["lr"]
steps = Config["model_config"]["stage_num"]
decay_rate = Config["model_config"]["decay_rate"]
lr = [lr_init * decay_rate ** (i // steps) for i in range(Iter)]

regularizers = Config["model_config"]["regularizers"]


# In[ ]:


time_start = time.time()
print("Begin second training.")
print("")
for it in range(Iter):

    sampler = Sampler(Config=Config, name="sampler")
    trainloader = [sampler.interior(), sampler.boundary(), sampler.initial()]

    bgk_solver = Solver(
        sol=bgk_sol,
        equation=bgk_eqn,
        trainloader=trainloader,
        regularizers=regularizers,
        ref=ref,
        name="bgk_solver",
    )

    opt = tf.keras.optimizers.Adam(lr[it])

    bgk_solver.compile(optimizer=opt)

    loss = bgk_solver.train_step()["total_loss"].numpy()
    res_bgk_eqn = bgk_solver.train_step()["bgk"].numpy()

    res_conservation_eqn_1 = bgk_solver.train_step()["conservation"][0].numpy()
    res_conservation_eqn_2 = bgk_solver.train_step()["conservation"][1].numpy()
    res_conservation_eqn_3 = bgk_solver.train_step()["conservation"][2].numpy()

    res_relaxation_eqn_1 = bgk_solver.train_step()["relaxation"][0].numpy()
    res_relaxation_eqn_2 = bgk_solver.train_step()["relaxation"][1].numpy()
    res_relaxation_eqn_3 = bgk_solver.train_step()["relaxation"][2].numpy()

    res_bc_rho = bgk_solver.train_step()["bc_rho"].numpy()
    res_bc_u = bgk_solver.train_step()["bc_u"].numpy()
    res_bc_T = bgk_solver.train_step()["bc_T"].numpy()

    res_ic_rho = bgk_solver.train_step()["ic_rho"].numpy()
    res_ic_u = bgk_solver.train_step()["ic_u"].numpy()
    res_ic_T = bgk_solver.train_step()["ic_T"].numpy()
    res_ic_f = bgk_solver.train_step()["ic_f"].numpy()

    error = np.array(bgk_solver.test_step(), dtype=float).reshape(1, -1)
    loss_record = np.concatenate((loss_record, loss * np.ones((1, 1))), axis=0)
    error_record = np.concatenate((error_record, error), axis=0)

    if it % 1 == 0:
        print(
            "[Iter: {:6d}/{:6d} - lr : {:.2e} and Loss: {:.2e}]".format(
                Iter + it + 1, Iter + Iter, lr[it], loss
            )
        )
        print(
            "[Error for density: {:.2e} - momentum: {:.2e} - energy: {:.2e}]".format(
                float(error[:, 0]), float(error[:, 1]), float(error[:, 2])
            )
        )
        print(
            "[BGK eqn: {:.2e} and Conservation - density: {:.2e} - momentum: {:.2e} - energy: {:.2e}]".format(
                res_bgk_eqn,
                res_conservation_eqn_1,
                res_conservation_eqn_2,
                res_conservation_eqn_3,
            )
        )
        print(
            "[Relaxation - rho: {:.2e} - u: {:.2e} - T: {:.2e}]".format(
                res_relaxation_eqn_1, res_relaxation_eqn_2, res_relaxation_eqn_3
            )
        )
        # print(
        #     "[Boundary - rho: {:.2e} - u: {:.2e} - T: {:.2e}]".format(
        #         res_bc_rho, res_bc_u, res_bc_T
        #     )
        # )
        print(
            "[Initial - rho: {:.2e} - u: {:.2e} - T: {:.2e} - f: {:.2e}]".format(
                res_ic_rho, res_ic_u, res_ic_T, res_ic_f
            )
        )

        if (it + 1) % 100 == 0:
            plot(iter=Iter + it + 1)

        if np.max(error) < 1e-2:
            print("Iteration step: ", Iter + it)
            break

np.savez(
    "./record/result.npz",
    loss=loss_record,
    error_rho=error_record[:, 0],
    error_momentum=error_record[:, 1],
    error_energy=error_record[:, 2],
)

print("")
print("Finished second training.")
time_end = time.time()
print("Total second time is: {:.2e}".format(time_end - time_start), "seconds")


# In[ ]:
weights = bgk_sol.get_weights()
np.savez(
    "./record/saved_weights.npz", model_weights=np.array(weights), allow_pickle=True
)

xmin = Config["dataset_config"]["x_range"][0]
xmax = Config["dataset_config"]["x_range"][1]
tmin = Config["dataset_config"]["t_range"][0]
tmax = Config["dataset_config"]["t_range"][1]

nx = 100
dx = float(xmax - xmin) / nx
ref_x = np.arange(xmin + dx / 2, xmax + dx / 2, dx).reshape((-1, 1))
ref_x = tf.convert_to_tensor(ref_x, dtype=tf.float32)


# In[ ]:


# plot for t = 0.1
ref_t = tmax * np.ones((nx, 1))
ref_t = tf.convert_to_tensor(ref_t, dtype=tf.float32)

rho_approx, u_approx, T_approx = bgk_sol.macrocall(
    inputs=tf.concat([ref_t, ref_x], axis=-1)
)
approx_density = rho_approx
approx_momentum = rho_approx * u_approx
approx_energy = 0.5 * rho_approx * (u_approx**2 + T_approx)

plt.plot(ref_x, ref_rho, "r", label="density")
plt.plot(ref_x, ref_momentum, "k", label="momentum")
plt.plot(ref_x, ref_energy, "g", label="energy")

plt.plot(ref_x, approx_density, "r*", markevery=4, label="approx density")
plt.plot(ref_x, approx_momentum, "k+", markevery=4, label="approx momentum")
plt.plot(ref_x, approx_energy, "gx", markevery=4, label="approx energy")

plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("macro(t = 0.1)")
plt.title("Approximate density, momentum, energy and reference solutions")
plt.savefig("./figure/solution_macro_t1.pdf")
plt.show()
plt.close()


# In[ ]:


def average_op(func, x, vwquads):
    # [x_dims, dims] -> [x_dims, 1, dims]
    x = tf.expand_dims(tf.concat(x, axis=-1), axis=-2)
    v, w = vwquads
    xv = tf.concat([x + 0.0 * v[..., 0:1], v + 0.0 * x[..., 0:1]], axis=-1)
    avg = tf.reduce_sum(func(xv) * w, axis=-2)
    return avg


# In[ ]:


v, w = np.polynomial.legendre.leggauss(32)
v = 0.5 * (v + 1.0) * 20 - 10
w = 0.5 * 20 * w
vquads = tf.convert_to_tensor(v[:, None], dtype=tf.float32)
wquads = tf.convert_to_tensor(w[:, None], dtype=tf.float32)

avg_f_1 = average_op(bgk_sol.fcall, [ref_t, ref_x], [vquads, wquads])
approx_density = avg_f_1
avg_f_2 = average_op(bgk_sol.fcall, [ref_t, ref_x], [vquads, wquads * vquads])
approx_momentum = avg_f_2
avg_f_3 = average_op(bgk_sol.fcall, [ref_t, ref_x], [
                     vquads, wquads * vquads**2])
approx_energy = 0.5 * avg_f_3

plt.plot(ref_x, ref_rho, "r", label="density")
plt.plot(ref_x, ref_momentum, "k", label="momentum")
plt.plot(ref_x, ref_energy, "g", label="energy")

plt.plot(ref_x, approx_density, "r*", markevery=4, label=r"$<f>$")
plt.plot(ref_x, approx_momentum, "k+", markevery=4, label=r"$<fv>$")
plt.plot(ref_x, approx_energy, "gx", markevery=4,
         label=r"$\frac{1}{2} <fv^2>$")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("macro (t = 0.1)")
plt.title("Approximate integrals of f and reference solutions")
plt.savefig("./figure/solution_f_t1.pdf")
plt.show()
plt.close()


# In[ ]:


def macro0(macro_l, macro_r, x):
    return macro_l + (macro_r - macro_l) * 0.5 * (tf.tanh(20.0 * x) + 1.0)


# In[ ]:


# plot for t = 0
ref_t = tmin * np.ones((nx, 1))
ref_t = tf.convert_to_tensor(ref_t, dtype=tf.float32)

rho_approx, u_approx, T_approx = bgk_sol.macrocall(
    inputs=tf.concat([ref_t, ref_x], axis=-1)
)
approx_density = rho_approx
approx_momentum = rho_approx * u_approx
approx_energy = 0.5 * rho_approx * (u_approx**2 + T_approx)

ref_rho = macro0(1.5, 0.625, ref_x)
ref_u = macro0(0.0, 0.0, ref_x)
ref_T = macro0(1.5, 0.75, ref_x)
ref_energy = 0.5 * ref_rho * (ref_u**2 + ref_T)

ref_momentum = ref_rho * ref_u
plt.plot(ref_x, ref_rho, "r", label="density")
plt.plot(ref_x, ref_momentum, "k", label="momentum")
plt.plot(ref_x, ref_energy, "g", label="energy")

plt.plot(ref_x, approx_density, "r*", markevery=4, label="approx density")
plt.plot(ref_x, approx_momentum, "k+", markevery=4, label="approx momentum")
plt.plot(ref_x, approx_energy, "gx", markevery=4, label="approx energy")

plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("macro (t = 0)")
plt.title("Approximate density, momentum, energy and reference solutions")
plt.savefig("./figure/solution_macro_t0.pdf")
plt.show()
plt.close()


# In[ ]:


v, w = np.polynomial.legendre.leggauss(32)
v = 0.5 * (v + 1.0) * 20 - 10
w = 0.5 * 20 * w
vquads = tf.convert_to_tensor(v[:, None], dtype=tf.float32)
wquads = tf.convert_to_tensor(w[:, None], dtype=tf.float32)

avg_f_1 = average_op(bgk_sol.fcall, [ref_t, ref_x], [vquads, wquads])
approx_density = avg_f_1
avg_f_2 = average_op(bgk_sol.fcall, [ref_t, ref_x], [vquads, wquads * vquads])
approx_momentum = avg_f_2
avg_f_3 = average_op(bgk_sol.fcall, [ref_t, ref_x], [
                     vquads, wquads * vquads**2])
approx_energy = 0.5 * avg_f_3

plt.plot(ref_x, ref_rho, "r", label="density")
plt.plot(ref_x, ref_momentum, "k", label="momentum")
plt.plot(ref_x, ref_energy, "g", label="energy")

plt.plot(ref_x, approx_density, "r*", markevery=4, label=r"$<f>$")
plt.plot(ref_x, approx_momentum, "k+", markevery=4, label=r"$<fv>$")
plt.plot(ref_x, approx_energy, "gx", markevery=4,
         label=r"$\frac{1}{2} <fv^2>$")
plt.grid()
plt.legend()
plt.xlabel("x")
plt.ylabel("macro (t = 0)")
plt.title("Approximate integrals of f and reference solutions")
plt.savefig("./figure/solution_f_t0.pdf")
plt.show()
plt.close()


# In[1]:


# jupyter notebook to python
# try:
#     get_ipython().system('jupyter nbconvert --to python example.ipynb')
# except:
#     pass


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:
