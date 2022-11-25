import tensorflow as tf
from tensorflow import keras


class Solver(keras.Model):
    def __init__(
        self,
        sol,
        equation,
        trainloader,
        regularizers,
        ref,
        name="solver",
        **kwargs
    ):
        super(Solver, self).__init__(name=name, **kwargs)
        self.sol = sol
        self.equation = equation
        self.trainloader = trainloader
        self.regularizers = regularizers
        self.ref = ref

    def compile(self, optimizer):
        super(Solver, self).compile(optimizer=optimizer)

    def train_step(self):

        trainloader_interior = self.trainloader[0]
        trainloader_boundary = self.trainloader[1]
        trainloader_initial = self.trainloader[2]

        with tf.GradientTape() as tape:

            res = self.equation.residual(
                sol=self.sol, inputs=trainloader_interior)
            res_bgk = res["equation"]
            res_conservation_1, res_conservation_2, res_conservation_3 = res["conservation"]
            res_relax_1, res_relax_2, res_relax_3 = res["relaxation"]

            res_boundary = self.equation.bc(
                sol=self.sol, inputs=trainloader_boundary)
            res_rho_l, res_u_l, res_T_l = res_boundary["bc_left"]
            res_rho_r, res_u_r, res_T_r = res_boundary["bc_right"]

            res_init = self.equation.ic(
                sol=self.sol, inputs=trainloader_initial
            )
            res_rho0, res_u0, res_T0, res_f0 = res_init["macro"]

            res_eqn_1 = tf.reduce_mean(res_bgk ** 2)

            res_eqn_2_1 = tf.reduce_mean(res_conservation_1 ** 2)
            res_eqn_2_2 = tf.reduce_mean(res_conservation_2 ** 2)
            res_eqn_2_3 = tf.reduce_mean(res_conservation_3 ** 2)

            res_eqn_3_1 = tf.reduce_mean(res_relax_1 ** 2)
            res_eqn_3_2 = tf.reduce_mean(res_relax_2 ** 2)
            res_eqn_3_3 = tf.reduce_mean(res_relax_3 ** 2)

            res_eqn = (
                self.regularizers[0] * res_eqn_1
                + (self.regularizers[1] * res_eqn_2_1 + self.regularizers[2]
                   * res_eqn_2_2 + self.regularizers[3] * res_eqn_2_3)
                + (self.regularizers[4] * res_eqn_3_1 + self.regularizers[5]
                   * res_eqn_3_2 + self.regularizers[6] * res_eqn_3_3)
            )

            res_bc_1 = tf.reduce_mean(res_rho_l ** 2) + tf.reduce_mean(
                res_rho_r ** 2
            )
            res_bc_2 = tf.reduce_mean(res_u_l ** 2) + tf.reduce_mean(
                res_u_r ** 2
            )
            res_bc_3 = tf.reduce_mean(res_T_l ** 2) + tf.reduce_mean(
                res_T_r ** 2
            )
            res_bc = (
                self.regularizers[7] * res_bc_1
                + self.regularizers[8] * res_bc_2
                + self.regularizers[9] * res_bc_3
            )

            res_ic_1 = tf.reduce_mean(res_rho0 ** 2)
            res_ic_2 = tf.reduce_mean(res_u0 ** 2)
            res_ic_3 = tf.reduce_mean(res_T0 ** 2)
            res_ic_4 = tf.reduce_mean(res_f0 ** 2)

            res_ic = self.regularizers[10] * res_ic_1 + \
                self.regularizers[11] * res_ic_2 + \
                self.regularizers[12] * res_ic_3 + \
                self.regularizers[13] * res_ic_4

            loss = res_eqn + res_bc + res_ic

        risk = {}
        risk.update({"total_loss": loss})
        risk.update({"bgk": res_eqn_1})
        risk.update({"conservation": (res_eqn_2_1, res_eqn_2_2, res_eqn_2_3)})
        risk.update({"relaxation": (res_eqn_3_1, res_eqn_3_2, res_eqn_3_3)})
        risk.update({"bc_rho": res_bc_1})
        risk.update({"bc_u": res_bc_2})
        risk.update({"bc_T": res_bc_3})
        risk.update({"ic_rho": res_ic_1})
        risk.update({"ic_u": res_ic_2})
        risk.update({"ic_T": res_ic_3})
        risk.update({"ic_f": res_ic_4})

        gradients = tape.gradient(loss, self.trainable_variables)
        # Update weights
        self.optimizer.apply_gradients(
            zip(gradients, self.trainable_variables))

        return risk

    def test_step(self):
        err_rho, err_momentum, err_energy = self.equation.val(
            sol=self.sol, ref=self.ref
        )
        return err_rho, err_momentum, err_energy
