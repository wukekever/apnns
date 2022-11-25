import tensorflow as tf
from tensorflow.keras import layers, Sequential
from math import pi

pi = tf.cast(pi, dtype=float)

# 1. ResNet
# Residual block


class ResidualBlock(layers.Layer):
    def __init__(
        self, units, activation, kernel_initializer, name="residual_block", **kwargs
    ):
        super(ResidualBlock, self).__init__()
        self._units = units
        self._activation = activation
        self._kernel_initializer = kernel_initializer

        self.layers = [
            layers.Dense(
                unit,
                activation=self._activation,
                kernel_initializer=self._kernel_initializer,
                dtype="float32",
            )
            for unit in self._units
        ]

    def call(self, inputs):
        residual = inputs
        for layer in self.layers:
            inputs = layer(inputs)

        residual += inputs

        return residual


class Sol_ResNet(layers.Layer):
    def __init__(
        self,
        units_f,
        units_rho,
        units_u,
        units_T,
        activation,
        kernel_initializer,
        name="bgk_sol",
        **kwargs
    ):
        super(Sol_ResNet, self).__init__()
        self._units_f = units_f
        self._units_rho = units_rho
        self._units_u = units_u
        self._units_T = units_T
        self._activation = activation
        self._kernel_initializer = kernel_initializer

    def build(self):
        # nn for f(t, x, v)
        self.f_in_layer = layers.Dense(
            units=self._units_f[0],
            kernel_initializer=self._kernel_initializer,
            dtype="float32",
        )

        self.f_residual_blocks = Sequential(
            [
                ResidualBlock(
                    units=self._units_f[i : i + 2],
                    activation=self._activation,
                    kernel_initializer=self._kernel_initializer,
                )
                for i in range(1, len(self._units_f) - 1, 2)
            ]
        )

        self.f_out_layer = layers.Dense(1)

        # nn for rho(t, x)
        self.rho_in_layer = layers.Dense(
            self._units_rho[0],
            kernel_initializer=self._kernel_initializer,
            dtype="float32",
        )

        self.rho_residual_blocks = Sequential(
            [
                ResidualBlock(
                    units=self._units_rho[i : i + 2],
                    activation=self._activation,
                    kernel_initializer=self._kernel_initializer,
                )
                for i in range(1, len(self._units_rho) - 1, 2)
            ]
        )

        self.rho_out_layer = layers.Dense(1)

        # nn for u(t, x)
        self.u_in_layer = layers.Dense(
            self._units_u[0],
            kernel_initializer=self._kernel_initializer,
            dtype="float32",
        )

        self.u_residual_blocks = Sequential(
            [
                ResidualBlock(
                    units=self._units_u[i : i + 2],
                    activation=self._activation,
                    kernel_initializer=self._kernel_initializer,
                )
                for i in range(1, len(self._units_u) - 1, 2)
            ]
        )

        self.u_out_layer = layers.Dense(1)

        # nn for T(t, x)
        self.T_in_layer = layers.Dense(
            self._units_T[0],
            kernel_initializer=self._kernel_initializer,
            dtype="float32",
        )

        self.T_residual_blocks = Sequential(
            [
                ResidualBlock(
                    units=self._units_T[i : i + 2],
                    activation=self._activation,
                    kernel_initializer=self._kernel_initializer,
                )
                for i in range(1, len(self._units_T) - 1, 2)
            ]
        )

        self.T_out_layer = layers.Dense(1)

    # inputs shape: [None, dims = 3]

    def fcall(self, inputs):
        t, x, v = inputs[..., 0:1], inputs[..., 1:2], inputs[..., 2:]
        t = t / 0.1
        normal_v = tf.cast(v, dtype=float) / 10.0  # normalization for v: [-1, 1]

        inputs = tf.concat([t, x, normal_v], axis=-1)

        f = self.f_in_layer(inputs)

        f = self.f_residual_blocks(f)

        # f = tf.exp(self.f_out_layer(f)) * tf.exp(-normal_v**2 / 25.0)
        f = tf.math.log(
            1.0 + tf.exp(self.f_out_layer(f))
        )  # * tf.exp(-(normal_v ** 2) / 25.0)

        return f

    # inputs shape: [None, dims = 2]
    def macrocall(self, inputs):
        t, x = inputs[..., 0:1], inputs[..., 1:2]
        t = t / 0.1
        indicator_1 = tf.cast((0.5 - x), dtype=float)
        indicator_2 = tf.cast((0.5 + x), dtype=float)

        inputs = tf.concat([t, x], axis=-1)

        rho = self.rho_in_layer(inputs)

        rho = self.rho_residual_blocks(rho)

        # exact
        rho = tf.exp(
            tf.math.log(1.5) * indicator_1
            + tf.math.log(0.625) * indicator_2
            + indicator_1 * indicator_2 * self.rho_out_layer(rho)
        )

        # # not exact
        # rho = tf.exp(
        #     self.rho_out_layer(rho)
        # )

        u = self.u_in_layer(inputs)

        u = self.u_residual_blocks(u)

        # exact
        u = (indicator_1 * indicator_2) ** 0.5 * self.u_out_layer(
            u
        )  # * tf.cast(t, dtype=float)
        # u = (indicator_1 * indicator_2)**0.5 * tf.exp(self.u_out_layer(u))

        # # not exact
        # u = self.u_out_layer(u)

        _T = self.T_in_layer(inputs)

        _T = self.T_residual_blocks(_T)

        # exact
        _T = tf.exp(
            tf.math.log(1.5) * indicator_1
            + tf.math.log(0.75) * indicator_2
            + indicator_1 * indicator_2 * self.T_out_layer(_T)
        )

        # # not exact
        # _T = tf.exp(
        #     self.T_out_layer(_T)
        # )

        return rho, u, _T

    def get_config(self):
        config = {
            "units_f": self.units_f,
            "units_rho": self.units_rho,
            "units_u": self.units_u,
            "units_T": self.units_T,
            "activation": self._activation,
            "kernel_initializer": self.kernel_initializer,
        }
        base_config = super(Sol_ResNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_weights(self):
        weights = super(Sol_ResNet, self).get_weights()
        return weights

    def set_weights(self, weights):
        weights = super(Sol_ResNet, self).set_weights(weights)
        return weights


# 2. Fully-connected net or Feedforward neural network


class Sol_FCNet(layers.Layer):
    def __init__(
        self,
        units_f,
        units_rho,
        units_u,
        units_T,
        activation,
        kernel_initializer,
        name="bgk_sol",
        **kwargs
    ):
        super(Sol_FCNet, self).__init__()
        self._units_f = units_f
        self._units_rho = units_rho
        self._units_u = units_u
        self._units_T = units_T
        self._activation = activation
        self._kernel_initializer = kernel_initializer

    def build(self):
        self.f_dense_layers = [
            layers.Dense(
                unit,
                activation=self._activation,
                kernel_initializer=self._kernel_initializer,
                dtype="float32",
            )
            for unit in self._units_f
        ]
        self.f_out_layer = layers.Dense(1)

        self.rho_dense_layers = [
            layers.Dense(
                unit,
                activation=self._activation,
                kernel_initializer=self._kernel_initializer,
                dtype="float32",
            )
            for unit in self._units_rho
        ]
        self.rho_out_layer = layers.Dense(1)

        self.u_dense_layers = [
            layers.Dense(
                unit,
                activation=self._activation,
                kernel_initializer=self._kernel_initializer,
                dtype="float32",
            )
            for unit in self._units_u
        ]
        self.u_out_layer = layers.Dense(1)

        self.T_dense_layers = [
            layers.Dense(
                unit,
                activation=self._activation,
                kernel_initializer=self._kernel_initializer,
                dtype="float32",
            )
            for unit in self._units_T
        ]
        self.T_out_layer = layers.Dense(1)

    # inputs shape: [None, dims = 3]

    def fcall(self, inputs):

        t, x, v = inputs[..., 0:1], inputs[..., 1:2], inputs[..., 2:]
        # t = t / 0.1
        # normalization for v: [-1, 1]
        normal_v = tf.cast(v, dtype=float) / 1.0
        f = tf.concat([t, x, normal_v], axis=-1)
        for layer in self.f_dense_layers:
            f = layer(f)

        # f = tf.exp(self.f_out_layer(f))

        f = tf.exp(self.f_out_layer(f)) * tf.exp(-(normal_v**2) / 25.0)

        return f

    # inputs shape: [None, dims = 2]
    def macrocall(self, inputs):
        t, x = inputs[..., 0:1], inputs[..., 1:2]
        # t = t / 0.1
        indicator_1 = tf.cast((0.5 - x), dtype=float)
        indicator_2 = tf.cast((0.5 + x), dtype=float)

        inputs = tf.concat([t, x], axis=-1)

        rho = inputs
        for layer in self.rho_dense_layers:
            rho = layer(rho)

        # # exact
        # rho = tf.exp(
        #     tf.math.log(1.5) * indicator_1
        #     + tf.math.log(0.625) * indicator_2
        #     + indicator_1 * indicator_2 * self.rho_out_layer(rho)
        # )

        # not exact
        rho = tf.exp(self.rho_out_layer(rho))

        u = inputs
        for layer in self.u_dense_layers:
            u = layer(u)

        # # exact
        # u = (indicator_1 * indicator_2)**0.5 * self.u_out_layer(u)

        # not exact
        u = self.u_out_layer(u)

        _T = inputs
        for layer in self.T_dense_layers:
            _T = layer(_T)

        # # exact
        # _T = tf.exp(
        #     tf.math.log(1.5) * indicator_1
        #     + tf.math.log(0.75) * indicator_2
        #     + indicator_1 * indicator_2 * self.T_out_layer(_T)
        # )

        # not exact
        _T = tf.exp(self.T_out_layer(_T))

        return rho, u, _T

    def get_config(self):
        config = {
            "units_f": self.units_f,
            "units_rho": self.units_rho,
            "units_u": self.units_u,
            "units_T": self.units_T,
            "activation": self._activation,
            "kernel_initializer": self.kernel_initializer,
        }
        base_config = super(Sol_FCNet, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_weights(self):
        weights = super(Sol_FCNet, self).get_weights()
        return weights

    def set_weights(self, weights):
        weights = super(Sol_FCNet, self).set_weights(weights)
        return weights
