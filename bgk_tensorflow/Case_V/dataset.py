import tensorflow as tf


class Sampler(object):
    def __init__(self, Config, name="sampler", **kwargs):

        self.interior_samples = Config["dataset_config"]["interior_samples"]
        self.boundary_samples = Config["dataset_config"]["boundary_samples"]
        self.initial_samples = Config["dataset_config"]["initial_samples"]

        self.t_range = Config["dataset_config"]["t_range"]
        self.x_range = Config["dataset_config"]["x_range"]
        self.v_range = Config["dataset_config"]["v_range"]

        self.rate = 1.0

    def interior(self):
        # t = tf.random.uniform((self.interior_samples, 1)) * self.t_range[-1]
        t = self.t_range[0] + tf.random.uniform(
            (self.interior_samples, 1)) * (self.t_range[-1] - self.t_range[0])


        # x = self.x_range[0] + tf.random.uniform(
        #     (self.interior_samples, 1)) * (self.x_range[-1] - self.x_range[0])
        x_l = self.x_range[0] + tf.random.uniform(
            (self.interior_samples // 2, 1)) * (0.0 - self.x_range[0]) * self.rate
        x_r = self.x_range[-1] - tf.random.uniform(
            (self.interior_samples // 2, 1)) * (self.x_range[-1] - 0.0) * self.rate
        x = tf.concat([x_l, x_r], axis=0)
        v = self.v_range[0] + tf.random.uniform(
            (self.interior_samples, 1)) * (self.v_range[-1] - self.v_range[0])
        return t, x, v

    def boundary(self):
        # t = tf.random.uniform((self.boundary_samples, 1)) * self.t_range[-1]
        t = self.t_range[0] + tf.random.uniform(
            (self.boundary_samples, 1)) * (self.t_range[-1] - self.t_range[0])
        return t

    def initial(self):
        # x = self.x_range[0] + tf.random.uniform(
        #     (self.initial_samples, 1)) * (self.x_range[-1] - self.x_range[0])
        x_l = self.x_range[0] + tf.random.uniform(
            (self.initial_samples // 2, 1)) * (0.0 - self.x_range[0]) * self.rate
        x_r = self.x_range[-1] - tf.random.uniform(
            (self.initial_samples // 2, 1)) * (self.x_range[-1] - 0.0) * self.rate
        x = tf.concat([x_l, x_r], axis=0)
        v = self.v_range[0] + tf.random.uniform(
            (self.initial_samples, 1)) * (self.v_range[-1] - self.v_range[0])
        return x, v
