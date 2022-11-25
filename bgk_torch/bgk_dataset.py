import torch


class Sampler(object):
    def __init__(self, config, name="sampler", **kwargs):

        self.interior_samples = config["dataset_config"]["interior_samples"]
        self.boundary_samples = config["dataset_config"]["boundary_samples"]
        self.initial_samples = config["dataset_config"]["initial_samples"]

        self.t_range = config["physical_config"]["t_range"]
        self.x_range = config["physical_config"]["x_range"]
        self.v_range = config["physical_config"]["v_range"]

        # device setting
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device(
            "cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu"
        )

        # self.rate = 0.99

    def interior(self):
        t = torch.rand((self.interior_samples, 1)).to(
            self.device) * self.t_range[-1]
        x = self.x_range[0] + torch.rand((self.interior_samples, 1)).to(
            self.device) * (self.x_range[-1] - self.x_range[0])
        # x_l = self.x_range[0] + torch.rand((self.interior_samples // 2, 1)).to(
        #     self.device) * (self.x_range[-1] - 0) * self.rate
        # x_r = self.x_range[1] - torch.rand((self.interior_samples // 2, 1)).to(
        #     self.device) * (self.x_range[1] - 0) * self.rate
        # x = torch.cat([x_l, x_r], 0)
        v = self.v_range[0] + torch.rand((self.interior_samples, 1)).to(
            self.device) * (self.v_range[-1] - self.v_range[0])
        return (t, x, v)

    def boundary(self):
        t = torch.rand((self.boundary_samples, 1)).to(
            self.device) * self.t_range[-1]
        return t

    def initial(self):
        x = self.x_range[0] + torch.rand((self.initial_samples, 1)).to(
            self.device) * (self.x_range[-1] - self.x_range[0])
        # x_l = self.x_range[0] + torch.rand((self.initial_samples // 2, 1)).to(
        #     self.device) * (self.x_range[-1] - 0) * self.rate
        # x_r = self.x_range[1] - torch.rand((self.initial_samples // 2, 1)).to(
        #     self.device) * (self.x_range[1] - 0) * self.rate
        # x = torch.cat([x_l, x_r], 0)
        v = self.v_range[0] + torch.rand((self.initial_samples, 1)).to(
            self.device) * (self.v_range[-1] - self.v_range[0])
        return (x, v)
