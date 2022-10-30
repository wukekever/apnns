import torch


class Sampler(object):
    def __init__(self, config, name="sampler", **kwargs):

        self.interior_samples = config["dataset_config"]["interior_samples"]
        self.boundary_samples = config["dataset_config"]["boundary_samples"]
        self.initial_samples = config["dataset_config"]["initial_samples"]

        self.z_range = config["physical_config"]["z_range"]
        self.t_range = config["physical_config"]["t_range"]
        self.x_range = config["physical_config"]["x_range"]
        self.v_range = config["physical_config"]["v_range"]
        device_ids = config["model_config"]["device_ids"]
        self.device = torch.device("cuda:{:d}".format(device_ids[0]) if torch.cuda.is_available() else "cpu")

        self.uq_dimension = config["physical_config"]["uq_dimension"]

    def interior(self):
        z = self.z_range[0] + torch.rand((self.interior_samples, self.uq_dimension)).to(self.device) * (self.z_range[-1] - self.z_range[0])
        t = torch.rand((self.interior_samples, 1)).to(self.device) * self.t_range[-1]
        x = self.x_range[0] + torch.rand((self.interior_samples, 1)).to(self.device) * (self.x_range[-1] - self.x_range[0])
        v = self.v_range[0] + torch.rand((self.interior_samples, 1)).to(self.device) * (self.v_range[-1] - self.v_range[0])
        return (z, t, x, v)

    def boundary(self):
        z = self.z_range[0] + torch.rand((self.boundary_samples, self.uq_dimension)).to(self.device) * (self.z_range[-1] - self.z_range[0])
        t = torch.rand((self.boundary_samples, 1)).to(self.device) * self.t_range[-1]
        v_l = torch.rand((self.boundary_samples, 1)).to(self.device) * self.v_range[-1]
        v_r = - torch.rand((self.boundary_samples, 1)).to(self.device) * self.v_range[-1]
        return (z, t, v_l, v_r)

    def initial(self):
        z = self.z_range[0] + torch.rand((self.initial_samples, self.uq_dimension)).to(self.device) * (self.z_range[-1] - self.z_range[0])
        x = self.x_range[0] + torch.rand((self.initial_samples, 1)).to(self.device) * (self.x_range[-1] - self.x_range[0])
        v = self.v_range[0] + torch.rand((self.initial_samples, 1)).to(self.device) * (self.v_range[-1] - self.v_range[0])
        return (z, x, v)
