import torch
import torch.nn as nn
import numpy as np
from nerf.renderer import NeRFRenderer
from nerf.base import ImplicitSurface, RadianceNet


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 num_layers=8,
                 hidden_dim=128,
                 W_geo_feat=-1,
                 variance_init=0.05,
                 speed_factor=10.0,
                 ):

        super().__init__(opt)

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor

        self.implicit_surface = ImplicitSurface(
            W_geo_feat=W_geo_feat, input_ch=3, obj_bounding_size=2.0)

        if W_geo_feat < 0:
            W_geo_feat = self.implicit_surface.W
        self.radiance_net = RadianceNet(
            W_geo_feat=W_geo_feat)


    # add a density blob to the scene center
    def gaussian(self, x):
        # x: [B, N, 3]

        d = (x ** 2).sum(-1)
        g = 5 * torch.exp(-d / (2 * 0.2 ** 2))

        return g

    def background(self, d):

        rgbs = torch.zeros(d.size(), dtype=d.dtype).to(d.device)

        return rgbs

    def forward_s(self):
        return torch.exp(self.ln_s * self.speed_factor)

    def forward(self, x, d, l=None, ratio=1, shading='albedo'):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], view direction, nomalized in [-1, 1]
        # l: [3], plane light direction, nomalized in [-1, 1]

        sdf, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiances = self.radiance_net.forward(x, d, nablas, geometry_feature)

        return sdf, radiances, nablas

    def forward_radiance(self, x: torch.Tensor, view_dirs: torch.Tensor):
        _, nablas, geometry_feature = self.implicit_surface.forward_with_nablas(x)
        radiance = self.radiance_net.forward(x, view_dirs, nablas, geometry_feature)
        return radiance

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        sdf = self.implicit_surface.forward(x)

        return {
            'sigma': sdf,
        }

    def density_with_nablas(self, x, return_dict=True, if_detach=True, has_grad_bypass: bool = None):
        sdf, nabla, h = self.implicit_surface.forward_with_nablas(x)

        if return_dict:
            return {
                'sigma': sdf,
                'nabla': nabla,
                'h': h,
            }
        return sdf, nabla, h

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.implicit_surface.parameters(), 'lr': lr},
            {'params': self.radiance_net.parameters(), 'lr': lr},
            {'params': self.ln_s, 'lr': lr},
        ]

        return params


