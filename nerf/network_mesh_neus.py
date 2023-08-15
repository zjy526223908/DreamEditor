import torch
import torch.nn as nn
from torch import autograd

from .renderer import NeRFRenderer

import numpy as np
import open3d as o3d
import frnn
from .base import ImplicitSurface_Mesh, RadianceNet_Mesh
from .encoding import get_embedder


class MeshSampler:
    def __init__(self, mesh, device, opt):

        self.mesh = mesh
        self.opt = opt
        self.device=  device

        self.mesh.compute_vertex_normals()
        self.mesh_vertices = torch.FloatTensor(np.asarray(self.mesh.vertices)).to(device)
        self.mesh_triangles = torch.FloatTensor(np.asarray(self.mesh.triangles)).to(device)
        self.vertex_normals = torch.FloatTensor(np.asarray(self.mesh.vertex_normals)).to(device)

        _, _, _, self.grid = frnn.frnn_grid_points(
            self.mesh_vertices.unsqueeze(0),
            self.mesh_vertices.unsqueeze(0),
            None,
            None,
            K=32,
            r=self.opt.far_point_thresh*2,
            grid=None,
            return_nn=False,
            return_sorted=True,
        )


    def update_normal(self, normal):
        self.vertex_normals = torch.FloatTensor(normal).float().to(self.device).contiguous()

    def compute_distance(self, xyz, if_detach, vertices, K=8, indicator_weight=0.1):

        indicator_vec = self.vertex_normals

        dis, indices, _, _ = frnn.frnn_grid_points(
            xyz.unsqueeze(0),
            vertices.unsqueeze(0),
            None,
            None,
            K=K,
            r=self.opt.far_point_thresh*2,
            grid=self.grid,
            return_nn=False,
            return_sorted=True,
        )  # (1,M,K)

        if if_detach:
            dis = dis.detach()
            indices = indices.detach()

        dis = dis.sqrt()
        weights = 1 / (dis + 1e-7)
        weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
        indices = indices.squeeze(0)  # (M, K)
        weights = weights.squeeze(0)  # (M, K)


        w1 = indicator_weight
        dir_vec = xyz.unsqueeze(-2) - vertices[indices]
        w2 = torch.norm(dir_vec, dim=-1, keepdim=True)
        middle_vec = (indicator_vec[indices] * w1 + dir_vec * w2) / (w1 + w2)
        distance = weights.unsqueeze(-1) * torch.sum(
            dir_vec * middle_vec,
            dim=-1,
            keepdim=True,
        )
        distance = torch.sum(distance, dim=-2)

        return distance, indices, weights

    def get_frnn_dis_only(
            self,
            xyz,
            vertices,
            K=8,
    ):

        dis, indices, _, _ = frnn.frnn_grid_points(
            xyz.unsqueeze(0),
            vertices.unsqueeze(0),
            None,
            None,
            K=K,
            r=self.opt.far_point_thresh*2,
            grid=self.grid,
            return_nn=False,
            return_sorted=True,
        )  # (1,M,K)

        return dis

    def compute_distance_frnn_nearpoint(
            self,
            xyz,
            if_detach,
            vertices,
            K=8,
            indicator_weight=0.1,
    ):

        indicator_vec = self.vertex_normals

        dis, indices, _, _ = frnn.frnn_grid_points(
            xyz.unsqueeze(0),
            vertices.unsqueeze(0),
            None,
            None,
            K=K,
            r=self.opt.far_point_thresh*2,
            grid=self.grid,
            return_nn=False,
            return_sorted=True,
        )  # (1,M,K)

        if if_detach:
            dis = dis.detach()
            indices = indices.detach()

        ori_dis = dis.clone()
        ori_indices = indices.clone()

        dis = dis.sqrt()
        weights = 1 / (dis + 1e-7)
        weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
        indices = indices.squeeze(0)  # (M, K)
        weights = weights.squeeze(0)  # (M, K)

        w1 = indicator_weight
        dir_vec = xyz.unsqueeze(-2) - vertices[indices]
        w2 = torch.norm(dir_vec, dim=-1, keepdim=True)
        middle_vec = (indicator_vec[indices] * w1 + dir_vec * w2) / (w1 + w2)
        distance = weights.unsqueeze(-1) * torch.sum(
            dir_vec * middle_vec,
            dim=-1,
            keepdim=True,
        )
        distance = torch.sum(distance, dim=-2)

        N_full = distance.shape[0]
        mask = distance.squeeze().abs() < self.opt.far_point_thresh
        ind_full = torch.arange(N_full).cuda()
        ind_occu = ind_full[mask]

        xyz = xyz[ind_occu]
        xyz = xyz.requires_grad_(True)

        dis = ori_dis[:, ind_occu]
        indices = ori_indices[:, ind_occu]

        dis = dis.sqrt()
        weights = 1 / (dis + 1e-7)
        weights = weights / torch.sum(weights, dim=-1, keepdims=True)  # (1,M,K)
        indices = indices.squeeze(0)  # (M, K)
        weights = weights.squeeze(0)  # (M, K)

        w1 = indicator_weight
        dir_vec = xyz.unsqueeze(-2) - vertices[indices]
        w2 = torch.norm(dir_vec, dim=-1, keepdim=True)
        middle_vec = (indicator_vec[indices] * w1 + dir_vec * w2) / (w1 + w2)
        distance = weights.unsqueeze(-1) * torch.sum(
            dir_vec * middle_vec,
            dim=-1,
            keepdim=True,
        )
        distance = torch.sum(distance, dim=-2)

        return distance, indices, weights, xyz, ind_occu




class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 opt,
                 device=torch.device('cuda'),
                 num_layers=4,
                 hidden_dim=128,
                 skips=[2],
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 variance_init=0.05,
                 speed_factor=10.0,
                 learn_indicator_weight=True
                 ):

        super().__init__(opt)

        self.opt = opt
        self.device = device

        self.mesh_sampler = MeshSampler(o3d.io.read_triangle_mesh(opt.prior_mesh), device, self.opt)
        self.ln_s = nn.Parameter(data=torch.Tensor([-np.log(variance_init) / speed_factor]), requires_grad=True)
        self.speed_factor = speed_factor
        self.far_point_thresh = opt.far_point_thresh
        self.if_direction = opt.if_direction
        self.fine_sigma_net = opt.fine_sigma_net
        self.fine_color_net = opt.fine_color_net
        self.learn_vector = opt.learn_vector

        mesh_vertices = self.mesh_sampler.mesh_vertices.float().clone()
        self.mesh_vertices = nn.Parameter(mesh_vertices)

        mesh_triangles = self.mesh_sampler.mesh_triangles.float().clone()
        self.mesh_triangles = nn.Parameter(mesh_triangles)

        mesh_vertices_offset = torch.zeros(mesh_vertices.size())
        self.mesh_vertices_offset = nn.Parameter(mesh_vertices_offset)

        # indicator weight for frnn projected distance
        self.learn_indicator_weight = learn_indicator_weight
        if self.learn_indicator_weight:
            self.indicator_weight_raw = nn.Parameter(
                torch.Tensor([-2]), requires_grad=True
            )

        num_vertices = np.asarray(self.mesh_sampler.mesh.vertices).shape[0]

        self.geometry_features = nn.Parameter(
            torch.randn(num_vertices, opt.geometry_dim, dtype=torch.float32)
        )
        self.color_features = nn.Parameter(
            torch.randn(num_vertices, opt.color_dim, dtype=torch.float32)
        )

        if self.opt.color_en:
            self.cfea_encoder, self.cfea_in_dim = get_embedder(2, opt.color_dim)
            cfea_in_dim = self.cfea_in_dim
        else:
            cfea_in_dim = opt.color_dim

        if self.opt.geometry_en:
            self.gfea_encoder, self.gfea_in_dim = get_embedder(2, opt.geometry_dim)
            gfea_in_dim = self.gfea_in_dim
        else:
            gfea_in_dim = opt.geometry_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.encoder, self.in_dim = get_embedder(8, 1)

        self.sigma_net = ImplicitSurface_Mesh(opt.sigma_net_w, opt.sigma_net_d, [], W_geo_feat=-1,
            input_ch=gfea_in_dim + self.in_dim, obj_bounding_size=1.0)

        self.dir_encoder, self.dir_in_dim = get_embedder(4, 3)

        if self.if_direction:
            self.color_net = RadianceNet_Mesh(opt.color_net_d, opt.color_net_w, [], self.dir_in_dim + cfea_in_dim + self.in_dim + 3 )
        else:
            self.color_net = RadianceNet_Mesh(opt.color_net_d, opt.color_net_w, [], cfea_in_dim + self.in_dim + 3)


    def forward_s(self):
        # print(self.ln_s)

        return torch.exp(self.ln_s * self.speed_factor)

    def forward_indicator_weight(self):
        return torch.sigmoid(self.indicator_weight_raw)

    def compute_distance(self, xyz, if_detach=True):
        _ds, _indices, _weights = self.mesh_sampler.compute_distance(
            xyz, if_detach, self.mesh_vertices + self.mesh_vertices_offset
        )
        _ds = _ds.reshape(*xyz.shape[:-1], -1)
        _indices = _indices.reshape(*xyz.shape[:-1], -1)
        _weights = _weights.reshape(*xyz.shape[:-1], -1)
        return _ds, _indices, _weights

    def interpolation(
            self, features: torch.Tensor, indices: torch.Tensor, weights: torch.Tensor
    ):
        fg = torch.sum(features[indices] * weights.unsqueeze(-1), dim=-2)
        return fg

    def forward(self, x, d, return_ind_occu = False):

        sigma, nablas, geometry_feature, ind_occu = self.density_with_nablas(x, return_dict=False)

        ds, indices, weights = self.compute_distance(x)

        mask = ds.squeeze().abs() < self.far_point_thresh

        fea_d = self.encoder(ds)
        fea_c = self.interpolation(self.color_features, indices, weights)
        if self.opt.color_en:
            fea_c = self.cfea_encoder(fea_c)

        if self.if_direction:
            fea_view = self.dir_encoder(d)
            c_input = torch.cat([fea_d, fea_view, fea_c, nablas], dim=-1)
        else:
            c_input = torch.cat([fea_d, fea_c, nablas], dim=-1)

        h_albedo, _ = self.color_net(c_input)

        albedo = h_albedo
        # albedo = torch.sigmoid(h_albedo)
        if return_ind_occu:
            return sigma, albedo, nablas, ind_occu
        return sigma, albedo, nablas

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        x = x.float()
        ds, indices, weights = self.compute_distance(x)

        N_full = ds.shape[0]
        mask = ds.squeeze().abs() < self.far_point_thresh
        ind_full = torch.arange(N_full).cuda()
        ind_occu = ind_full[mask]
        ds = ds[ind_occu]
        indices = indices[ind_occu]
        weights = weights[ind_occu]

        fea_d = self.encoder(ds)
        fea_g = self.interpolation(self.geometry_features, indices, weights)

        if self.opt.geometry_en:
            fea_g = self.gfea_encoder(fea_g)

        # print(fea_d.size(), fea_g.size())
        # sys.exit()

        sigma, h = self.sigma_net(torch.cat([fea_d, fea_g], dim=-1))

        sigma_full = torch.ones((N_full), device=sigma.device, dtype=sigma.dtype)
        sigma_full[ind_occu] = sigma

        return {
            'sigma': sigma_full,
            'sdf_mask': mask,
        }

    def density_mesh(self, x):
        # x: [N, 3], in [-bound, bound]

        ds, indices, weights = self.compute_distance(x)

        # N_full = ds.shape[0]
        # mask = ds.squeeze().abs() < self.far_point_thresh
        # ind_full = torch.arange(N_full).cuda()
        # ind_occu = ind_full[mask]

        fea_d = self.encoder(ds)
        fea_g = self.interpolation(self.geometry_features, indices, weights)
        if self.opt.geometry_en:
            fea_g = self.gfea_encoder(fea_g)
        sigma, h = self.sigma_net(torch.cat([fea_d, fea_g], dim=-1))

        # sigma_full = -torch.ones((N_full, 1), device=sigma.device, dtype=sigma.dtype)
        # sigma_full[mask] = sigma[mask]

        return {
            'sigma': ds,
        }


    def density_with_nablas(self, x, return_dict=True, if_detach=True, has_grad_bypass: bool = None):
        x = x.float()
        has_grad = torch.is_grad_enabled() if has_grad_bypass is None else has_grad_bypass

        with torch.enable_grad():
            # x = x.requires_grad_(True)
            N_full =  x.shape[0]
            ds, indices, weights, xyz, ind_occu = self.mesh_sampler.compute_distance_frnn_nearpoint(
                x, if_detach, self.mesh_vertices + self.mesh_vertices_offset, self.opt.K)

            fea_d = self.encoder(ds)

            fea_g = self.interpolation(self.geometry_features, indices, weights)

            if self.opt.geometry_en:
                fea_g = self.gfea_encoder(fea_g)

            sigma, h = self.sigma_net(torch.cat([fea_d, fea_g], dim=-1))

            nabla = autograd.grad(
                sigma,
                xyz,
                torch.ones_like(sigma, requires_grad=False, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]

        sigma_full = torch.ones((N_full), device=sigma.device, dtype=sigma.dtype)
        nabla_full = torch.zeros((N_full, nabla.size()[1]), device=nabla.device, dtype=nabla.dtype)
        h_full = torch.ones((N_full, h.size()[1]), device=h.device, dtype=h.dtype)

        sigma_full[ind_occu] = sigma
        nabla_full[ind_occu] = nabla
        h_full[ind_occu] = h


        if not has_grad:
            sigma_full = sigma_full.detach()
            nabla_full = nabla_full.detach()
            h_full = h_full.detach()
        if return_dict:
            return {
                'sigma': sigma_full,
                'nabla': nabla_full,
                'h': h_full,
                'ind_occu': ind_occu,
            }
        return sigma_full, nabla_full, h_full, ind_occu

    def forward_radiance(self, x, d, return_sdf=False):
        x = x.float()
        d = d.float()
        N_full = x.shape[0]
        has_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            # x = x.requires_grad_(True)
            ds, indices, weights, xyz, ind_occu = self.mesh_sampler.compute_distance_frnn_nearpoint(
                x, True, self.mesh_vertices + self.mesh_vertices_offset, self.opt.K)
            fea_d = self.encoder(ds)
            fea_g = self.interpolation(self.geometry_features, indices, weights)
            if self.opt.geometry_en:
                fea_g = self.gfea_encoder(fea_g)
            sigma, h = self.sigma_net(torch.cat([fea_d, fea_g], dim=-1))

            nabla = autograd.grad(
                sigma,
                xyz,
                torch.ones_like(sigma, requires_grad=False, device=x.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True)[0]

        fea_d = self.encoder(ds)
        fea_c = self.interpolation(self.color_features, indices, weights)
        if self.opt.color_en:
            fea_c = self.cfea_encoder(fea_c)

        if self.if_direction:
            fea_view = self.dir_encoder(d[ind_occu])
            c_input = torch.cat([fea_d, fea_view, fea_c, nabla], dim=-1)
        else:
            c_input = torch.cat([fea_d, fea_c, nabla], dim=-1)
        h_albedo, _ = self.color_net(c_input)
        albedo = torch.zeros((N_full, h_albedo.size()[1]), device=h_albedo.device, dtype=h_albedo.dtype)
        albedo[ind_occu] = h_albedo

        if return_sdf:
            sigma_full = torch.ones((N_full), device=sigma.device, dtype=sigma.dtype)
            nabla_full = torch.zeros((N_full, nabla.size()[1]), device=nabla.device, dtype=nabla.dtype)
            sigma_full[ind_occu] = sigma
            nabla_full[ind_occu] = nabla

            return sigma_full, albedo, nabla_full, ind_occu
        else:
            return albedo, ind_occu

    def background(self, d):

        rgbs = torch.zeros(d.size(), dtype=d.dtype).to(d.device)

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.color_features, 'lr': lr},
            {'params': self.geometry_features, 'lr': lr},
        ]

        if self.opt.fine_ln:
            params.append({'params': self.ln_s, 'lr': 0.1*lr})

        if self.fine_color_net:
            params.append({'params': self.color_net.parameters(), 'lr': lr})

        if self.fine_sigma_net:
            params.append({'params': self.sigma_net.parameters(), 'lr': 0.1*lr})


        return params

    def get_params_diss(self, lr):

        params = [
            {'params': self.color_features, 'lr': lr},
            {'params': self.geometry_features, 'lr': lr},
        ]

        if self.fine_color_net:
            params.append({'params': self.color_net.parameters(), 'lr': 0.01*lr})

        if self.opt.fine_sigma_net:
            params.append({'params': self.sigma_net.parameters(), 'lr': lr})

        if self.opt.fine_ln:
            params.append({'params': self.ln_s, 'lr': lr})

        return params

    def get_params_geo(self, lr):

        params = [
            {'params': self.geometry_features, 'lr': lr},
        ]

        if self.opt.fine_sigma_net:
            params.append({'params': self.sigma_net.parameters(), 'lr': lr})

        if self.opt.fine_ln:
            params.append({'params': self.ln_s, 'lr': lr})

        return params

    def get_text_params(self, lr):

        params = [
            {'params': self.color_features, 'lr': lr},
        ]

        if self.fine_color_net:
            params.append({'params': self.color_net.parameters(), 'lr': lr})

        if self.opt.fine_sigma_net:
            params.append({'params': self.sigma_net.parameters(), 'lr': lr})

        if self.opt.fine_geo:
            params.append({'params': self.geometry_features, 'lr': lr})

        return params

    def get_text_geo_params(self, lr):

        params = [
            {'name': 'color_features','params': self.color_features, 'lr': lr*self.opt.learn_color_weight},

        ]

        if not self.opt.fix_geometry_features:
            print('train geometry_features')
            params.append(  {'name': 'geometry_features', 'params': self.geometry_features, 'lr': lr*self.opt.learn_sigma_weight})


        print("learn_vector_weight, {}".format(self.opt.learn_vector_weight))
        if self.learn_vector:
            params.append({'name': 'mesh_vertices_offset', 'params': self.mesh_vertices_offset,
                           'lr': lr*self.opt.learn_vector_weight})


        return params


    def get_mesh_vertices_offset(self, lr):
        params = []
        params.append({'name': 'mesh_vertices_offset', 'params': self.mesh_vertices_offset,
                       'lr':  lr })
        return params

    def get_text_geo_params_only(self, lr):
        params = [
            {'name': 'color_features', 'params': self.color_features, 'lr': lr * self.opt.learn_color_weight},
        ]
        if not self.opt.fix_geometry_features:
            print('train geometry_features')
            params.append(  {'name': 'geometry_features', 'params': self.geometry_features, 'lr': lr*self.opt.learn_sigma_weight})
        return params

    def get_vector_only(self, lr):
        params = [
            {'name': 'mesh_vertices_offset', 'params': self.mesh_vertices_offset, 'lr': lr * self.opt.learn_vector_weight}
        ]
        return params

