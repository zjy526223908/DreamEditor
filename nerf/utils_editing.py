import os
import glob

import tqdm
import imageio
import open3d as o3d
import random
import pymeshlab
import tensorboardX
import numpy as np

from nerf.provider import  SphericalSamplingDataset
import time

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from rich.console import Console

from nerf.network_mesh_neus import MeshSampler

from .loss import LaplacianLoss, FlattenLoss, ARAPLoss, NormLoss, EdgeLoss


class Trainer(object):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model,  # network
                 guidance,  # guidance network
                 background_model=None,
                 opt_back=None,
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 ema_decay=None,  # if use EMA, set the decay
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=1,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

        self.name = name
        self.opt = opt
        self.opt_back = opt_back

        if not opt.test:
            self.mask_mesh = o3d.io.read_triangle_mesh(opt.prior_mask_mesh)
            self.mask_mesh_sampler = MeshSampler(self.mask_mesh, device, self.opt)

            mask_vector_info_path =  opt.mesh_mask_info
            self.mesh_info_dict = np.load(mask_vector_info_path,  allow_pickle=True).item()

            self.mask = torch.LongTensor(self.mesh_info_dict['editing_mask']).to(self.device)
            self.mask_vector_noedge = torch.LongTensor(self.mesh_info_dict['editing_mask_noedge']).to(self.device).unsqueeze(1).repeat(1, 3)
            self.mask_fea = torch.LongTensor(self.mesh_info_dict['editing_mask']).to(self.device).unsqueeze(1).repeat(1, opt.color_dim)
            self.mask_vector = torch.LongTensor(self.mesh_info_dict['editing_mask']).to(self.device).unsqueeze(1).repeat(1, 3)

        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if not opt.test:
            # guide model
            self.guidance = guidance

            if background_model:
                background_model.to(self.device)
            self.background_model = background_model
            if opt.backnerf_path:
                backnerf_checkpoint_dict = torch.load(opt.backnerf_path, map_location=self.device)
                self.background_model.load_state_dict(backnerf_checkpoint_dict['model'], strict=False)
                self.background_model.eval()

            # loss
            prior_mask_mesh = o3d.io.read_triangle_mesh(opt.prior_mask_mesh)
            center_point = torch.tensor(np.mean(np.asarray(prior_mask_mesh.vertices), axis=0)).float().to(device)


            self.norm = None
            self.laplacian_loss = None
            self.flatten_loss = None
            self.norm_loss = None
            self.edge_loss = None
            self.arap_loss = None
            self.define_mesh_loss(prior_mask_mesh, if_norm=True)

            # text prompt
            if self.guidance is not None:
                for p in self.guidance.parameters():
                    p.requires_grad = False
                self.prepare_text_embeddings()
            else:
                self.text_z = None

            if isinstance(criterion, nn.Module):
                criterion.to(self.device)
            self.criterion = criterion

            optimizer1 = lambda model: torch.optim.Adam(model.get_text_geo_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
            optimizer2 = lambda model: torch.optim.Adam(model.get_vector_only(opt.lr), betas=(0.9, 0.99), eps=1e-15)
            optimizer3 = lambda model: torch.optim.Adam(model.get_text_geo_params_only(opt.lr), betas=(0.9, 0.99), eps=1e-15)

            self.scheduler_f = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

            self.optimizer1 = optimizer1(self.model)
            self.optimizer2 = optimizer2(self.model)
            self.optimizer3 = optimizer3(self.model)



            self.lr_scheduler1 = self.scheduler_f(self.optimizer1)
            self.lr_scheduler2 = self.scheduler_f(self.optimizer2)
            self.lr_scheduler3 = self.scheduler_f(self.optimizer3)


        self.ema = None

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
        self.flag = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [],  # metrics[0], or valid_loss
            "checkpoints": [],  # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
        }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")


            self.ckpt_path = os.path.join(self.workspace, 'checkpoints')
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)

            if not opt.test:
                o3d.io.write_triangle_mesh(os.path.join(workspace, 'tmp.ply'), self.mask_mesh)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        self.log(self.opt)


        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else:  # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)


        if self.flag == 0 and self.epoch == 0:
            print('loading pre mesh net')
            self.load_checkpoint(checkpoint=self.opt.load_path, model_only=True)

        self.ori_color_features = self.model.color_features.clone().data
        self.ori_geometry_features = self.model.geometry_features.clone().data


        if self.opt.hump_mesh_dis > 0 and self.epoch==0:

            if self.opt.hump_mesh_type=='semicircle':
                t_mask = self.mask_vector_noedge[:, 0][self.mask == 1]
                dis1 = F.pairwise_distance(self.model.mesh_vertices.data[self.mask_vector_noedge[:, 0] == 1], center_point, 2)
                tmp = self.opt.hump_mesh_dis * self.opt.hump_mesh_dis - dis1 * dis1
                tmp[tmp < 0] = 0
                dis2 = tmp**0.5

                self.model.mesh_vertices_offset.data[self.mask_vector_noedge[:, 0] == 1] = dis2.reshape(-1, 1).repeat(1, 3) * self.norm[t_mask==1].cuda().clone().float()
                # self.model.mesh_vertices_offset.data[self.mask_vector_noedge[:, 0] == 1] -= 0.004 * self.norm[t_mask==1].cuda().clone().float()
            elif self.opt.hump_mesh_type=='mean_up':
                t_mask = self.mask_vector_noedge[:, 0][self.mask == 1]
                self.model.mesh_vertices_offset.data[self.mask_vector_noedge[:, 0] == 1] = \
                    self.opt.hump_mesh_dis * torch.mean(self.norm[t_mask==1], dim=0).cuda().clone().float()
            elif self.opt.hump_mesh_type=='up':
                t_mask = self.mask_vector_noedge[:, 0][self.mask == 1]
                self.model.mesh_vertices_offset.data[self.mask_vector_noedge[:, 0] == 1] = \
                    self.opt.hump_mesh_dis * self.norm.cuda().clone().float()[t_mask == 1]
            else:
                print('no this operation')
                sys.exit()

            self.smooth_mesh()
            self.save_edit_mesh()
            self.update_mask_mesh_sampler((self.model.mesh_vertices + self.model.mesh_vertices_offset))
            self.update_model_mesh_sampler((self.model.mesh_vertices + self.model.mesh_vertices_offset))
            self.save_checkpoint(full=True, best=False)


    def define_mesh_loss(self, mesh, if_norm=True):
        vertices = torch.FloatTensor(np.asarray(mesh.vertices))
        triangles = torch.LongTensor(np.asarray(mesh.triangles))
        mesh.compute_vertex_normals()
        norm = torch.FloatTensor(np.asarray(mesh.vertex_normals))
        self.norm = norm
        self.laplacian_loss = LaplacianLoss(vertices, triangles, size_factor=self.opt.size_factor).to(self.device)
        self.flatten_loss = FlattenLoss(triangles).to(self.device)
        if if_norm:
            self.norm_loss = NormLoss(norm).to(self.device)
        self.edge_loss = EdgeLoss(triangles, size_factor=self.opt.size_factor).to(self.device)

        if self.opt.lambda_arap > 0:
            self.arap_loss = ARAPLoss(vertices, triangles, size_factor=self.opt.size_factor).to(self.device)


    # calculate the text embs.
    def prepare_text_embeddings(self):

        if self.opt.text is None:
            self.log(f"[WARN] text prompt is not provided.")
            self.text_z = None
            return

        # self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])

        if not self.opt.dir_text:
            self.text_z = self.guidance.get_text_embeds([self.opt.text], [self.opt.negative])
        else:
            self.text_z = []
            for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
                # construct dir-encoded text
                text = f"{self.opt.text}, {d} view"

                negative_text = f"{self.opt.negative}"

                # explicit negative dir-encoded text
                if self.opt.negative_dir_text:
                    if negative_text != '': negative_text += ', '

                    if d == 'back':
                        negative_text += "front view"
                    elif d == 'front':
                        negative_text += "back view"
                    elif d == 'side':
                        negative_text += "front view, back view"
                    elif d == 'overhead':
                        negative_text += "bottom view"
                    elif d == 'bottom':
                        negative_text += "overhead view"

                text_z = self.guidance.get_text_embeds([text], [negative_text])
                self.text_z.append(text_z)

    def __del__(self):
        if self.log_ptr:
            self.log_ptr.close()

    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute:
                # print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr:
                print(*args, file=self.log_ptr)
                self.log_ptr.flush()  # write immediately to file

    ### ------------------------------

    def update_mask_mesh_sampler(self, vector):

        vertices = vector[self.mask == 1].detach().cpu().numpy()
        self.mask_mesh.vertices = o3d.utility.Vector3dVector(vertices)
        self.mask_mesh_sampler = MeshSampler(self.mask_mesh, self.device, self.opt)


    def update_model_mesh_sampler(self, vector):

        vertices = vector.detach().cpu().numpy()
        new_mesh_o = o3d.geometry.TriangleMesh()
        new_mesh_o.vertices = o3d.utility.Vector3dVector(vertices)
        new_mesh_o.triangles = o3d.utility.Vector3iVector(self.model.mesh_triangles.long().detach().cpu().numpy())
        self.model.mesh_sampler = MeshSampler(new_mesh_o, self.device, self.opt)


    def smooth_mesh(self):
        v_o = np.asanyarray((self.model.mesh_vertices + self.model.mesh_vertices_offset).detach().cpu().numpy())
        t_o = np.asanyarray(self.model.mesh_triangles.long().detach().cpu().numpy())

        c_f = np.zeros((t_o.shape[0], 4))
        mask = self.mask.long().detach().cpu().numpy()
        old_fix_id = []
        old_edit_id = []
        for idx, f in enumerate(t_o):
            count = mask[f[0]] + mask[f[1]] + mask[f[2]]
            if count == 3:
                c_f[idx][0] = 1
                old_edit_id.append(idx)
            else:
                old_fix_id.append(idx)
        pymesh = pymeshlab.Mesh(v_o, t_o, f_color_matrix=c_f)

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymesh)
        ms.compute_selection_by_condition_per_face(condselect='(fr == 255)')
        ms.meshing_repair_non_manifold_edges(method=0)
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=self.opt.stepsmoothnum,  selected=True)
        pymesh = ms.current_mesh()
        new_v = pymesh.vertex_matrix()

        # 更新顶点向量
        new_mesh_o = o3d.geometry.TriangleMesh()
        new_mesh_o.vertices = o3d.utility.Vector3dVector(new_v)
        new_mesh_o.triangles = o3d.utility.Vector3iVector(self.model.mesh_triangles.long().detach().cpu().numpy())
        self.model.mesh_sampler = MeshSampler(new_mesh_o, self.device, self.opt)

        # 更新顶点以及特征
        self.model.mesh_vertices_offset.data[self.mask==1] = (torch.FloatTensor(new_v).to(self.device)
              - self.model.mesh_vertices)[self.mask==1]

        torch.cuda.empty_cache()



    def train_step(self, data):

        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]
        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']


        if self.opt.if_mask:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.opt_back.fp16):
                    if self.opt.backnerf_path:
                        outputs = self.background_model.render_cuda(rays_o, rays_d)
                    else:
                        outputs = self.model.render_cuda(rays_o, rays_d)
                    rgb = outputs['image'].reshape(B, H * W, 3)
            rgb = rgb.cpu().numpy()
            rgb = torch.tensor(rgb).cuda()


        if self.opt.if_mask:
            sample_points = self.model.sample_pts_linnear(rays_o, rays_d, **vars(self.opt))

            N, R = sample_points.size()[:2]

            sample_points_dist = self.mask_mesh_sampler.get_frnn_dis_only(
                sample_points.reshape(-1, 3), self.mask_mesh_sampler.mesh_vertices, 1)

            sample_points_dist = sample_points_dist.reshape(N, R).abs()
            sample_points_dist, _ = torch.min(sample_points_dist, dim=-1)

            mask = sample_points_dist.squeeze() ** 0.5 < self.opt.mask_thresh
            select_inds = torch.arange(mask.shape[0]).to(self.device)
            mask_inds = select_inds[mask == True]

            if len(mask_inds) > self.opt.batch_rays:
                s_index = random.randint(0, len(mask_inds) - self.opt.batch_rays)
                y_mask_inds = mask_inds[s_index:s_index + self.opt.batch_rays]
                n_mask_inds = torch.cat((mask_inds[:s_index], mask_inds[s_index + self.opt.batch_rays:]))


                rays_o_y = rays_o[:, y_mask_inds, :]
                rays_d_y = rays_d[:, y_mask_inds, :]
                rays_o_n = rays_o[:, n_mask_inds, :]
                rays_d_n = rays_d[:, n_mask_inds, :]

            else:
                rays_o = rays_o[:, mask == True, :]
                rays_d = rays_d[:, mask == True, :]



        if self.opt.if_mask:
            if len(mask_inds) > self.opt.batch_rays:
                with torch.no_grad():
                    outputs_n = self.model.render(rays_o_n, rays_d_n, staged=False, perturb=True, force_all_rays=True,
                                                    **vars(self.opt))

                outputs = self.model.render(rays_o_y, rays_d_y, staged=False, perturb=True, force_all_rays=True,
                                                **vars(self.opt))
                pred_rgb = torch.ones((B, N, 3)).cuda() * 0.5
                pred_rgb[:, y_mask_inds, :] = outputs['image']
                pred_rgb[:, n_mask_inds, :] = outputs_n['image']
                pred_rgb[:, mask == False, :] = rgb[:, mask == False, :]

            else:
                outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, **vars(self.opt))

                pred_rgb = torch.ones((B, N, 3)).cuda() * 0.5
                pred_rgb[:, mask == True, :] = outputs['image']
                pred_rgb[:, mask == False, :] = rgb[:, mask == False, :]


            pred_rgb = pred_rgb.reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()  # [1, 3, H, W]
        else:
            outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True,
                                        **vars(self.opt))
            pred_rgb = outputs['image'].reshape(B, H, W, 3).permute(0, 3, 1, 2).contiguous()
            mask = torch.tensor([0])

        # text embeddings
        if self.opt.dir_text:
            dirs = data['dir']  # [B,]          print(dirs)
            text_z = self.text_z[dirs]
        else:
            text_z = self.text_z


        loss_dict = {}

        loss = self.guidance.train_step(text_z, pred_rgb, ratio = (self.global_step/self.opt.iters), guidance_scale=self.opt.guidance_scale)

        if self.opt.lambda_eikonal > 0:
            loss_eikonal = outputs['loss_eikonal']

            loss += self.opt.lambda_eikonal * loss_eikonal
            loss_dict.update({
                "loss_eikonal": loss_eikonal.item()
            })

        # occupancy loss
        pred_ws = outputs['weights_sum']  # .reshape(B, 1, H, W)

        sdf = outputs['rad_sdf']


        # print(F.pairwise_distance((self.model.mesh_vertices + self.model.mesh_vertices_offset)[self.mask == 1],
        #                           self.model.mesh_vertices[self.mask == 1], 2 ).mean().item())

        if self.opt.lambda_entropy > 0:
            alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
            alphas = alphas ** 2 # skewed entropy, favors 0 over 1
            # alphas = (1-alphas) ** 2 # skewed entropy, favors 1 over 0
            # loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()

            loss_entropy = (- (1 - alphas) * torch.log2(1 - alphas)).mean()

            loss = loss + self.opt.lambda_entropy * loss_entropy
            loss_dict.update({'entropy': loss_entropy.item()})


        return pred_rgb, pred_ws, loss, mask.sum().item(), loss_dict

    def eval_step(self, data):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        # rays, rgbs = data
        # rgbs = rgbs.cuda()
        # rays_o = rays.origins.cuda()  # [B, N, 3]
        # rays_d = rays.directions.cuda()  # [B, N, 3]
        #
        # rgbs = rgbs.reshape(-1, 3).unsqueeze(0)
        # rays_o = rays_o.reshape(-1, 3).unsqueeze(0)
        # rays_d = rays_d.reshape(-1, 3).unsqueeze(0)
        #
        # B, N = rays_o.shape[:2]
        # H, W = self.opt.H, self.opt.W

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None


        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=False, light_d=light_d, bg_color_test=1.0,
                                        ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                        **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_ws = outputs['weights_sum'].reshape(B, H, W)
        pred_normal = outputs['normal_map'].reshape(B, H, W, 3)
        # mask_ws = outputs['mask'].reshape(B, H, W) # near < far

        # loss_ws = pred_ws.sum() / mask_ws.sum()
        # loss_ws = pred_ws.mean()

        alphas = (pred_ws).clamp(1e-5, 1 - 1e-5)
        # alphas = alphas ** 2 # skewed entropy, favors 0 over 1
        loss_entropy = (- alphas * torch.log2(alphas) - (1 - alphas) * torch.log2(1 - alphas)).mean()

        loss = self.opt.lambda_entropy * loss_entropy

        return pred_rgb, pred_depth, pred_normal, loss

    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_normal = outputs['normal_map'].reshape(B, H, W, 3)

        return pred_rgb, pred_depth, pred_normal


    def test_step(self, data, bg_color=None, perturb=False):
        rays_o = data['rays_o']  # [B, N, 3]
        rays_d = data['rays_d']  # [B, N, 3]

        B, N = rays_o.shape[:2]
        H, W = data['H'], data['W']

        if bg_color is not None:
            bg_color = bg_color.to(rays_o.device)
        else:
            bg_color = torch.ones(3, device=rays_o.device)  # [3]

        shading = data['shading'] if 'shading' in data else 'albedo'
        ambient_ratio = data['ambient_ratio'] if 'ambient_ratio' in data else 1.0
        light_d = data['light_d'] if 'light_d' in data else None
        outputs = self.model.render(rays_o, rays_d, staged=True, perturb=perturb, light_d=light_d,
                                    ambient_ratio=ambient_ratio, shading=shading, force_all_rays=True,
                                    **vars(self.opt))

        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        pred_normal = outputs['normal_map'].reshape(B, H, W, 3)

        return pred_rgb, pred_depth, pred_normal

    def save_edit_mesh(self):

        mesh = o3d.geometry.TriangleMesh()

        save_path = os.path.join(self.workspace, 'mesh')
        os.makedirs(save_path, exist_ok=True)

        mesh.vertices = o3d.utility.Vector3dVector(
            (self.model.mesh_vertices + self.model.mesh_vertices_offset).data.detach().cpu().numpy())
        mesh.triangles = o3d.utility.Vector3iVector(
            self.model.mesh_triangles.data.long().detach().cpu().numpy())

        o3d.io.write_triangle_mesh(os.path.join(self.workspace, 'mesh', 'edit_vertices_mesh.ply'), mesh)


    def save_mesh(self, save_path=None, resolution=128):


        if save_path is None:
            save_path = os.path.join(self.workspace, 'mesh')

        self.log(f"==> Saving mesh to {save_path}")

        os.makedirs(save_path, exist_ok=True)

        if 'sdf' in self.opt.backbone:
            self.model.export_mesh_sdf(save_path, resolution=resolution)
        else:
            self.model.export_mesh(save_path, resolution=resolution)

        self.log(f"==> Finished saving mesh.")

    ### ------------------------------

    def train(self, train_loader, valid_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()

        for epoch in range(self.epoch + 1, max_epochs + 1):
            self.epoch = epoch

            if self.epoch > self.opt.stop_vector_epoch and self.opt.learn_vector == True:
                print('*'*100)
                print('stop vector move, smooth and subdivide mesh')
                print('*' * 100)
                self.opt.learn_vector = False

                self.update_mask_mesh_sampler((self.model.mesh_vertices + self.model.mesh_vertices_offset))
                train_loader = SphericalSamplingDataset(self.opt, device=self.device, R_path=self.opt.R_path, type='train',
                                                        H=self.opt.up_h, W=self.opt.up_w, size=100).dataloader()
                self.opt.mask_thresh = self.opt.up_mask_thresh


            self.train_one_epoch(train_loader)

            if self.opt.learn_vector:
                if self.epoch % self.opt.update_mesh_interval ==0:
                    print('update_mask_mesh_sampler')
                    if self.opt.if_smooth_mesh:
                        self.smooth_mesh()
                    self.update_mask_mesh_sampler((self.model.mesh_vertices + self.model.mesh_vertices_offset))

                    # 更新mesh loss
                    tmp_mask_mesh = o3d.io.read_triangle_mesh(os.path.join(self.workspace, 'tmp.ply'))
                    vertices = (self.model.mesh_vertices + self.model.mesh_vertices_offset)
                    vertices = vertices[self.mask == 1].detach().cpu().numpy()
                    tmp_mask_mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    o3d.io.write_triangle_mesh(os.path.join(self.workspace, 'tmp.ply'), tmp_mask_mesh)
                    new_mask_mesh = o3d.io.read_triangle_mesh(os.path.join(self.workspace, 'tmp.ply'))
                    self.define_mesh_loss(new_mask_mesh, if_norm=False)


            self.save_checkpoint(full=True, best=False)
            self.save_edit_mesh()

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None, write_video=True):

        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        os.makedirs(save_path, exist_ok=True)

        self.log(f"==> Start Test, save results to {save_path}")

        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                         bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()

        if write_video:
            all_preds = []
            all_preds_depth = []
            all_preds_normal = []
        with torch.no_grad():

            for i, data in enumerate(loader):

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal = self.test_step(data)

                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

                pred_depth = preds_depth[0].detach().cpu().numpy()
                pred_depth = (pred_depth * 255).astype(np.uint8)

                pred_normal = preds_normal[0].detach().cpu().numpy()
                pred_normal = ((pred_normal / 2 + 0.5) * 255).astype(np.uint8)

                if write_video:
                    all_preds.append(pred)
                    all_preds_depth.append(pred_depth)
                    all_preds_normal.append(pred_normal)

                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_normal.png'),
                                cv2.cvtColor(pred_normal, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'),
                                cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

                pbar.update(loader.batch_size)

        if write_video:
            all_preds = np.stack(all_preds, axis=0)
            all_preds_depth = np.stack(all_preds_depth, axis=0)
            all_preds_normal = np.stack(all_preds_normal, axis=0)

            imageio.mimwrite(os.path.join(save_path, f'{name}_normal.mp4'), all_preds_normal, fps=25, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_rgb.mp4'), all_preds, fps=25, quality=8,
                             macro_block_size=1)
            imageio.mimwrite(os.path.join(save_path, f'{name}_depth.mp4'), all_preds_depth, fps=25, quality=8,
                             macro_block_size=1)

        self.log(f"==> Finished Test.")


    def train_one_epoch(self, loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer1.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0
        print_str = ""
        for data in loader:
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            if self.local_step == 100:
                break

            loss_dict = {}

            # SDS loss
            self.optimizer1.zero_grad()
            self.optimizer3.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss, mask_sum, loss_dict_tmp = self.train_step(data)
            loss_dict.update(loss_dict_tmp)
            self.scaler.scale(loss).backward()

            if self.flag == 0 and self.opt.learn_vector:
                self.scaler.step(self.optimizer1)
                self.scaler.update()
                if self.scheduler_update_every_step:
                    self.lr_scheduler1.step()
            elif self.flag == 2 or not self.opt.learn_vector:
                self.scaler.step(self.optimizer3)
                self.scaler.update()
                if self.scheduler_update_every_step:
                    self.lr_scheduler3.step()

            # mesh reg loss
            if self.flag ==0 and self.opt.learn_vector==True:
                self.optimizer2.zero_grad()
                loss = 0

                if self.opt.lambda_arap > 0:
                    loss_arap = self.arap_loss(
                        (self.model.mesh_vertices + self.model.mesh_vertices_offset)[self.mask == 1],
                        self.model.mesh_vertices[self.mask == 1]
                    )

                    loss += self.opt.lambda_arap * loss_arap
                    loss_dict.update({'arap': 100 * loss_arap.item()})

                if self.opt.lambda_laplacian > 0:
                    loss_laplacian = self.laplacian_loss(
                        (self.model.mesh_vertices + self.model.mesh_vertices_offset)[self.mask == 1]
                    )
                    loss = loss + self.opt.lambda_laplacian * loss_laplacian
                    loss_dict.update({'laplacian': loss_laplacian.item()})

                if self.opt.lambda_norm != 0:
                    loss_norm = self.norm_loss(self.model.mesh_vertices_offset[self.mask == 1])
                    loss = loss + self.opt.lambda_norm * loss_norm
                    loss_dict.update({'norm': loss_norm.item()})

                if self.opt.lambda_edge > 0:
                    loss_edge = self.edge_loss(
                        (self.model.mesh_vertices + self.model.mesh_vertices_offset)[self.mask == 1])
                    loss = loss + self.opt.lambda_edge * loss_edge
                    loss_dict.update({'edge': loss_edge.item()})

                if self.opt.lambda_flatten > 0:
                    loss_flatten = self.flatten_loss(
                        (self.model.mesh_vertices + self.model.mesh_vertices_offset)[self.mask == 1]
                    )
                    loss = loss + self.opt.lambda_flatten * loss_flatten
                    loss_dict.update({'flatten': loss_flatten.item()})

                self.scaler.scale(loss).backward()

                self.scaler.step(self.optimizer2)
                self.scaler.update()
                if self.scheduler_update_every_step:
                    self.lr_scheduler2.step()


            total_loss += mask_sum

            self.model.color_features.data = (1 - self.mask_fea) * self.ori_color_features + \
                                             self.mask_fea * self.model.color_features.data
            self.model.geometry_features.data = (1 - self.mask_fea) * self.ori_geometry_features + \
                                                self.mask_fea * self.model.geometry_features.data


            self.model.geometry_features.data = torch.where(torch.isnan(self.model.geometry_features.data),
                                                      self.ori_geometry_features,
                                                      self.model.geometry_features.data)

            self.model.color_features.data = torch.where(torch.isnan(self.model.color_features.data),
                                                            self.ori_color_features,
                                                            self.model.color_features.data)

            if self.opt.learn_vector:
                self.model.mesh_vertices_offset.data = self.model.mesh_vertices_offset.data.clamp(
                    -self.opt.offset_thresh, self.opt.offset_thresh)
                self.model.mesh_vertices_offset.data = self.mask_vector_noedge * self.model.mesh_vertices_offset.data


            if self.scheduler_update_every_step:
                print_str = f""
                for k, v in loss_dict.items():
                    print_str += "{} : {:.4f}, ".format(k, v)

                optimizer = self.optimizer1
                if self.flag == 2 or not self.opt.learn_vector:
                    optimizer = self.optimizer3
                if self.flag == 0 and self.opt.learn_vector == True:
                    optimizer = self.optimizer2

                pbar.set_description(
                    print_str + f"mask_sum={mask_sum:.0f} ({total_loss / self.local_step:.0f}), lr={optimizer.param_groups[0]['lr']:.6f}")
            else:
                pbar.set_description(f"mask_sum={mask_sum:.0f} ({total_loss / self.local_step:.0f})")
            pbar.update(loader.batch_size)

        self.log(print_str)
        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        pbar.close()
        if self.report_metric_at_train:
            for metric in self.metrics:
                self.log(metric.report(), style="red")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="train")
                metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler1, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_schedule1.step(average_loss)
            else:
                self.lr_schedule1.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate {self.workspace} at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            for data in loader:
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, preds_normal, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size

                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in
                                  range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in
                                        range(self.world_size)]  # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:
                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_rgb.png')
                    save_path_depth = os.path.join(self.workspace, 'validation',
                                                   f'{name}_{self.local_step:04d}_depth.png')
                    save_path_normal = os.path.join(self.workspace, 'validation',
                                                    f'{name}_{self.local_step:04d}_normal.png')
                    # self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    pred = preds[0].detach().cpu().numpy()
                    pred = (pred * 255).astype(np.uint8)

                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    pred_depth = (pred_depth * 255).astype(np.uint8)

                    preds_normal = preds_normal[0].detach().cpu().numpy()
                    preds_normal = ((preds_normal / 2 + 0.5) * 255).astype(np.uint8)

                    cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_normal, cv2.cvtColor(preds_normal, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, pred_depth)

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                    pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(
                    result if self.best_mode == 'min' else - result)  # if max mode, use -result
            else:
                self.stats["results"].append(average_loss)  # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'flag': self.flag,
            'stats': self.stats,
            'mask' : self.mask.detach().cpu().numpy(),
            'mask_vector_noedge': self.mask_vector_noedge.detach().cpu().numpy(),
            'mask_fea': self.mask_fea.detach().cpu().numpy(),
            'mask_vector': self.mask_vector.detach().cpu().numpy(),
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer1.state_dict()
            state['lr_scheduler'] = self.lr_scheduler1.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()

        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{name}.pth"

            self.stats["checkpoints"].append(file_path)

            if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                old_ckpt = os.path.join(self.ckpt_path, self.stats["checkpoints"].pop(0))
                if os.path.exists(old_ckpt):
                    os.remove(old_ckpt)

            torch.save(state, os.path.join(self.ckpt_path, file_path))

        else:
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()
                    if self.ema is not None:
                        self.ema.restore()

                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")

    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)

        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict, strict=False)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)

        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")

        if self.ema is not None and 'ema' in checkpoint_dict:
            try:
                self.ema.load_state_dict(checkpoint_dict['ema'])
                self.log("[INFO] loaded EMA.")
            except:
                self.log("[WARN] failed to loaded EMA.")

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']

        if 'mask' in checkpoint_dict.keys():
            self.mask = torch.LongTensor(checkpoint_dict['mask']).to(self.device)
            self.mask_vector_noedge = torch.LongTensor(checkpoint_dict['mask_vector_noedge']).to(self.device)
            self.mask_fea = torch.LongTensor(checkpoint_dict['mask_fea']).to(self.device)
            self.mask_vector = torch.LongTensor(checkpoint_dict['mask_vector']).to(self.device)


        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        if 'flag' in checkpoint_dict.keys():
            self.flag = checkpoint_dict['flag']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")

        if self.optimizer1 and 'optimizer' in checkpoint_dict:
            try:
                self.optimizer1.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")

        if self.lr_scheduler1 and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler1.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")

        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")
