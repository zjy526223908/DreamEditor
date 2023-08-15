import os

import tqdm
import tensorboardX
import open3d as o3d
import numpy as np

import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from nerf.network_neus import NeRFNetwork as NeusNetwork
from nerf.network_mesh_neus import NeRFNetwork as MeshNeusNetwork
from rich.console import Console


from nerf.utils_init_nerf import Trainer_Nerf

class Trainer_Diss(Trainer_Nerf):
    def __init__(self,
                 name,  # name of this experiment
                 opt,  # extra conf
                 model:MeshNeusNetwork,  # network
                 model_teacher:NeusNetwork,
                 criterion=None,  # loss function, if None, assume inline implementation in train_step
                 optimizer=None,  # optimizer
                 ema_decay=None,  # if use EMA, set the decay
                 lr_scheduler=None,  # scheduler
                 metrics=[],
                 # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0,  # which GPU am I
                 world_size=1,  # total num of GPUs
                 device=None,  # device to use, usually setting to None is OK. (auto choose device)
                 mute=False,  # whether to mute all print
                 fp16=False,  # amp optimize level
                 eval_interval=1,  # eval once every $ epoch
                 max_keep_ckpt=5,  # max num of saved ckpts in disk
                 workspace='workspace',  # workspace to save logs & ckpts
                 best_mode='min',  # the smaller/larger result, the better
                 use_loss_as_metric=True,  # use loss as the first metric
                 report_metric_at_train=False,  # also report metrics at training
                 use_checkpoint="latest",  # which ckpt to use at init time
                 use_tensorboardX=True,  # whether to use tensorboard for logging
                 scheduler_update_every_step=False,  # whether to call scheduler.step() after every train step
                 ):

        self.name = name
        self.opt = opt
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
        self.device = device if device is not None else torch.device(
            f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        model_teacher.to(self.device)
        self.model_teacher = model_teacher

        print('load teacher model from {}'.format(self.opt.teacher_path))
        checkpoint_dict = torch.load(self.opt.teacher_path, map_location=self.device)
        self.model_teacher.load_state_dict(checkpoint_dict['model'], strict=False)

        if self.opt.prenet_path:
            print('load color sigma model from {}'.format(self.opt.prenet_path))
            checkpoint_dict = torch.load(self.opt.prenet_path, map_location=self.device)
            self.model.load_state_dict(checkpoint_dict['model'], strict=False)

        mesh = o3d.io.read_triangle_mesh(self.opt.prior_mesh)
        mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        self.scene = o3d.t.geometry.RaycastingScene()
        _ = self.scene.add_triangles(mesh)


        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion
        self.img2mse = lambda x, y : torch.mean((x - y) ** 2)

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4)  # naive adam
        else:
            self.optimizer = optimizer(self.model)


        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1)  # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)

        self.ema = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 0
        self.global_step = 0
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

        self.log(self.opt)

        self.log(
            f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

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

    def train(self, train_loader, sample_loader, valid_loader, max_epochs):

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))

        start_t = time.time()
        self.save_checkpoint(full=True, best=False)

        for epoch in range(self.epoch + 1, max_epochs + 1):

            self.epoch = epoch

            self.train_one_epoch(train_loader, sample_loader)

            if self.epoch % self.eval_interval == 0:
                self.save_checkpoint(full=True, best=False)

            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=True, best=False)

        end_t = time.time()

        self.log(f"[INFO] training takes {(end_t - start_t) / 60:.4f} minutes.")

        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()


    def train_one_epoch(self, loader, sample_loader):
        self.log(
            f"==> Start Training {self.workspace} Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size,
                             bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data ,sample_data in zip(loader, sample_loader):

            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % self.opt.update_extra_interval == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    if 'sdf' in self.opt.backbone:
                        self.model.update_extra_state_sdf()
                    else:
                        self.model.update_extra_state()

            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                pred_rgbs, pred_ws, loss,  loss_dict = self.train_step(data)
                loss_sample, loss_dict_sample = self.train_sample_step(sample_data)

                loss = loss + loss_sample
                loss_dict.update(loss_dict_sample)


            self.scaler.scale(loss).backward()

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss_dict['loss_c']
            total_loss += loss_val

            if self.local_rank == 0:

                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    print_str = f""
                    for k, v in loss_dict.items():
                        print_str += "{} : {:.4f}, ".format(k, v)

                    pbar.set_description(
                        print_str+f" ({total_loss / self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss / self.local_step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}. average_loss {average_loss}")


    def train_step(self, data):

        # data:
        rgbs, mask, rays_o, rays_d, H, W  = data

        if not self.opt.if_data_cuda:
            rgbs = rgbs.to(self.device)
            mask = mask.to(self.device)
            rays_o = rays_o.to(self.device)
            rays_d = rays_d.to(self.device)

        if self.opt.if_mask:
            select_inds = torch.arange(rays_o.shape[1]).to(self.device)
            mask_pos_inds = select_inds[mask.reshape(-1) == True]
            mask_neg_inds = select_inds[mask.reshape(-1) == False]

            select_inds_pos = np.random.choice(mask_pos_inds.cpu().numpy(), size=[int(self.opt.batch_rays* 1.0)], replace=False)

            if mask_neg_inds.size()[0] > int(self.opt.batch_rays * 1.0):
                select_inds_neg = np.random.choice(mask_neg_inds.cpu().numpy(), size=[int(self.opt.batch_rays * 1.0)],
                                                   replace=False)
                select_inds = np.concatenate((select_inds_pos, select_inds_neg))
            else:
                select_inds = select_inds_pos
        else:
            select_inds = np.random.choice(rays_o.shape[1], size=[self.opt.batch_rays], replace=False)

        rgbs = rgbs[:, select_inds, :]
        rays_o = rays_o[:, select_inds, :]
        rays_d = rays_d[:, select_inds, :]

        B, N = rays_o.shape[:2]

        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image']

        pred_sdf = outputs['rad_sdf']
        xyzs = outputs['xyzs']
        ind_occu = outputs['ind_occu']

        mask_volume = outputs['weights_sum'].reshape(B, N)
        mask_volume = torch.clamp(mask_volume, 1e-5, 1 - 1e-5)

        with torch.no_grad():
            t_sdf = self.model_teacher.density(xyzs.reshape(-1, 3)[ind_occu])['sigma']

        loss_sdf = F.l1_loss(pred_sdf.reshape(-1)[ind_occu], t_sdf.reshape(-1), reduction='mean')

        pred_norm = outputs['normal']
        pred_norm = torch.norm(pred_norm, dim=-1).reshape(-1)
        pred_norm = pred_norm[pred_norm > 0]
        loss_eikonal = F.mse_loss(pred_norm, pred_norm.new_ones(pred_norm.shape), reduction='mean')

        if self.opt.if_mask:
            mask = mask.reshape(B, -1)
            select_inds = torch.tensor(select_inds).reshape(B, -1).to(self.device)
            target_mask = torch.gather(mask, 1, select_inds)

            loss_m = F.mse_loss(mask_volume, target_mask, reduction='mean')


            if mask_neg_inds.size()[0] > int(self.opt.batch_rays * 1.0):
                loss_c = F.mse_loss(pred_rgb.reshape(-1, 3)[:int(N/2)], rgbs.reshape(-1, 3)[:int(N/2)], reduction='mean')
            else:
                loss_c = F.mse_loss(pred_rgb.reshape(-1, 3), rgbs.reshape(-1, 3), reduction='mean')

            loss = loss_c + loss_sdf + 0.1 * loss_m + self.opt.lambda_eikonal * loss_eikonal

            loss_dict = {
                'loss_c': loss_c.item(),
                'loss_m': loss_m.item(),
                'loss_sdf': loss_sdf.item(),
                'loss_eikonal': loss_eikonal.item()
            }

            return pred_rgb, mask_volume, loss, loss_dict
        else:
            loss_c = F.mse_loss(pred_rgb.reshape(-1, 3), rgbs.reshape(-1, 3), reduction='mean')
            loss = loss_c + loss_sdf + self.opt.lambda_eikonal * loss_eikonal

            loss_dict = {
                'loss_c': loss_c.item(),
                'loss_sdf': loss_sdf.item(),
                'loss_eikonal': loss_eikonal.item()
            }

            return pred_rgb, mask_volume, loss, loss_dict

    def train_sample_step(self, data):

        rays_o_all = data['rays_o']  # [B, N, 3]
        rays_d_all = data['rays_d']  # [B, N, 3]
        B, N = rays_o_all.shape[:2]

        rays = torch.cat((rays_o_all, rays_d_all), dim=2)
        rays = rays.resize(N, 6).cpu().numpy()
        ans = self.scene.cast_rays(rays)
        depth_gt = ans['t_hit'].numpy()
        depth_gt = torch.tensor(depth_gt).cuda()

        mask = torch.ones_like(depth_gt).cuda()
        mask[depth_gt > 999] = 0
        rays_o_all = rays_o_all[:,mask==1]
        rays_d_all = rays_d_all[:,mask == 1]
        depth_gt = depth_gt[mask == 1]

        select_inds = np.random.choice(rays_o_all.shape[1], size=[self.opt.batch_rays], replace=False)

        rays_o = rays_o_all[:, select_inds, :]
        rays_d = rays_d_all[:, select_inds, :]
        depth_gt = depth_gt[select_inds]

        outputs = self.model.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, **vars(self.opt))
        pred_rgb = outputs['image']
        pred_depth = outputs['depth']
        pred_sdf = outputs['rad_sdf']
        pred_radiances = outputs['radiances']

        xyz_mid = outputs['xyz_mid']
        dirs_mid = outputs['dirs_mid']
        ind_occu_mid = outputs['ind_occu_mid']

        xyzs = outputs['xyzs']
        ind_occu = outputs['ind_occu']

        pred_norm = outputs['normal']
        pred_norm = torch.norm(pred_norm, dim=-1).reshape(-1)
        pred_norm = pred_norm[pred_norm > 0]
        loss_eikonal = F.mse_loss(pred_norm, pred_norm.new_ones(pred_norm.shape), reduction='mean')

        with torch.no_grad():
            outputs_t = self.model_teacher.render(rays_o, rays_d, staged=False, perturb=True, force_all_rays=True, **vars(self.opt))
            t_rgb = outputs_t['image']
            t_sdf = self.model_teacher.density(xyzs.reshape(-1, 3)[ind_occu])['sigma']
            t_radiances = self.model_teacher.forward_radiance(xyz_mid.reshape(-1, 3)[ind_occu_mid], dirs_mid.reshape(-1, 3)[ind_occu_mid])

        loss_radiances = F.mse_loss(pred_radiances.reshape(-1, 3)[ind_occu_mid], t_radiances.reshape(-1, 3), reduction='mean')
        loss_sdf = F.l1_loss(pred_sdf.reshape(-1)[ind_occu], t_sdf.reshape(-1), reduction='mean')

        loss_c = F.mse_loss(pred_rgb.reshape(-1, 3), t_rgb.reshape(-1, 3), reduction='mean')

        loss = self.opt.lambda_eikonal * loss_eikonal + \
               loss_sdf + self.opt.lambda_diss * loss_c  + self.opt.lambda_diss * loss_radiances

        loss_dict = {
            # 'loss_eikonal_sample': loss_eikonal.item(),
            'loss_radiances_sample': loss_radiances.item(),
            'loss_c_sample': loss_c.item(),
            'loss_sdf_sample': loss_sdf.item(),
        }

        return loss, loss_dict



