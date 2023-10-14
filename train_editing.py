import copy
import torch
import numpy as np
import argparse

from nerf.provider import SphericalSamplingDataset

from nerf.utils_editing import Trainer

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default=None, help="text prompt")
    parser.add_argument('--negative', default='', type=str, help="negative text prompt")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--save_vedio', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=1, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--backnerf_path', type=str, default=None)
    parser.add_argument('--guidance', type=str, default='stable-diffusion', help='choose from [stable-diffusion, clip]')
    parser.add_argument('--sd_path', type=str, default='/mnt/d/project/sd_pre/textual_inversion_doll_unet_200')
    parser.add_argument('--sd_img_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sd_max_step', type=float, default=0.98, help="sd_max_step")

    ### training options
    parser.add_argument('--iters', type=int, default=2000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--guidance_scale', type=float, default=100.0, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024,
                        help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64,
                        help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=64,
                        help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--up_num_steps', type=int, default=64,
                        help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--up_upsample_steps', type=int, default=64,
                        help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=64,
                        help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--update_mesh_interval', type=int, default=1)
    parser.add_argument('--max_ray_batch', type=int, default=1024,
                        help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")
    parser.add_argument('--learn_vector', type=bool, default=False,
                        help="training iters that only use albedo shading")
    parser.add_argument('--offset_thresh', type=float, default=0.05)
    parser.add_argument('--learn_vector_weight', type=float, default=0.1,
                        help="training iters that only use albedo shading")
    parser.add_argument('--learn_sigma_weight', type=float, default=0.1,
                        help="training iters that only use albedo shading")
    parser.add_argument('--learn_color_weight', type=float, default=1.0,
                        help="training iters that only use albedo shading")
    # model options
    parser.add_argument('--far_point_thresh', type=float, default=0.05)
    parser.add_argument('--mask_thresh', type=float, default=0.05)
    parser.add_argument('--up_mask_thresh', type=float, default=0.05)
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--uniform_sphere_rate', type=float, default=0.5,
                        help="likelihood of sampling camera location uniformly on the sphere surface area")
    # network backbone
    parser.add_argument("--if_direction", type=bool, default=True)
    parser.add_argument('--K', type=int, default=8, help="GUI height")
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--prior_mesh', type=str, default=None)
    parser.add_argument('--prior_mask_mesh', type=str, default=None)
    parser.add_argument('--mesh_mask_info', type=str, default=None)
    parser.add_argument('--geometry_dim', type=int, default=128, help="GUI width")
    parser.add_argument('--color_dim', type=int, default=128, help="GUI height")
    parser.add_argument('--color_en', type=bool, default=False, help="GUI height")
    parser.add_argument('--geometry_en', type=bool, default=False, help="GUI width")
    parser.add_argument('--sigma_net_d', type=int, default=2, help="GUI width")
    parser.add_argument('--sigma_net_w', type=int, default=256
                        , help="GUI height")
    parser.add_argument('--color_net_d', type=int, default=3, help="GUI width")
    parser.add_argument('--color_net_w', type=int, default=256, help="GUI height")
    parser.add_argument('--backbone', type=str, default='mesh_neus',
                        help="nerf backbone, choose from [mesh, mesh_neus, grid, vanilla]")
    parser.add_argument("--if_smooth_mesh", type=bool, default=False)
    parser.add_argument('--stepsmoothnum', type=int, default=1, help="GUI width")
    parser.add_argument("--if_mask", type=bool, default=True)
    parser.add_argument("--if_smooth", type=bool, default=False)
    parser.add_argument("--fine_sigma_net", type=bool, default=False)
    parser.add_argument("--fine_color_net", type=bool, default=False)
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=128, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=128, help="render height for NeRF in training")
    parser.add_argument('--up_w', type=int, default=192, help="render width for NeRF in training")
    parser.add_argument('--up_h', type=int, default=192, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")

    ### dataset options
    parser.add_argument("--pose_sample_strategy", type=str, default='front_back',
                        help='input data directory')
    parser.add_argument("--R_path", type=str, default=None,
                        help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--batch_rays', type=int, default=1024, help="GUI width")
    parser.add_argument('--up_batch_rays', type=int, default=1024, help="GUI width")
    parser.add_argument('--num_work', type=int, default=4, help="GUI width")

    parser.add_argument('--bound', type=float, default=1.0, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--bg_bound', type=float, default=1.0, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--object_bound', type=float, nargs='+', default=None)
    parser.add_argument('--dt_gamma', type=float, default=0,
                        help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[1.4, 1.6],
                        help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[65, 65], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-90, 90], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="training camera fovy range")
    parser.add_argument('--dir_text', action='store_true',
                        help="direction-encode the text prompt, by appending front/side/back/overhead view")
    parser.add_argument('--negative_dir_text', action='store_true', help="also use negative dir text prompt.")
    parser.add_argument('--angle_overhead', type=float, default=25, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60,
                        help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")
    parser.add_argument("--fix_geometry_features", action='store_true')

    parser.add_argument('--hump_mesh_dis', type=float, default=-1., help="loss scale for alpha entropy")
    parser.add_argument('--hump_mesh_type', type=str, default='semicircle', help="loss scale for alpha entropy")
    parser.add_argument('--stop_vector_epoch', type=int, default=50, help="loss scale for alpha entropy")
    parser.add_argument('--size_factor', type=float, default=1.0, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_eikonal', type=float, default=1e-2, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_entropy', type=float, default=0, help="loss scale for alpha entropy")
    parser.add_argument('--lambda_opacity', type=float, default=0, help="loss scale for alpha value")
    parser.add_argument('--lambda_orient', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--lambda_smooth', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--lambda_laplacian', type=float, default=1e-4, help="loss scale for orientation")
    parser.add_argument('--lambda_norm', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--lambda_edge', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--lambda_flatten', type=float, default=0, help="loss scale for orientation")
    parser.add_argument('--lambda_arap', type=float, default=100, help="loss scale for orientation")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=264, help="GUI width")
    parser.add_argument('--H', type=int, default=264, help="GUI height")
    parser.add_argument('--radius', type=float, default=1, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=70, help="default GUI camera fovy")
    parser.add_argument('--light_theta', type=float, default=70,
                        help="default GUI light direction in [0, 180], corresponding to elevation [90, -90]")
    parser.add_argument('--light_phi', type=float, default=0, help="default GUI light direction in [0, 360), azimuth")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    opt = parser.parse_args()

    if opt.O:
        opt.cuda_ray = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'mesh_neus':
        from nerf.network_mesh_neus import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt, device)

    print(model)

    if opt.backnerf_path and not opt.test:
        opt_back = copy.deepcopy(opt)
        from nerf.network_grid import NeRFNetwork as BackNetwork

        opt_back.bound = 1.0
        opt_back.fp16 = True
        opt_back.cuda_ray = True
        background_model = BackNetwork(opt_back, device)
    else:
        background_model = None

    if opt.test:
        guidance = None  # no need to load guidance model at test

        trainer = Trainer('df', opt, model, guidance, background_model, opt, device=device, workspace=opt.workspace,
                          fp16=opt.fp16,
                          use_checkpoint=opt.ckpt)

        if opt.save_mesh:
            trainer.save_edit_mesh()
            trainer.save_mesh(resolution=128)

        if opt.save_vedio:
            test_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, type='test', H=opt.H, W=opt.W,
                                                   size=50).dataloader()
            trainer.test(test_loader)

    else:

        guidance = None

        if opt.guidance == 'stable-diffusion':
            from nerf.sd import StableDiffusion

            guidance = StableDiffusion(opt, device, sd_path=opt.sd_path)
        else:
            raise NotImplementedError(f'--guidance {opt.guidance} is not implemented.')

        train_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, type='train', H=opt.h, W=opt.w,
                                                size=100).dataloader()

        trainer = Trainer('df', opt, model, guidance, background_model, opt_back, device=device,
                          workspace=opt.workspace,
                          ema_decay=False, fp16=opt.fp16, use_checkpoint=opt.ckpt,
                          eval_interval=opt.eval_interval, scheduler_update_every_step=True)

        valid_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, type='test', H=opt.H, W=opt.W,
                                                size=4).dataloader()

        max_epoch = np.ceil(opt.iters / 100).astype(np.int32)
        trainer.train(train_loader, valid_loader, max_epoch)
