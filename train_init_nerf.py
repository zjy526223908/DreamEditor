import argparse
import torch
import torch.optim as optim
import numpy as np
from nerf.provider import NeRFDataset, SphericalSamplingDataset
from nerf.provider_utils import seed_everything
from nerf.utils_init_nerf import Trainer_Nerf


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --dir_text")
    parser.add_argument('-O2', action='store_true', help="equals --fp16 --dir_text")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--save_mesh', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--save_video', action='store_true', help="export an obj mesh with texture")
    parser.add_argument('--eval_interval', type=int, default=50, help="evaluate on the valid set every interval epochs")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=300000, help="training iters")
    parser.add_argument('--lr', type=float, default=5e-4, help="initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=64, help="num steps up-sampled per ray (only valid when not using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=100, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when not using --cuda_ray)")

    # model options
    parser.add_argument('--prenet_path', type=str, default=None)
    parser.add_argument('--bg_color', type=float, nargs='+', default=None)
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")

    # network backbone
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--geometry_dim', type=int, default=128, help="GUI width")
    parser.add_argument('--color_dim', type=int, default=128, help="GUI height")
    parser.add_argument('--color_en', type=bool, default=False, help="GUI height")
    parser.add_argument('--geometry_en', type=bool, default=False, help="GUI width")
    parser.add_argument('--sigma_net_d', type=int, default=2, help="GUI width")
    parser.add_argument('--sigma_net_w', type=int, default=256, help="GUI height")
    parser.add_argument('--color_net_d', type=int, default=3, help="GUI width")
    parser.add_argument('--color_net_w', type=int, default=256, help="GUI height")
    parser.add_argument('--backbone', type=str, default='grid', help="nerf backbone, choose from [mesh, mesh_sdf, grid, vanilla]")
    parser.add_argument("--if_data_cuda", action='store_false')
    parser.add_argument("--if_direction", type=bool, default=False)
    parser.add_argument("--if_bg_model", type=bool, default=False)
    parser.add_argument("--if_mask", type=bool, default=False)
    parser.add_argument("--if_smooth", type=bool, default=False)
    # rendering resolution in training, decrease this if CUDA OOM.
    parser.add_argument('--w', type=int, default=400, help="render width for NeRF in training")
    parser.add_argument('--h', type=int, default=300, help="render height for NeRF in training")
    parser.add_argument('--scale', type=float, default=1.0, help="render height for NeRF in training")
    parser.add_argument('--jitter_pose', action='store_true', help="add jitters to the randomly sampled camera poses")

    ### dataset options
    parser.add_argument("--data_path", type=str, default='/mnt/d/dataset/data_DTU/dtu_scan105/',
                        help='input data directory')
    parser.add_argument("--pose_path", type=str, default=None, help='input data directory')
    parser.add_argument("--data_type", type=str, default='dtu', help='input data')
    parser.add_argument("--if_sphere", type=bool, default=False)
    parser.add_argument("--R_path", type=str, default=None, help='input data directory')
    parser.add_argument('--batch_size', type=int, default=1, help="GUI width")
    parser.add_argument('--batch_rays', type=int, default=512, help="GUI width")
    parser.add_argument('--train_resolution_level', type=float, default=1, help="GUI width")
    parser.add_argument('--eval_resolution_level', type=float, default=4, help="GUI width")
    parser.add_argument('--num_work', type=int, default=0, help="GUI width")
    parser.add_argument('--train_batch_type', type=str, default='all_images')
    parser.add_argument('--val_batch_type', type=str, default='all_images')
    parser.add_argument('--bound', type=float, default=2., help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--scene_scale', type=float, default=0.33, help="assume the scene is bounded in box(-bound, bound)")
    parser.add_argument('--min_near', type=float, default=0.01, help="minimum near distance for camera")
    parser.add_argument('--radius_range', type=float, nargs='*', default=[0.15, 0.15], help="training camera radius range")
    parser.add_argument('--fovy_range', type=float, nargs='*', default=[50, 70], help="training camera fovy range")
    parser.add_argument('--phi_range', type=float, nargs='*', default=[-180, 180], help="training camera fovy range")
    parser.add_argument('--theta_range', type=float, nargs='*', default=[60, 90], help="training camera fovy range")
    parser.add_argument('--angle_overhead', type=float, default=30, help="[0, angle_overhead] is the overhead region")
    parser.add_argument('--angle_front', type=float, default=60, help="[0, angle_front] is the front region, [180, 180+angle_front] the back region, otherwise the side region.")

    parser.add_argument('--lambda_eikonal', type=float, default=1e-2, help="loss scale for alpha entropy")

    ### GUI options
    parser.add_argument('--W', type=int, default=400, help="GUI width")
    parser.add_argument('--H', type=int, default=300, help="GUI height")

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True

    elif opt.O2:
        opt.fp16 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    if opt.backbone == 'grid':
        from nerf.network_grid import NeRFNetwork
    elif opt.backbone == 'neus':
        from nerf.network_neus import NeRFNetwork
    else:
        raise NotImplementedError(f'--backbone {opt.backbone} is not implemented!')

    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt, device)

    print(model)

    if opt.test:
        guidance = None # no need to load guidance model at test

        trainer = Trainer_Nerf('df', opt, model, guidance, device=device, workspace=opt.workspace, fp16=opt.fp16, use_checkpoint=opt.ckpt)

        if opt.save_video:
            test_loader = SphericalSamplingDataset(opt, device=device, R_path=opt.R_path, type='test', H=256, W=256, size=100).dataloader()
            trainer.test(test_loader, if_gui=True)

        if opt.save_mesh:
            trainer.save_mesh(resolution=512)

        test_loader = NeRFDataset(opt, device=device, R_path=opt.R_path, type='val', H=opt.H, W=opt.W, size=1000).dataloader()
        trainer.test(test_loader, if_gui=False)

    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, R_path=opt.R_path, type='train', H=opt.h, W=opt.w, size=500).dataloader()
        valid_loader = NeRFDataset(opt, device=device, R_path=opt.R_path, type='val', H=opt.H, W=opt.W, size=10).dataloader()

        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        trainer = Trainer_Nerf('df', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, ema_decay=None, fp16=opt.fp16, lr_scheduler=scheduler, use_checkpoint=opt.ckpt, eval_interval=opt.eval_interval, scheduler_update_every_step=True)



        max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        trainer.train(train_loader, valid_loader, max_epoch)

        # also test
        test_loader = NeRFDataset(opt, device=device, R_path=opt.R_path, type='test', H=opt.H, W=opt.W, size=100).dataloader()
        trainer.test(test_loader)

        if opt.save_mesh:
            trainer.save_mesh(resolution=256)