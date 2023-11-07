# step1: distill to meshneus

# train init neus with dtu mask
#python train_init_nerf.py -O2 --workspace ./res/dtu_scan105-neus-mask   \
#--lambda_eikonal 0.01 --batch_rays 512 --if_mask True \
#--backbone neus --train_resolution_level 1 --eval_resolution_level 4 \
#--data_type 'dtu' --data_path ./data/dtu_scan105 --H 1200 --W 1600  --eval_interval 100  --bound 1.0


# train background instant-ngp
python train_init_nerf.py -O --workspace ./res/dtu_scan105-grid  \
--batch_rays 1024 --iters 30000 \
--backbone grid --train_resolution_level 1 --eval_resolution_level 4 \
--data_type 'dtu' --data_path ./data/dtu_scan105 --H 1200 --W 1600  --eval_interval 20  --bound 1.0

# train init neus
python train_init_nerf.py -O2 --workspace ./res/dtu_scan105-neus   \
--lambda_eikonal 0.1 --batch_rays 512 \
--backbone neus --train_resolution_level 1 --eval_resolution_level 4 \
--data_type 'dtu' --data_path ./data/dtu_scan105 --H 1200 --W 1600  --eval_interval 100  --bound 1.0


# train diss

python train_distill.py  --backbone mesh_neus \
  --prior_mesh ./data/dtu_scan105/doll.ply   --iters 45000 \
  --train_resolution_level 1 --eval_resolution_level 4 --w 1600 --h 1200    -O2 \
  --far_point_thresh 0.05 --lambda_eikonal 0.01    \
  --lambda_diss 1. \
  --teacher_path res/dtu_scan105-neus/checkpoints/df_ep4600.pth   \
  --sample_R_path  data/dtu_scan105/Orient_R.npy \
  --R_path  '' \
  --fine_sigma_net True --fine_ln True  \
  --prenet_path ./pre_mesh_net/pre_color_net.pth  \
  --radius_range 0.45 0.55  --fovy_range 50 50 --pose_sample_strategy '360'  --phi_range -120 120   --theta_range 30 75 \
  --data_type  'dtu' --data_path ./data/dtu_scan105 --H 300 --W 400  \
  --num_steps 64 --upsample_steps 64  \
  --workspace ./res/diss-dtu_scan105-doll --eval_interval 100 --bound 1.

python train_distill.py  --backbone mesh_neus \
  --prior_mesh ./data/dtu_scan105/doll.ply   --iters 300000 \
  --train_resolution_level 1 --eval_resolution_level 4 --w 1600 --h 1200    -O2 \
  --far_point_thresh 0.05 --lambda_eikonal 0.01    \
  --lambda_diss 1. \
  --teacher_path res/dtu_scan105-neus/checkpoints/df_ep4600.pth   \
  --sample_R_path  data/dtu_scan105/Orient_R.npy \
  --R_path  '' \
  --prenet_path ./pre_mesh_net/pre_color_net.pth  \
  --radius_range 0.45 0.55  --fovy_range 50 50 --pose_sample_strategy '360'  --phi_range -120 120   --theta_range 30 75 \
  --data_type  'dtu' --data_path ./data/dtu_scan105 --H 300 --W 400  \
  --num_steps 64 --upsample_steps 64  \
  --workspace ./res/diss-dtu_scan105-doll --eval_interval 100 --bound 1.


