# step3
python3 train_editing.py  \
--if_mask True  --iters 5000 \
--backbone mesh_neus -O2  --far_point_thresh 0.05 --mask_thresh 0.02 --up_mask_thresh 0.02 --offset_thresh 0.1 \
--learn_vector_weight 0.001 --learn_sigma_weight 0.1 --learn_color_weight 1.0 \
--lr 1e-2 --w 256 --h 256 --up_w 256 --up_h 256 --batch_rays 8192  --bound 1.0  \
--num_steps 64 --upsample_steps 64 --learn_vector True  \
--seed 1 --stop_vector_epoch 10 --eval_interval 4  \
--R_path data/dtu_scan105/Orient_R.npy \
--backnerf_path res/dtu_scan105-grid/checkpoints/df_ep0460.pth  \
--load_path res/diss-dtu_scan105-doll/checkpoints/df_ep4600.pth \
--prior_mesh data/dtu_scan105/doll.ply \
--lambda_arap 1e-4	 --lambda_laplacian 1e-4 --lambda_eikonal 1 \
--text "a photo of a <ktn> plush toy wearing black sunglasses"  	\
--sd_path res/dtu_scan105-neus/dreambooth_res  \
--mesh_mask_info data/dtu_scan105/locating/sunglasses/editing_region_info.npy	\
--prior_mask_mesh data/dtu_scan105/locating/sunglasses/editing_region_mask.ply \
--radius_range 0.45 0.55 --fovy_range 45 55 --pose_sample_strategy '360' \
--phi_range -45 45 --theta_range 45 75  --sd_img_size 512 \
--workspace ./res_doll/sunglasses

python3 train_editing.py  \
--if_mask True  --iters 4000 \
--backbone mesh_neus  --far_point_thresh 0.05 --mask_thresh 0.02 --up_mask_thresh 0.02 --offset_thresh 0.25 \
--learn_vector_weight 0.1 --learn_sigma_weight 0.1 --learn_color_weight 1.0 \
--lr 1e-2 --w 256 --h 256 --up_w 256 --up_h 256 --batch_rays 16000  --bound 1.0  \
--num_steps 64 --upsample_steps 32 --learn_vector True  \
--seed 1 --stop_vector_epoch 20 --eval_interval 4  \
--R_path data/dtu_scan105/Orient_R.npy \
--backnerf_path res/dtu_scan105-grid/checkpoints/df_ep0460.pth  \
--load_path res/diss-dtu_scan105-doll/checkpoints/df_ep4600.pth \
--prior_mesh data/dtu_scan105/doll.ply \
--lambda_arap 0	 --lambda_laplacian 1e-10 --lambda_eikonal 1 \
--text "a photo of a <ktn> plush toy wearing a red top hat" 	\
--sd_path res/dtu_scan105-neus/dreambooth_res  \
--mesh_mask_info data/dtu_scan105/locating/hat/editing_region_info.npy	\
--prior_mask_mesh data/dtu_scan105/locating/hat/editing_region_mask.ply \
--radius_range 0.45 0.55 --fovy_range 45 55 --pose_sample_strategy '360' \
--phi_range -120 120 --theta_range 30 90 --sd_img_size 512 \
--workspace ./res_doll/red-tophat


python3 train_editing.py  \
--if_mask True  --iters 4000  \
--backbone mesh_neus  --far_point_thresh 0.05 --mask_thresh 0.01 --up_mask_thresh 0.01 --offset_thresh 0.25 \
--learn_vector_weight 0.01 --learn_sigma_weight 0.1 --learn_color_weight 1.0 \
--lr 1e-2 --w 256 --h 256 --up_w 256 --up_h 256 --batch_rays 8192  --bound 1.0  \
--num_steps 64 --upsample_steps 64 --learn_vector True  \
--seed 1 --stop_vector_epoch 20 --eval_interval 4  \
--R_path data/dtu_scan105/Orient_R.npy \
--backnerf_path res/dtu_scan105-grid/checkpoints/df_ep0460.pth  \
--load_path res/diss-dtu_scan105-doll/checkpoints/df_ep4600.pth \
--prior_mesh data/dtu_scan105/doll.ply \
--lambda_arap 1e-4	 --lambda_laplacian 1e-4 --lambda_eikonal 1 \
--text "a photo of a <ktn> plush toy wearing a blue apron" 	\
--sd_path res/dtu_scan105-neus/dreambooth_res  \
--mesh_mask_info data/dtu_scan105/locating/apron/editing_region_info.npy	\
--prior_mask_mesh data/dtu_scan105/locating/apron/editing_region_mask.ply \
--radius_range 0.45 0.55 --fovy_range 45 55 --pose_sample_strategy '360' \
--phi_range -45 45 --theta_range 60 90  --sd_img_size 512 \
--workspace ./res_doll/blue-apron



python3 train_editing.py  \
--if_mask True  --iters 4000  --fp16 \
--backbone mesh_neus  --far_point_thresh 0.05 --mask_thresh 0.01 --up_mask_thresh 0.01 --offset_thresh 0.25 \
--learn_vector_weight 0 --learn_sigma_weight 0 --learn_color_weight 1.0 \
--lr 1e-2 --w 192 --h 192 --up_w 256 --up_h 256 --batch_rays 16000  --bound 1.0  \
--num_steps 64 --upsample_steps 32   \
--seed 1 --stop_vector_epoch 20 --eval_interval 4  \
--R_path data/dtu_scan105/Orient_R.npy \
--backnerf_path res/dtu_scan105-grid/checkpoints/df_ep0460.pth  \
--load_path res/diss-dtu_scan105-doll/checkpoints/df_ep4600.pth \
--prior_mesh data/dtu_scan105/doll.ply \
--lambda_arap 1e-4	 --lambda_laplacian 1e-4 --lambda_eikonal 1 \
--text "a photo of a <ktn> gold toy made of gold" 	\
--sd_path res/dtu_scan105-neus/dreambooth_res  \
--mesh_mask_info data/dtu_scan105/locating/toy/editing_region_info.npy	\
--prior_mask_mesh data/dtu_scan105/locating/toy/editing_region_mask.ply \
--radius_range 0.50 0.50 --fovy_range 50 50 --pose_sample_strategy '360' \
--phi_range -90 90 --theta_range 60 90  --sd_img_size 512 \
--workspace ./res_doll/toy-gold

#"a photo of a <ktn> sliver toy made of sliver"
#"a photo of a <ktn> plush toy with tiger pattern"
#"a photo of a <ktn> plush toy with leopard pattern"