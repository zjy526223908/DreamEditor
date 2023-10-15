# dog
# black sunglasses
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dog/Orient_R.npy \
#--load_path res_model/dog/sunglasses/lasted.pth \
#--prior_mesh res_model/dog/sunglasses/dog_sunglasses.ply \
#--text "a photo of a <ktn> dog wearing black sunglasses"  	\
#--radius_range 1.0 1.0 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -60 60 --theta_range 75 75   --bg_color 0.1 0.2 0.6  \
#--workspace ./res_dog/sunglasses-test

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dog/Orient_R.npy \
#--load_path res_model/dog/rose/lasted.pth \
#--prior_mesh res_model/dog/rose/dog_rose.ply \
#--text "a photo of a <ktn> dog taking a red rose in the mouth"  	\
#--radius_range 1.0 1.0 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -60 60 --theta_range 75 75   --bg_color 0.1 0.2 0.6  \
#--workspace ./res_dog/rose-test

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--num_steps 64 --upsample_steps 64   \
#--R_path data/dog/Orient_R.npy \
#--load_path res_model/dog/sliver_dog/lasted.pth \
#--prior_mesh data/dog/dog.ply \
#--text "a photo of a silver <ktn> dog"  	\
#--radius_range 1.0 1.0 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -60 60 --theta_range 75 75   --bg_color 0.1 0.2 0.6  \
#--workspace ./res_dog/sliver_dog-test

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dog/Orient_R.npy \
#--load_path res_model/dog/laughing/lasted.pth \
#--prior_mesh res_model/dog/laughing/dog_laughing.ply \
#--text "a photo of a laughing <ktn> dog"  	\
#--radius_range 1.0 1.0 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -60 60 --theta_range 75 75   --bg_color 0.1 0.2 0.6  \
#--workspace ./res_dog/laughing-test

# doll
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dtu_scan105/Orient_R.npy \
#--load_path res_model/doll/apron/lasted.pth \
#--prior_mesh data/dtu_scan105/doll.ply \
#--text "a photo of a <ktn> plush toy wearing a blue apron"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 75 75     \
#--workspace ./res_doll/apron

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dtu_scan105/Orient_R.npy \
#--load_path res_model/doll/gold_doll/lasted.pth \
#--prior_mesh data/dtu_scan105/doll.ply \
#--text  "a photo of a <ktn> gold toy made of gold"   	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 75 75     \
#--workspace ./res_doll/gold

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dtu_scan105/Orient_R.npy \
#--load_path res_model/doll/ironmam_doll/lasted.pth \
#--prior_mesh data/dtu_scan105/doll.ply \
#--text  "a photo of a ironmam <ktn> toy"   	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 75 75     \
#--workspace ./res_doll/ironmam_doll

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dtu_scan105/Orient_R.npy \
#--load_path res_model/doll/red_top_hat/lasted.pth \
#--prior_mesh res_model/doll/red_top_hat/hat_red.ply \
#--text  "a photo of a <ktn> plush toy wearing a red top hat"    	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 75 75     \
#--workspace ./res_doll/red_top_hat

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dtu_scan105/Orient_R.npy \
#--load_path res_model/doll/sunglasses/lasted.pth \
#--prior_mesh res_model/doll/sunglasses/sunglasses_doll.ply \
#--text "a photo of a <ktn> plush toy wearing black sunglasses"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 75 75     \
#--workspace ./res_doll/sunglasses

#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--R_path data/dtu_scan105/Orient_R.npy \
#--load_path res_model/doll/tiger_doll/lasted.pth \
#--prior_mesh data/dtu_scan105/doll.ply \
#--text  "a photo of a <ktn> plush toy with tiger pattern"   	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 75 75     \
#--workspace ./res_doll/tiger_doll

# horse
#box
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/horse/box/lasted.pth \
#--prior_mesh data/stone_horse/horse.ply \
#--text "a photo of a <ktn> horse standing on a wooden box"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 60 60     \
#--workspace ./res_horse/box

# deer
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/horse/deer/lasted.pth \
#--prior_mesh data/stone_horse/horse.ply \
#--text "a photo of a <ktn> deer"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 60 60   \
#--workspace ./res_horse/deer

# giraffe
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/horse/girafffe/lasted.pth \
#--prior_mesh data/stone_horse/horse.ply \
#--text "a photo of a <ktn> giraffe"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 60 60     \
#--workspace ./res_horse/giraffe

# hair
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/horse/hair/lasted.pth \
#--prior_mesh data/stone_horse/horse.ply \
#--text "a photo of a <ktn> horse with rainbow hair"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 60 60     \
#--workspace ./res_horse/hair

# robot
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/horse/robot/lasted.pth \
#--prior_mesh data/stone_horse/horse.ply \
#--text "a photo of a <ktn> robot horse"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 60 60     \
#--workspace ./res_horse/robot

# zebra
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/horse/zebra/lasted.pth \
#--prior_mesh data/stone_horse/horse.ply \
#--text "a photo of a <ktn> zebra"  	\
#--radius_range 0.5 0.5 --fovy_range 50 50 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 60 60     \
#--workspace ./res_horse/zebra

# face

# clown
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/face/clown/lasted.pth \
#--prior_mesh data/face/face.ply \
#--R_path data/face/R_test.npy \
#--text "a face photo of a <ktn> clown"  	\
#--radius_range 0.4 0.4 --fovy_range 40 40 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 90 90 --bg_color 1.0 1.0 1.0     \
#--workspace ./res_face/clown

# cowboy hat
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/face/hat_cowboy/lasted.pth \
#--prior_mesh res_model/face/hat_cowboy/cowboy.ply \
#--R_path data/face/R_test.npy \
#--text "a face photo of a <ktn> man wearing a cowboy hat"  	\
#--radius_range 0.4 0.4 --fovy_range 40 40 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 90 90 --bg_color 1.0 1.0 1.0     \
#--workspace ./res_face/hat_cowboy

# navy hat
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/face/hat_navy/lasted.pth \
#--prior_mesh data/face/face.ply \
#--R_path data/face/R_test.npy \
#--text "a face photo of a <ktn> man wearing a navy hat"  	\
#--radius_range 0.4 0.4 --fovy_range 40 40 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 90 90 --bg_color 1.0 1.0 1.0     \
#--workspace ./res_face/hat_navy

# moustache
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/face/moustache/lasted.pth \
#--prior_mesh res_model/face/moustache/moustache.ply \
#--R_path data/face/R_test.npy \
#--text "a face photo of a <ktn> man with black moustache"  	\
#--radius_range 0.4 0.4 --fovy_range 40 40 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 90 90 --bg_color 1.0 1.0 1.0     \
#--workspace ./res_face/moustache

# sunglasses
#python3 train_editing.py  --test --save_vedio \
#--far_point_thresh 0.05  \
#--load_path res_model/face/sunglasses/lasted.pth \
#--prior_mesh res_model/face/sunglasses/glass.ply \
#--R_path data/face/R_test.npy \
#--text "a face photo of a <ktn> man wearing black sunglasses"  	\
#--radius_range 0.4 0.4 --fovy_range 40 40 --pose_sample_strategy '360' \
#--phi_range -45 45 --theta_range 90 90 --bg_color 1.0 1.0 1.0     \
#--workspace ./res_face/glass