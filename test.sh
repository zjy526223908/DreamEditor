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