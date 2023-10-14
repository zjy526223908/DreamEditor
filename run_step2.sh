# step2: locating editing region

# sample views
python sample_views.py -O2   \
  --R_path data/dtu_scan105/Orient_R.npy \
  --radius_range 0.5 0.5  --fovy_range 50 50  --phi_list -45 -30 0 30 45   --theta_list 60 75 \
  --backbone neus  \
  --data_type 'dtu' --data_path ./data/dtu_scan105 --H 1200 --W 1600  --eval_interval 20  --bound 1.0 \
  --workspace ./res/dtu_scan105-neus

# dreambooth

cd personalization
export MODEL_NAME="stabilityai/stable-diffusion-2-1-base"
export OUTPUT_DIR='../res/dtu_scan105-neus/dreambooth_res_2'

python train_dreambooth.py \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision fp16 \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir '../res/dtu_scan105-neus/sample_views' \
  --class_data_dir '../res/dtu_scan105-neus/class_samples' \
  --instance_prompt 'a photo of a <ktn> plush toy' \
  --class_prompt 'a photo of a plush toy' \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --validation_prompt="a photo of a <ktn> plush toy wearing black sunglasses" \
  --validation_images ../res/dtu_scan105-neus/sample_views/75_0.png ../res/dtu_scan105-neus/sample_views/75_30.png ../res/dtu_scan105-neus/sample_views/75_-30.png \
  --resolution=512 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --checkpointing_steps=1000 \
  --validation_steps=40 \
  --max_train_steps=400


# locating editing region
cd locating
python locating_editing_region.py \
  --workspace_path '../data/dtu_scan105/locating2/sunglasses' \
  --sd_path '../res/dtu_scan105-neus/dreambooth_res' \
  --output_images_path '../res/dtu_scan105-neus/sample_views'  --orient_r_path  '../data/dtu_scan105/Orient_R.npy'  \
  --mesh_path '../data/dtu_scan105/doll.ply' \
  --radius_list 0.5  --fovy_list 50 --phi_list 0  --theta_list 75  \
  --prompt "a photo of a <plush> toy wearing black sunglasses"   \
  --key_word "sunglasses"  --threshold 0.75

python locating_editing_region.py \
  --workspace_path '../data/dtu_scan105/locating/hat' \
  --sd_path '../res/dtu_scan105-neus/dreambooth_res' \
  --output_images_path '../res/dtu_scan105-neus/sample_views'  --orient_r_path  '../data/dtu_scan105/Orient_R.npy'  \
  --mesh_path '../data/dtu_scan105/doll.ply' \
  --radius_list 0.5  --fovy_list 50 --phi_list -90 0 90 -180 --theta_list 60  \
  --prompt "a photo of a <ktn> plush toy wearing a hat"   \
  --key_word "hat"  --threshold 0.75

python locating_editing_region.py \
  --workspace_path '../data/dtu_scan105/locating2/apron' \
  --sd_path '../res/dtu_scan105-neus/dreambooth_res' \
  --output_images_path '../res/dtu_scan105-neus/sample_views'  --orient_r_path  '../data/dtu_scan105/Orient_R.npy'  \
  --mesh_path '../data/dtu_scan105/doll.ply' \
  --radius_list 0.5  --fovy_list 50 --phi_list 0 --theta_list 75  \
  --prompt "a photo of a <ktn> plush toy wearing a blue apron"   \
  --key_word "apron"  --threshold 0.6

python locating_editing_region.py \
  --workspace_path '../data/dtu_scan105/locating/toy' \
  --sd_path '../res/dtu_scan105-neus/dreambooth_res' \
  --output_images_path '../res/dtu_scan105-neus/sample_views'  --orient_r_path  '../data/dtu_scan105/Orient_R.npy'  \
  --mesh_path '../data/dtu_scan105/doll.ply' \
  --radius_list 0.5  --fovy_list 50 --phi_list -180 -90 0 90 --theta_list 75  \
  --prompt "a photo of a <ktn> plush toy"   \
  --key_word "plush"  --threshold 0.5 --max_depth 20

