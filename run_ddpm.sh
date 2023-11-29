SAMPLE_FLAGS="--batch_size 8"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
CUDA_VISIBLE_DEVICES=0 python uie_sample.py $MODEL_FLAGS --uie_dataset U45 --classifier_scale 4000.0 --model_path models/256x256_diffusion_uncond.pt $SAMPLE_FLAGS
