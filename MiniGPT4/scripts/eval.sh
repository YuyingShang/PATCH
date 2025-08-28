export CUDA_VISIBLE_DEVICES=1
torchrun --master-port 7878 --nproc_per_node 1 eval_vqa.py --cfg-path eval_configs/minigptv2_pope.yaml --dataset pope