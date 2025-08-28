export CUDA_VISIBLE_DEVICES=0

prompt_list=(
    "According to the previous object detection results, please answer the following question with 'yes' or 'no':"
    "According to the previous object detection results, please answer the following question:"
)

lr=2e-3
dataset=pope

prompt_id=0 # or 1
init_mode='text' # or random
virtual_tokens_num=20
info='obj+position' # or 'obj'
pos='middle' #or 'nono' or 'late'

output_dir="output path"

python -m torch.distributed.run --nproc-per-node 1 --master-port 6678 train.py --cfg-path train_configs/minigptv2_finetune.yaml \
    --dataset ${dataset} \
    --options \
        run.init_lr="${lr}" \
        model.extra_kwargs.virtual_tokens_num="${virtual_tokens_num}" \
        model.extra_kwargs.init_mode="${init_mode}" \
        model.extra_kwargs.prompt_id="${prompt_id}" \
        datasets.pope.extra_kwargs.pos="${pos}" \
        datasets.pope.extra_kwargs.info="${info}"\
        run.output_dir="${output_dir}" \
