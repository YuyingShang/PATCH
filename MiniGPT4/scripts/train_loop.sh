export CUDA_VISIBLE_DEVICES=0

prompt_list=(
    "According to the previous object detection results, please answer the following question with 'yes' or 'no':"
    "According to the previous object detection results, please answer the following question:"
)


lrs=(
    2e-3
    5e-3
    8e-3
)

# virtual_tokens_nums=(
#     1
#     10
#     30
#     50
#     100
# )



for lr in ${lrs[@]}
do
    echo "lr: ${lr}"

    dataset=pope

    prompt_id=0
    init_mode='text' # random
    virtual_tokens_num=20
    info='obj+position'
    pos='middle'

    output_dir="output path"

    torchrun --nproc-per-node 1 --master-port 9987 train.py --cfg-path train_configs/minigptv2_finetune.yaml \
        --dataset ${dataset} \
        --options \
            run.init_lr="${lr}" \
            run.output_dir="${output_dir}" \
            model.extra_kwargs.virtual_tokens_num="${virtual_tokens_num}" \
            model.extra_kwargs.init_mode="${init_mode}" \
            model.extra_kwargs.prompt_id="${prompt_id}" \
            datasets.pope.extra_kwargs.pos="${pos}" \
            datasets.pope.extra_kwargs.info="${info}"
done
