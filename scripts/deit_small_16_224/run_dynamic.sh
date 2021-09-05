cd ../..

ARCH="deit_small"

INIT_LR="5e-4"
LR_SCHEDULE="cosine"
GLOBAL_BATCH_SIZE="512"
LOCAL_BATCH_SIZE="128"
EPOCHS="30"
WARMUP="8"

TEMPTERATURE="2"
PENALTY="0.05"

LOAD_CKPT="xxxxxx"
SAVE_FOLDER="./checkpoints/deit_small_16_224/multihead_245_fix_dynamic_30epoch"


mkdir -p ${SAVE_FOLDER}

#python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_soft_vit.py --arch ${ARCH} --input-size 224 --lr ${INIT_LR} --batch-size ${LOCAL_BATCH_SIZE} --data-path /mnt/dataset/imagenet --epochs ${EPOCHS} --dist-eval --distill --temperature ${TEMPTERATURE} --ratiow ${PENALTY} --output_dir ${SAVE_FOLDER} 2>&1 | tee -i ${SAVE_FOLDER}/all_log.txt


python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit.py --output_dir ${SAVE_FOLDER} --arch deit_small --input-size 224 --batch-size 96 --data-path /mnt/dataset/imagenet --epochs 30 --dist-eval --distill --base_rate 0.7 2>&1 | tee -i ${SAVE_FOLDER}/all_log.txt