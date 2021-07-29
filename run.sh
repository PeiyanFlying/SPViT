conda create -n DynamicVit python=3.6
conda activate DynamicVit
conda deactivate

conda install pytorch==1.7.0 torchvision==0.8.1 torchaudio==0.7.0 -c pytorch
pip3 install timm==0.4.5

tmux attach -t 4

### Evaluation

To evaluate a pre-trained DynamicViT model on the ImageNet validation set with a single GPU, run:
```
python3 infer.py --data-path /data/ImageNet/ --arch arch_name --model-path /path/to/model --base_rate 0.7
```

### Training

To train DynamicViT models on ImageNet, run:

DeiT-small
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_dynamic_vit.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 20 --data-path /data/imagenet/--epochs 30 --dist-eval --distill --base_rate 0.7
```

LV-ViT-S
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_dynamic_vit.py  --output_dir logs/dynamic-vit_lvvit-s --arch lvvit_s --input-size 224 --batch-size 64 --data-path /data/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7
```

LV-ViT-M
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_dynamic_vit.py  --output_dir logs/dynamic-vit_lvvit-m --arch lvvit_m --input-size 224 --batch-size 48 --data-path /data/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7
```

You can train models with different keeping ratio by adjusting ```base_rate```. DynamicViT can also achieve comparable performance with only 15 epochs training (around 0.1% lower accuracy compared to 30 epochs).


###### 以下是正在开发的代码
# debug for channel merge
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_dy_channel.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 48 --data-path /home/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7 --addl1 --pruning_ratio 0.2

CUDA_VISIBLE_DEVICES=1 python3 main_dy_channel.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 2 --data-path /home/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7 --addl1 --pruning_ratio 0.2

# debug for soft mask merge


CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python3 main_dynamic_vit.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 2 --data-path /home/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7

python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_soft_vit.py --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 96 --data-path /data/ImageNet_new/ --epochs 30 --dist-eval --distill --base_rate 0.7 2>&1 | tee -i test.log

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_soft_vit.py --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 384 --data-path /data/ImageNet_new/ --epochs 30 --dist-eval --distill --base_rate 0.7 2>&1 | tee -i fine_soft.log
CUDA_VISIBLE_DEVICES=2 python3 main_dynamic_vit.py --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 2 --data-path /data/ImageNet_new/ --epochs 30 --dist-eval --distill --base_rate 0.7 2>&1 | tee -i fine_orig.log

