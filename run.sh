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

### Training for 1 token_keep

To train DynamicViT models on ImageNet, run:

DeiT-small
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 20 --data-path /data/imagenet/--epochs 30 --dist-eval --distill --base_rate 0.7
```

LV-ViT-S
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit.py  --output_dir logs/dynamic-vit_lvvit-s --arch lvvit_s --input-size 224 --batch-size 64 --data-path /data/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7
```

LV-ViT-M
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit.py  --output_dir logs/dynamic-vit_lvvit-m --arch lvvit_m --input-size 224 --batch-size 48 --data-path /data/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7
```


### Training for 3 token_keep

To train DynamicViT models on ImageNet, run:

DeiT-small
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit_3keep.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 20 --data-path /data/imagenet/--epochs 30 --dist-eval --distill --base_rate 0.7
```

LV-ViT-S
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit_3keep.py  --output_dir logs/dynamic-vit_lvvit-s --arch lvvit_s --input-size 224 --batch-size 64 --data-path /data/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7
```

LV-ViT-M
```
python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit_3keep.py  --output_dir logs/dynamic-vit_lvvit-m --arch lvvit_m --input-size 224 --batch-size 48 --data-path /data/ImageNet/ --epochs 30 --dist-eval --distill --base_rate 0.7
```

You can train models with different keeping ratio by adjusting ```base_rate```. DynamicViT can also achieve comparable performance with only 15 epochs training (around 0.1% lower accuracy compared to 30 epochs).
