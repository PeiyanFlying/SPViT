## 环境配置:

conda create -n DynamicVit python=3.6

conda activate DynamicVit

conda deactivate

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 -c pytorch

pip3 install timm==0.4.5


## 下载prertained model
见 download_pretrain.sh



## 命令

举例：跑deit-small, 用3keep+senet的代码

python3 -u -m torch.distributed.launch --nproc_per_node=8 --use_env main_l2_vit_3keep_senet.py --output_dir logs/3keep_senet --arch deit_small --input-size 224 --batch-size 96 --data-path /data/ImageNet_new/ --epochs 30 --dist-eval --distill --base_rate 0.7 2>&1 | tee -i 3keep_senet.log


## 调参 link
https://docs.google.com/spreadsheets/d/1k25sS_-mmQyIvpIrn32GUw3eRuYcCy0cN0OSOq0QGFI/edit?usp=sharing

## score 数据记录
https://drive.google.com/drive/folders/1diICKopeYL7H84Wsr0Xxh30e9xh6RX2d?usp=sharing

## 可选择是否使用全精度训练。关闭 amp 功能。在 engine_l2.py 的 train_one_epoch() 和 evaluate()
![](fig/1.png)

## vit.py 文件改动，生成 vit_l2.py [对于 one keep token]

### 生成 multihead-predictor 类
### VisionTransformerDiffPruning 的 forward()

## vit.py 文件改动，生成 vit_l2_3keep.py [对于 three keep tokens]


### 生成 multihead-predictor 类
### VisionTransformerDiffPruning 的 forward()

###环境：因为加入了torch.nan_to_num(x, nan=4.0)，需要用torch 1.8.0
