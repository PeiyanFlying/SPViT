import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from torch.nn import parameter

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss, DiffPruningLoss, DistillDiffPruningLoss
from samplers import RASampler
import utils
from functools import partial
import torch.nn.functional as F
# import torch.nn as nn
from soft_mask.soft2.vit_soft import VisionTransformerDiffPruning, VisionTransformerTeacher, _cfg, checkpoint_filter_fn
# from vit import VisionTransformerDiffPruning, VisionTransformerTeacher, _cfg, checkpoint_filter_fn
from lvvit import LVViTDiffPruning, LVViT_Teacher
import math
import shutil



base_rate = 0.7
KEEP_RATE = [base_rate, base_rate ** 2, base_rate ** 3]


PRUNING_LOC = [3,6,9]
print('token_ratio =', KEEP_RATE, 'at layer', PRUNING_LOC)
model = VisionTransformerDiffPruning(
    patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
    pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE, distill=True
    )
model_path = 'checkpoint.pth'
checkpoint = torch.load(model_path, map_location="cpu")
ckpt = checkpoint_filter_fn(checkpoint, model)
model.default_cfg = _cfg()
missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False) # 知道问题出在哪儿了。很可以。

# 只以此形式存下了 keep_threshold_base 没存
print(checkpoint['model']['module.score_predictor.0.keep_threshold']-0.7)
print(checkpoint['model']['module.score_predictor.1.keep_threshold']-0.7)
print(checkpoint['model']['module.score_predictor.2.keep_threshold']-0.7)

# data  =  torch.randn([2, 3, 224, 224])
#
# model.eval()
# a = model(data)
# print('ok')

# print(model.score_predictor[0].keep_threshold+model.score_predictor[0].keep_threshold_base)
# print(model.score_predictor[1].keep_threshold+model.score_predictor[1].keep_threshold_base)
# print(model.score_predictor[1].keep_threshold+model.score_predictor[1].keep_threshold_base)


# logits = torch.Tensor([8.8915e-04,3.8466e-01,3.7118e-01,3.3678e-01,8.0558e-11,6.6097e-04,1.6053e-18,4.1310e-01,2.3311e-01,4.2189e-01,9.4778e-16,\
#                        1.0022e-03,4.9389e-19,2.6817e-01,3.1232e-01,4.1046e-01,1.7933e-06,3.3068e-17,5.0708e-12,6.1491e-12,3.9709e-16,1.1515e-05,\
#                        3.8529e-05,4.1626e-07,8.3702e-10,2.2027e-01,2.6299e-01,2.8590e-01,2.5617e-01,1.9497e-01,1.7025e-01,4.1991e-05,3.0012e-05,\
#                        4.3610e-03])*1000
# print(logits)
# a = F.gumbel_softmax(logits, hard=False)
# b =  F.gumbel_softmax(logits, hard=True)
# print(a)
# print(b)

# data = torch.randn([2, 2, 3, 3])
# policy = torch.Tensor([[[0.2],[0.8],[0.1]],[[0.1],[0.3],[0.9]]])
#
# print('===data===')
# print(data)

def softmax_with_policy(attn, policy, eps=1e-6):
    B, N, _ = policy.size()
    B, H, N, N = attn.size()
    attn_policy = policy.reshape(B, 1, 1, N)  # * policy.reshape(B, 1, N, 1)
    eye = torch.eye(N, dtype=attn_policy.dtype, device=attn_policy.device).view(1, 1, N, N)
    attn_policy = attn_policy + (1.0 - attn_policy) * eye
    # print('===attn_policy===')
    # print(attn_policy)
    # print(attn_policy) # 这个得到了我们想要的attention mask 模式/现在需要确定的是 sparse attn 之后，sparse attention 是啥样的。原来是概率的重新分配
    max_att = torch.max(attn, dim=-1, keepdim=True)[0]
    # print('===max_att===')
    # print(max_att)
    attn = attn - max_att
    # print('===attn===')
    # print(attn)
    # attn = attn.exp_() * attn_policy
    # return attn / attn.sum(dim=-1, keepdim=True)

    # for stable training
    attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
    attn = (attn + eps/N) / (attn.sum(dim=-1, keepdim=True) + eps)
    # print('===results===')
    return attn.type_as(max_att)




