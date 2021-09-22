import math
import numpy as np
 
def PatchEmbed(N, P, C):
    return N * P * P * 3 * C

def Head(C):
    return C * 1000

def AuxHead(N, C):
    return N * C * 1000

def Deit_Block(N, C, R=4):
    return 4 * N * C ** 2 + 2 * N ** 2 * C + 2 * R * N * C ** 2

def PredictorLG(N, C):
    return 5 / 8 * N * C ** 2 + N * C // 2

def DynamicViT(img_size=224, P=16, H=14, C=384, rate=1.0):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(4):
        #print('i',i)
        blocks += 3 * Deit_Block(N, C)  #DDDP DDDP DDDP DDD-break
        if i == 3:
            break
        predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1
        #print('N',N)

    head = Head(C)
    return pe + blocks + predictor + head


def Dynamic_Soft_Mask_ViT(img_size=224, P=16, H=14, C=384, sparse=[0.3,0.3,0.3]):
    assert img_size == P * H
    N = H * H + 1
    N_org = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 0
    predictor = 0

    for i in range(4):
        print('i',i)
        print('before prune N',N)
        blocks += 3 * Deit_Block(N, C)
        if i == 3:
            break
        predictor += PredictorLG(N, C)
        #print('sparse[i]',sparse[i])
        N = int((N_org - 1) * (1-sparse[i])) + 1  
        print('after prune N',N)

    head = Head(C)
    return pe + blocks + predictor + head

def PatchEmbed4_2(C):
    return 3 * 64 * 49 * 112 ** 2 + 64 ** 2 * 9 * 112 ** 2 * 2 + 64 * C * 112 ** 2

def Dynamic_LV_ViT(img_size=224, P=16, H=14, C=384, rate=1.0, depth=16, mlp_ratio=3):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed4_2(C)
    blocks = 0
    predictor = 0

    for i in range(4):
        blocks += depth // 4 * Deit_Block(N, C, R=mlp_ratio)
        if i == 3:
            break
        predictor += PredictorLG(N, C)
        N = int((N - 1) * rate) + 1

    head1 = Head(C)
    aux_head = (N - 1) * C * 1000
    return pe + blocks + predictor + head1 + aux_head

def DeiT12(img_size=224, P=16, H=14, C=384):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 12 * Deit_Block(N, C)
    head = Head(C)
    return pe + blocks + head

def DeiT_Base(img_size=224, P=16, H=14, C=768):  #added
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 12 * Deit_Block(N, C)
    head = Head(C)
    return pe + blocks + head

def LV_ViT(img_size=224, P=16, H=14, C=384, rate=1.0, depth=16, mlp_ratio=3):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed4_2(C)
    blocks = depth * Deit_Block(N, C, R=mlp_ratio)
    head1 = Head(C)
    aux_head = (N - 1) * C * 1000
    return pe + blocks + head1 + aux_head


def AvgPool12(img_size=224, P=16, H=14, C=384):
    assert img_size == P * H
    N = H * H + 1
    pe = PatchEmbed(N, P, C)
    blocks = 6 * Deit_Block(N, C)
    N = (N - 1) // 4 + 1
    blocks += 6 * Deit_Block(N, C)
    head = Head(C)
    return pe + blocks + head

def print_dynamic_vit():

    #hello
    #print('DeiT 12/192', DeiT12(C=192) / 1e9)
    #print('DeiT 12/256', DeiT12(C=256) / 1e9)
    #print('DeiT 12/320', DeiT12(C=320) / 1e9)
    #print('DeiT 12/384', DeiT12(C=384) / 1e9)
    #print('DeiT 12/768', DeiT_Base(C=768) / 1e9)

    print('AvgPool 12/384', AvgPool12(C=384) / 1e9)
    print('DynamicViT 384/0.7', DynamicViT(C=384, rate=0.7) / 1e9)
    print('Dynamic_Soft_Mask_ViT 384', Dynamic_Soft_Mask_ViT(C=384, sparse=[0.54,0.72,0.85]) / 1e9) #change sparse here
    #print('DynamicViT 320/0.7', DynamicViT(C=320, rate=0.7) / 1e9)
    #print('DynamicViT 256/0.7', DynamicViT(C=256, rate=0.7) / 1e9)
    #print('-' * 10)
    #print('DynamicViT 384/1.0', DynamicViT(C=384, rate=1.0) / 1e9)
    #print('DynamicViT 384/0.9', DynamicViT(C=384, rate=0.9) / 1e9)
    #print('DynamicViT 384/0.8', DynamicViT(C=384, rate=0.8) / 1e9)
    #print('DynamicViT 384/0.7', DynamicViT(C=384, rate=0.7) / 1e9)
    #print('-' * 10)
    #print('LV ViT-S', LV_ViT(C=384, depth=16) / 1e9)
    #print('LV ViT-M', LV_ViT(C=512, depth=20) / 1e9)
    #print('-' * 10)
    #for rate in [1.0, 0.9, 0.8, 0.7, 0.5]:
    #    print(f'Dynamic LV ViT-S/{rate}', Dynamic_LV_ViT(C=384, depth=16, rate=rate) / 1e9)
    #for rate in [1.0, 0.9, 0.8, 0.7]:
    #    print(f'Dynamic LV ViT-M/{rate}', Dynamic_LV_ViT(C=512, depth=20, rate=rate) / 1e9)

if __name__ == '__main__':
    print_dynamic_vit()
