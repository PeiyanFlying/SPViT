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
