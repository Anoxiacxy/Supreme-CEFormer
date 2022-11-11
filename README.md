# 高效的卷积变形金钢（Convolutional Efficient Transformer）

这是关于[此论文](/paper.pdf) 的非官方代码复现，其中额外参考的资料有[轻量化卷积模块的替代](/2106.14881.pdf)，[自适应token采样](/2111.15667.pdf)。 

## 安装

```bash
pip install -r requirements.txt
```

## 文件结构说明

- `models` 目录保存了所有用到的模型，其中的 `ceformer` 是论文的模型实现。
- `datasets` 目录用于保存所有数据集的代码部分
- `data` 目录用于保存所有数据集的数据部分
- `tf-logs` 目录用于保存所有运行的结果

## 运行

### 推荐命令

```bash
python main.py fit --model ceformer --dataset imagenet --model_embed_dim 384
python main.py test --model ceformer --dataset imagenet --model_embed_dim 384 --ckpt_path /path/to/your/'.ckpt file'
```

### 所有支持的参数

```bash
usage: main.py [-h] [--model {ceformer,vit}] [--model_attention {e-attention,basic}] [--model_feedforward {enhanced,basic}]
               [--model_num_layers MODEL_NUM_LAYERS] [--model_embed_dim MODEL_EMBED_DIM] [--model_hidden_dim MODEL_HIDDEN_DIM]
               [--model_prune_rate MODEL_PRUNE_RATE] [--model_sampler_strategy {adaptive,top,random,distribution}]
               [--model_stem {thin_conv,conv,conv_s,conv_m,conv_l,patchify,alter1,alter2,alter3,alter4}] [--params] [--flops]
               [--learning_rate LEARNING_RATE] [--dataset {cifar10,cifar100,minist,imagenet}] [--ckpt_path CKPT_PATH]
               [--model_ckpt_path MODEL_CKPT_PATH] [--batch_size BATCH_SIZE] [--accumulate_batches ACCUMULATE_BATCHES]
               [--precision PRECISION]
               <phase>

```


### 参数详细说明

```bash
positional arguments:
  <phase>

optional arguments:
  -h, --help            show this help message and exit
  --model {ceformer,vit}
  --model_attention {e-attention,basic}
  --model_feedforward {enhanced,basic}
  --model_num_layers MODEL_NUM_LAYERS
  --model_embed_dim MODEL_EMBED_DIM
  --model_hidden_dim MODEL_HIDDEN_DIM
  --model_prune_rate MODEL_PRUNE_RATE
  --model_sampler_strategy {adaptive,top,random,distribution}
  --model_stem {thin_conv,conv,conv_s,conv_m,conv_l,patchify,alter1,alter2,alter3,alter4}
  --params
  --flops
  --learning_rate LEARNING_RATE
  --dataset {cifar10,cifar100,minist,imagenet}
                        dataset to fit
  --ckpt_path CKPT_PATH
                        path to your checkpoints
  --model_ckpt_path MODEL_CKPT_PATH
                        path to your checkpoints
  --batch_size BATCH_SIZE
  --accumulate_batches ACCUMULATE_BATCHES
  --precision PRECISION

```

### 一些使用的例子

```bash
# 使用 VisionTransformer 模型进行训练
python main.py fit --model vit
# 在 CEFormer 中使用基礎 attention 模块
python main.py fit --model ceformer --model_attention basic
# 調節模型的層數，token維度，隱藏層大小
python main.py fit --model_num_layers 24 --model_embed_dim 384 --model_hidden_dim 512
# 調節采樣的 token 比例
python main.py fit --model_prune_rate 0.9
# batch_size x accumulate_batches 得到的是一次更新的縂數據量，論文中的是 1024
# 顯存不夠的時候可以適當減小 batch_size，但也要對應增大 accumulate_batches
python main.py fit --batch_size 32 --accumulate_batches 32
```

