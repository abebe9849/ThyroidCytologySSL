# ThyroidCytologySSL


## Requirements
- **Operating system**: Testing has been performed on Ubuntu 20.04.
- Python == 3.9
- PyTorch == 1.12.0

## Pretrained Models
The checkpoint and loss logs for the DINO pre-trained model are located [here](https://www.kaggle.com/datasets/abebe9849/kumadinoe300s). 
## Training DINO

１．Prepare a CSV with a column named 'file_path' that includes the absolute paths of all images, and modify the following section accordingly
https://github.com/abebe9849/ThyroidCytologySSL/blob/1da607a94fccfee592e9b6cb5dadee155c4c0af8/dino/main_dino.py#L134
```
aimed_df = pd.read_csv("/home/abe/kuma-ssl/data/all_df.csv")
```
2. training 
```
cd dino
python -m torch.distributed.launch --nproc_per_node=2 main_dino.py --arch vit_small --batch_size_per_gpu 256
```
If you experience NaN values in the DINO loss, please set `fp_16` to `False`, and also reduce the value of the gradient clipping.


## Training MAE
１．Prepare a CSV with a column named 'file_path' that includes the absolute paths of all images, and modify the following section accordingly
https://github.com/abebe9849/ThyroidCytologySSL/blob/f1721e6d7563f979404aa5ae929eddeffb2458e9/mae/main_pretrain.py#L109

```
aimed_df = pd.read_csv("/home/abe/kuma-ssl/data/all_df.csv")
```
2. training 
```
cd mae
python -m torch.distributed.launch --nproc_per_node=2 main_pretrain.py --arch vit_small --batch_size_per_gpu 256
```

## get embedding from pretrained ViT→TSNE

### ImageNet pretrained
```
dino/get_embed_imnet.py
```
### DINO pretrained
```
dino/get_embed.py --pretrained_weights [dino_vits_chechpoint.pth]
```

## visualize attention map

```
dino/visualize_attention.py --pretrained_weights [dino_vits_chechpoint.pth] --image_path [image path]
```

## evaluate ViT-S

### DINO-ViT-S
100% labels
```
python dino/eval_line_all_8cls.py
```
10% or 1% labels 5seed
```
python dino/eval_line_few_8cls.py --rate [10 or 1]
```
### ImageNet--ViT-S
100% labels
```
python dino/imnet_fine_10_8cls.py
```
10% or 1% labels 5seed
```
python dino/imnet_fine_few_8cls.py --rate [10 or 1]
```











