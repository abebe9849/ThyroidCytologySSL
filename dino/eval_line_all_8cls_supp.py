# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import json
from pathlib import Path
import pandas as pd
import sklearn.metrics as metrics
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
import timm
from PIL import Image
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torch


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""
    def __init__(self, block, layers, cifar_head=False, hparams=None):
        super().__init__(block, layers)
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
            self.relu = nn.ReLU(inplace=True)
        self.hparams = hparams

        print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [3, 4, 6, 3], cifar_head=cifar_head, hparams=hparams)

class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df,train=True,transform=None):
        self.df = df
        self.transform = transform

        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file_path'].values[idx]
        image = Image.open(file_path)
        
        image = self.transform(image)
        

        label_ = self.df['label'].values[idx]
        label = torch.tensor(label_).long()
        return image,label
import random
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_torch()
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import math
import torch
import torch.nn as nn
from functools import partial, reduce
from operator import mul

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.layers import PatchEmbed

__all__ = [
    'vit_small', 
    'vit_base',
    'vit_conv_small',
    'vit_conv_base',
]


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=False, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'qkv' in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(6. / float(m.weight.shape[0] // 3 + m.weight.shape[1]))
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        #assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False



def vit_small_mocov3(**kwargs):
    model = VisionTransformerMoCo(
        patch_size=16, embed_dim=384, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
from transformers import CLIPProcessor, CLIPModel
def eval_linear(args):
    #utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.model_name == "UNI":
        model = timm.create_model("hf-hub:MahmoodLab/UNI", pretrained=True, init_values=1e-5, dynamic_img_size=True,num_classes=0)
        embed_dim = model.num_features
    elif args.model_name == "gigapath":
        model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True,num_classes=0)
        embed_dim = model.num_features
    elif args.model_name == "plip":
        model = CLIPModel.from_pretrained("vinid/plip")
        embed_dim = 512
    elif args.model_name == "simclr":
        ckpt = torch.load("./checkpoint_simclr.pth.tar", map_location="cpu")

        model = ResNet50(cifar_head=False, hparams=None)
        new_state_dict = {}
        for k, v in ckpt['state_dict'].items():
            if k.startswith('convnet.'):
                new_key = k.replace('convnet.', '')
                new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
        embed_dim = 2048
    elif args.model_name == "mocov2":
        model_PATH = "./checkpoint_mocov2.pth.tar"
        checkpoint = torch.load(model_PATH, map_location="cpu")
        model = models.resnet50()
        # rename moco pre-trained keys
        state_dict = checkpoint["state_dict"]
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith("module.encoder_q") and not k.startswith(
                "module.encoder_q.fc"
            ):
                state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        model.fc = nn.Identity()
        embed_dim = 2048
    elif args.model_name == "mocov3":
        model = vit_small_mocov3()
        model.head = nn.Identity()
        embed_dim = 384
        checkpoint = torch.load("checkpoint_mocov3.pth.tar", map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only base_encoder up to before the embedding layer
            if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % "head"):
                # remove prefix
                state_dict[k[len("module.base_encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
    elif args.model_name == "maxvit":
        model = timm.create_model("maxvit_tiny_rw_224", pretrained=False,img_size = 96,num_classes=0)
        embed_dim = model.num_features
        
    
    model.cuda()
    model.eval()
    # load weights to evaluate
    #utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    #linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(224, interpolation=3),
        #pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    FOLD = args.fold
    
    df = pd.read_csv("/home/abe/kuma-ssl/data/folds8.csv")
    tra_df = df[df["fold"]!=FOLD].reset_index(drop=True)

    val_df = df[df["fold"]==FOLD].reset_index(drop=True)
    dataset_val = TrainDataset(val_df, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu*4,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_df = pd.read_csv("/home/abe/kuma-ssl/data/test8.csv")


    dataset_tesst = TrainDataset(test_df, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_tesst,
        batch_size=args.batch_size_per_gpu*4,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(linear_classifier, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = TrainDataset(tra_df, transform=train_transform)
    #dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        #sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    
    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    best_state = None
    best_score = 0
    best_preds = None
    
    patient=20
    for epoch in range(start_epoch, args.epochs):
        #train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats,preds,valid_labels = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            one_hot_label = np.eye(8)[valid_labels.astype(np.int64)]
            pr_auc = 0
            pr_auc_each = []
            for i in range(8):
                label_ = one_hot_label[:,i]
                pred_ = preds[:,i]
                precision, recall, thresholds = metrics.precision_recall_curve(label_, pred_)
                each =  metrics.auc(recall, precision)/8
                pr_auc_each.append(each)
                pr_auc +=each
            if pr_auc>best_score:#pr_auc best
                best_score = pr_auc
                best_preds = preds
                patient=20
                torch.save(linear_classifier.state_dict(), os.path.join(args.output_dir,"linear_w.pth"))
            else:
                patient-=1
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats)+ f"best pr_auc:{best_score}"+ "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best pr_auc":best_score,
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

        if patient==0:break
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    
    state_dict = torch.load( os.path.join(args.output_dir,"linear_w.pth"),map_location="cuda")
    linear_classifier.load_state_dict(state_dict)
    linear_classifier.eval()
    test_stats,best_preds,valid_labels = validate_network(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
    for i in range(8):
        col = f"pred_{i}"
        val_df[col]=best_preds[:,i]

    val_df.to_csv( os.path.join(args.output_dir,f"oof_fold{args.fold}_dino.csv"),index=False)

    state_dict = torch.load( os.path.join(args.output_dir,"linear_w.pth"),map_location="cuda")
    linear_classifier.load_state_dict(state_dict)
    linear_classifier.eval()
    _,preds,tes_labels = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)

    for i in range(8):
        col = f"pred_{i}"
        test_df[col]=preds[:,i]
    test_df.to_csv( os.path.join(args.output_dir,f"sub_fold{args.fold}_dino.csv"),index=False)

def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "plip" in args.model_name:
                output = model.get_image_features(inp)
            else:
                output = model(inp)
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    preds = []
    valid_labels = []
    softmax= nn.Softmax(dim=1)
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "plip" in args.model_name:
                output = model.get_image_features(inp)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        preds.append(softmax(output).to('cpu').numpy())
        valid_labels.append(target.to('cpu').numpy())

        if linear_classifier.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 3))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.num_labels >= 5:
            metric_logger.meters['acc3'].update(acc5.item(), n=batch_size)
    if linear_classifier.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc3, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()},preds,valid_labels


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='/home/abe/kuma-ssl/dino/exp202/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--model_name", default="UNI", type=str, help="Model name")
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/home/abe/kuma-ssl/dino/exp202/all8", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=8, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    ROOT = args.output_dir 
    
    args.fold = 0
    args.output_dir = ROOT+str(args.fold)    
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    args.fold = 1
    args.output_dir =ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    args.fold = 2
    args.output_dir = ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    args.fold = 3
    args.output_dir =ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    args.fold = 4
    args.output_dir =ROOT+str(args.fold)
    os.makedirs(args.output_dir,exist_ok=True)
    eval_linear(args)
    

    cols = [f"pred_{i}" for i in range(args.num_labels)]
    sub_pred_ = []
    oof_ = []
    for fold in [0,1,2,3,4]:
    
        sub = pd.read_csv(os.path.join(ROOT+str(fold),f"sub_fold{fold}_dino.csv"))
        print(sub[cols].to_numpy().shape)
        sub_pred_.append(sub[cols].to_numpy())
        oof_.append(pd.read_csv(os.path.join(ROOT+str(fold),f"oof_fold{fold}_dino.csv")))
    sub_pred_ = np.mean(np.stack(sub_pred_),axis=0)
    oof_ = pd.concat(oof_,axis=0)
    oof_.to_csv(f"{ROOT}oof.csv",index=False)
    
    for i in range(args.num_labels):
        col = f"pred_mean_{i}"
        sub[col]=sub_pred_[:,i]
    sub.to_csv(f"{ROOT}sub.csv",index=False)
