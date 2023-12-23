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
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import json
from pathlib import Path
import pandas as pd
import sklearn.metrics as metrics
import torch,glob
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

from PIL import Image
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import numpy as np
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
import random,timm
def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
    
#all_densenet_models = timm.list_models('*vit*small*')

#print(all_densenet_models)
#exit()
seed_torch()
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def eval_linear(args):
    #utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    
    
    
    
    # load weights to evaluate
    model = timm.create_model("vit_small_patch16_224_in21k",pretrained=True,num_classes=8)
    model.cuda()

    model.train()
    print(model)
    print(f"Model {args.arch} built.")


    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(224, interpolation=3),
        #pth_transforms.Resize(256, interpolation=3),
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
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.evaluate:
        utils.load_pretrained_linear_weights(model, args.arch, args.patch_size)
        test_stats = validate_network(val_loader, model, args.n_last_blocks, args.avgpool_patchtokens)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return
    
    
    train_transform = pth_transforms.Compose([
        pth_transforms.RandomResizedCrop(224),
        pth_transforms.RandomHorizontalFlip(),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset_train = TrainDataset(tra_df, transform=train_transform)
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
        model.parameters(),
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
        state_dict=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    best_state = None
    best_score = 0
    best_preds = None

    patience = 20

    

    

    for epoch in range(start_epoch, args.epochs):
        #train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats,preds,valid_labels = validate_network(val_loader, model, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Accuracy at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
            one_hot_label = np.eye(8)[valid_labels.astype(np.int64)]
            pr_auc = 0
            for i in range(8):
                label_ = one_hot_label[:,i]
                pred_ = preds[:,i]
                precision, recall, thresholds = metrics.precision_recall_curve(label_, pred_)
                pr_auc += metrics.auc(recall, precision)/8
            if pr_auc>best_score:#pr_auc best
                best_score = pr_auc
                patience =20
                best_preds = preds
                torch.save(model.state_dict(), os.path.join(args.output_dir,"linear_w.pth"))
            else:
                patience -=1
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats)+ f"best pr_auc:{best_score}"+ "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best pr_auc":best_score,
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

        if patience==0:
            break
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


    state_dict = torch.load( os.path.join(args.output_dir,"linear_w.pth"),map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    _,best_preds,_ = validate_network(val_loader, model, args.n_last_blocks, args.avgpool_patchtokens)
    

    for i in range(8):
        col = f"pred_{i}"
        val_df[col]=best_preds[:,i]

    val_df.to_csv( os.path.join(args.output_dir,f"oof_fold{args.fold}_dino.csv"),index=False)
    
    

    state_dict = torch.load( os.path.join(args.output_dir,"linear_w.pth"),map_location="cuda")
    model.load_state_dict(state_dict)
    model.eval()
    print("MDOEL   ~~")
    _,preds,tes_labels = validate_network(test_loader, model, args.n_last_blocks, args.avgpool_patchtokens)

    for i in range(8):
        col = f"pred_{i}"
        test_df[col]=preds[:,i]
    test_df.to_csv( os.path.join(args.output_dir,f"sub_224_fold{args.fold}_dino.csv"),index=False)

def train(model, optimizer, loader, epoch, n, avgpool):
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        
        output = model(inp)

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
def validate_network(val_loader, model, n, avgpool):
    model.eval()
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
            
            output = model(inp)
        loss = nn.CrossEntropyLoss()(output, target)

        preds.append(softmax(output).to('cpu').numpy())
        valid_labels.append(target.to('cpu').numpy())

        
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 3))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        
        metric_logger.meters['acc3'].update(acc5.item(), n=batch_size)
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@3 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc3, losses=metric_logger.loss))
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
    parser.add_argument('--pretrained_weights', default='/home/abe/kuma-ssl/dino/exp002/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument("--seed", default=1000, type=int)
    parser.add_argument("--fold", default=0, type=int)
    parser.add_argument("--freeze", default=9, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/home/abe/kuma-ssl/dino/imnetViTS/all", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=8, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,str(args.fold))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    eval_linear(args)
