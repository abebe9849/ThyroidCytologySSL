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
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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

from PIL import Image
import warnings
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")
import numpy as np
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,df,train=True,transform=None):
        self.df = df
        self.transform = transform

        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df['file_path'].values[idx]
        image = Image.open(file_path)#大きい画像
                
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
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    #linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    #X = 224/256
    X = 1
    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize((int(960*X),int(1280*X)), interpolation=3),
        #pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    FOLD = args.fold
    
    test_df = pd.read_csv("/home/abe/kuma-ssl/precrop/test.csv")
    print(test_df.shape)
    test_df = test_df.sample(n=1000).reset_index(drop=True)
    



    dataset_tesst = TrainDataset(test_df,transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_tesst,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
    )



    state_dict = torch.load( os.path.join(args.output_dir,"linear_w.pth"),map_location="cuda")
    linear_classifier.load_state_dict(state_dict)
    linear_classifier.eval()
    preds,tes_labels = validate_network(test_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
    
    preds = np.stack(preds)
    
    
    n_patch = preds.shape[1]

    print(preds.shape)

    #np.save(os.path.join(args.output_dir,f"test_precrop_pred_n{n_patch}_fold{args.fold}.npy"),preds)
    np.save(os.path.join(args.output_dir,f"test_precrop_pred_n___{n_patch}_fold{args.fold}.npy"),preds)

    #test_df.to_csv( os.path.join(args.output_dir,f"precrop_fold{args.fold}_dino.csv"),index=False)

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
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
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
def sliding_window_split(image, window_size=224, stride=224):
    N, _, H, W = image.size()
    num_windows_h = (H - window_size) // stride + 1
    num_windows_w = (W - window_size) // stride + 1
    patches = []
    for i in range(num_windows_h):
        for j in range(num_windows_w):
            window = image[:, :, i*stride:i*stride+window_size, j*stride:j*stride+window_size]
            patches.append(window[0])
    if len(patches) > 0:
        return torch.stack(patches, dim=0)
    else:
        raise ValueError("スライディングウィンドウの結果が空です。ウィンドウサイズとストライドを調整してください。")



inp = torch.randn((1,3,960,1280))

tmp = sliding_window_split(inp,stride=224)
print(tmp.shape)



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
        
        inp = sliding_window_split(inp)
        

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        
        preds.append(softmax(output).to('cpu').numpy())
        valid_labels.append(target.to('cpu').numpy())


    
    return preds,valid_labels


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

import time

s = time.time()
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
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument("--fold", default=4, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/home/abe/kuma-ssl/dino/exp002/all8", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=8, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir,str(args.fold))
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.fold=0
    
    fold = 0

    eval_linear(args)
    args.fold=1
    eval_linear(args)
    args.fold=2
    eval_linear(args)
    args.fold=3
    eval_linear(args)
    args.fold=4
    eval_linear(args)
    

import lightgbm as lgb
import pickle,glob

for fold in [0,1,2,3,4]:
    with open(f"/home/abe/kuma-ssl/precrop/stacking/exp002_20patch/fold{fold}_lgbm.pkl", 'rb') as f:
        loaded_model = pickle.load(f)
        
    emb = np.load(os.path.join(args.output_dir,f"test_precrop_pred_n___20_fold{fold}.npy"))

    # 読み込んだモデルを使用して予測などを行う
    # 例: テストデータを使った予測
    a,b,c = emb.shape
    
    tra_x = emb.reshape(-1,b*c)
    y_pred = loaded_model.predict(tra_x)
    print(y_pred.shape,y_pred[:10])
    
print((time.time()-s)/1000)



#0.10633262920379638
#0.09676823306083679

