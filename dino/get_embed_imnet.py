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
import argparse
import json,glob
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
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import utils
import vision_transformer as vits

from PIL import Image
#from PIL import ImageFile
#ImageFile.LOAD_TRUNCATED_IMAGES = True
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
import timm

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
        
    model = timm.create_model("vit_small_patch16_224_in21k",pretrained=True,num_classes=0)
    
    model.cuda()
    model.eval()



    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


    test_df = pd.read_csv("/home/abe/kuma-ssl/data/test.csv")

    dataset_tesst = TrainDataset(test_df, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        dataset_tesst,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    embeds = validate_network_(test_loader, model, args.n_last_blocks, args.avgpool_patchtokens)

    np.save(os.path.join(args.output_dir,"ImageNet_ViT-S_.npy"),embeds)
    X_reduced = TSNE(n_components=2, random_state=2000,perplexity=10).fit_transform(embeds)
    np.save(os.path.join(args.output_dir,"ImageNet_ViT-S_TSNE.npy"),X_reduced)






@torch.no_grad()
def validate_network_(val_loader, model, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    preds = []
    valid_labels = []
    embeds = []
    softmax= nn.Softmax(dim=1)
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():

            output = model(inp)
        embeds.append(output.to('cpu').numpy())

        batch_size = inp.shape[0]
    embeds = np.concatenate(embeds)
    return embeds
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument("--fold", default=1, type=int)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="/home/abe/kuma-ssl/dino/imnetViTS", help='Path to save logs and checkpoints')
    args = parser.parse_args()
    #args.output_dir = os.path.join(args.output_dir,str(args.fold))
    eval_linear(args)
