import cupy as cp

from cuml import TSNE
from cuml import KMeans
from cuml.neighbors import NearestNeighbors

#from cuml import UMAP
#from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
folds =pd.read_csv("/home/abe/kuma-ssl/data/folds.csv")
import time
s = time.time()
features = cp.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/folds_embed_300e_vit_s.npy")


"""
それぞれの画像について近傍のK枚の画像を保持→そのままアノテーションにつかう


"""

file_paths = folds["file_path"].to_numpy()

nbrs = NearestNeighbors(n_neighbors=10).fit(features)
distances, indices = nbrs.kneighbors(features)

indices = cp.asnumpy(indices)

for i in range(10):
    tmp = indices[:,i]
    a = file_paths[tmp]
    col = f"NearestNeighbors_path_{i}"
    folds[col]=a

folds.to_csv("/home/abe/kuma-ssl/dino/exp002/exp002-embed/folds_nn.csv",index=False)





