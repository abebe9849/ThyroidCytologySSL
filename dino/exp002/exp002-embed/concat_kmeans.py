import cupy as cp

from cuml import TSNE
from cuml import KMeans
from cuml.neighbors import NearestNeighbors

#from cuml import UMAP
#from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
test =pd.read_csv("/home/abe/kuma-ssl/data/test.csv")
folds =pd.read_csv("/home/abe/kuma-ssl/data/folds.csv")
import time
s = time.time()
folds_features = cp.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/folds_embed_300e_vit_s.npy")
test_features = cp.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/test_embed_300e_vit_s.npy")

ccat_f = cp.concatenate([folds_features,test_features],axis=0)

print(ccat_f.shape,test_features.shape)

km = KMeans(n_clusters=30).fit(ccat_f)

folds_label = km.labels_[:folds_features.shape[0]]
print(cp.unique(folds_label),len(folds_label))




cp.save(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/kmeans30_label_ccat.npy",km.labels_)