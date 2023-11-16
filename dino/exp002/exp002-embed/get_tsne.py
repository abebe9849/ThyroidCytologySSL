import cupy as cp

from cuml import TSNE
from cuml import KMeans
from cuml.neighbors import NearestNeighbors

#from cuml import UMAP
#from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import time
s = time.time()



test_f = cp.load(f"/home/abe/kuma-ssl/dino/imnetViTS/exp002__embed_imnet_vits.npy")
X_reduced = TSNE(n_components=2, random_state=2000,perplexity=10).fit_transform(test_f)
cp.save(f"/home/abe/kuma-ssl/dino/imnetViTS/test_tsne_seed2000_p10.npy",X_reduced)

exit()
def func(ROOT):

    test_f = cp.load(f"{ROOT}/test_emb.npy")
    X_reduced = TSNE(n_components=2, random_state=2000,perplexity=10).fit_transform(test_f)
    cp.save(f"{ROOT}/test_tsne_seed2000_p10.npy",X_reduced)
    return 

root_list = ["/home/abe/kuma-ssl/dino/exp002/finetune8/all/9/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/all/10/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/all/11/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/10per/2022/11/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/1per/2022/11/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/10per/2022/10/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/1per/2022/10/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/10per/2022/9/0",
             "/home/abe/kuma-ssl/dino/exp002/finetune8/1per/2022/9/0"]

root_list = ["/home/abe/kuma-ssl/dino/imnetViTS/all/0",
             "/home/abe/kuma-ssl/dino/imnetViTS/10per/2022/0",
             "/home/abe/kuma-ssl/dino/imnetViTS/1per/2022/0",
]

for r in root_list:
    func(r)
#cp.save(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/cat_tsne_features10_dino.npy",X_reduced)

#X_reduced = TSNE(n_components=2, random_state=0,perplexity=10).fit_transform(features)
#X_reduced = UMAP(n_components=2, random_state=0).fit_transform(features)
#kmeans_float = KMeans(n_clusters=120).fit(features)

from tqdm import tqdm

"""
import matplotlib.pyplot as plt
distortions =[]
for i  in tqdm(range(1,2000)):                # 1~10クラスタまで一気に計算 
    km = KMeans(n_clusters=i)
    km.fit(features)                         # クラスタリングの計算を実行
    distortions.append(km.inertia_)

plt.plot(range(1,2000),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/elbo3.png")
"""





#cp.save(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/kmeans100_center_folds.npy",kmeans_float.cluster_centers_)
#cp.save(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/kmeans100_label_folds.npy",kmeans_float.labels_)
#cp.save(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/tsne_features10_dino.npy",X_reduced)



exit()
nbrs = NearestNeighbors(n_neighbors=100).fit(features)
distances, indices = nbrs.kneighbors(kmeans_float.cluster_centers_)

far_sample = indices[:,-9:]

f = indices[:,-4:]
n = indices[:,:5]

print(f.shape,n.shape)
fn = cp.concatenate([f,n],axis=1)
print(fn.shape)


nbrs = NearestNeighbors(n_neighbors=9).fit(features)
distances, indices = nbrs.kneighbors(kmeans_float.cluster_centers_)

print(type(indices))
index = cp.asnumpy(indices.ravel())
print(far_sample.shape)
print(far_sample)
far_sample = cp.asnumpy(far_sample.ravel())

fn = cp.asnumpy(fn.ravel())



select = folds.iloc[fn,:]

for i in range(5):
    train = folds[folds["fold"]!=i]
    train = train.sample(frac=0.01,random_state=2000).reset_index(drop=True)
    tmp = select[select["fold"]!=i]
    print(tmp.shape,train.shape)
    print(tmp["label"].value_counts())

    tmp.to_csv(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/fold_{i}_farnearsample.csv",index=False)


print(indices)

#第一近傍はおなじclsなのか
nbrs = NearestNeighbors(n_neighbors=2).fit(features)
distances, indices = nbrs.kneighbors(features)

indices = cp.asnumpy(indices)
labels = folds["label"].to_numpy()
cnt = 0
count = 0
for i in range(len(folds)):
    cls0=labels[i]

    #print(f"{i},{indices[i][1]}")
    if i==indices[i][1]:
        cnt=0
        
        continue
    count+=1
    cls1 = labels[indices[i][1]]
    #print(cls0,cls1)
    if cls0==cls1:
        cnt +=1

print("Acc:",cnt/count,count,cnt)
    


print(time.time()-s)