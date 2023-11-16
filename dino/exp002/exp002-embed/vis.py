import numpy as np
from sklearn.manifold import TSNE
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
import sys

X_reduced = np.load(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/tsne_features10_dino.npy")

root = "/home/u094724e/kuma_ssl/src/mae/exp100/10per/0"

predict = pd.read_csv(f"/home/abe/kuma-ssl/dino/exp002/exp002-1/sub.csv")
predict_pred = predict.iloc[:,-9:].values
predict_pred_th = np.argmax(predict_pred,axis=1)
predict["pred_label"]=predict_pred_th

predict["tsne0"] = X_reduced[:, 0]
predict["tsne1"] = X_reduced[:, 1]


sub = predict.copy()


plt.figure(figsize=(15,15))
clist = ["orange","pink","blue","brown","red","grey","yellow","green"]
clist = ["orange","pink","blue","brown","red","grey","yellow","green","black"]
cdict = dict(zip(range(predict["label"].nunique()),clist))
color_dict ={"orange":[255,165,0],"pink":[255,10,255],"blue":[0,0,255],
"brown":[153,51,0],"red":[255,0,0],"grey":[150,150,150],
"yellow":[255,255,0],"green":[0,125,0],"black":[0,0,0],
}


predict_other = predict[predict["pred_label"]!=8]

tn = predict[predict["pred_label"]==8]
tn = tn[tn["label"]==8]

predict = predict[predict["label"]!=8]
predict = predict[predict["pred_label"]==8]


for i in predict["label"].unique():
    tmp = predict[predict["label"]==i]
    cls_name = tmp["label_name"].unique()[0]
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[i], label=cls_name, s=25)




for i in tn["label"].unique():
    tmp = tn[tn["label"]==i]
    cls_name = tmp["label_name"].unique()[0]
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    #plt.scatter(x,y, color=cdict[i], label=cls_name, s=3)

for i in predict_other["label"].unique():
    tmp = predict_other[predict_other["label"]==i]
    cls_name = tmp["label_name"].unique()[0]
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y, color=cdict[i], label=cls_name, s=3)


def get_photo_id(x):
    x = x.split("/")[-1].split("_")[0][:-4]
    return x

sub["photo_id"]=sub["file_path"].apply(get_photo_id)





from sklearn.cluster import KMeans

predict["kmean_all"]=np.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/kmeans30_label.npy")

for i in range(30):
    tmp = predict[predict["label"]==i]
    x = tmp["tsne0"]
    y = tmp["tsne1"]
    plt.scatter(x,y,cmap ="hsv" , s=3)

plt.savefig(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/FN_tsne.png")

"""
plt.figure(figsize=(15,15))
for i in predict["label"].unique():
    tmp = predict[predict["label"]==i]
    cls_name = tmp["label_name"].unique()[0]
    if "PTL" not in cls_name:continue
    ptl_feature = tmp[["tsne0","tsne1"]].to_numpy()
    kmean_ptl = KMeans(n_clusters=4).fit(ptl_feature)

    plt.scatter(ptl_feature[:,0],ptl_feature[:,1], s=5,c=kmean_ptl.labels_)


plt.legend()
plt.savefig(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/PTL_tsne10_cls4.png")
"""
plt.figure(figsize=(15,15))
for i in predict["label"].unique():
    tmp = predict[predict["label"]==i]
    cls_name = tmp["label_name"].unique()[0]
    if "ATC" not in cls_name:continue
    ptl_feature = tmp[["tsne0","tsne1"]].to_numpy()
    kmean_atc = KMeans(n_clusters=2).fit(ptl_feature)

    plt.scatter(ptl_feature[:,0],ptl_feature[:,1], s=5,c=kmean_atc.labels_)

targets_cm  = ["ATC","FNA","FNC","FNO","MTC","POC","PTL","TPC","negative"]
target_di = dict(zip(targets_cm,range(len(targets_cm))))

import os,shutil
atc_df = predict[predict["label"]==target_di["ATC"]]
print(atc_df.columns)
DIR = "/home/abe/kuma-ssl/dino/exp002/exp002-embed"
for i in range(2):
    tmp_paths = atc_df[kmean_atc.labels_==i].file_path


    os.makedirs(f"{DIR}/atc_{i}", exist_ok=True)

    for x in tmp_paths:
        path = x.split("/")[-1]
        shutil.copyfile(x,f"{DIR}/atc_{i}/{path}")

    shutil.make_archive(f"{DIR}/atc_{i}", format='zip', root_dir=f"{DIR}/atc_{i}")

plt.legend()
plt.savefig(f"/home/abe/kuma-ssl/dino/exp002/exp002-embed/ATC.png")
sys.exit()