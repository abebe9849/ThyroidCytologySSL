import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

folds =pd.read_csv("/home/abe/kuma-ssl/data/test.csv")
kmeans_label = np.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/kmeans30_label.npy")


import cv2
def vis(paths):

    vsi = np.zeros((256*14,256*14,3)).astype(np.uint8)
    for i in range(14):
        for j in range(14):
            vsi[256*i:256*i+256,256*j:256*j+256,:]=cv2.imread(paths[i+j*14])[:,:,::-1]
    plt.figure(figsize=(22,22))
    plt.imshow(vsi)


def vis_10(paths):
    for L in range(10):
        vsi = np.zeros((256*14,256*14,3)).astype(np.uint8)
        for i in range(14):
            for j in range(14):
                vsi[256*i:256*i+256,256*j:256*j+256,:]=cv2.imread(paths[i+j*14+14*14*L])[:,:,::-1]
        plt.figure(figsize=(22,22))
        plt.imshow(vsi)
        plt.savefig(f"kmean0_{L}.png")

path = folds["file_path"].to_numpy()[kmeans_label==0]
vis_10(path)