import cv2
import pathlib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def imscatter(x, y, image_list, ax=None, zoom=0.5):
    if ax is None:
        ax = plt.gca()
    im_list = [OffsetImage(cv2.imread(p)[:,:,::-1], zoom=zoom) for p in image_list]
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, im in zip(x, y, im_list):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

kills_reduced = np.load("/home/abe/kuma-ssl/dino/exp002/exp002-embed/tsne_features10_dino.npy")

test = pd.read_csv("/home/abe/kuma-ssl/data/test.csv")

def get_photo_id(x):
    x = x.split("/")[-1].split("_")[0][:-4]
    return x

test["photo_id"]=test["file_path"].apply(get_photo_id)


path_l = test["file_path"].to_numpy()

# perplexity: 20
#fig, ax = plt.subplots(figsize=(40,40))
#imscatter(kills_reduced[:,0], kills_reduced[:,1], path_l, ax=ax, zoom=0.2)

from PIL import Image
from functools import reduce
from skimage.transform import resize
def plot_tiles(imgs, emb, grid_units=50, pad=2):


    imgs = [cv2.imread(p)[:,:,::-1] for p in imgs]
    # roughly 1000 x 1000 canvas
    cell_width = 10000 // grid_units
    s = grid_units * cell_width

    nb_imgs = len(imgs)

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx]               
                
                resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                #print(resized_tile.shape)
                
                
                #exit()
                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = resized_tile
                
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict

#canvas, img_idx_dict = plot_tiles(path_l, kills_reduced, grid_units=30)

#print(type(canvas))
plt.figure(figsize=(25,25))
#plt.imshow(canvas)
#plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/TSNE_1.png")
#plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/TSNE_2.png")

def plot_tiles_v2(imgs,IDS, emb, grid_units=50, pad=1):


    imgs = [cv2.imread(p)[:,:,::-1] for p in imgs]
    # roughly 1000 x 1000 canvas
    cell_width = 10000 // grid_units
    s = grid_units * cell_width

    nb_imgs = len(imgs)
    id_unique = list(np.unique(IDS))
    LEN_IDS = len(id_unique)
    id_unique = dict(zip(id_unique,range(len(id_unique))))

    
    

    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx] 

                id = IDS[img_idx]
                id_index = id_unique[id]
                X = (1/LEN_IDS)*id_index

                
                resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                out_bbox = np.full((cell_width,cell_width,3),(X,X,X)).astype(np.float64)
                out_bbox[pad:-pad,pad:-pad,:]=resized_tile
                y = j * cell_width
                x = i * cell_width

                #canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = resized_tile
                canvas[s - y - cell_width:s - y, x :x+cell_width] = out_bbox
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict

def plot_tiles_v3(imgs,IDS, emb, grid_units=50, pad=2):


    imgs = [cv2.imread(p)[:,:,::-1] for p in imgs]
    # roughly 1000 x 1000 canvas
    cell_width = 10000 // grid_units
    s = grid_units * cell_width

    nb_imgs = len(imgs)
    id_unique = list(np.unique(IDS))
    LEN_IDS = len(id_unique)
    id_unique = dict(zip(id_unique,range(len(id_unique))))


    embedding = emb.copy()

    # rescale axes to make things easier
    min_x, min_y = np.min(embedding, axis=0)
    max_x, max_y = np.max(embedding, axis=0)

    embedding[:, 0] = s * (embedding[:, 0] - min_x) / (max_x - min_x)
    embedding[:, 1] = s * (embedding[:, 1] - min_y) / (max_y - min_y)

    canvas = np.ones((s, s, 3))
    
    img_idx_dict = {}

    for i in range(grid_units):
        for j in range(grid_units):

            idx_x = (j * cell_width <= embedding[:, 1]) & (embedding[:, 1] < (j + 1) * cell_width)
            idx_y = (i * cell_width <= embedding[:, 0]) & (embedding[:, 0] < (i + 1) * cell_width)

            points = embedding[idx_y & idx_x]

            if len(points) > 0:

                img_idx = np.arange(nb_imgs)[idx_y & idx_x][0]  # take first available img in bin
                tile = imgs[img_idx]               
                id = IDS[img_idx]
                id_index = id_unique[id]
                X = (1/LEN_IDS)*id_index

                resized_tile = resize(tile, output_shape=(cell_width - 2 * pad, cell_width - 2 * pad, 3))
                out_bbox = np.full((cell_width - 2 * pad,cell_width - 2 * pad,3),(X,X,X)).astype(np.float64)
                y = j * cell_width
                x = i * cell_width

                canvas[s - y - cell_width+pad:s - y - pad, x + pad:x+cell_width - pad] = out_bbox
                
                img_idx_dict[img_idx] = (x, x + cell_width, s - y - cell_width, s - y)

    return canvas, img_idx_dict

#canvas, img_idx_dict = plot_tiles_v3(path_l,test["photo_id"].to_numpy(), kills_reduced, grid_units=30)
canvas, img_idx_dict = plot_tiles(path_l, kills_reduced, grid_units=30)


print(type(canvas))
plt.figure(figsize=(25,25))
plt.imshow(canvas)
plt.axis("off")
#ax = plt.gca()

#ax.axes.xaxis.set_ticklabels([])
#ax.axes.yaxis.set_ticklabels([])
#plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/TSNE_1.png")
plt.savefig("/home/abe/kuma-ssl/dino/exp002/exp002-embed/TSNE_1_per10.jpg")