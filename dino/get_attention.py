import os,time,glob
import pandas as pd
test = pd.read_csv("/home/abe/kuma-ssl/data/test.csv")

test_imgs = test["file_path"].to_numpy()



import subprocess

outdir = "/home/abe/kuma-ssl/dino/exp002/attention"

c = 0
for image in test_imgs:
    #print(image)
    if image.split("/")[-1].split(".")[0].split("_")[-2]=="n":continue
    if "FNC" in image:continue
    if "FNA" in image:continue
    c +=1
    os.system(f"python /home/abe/kuma-ssl/dino/visualize_attention_fine10.py --image_path {image}")
print(c)