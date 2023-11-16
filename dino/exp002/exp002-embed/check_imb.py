import pandas as pd


df = pd.read_csv("/home/abe/kuma-ssl/data/folds.csv")
df = df[df["fold"]!=0]
tra_df = df.sample(frac=0.01,random_state=3000).reset_index(drop=True)


print(tra_df["label"].value_counts())
fold0 = pd.read_csv("/home/abe/kuma-ssl/dino/exp002/exp002-embed/fold_0_farnearsample.csv")
fold0 = pd.read_csv("/home/abe/kuma-ssl/dino/exp002/exp002-embed/fold_0_farsample.csv")
fold0 = pd.read_csv("/home/abe/kuma-ssl/dino/exp002/exp002-embed/fold_0_nearsample.csv")
print(fold0.columns)
#print(fold0["label"].value_counts())



