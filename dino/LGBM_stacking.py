
import lightgbm as lgbm
import pandas as pd
import numpy as np
oof = pd.read_csv("/home/abe/kuma-ssl/precrop/oof.csv")
test = pd.read_csv("/home/abe/kuma-ssl/precrop/test.csv")

print(oof["photo"].unique()[:5])

import pickle



params = {
    'learning_rate': .1,
    'reg_lambda': 1.,
    'reg_alpha': .1,
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'max_depth': 5, 
    'n_estimators': 10000, 
    'colsample_bytree': .5, 
    'min_child_samples': 10,
    'subsample_freq': 3,
    'subsample': .9,
    'importance_type': 'gain', 
    'random_state': 71,
}




root = "/home/abe/kuma-ssl/precrop/stacking/exp002_20patch"
N=20
models = []
oof_pred = np.zeros((oof.shape[0],8), dtype=np.float32)
from sklearn.metrics import roc_auc_score,accuracy_score
import sklearn.metrics as metrics

from sklearn.metrics import classification_report

def get_score(gt,preds):
    one_hot_label = np.eye(8)[gt.astype(np.int64)]
    pr_auc = 0
    each = []
    for i in range(8):
        label_ = one_hot_label[:,i]
        pred_ = preds[:,i]
        precision, recall, thresholds = metrics.precision_recall_curve(label_, pred_)
        pr_auc += metrics.auc(recall, precision)/8
        each.append(metrics.auc(recall, precision))
    th_pred = np.argmax(preds,axis=1)
    acc = accuracy_score(gt,th_pred)

    return pr_auc,f"Acc:{acc}",each


test_pred = []
for fold in range(5):
    
    val_index = oof[oof["fold"]==fold].index
    tra_index = oof[oof["fold"]!=fold].index
    val_df = oof.loc[val_index].reset_index(drop=True)
    tra_df = oof.loc[tra_index].reset_index(drop=True)
    
    tra_y = tra_df["label"].values
    val_y = val_df["label"].values
    
    X = np.load(f"{root}/oof_precrop_pred_n{N}_fold{fold}.npy")#N,k,8
    test_X = np.load(f"{root}/test_precrop_pred_n{N}_fold{fold}.npy")#N,k,8

    a,b,c = X.shape
    
    tra_x = X[tra_index].reshape(-1,b*c)
    
    test_X = test_X.reshape(-1,b*c)
    
    val_x = X[val_index].reshape(-1,b*c)
    clf = lgbm.LGBMClassifier(**params)
    

    
    
    
    clf.fit(tra_x, tra_y,
                    eval_set=[(val_x, val_y)],  
                    early_stopping_rounds=100,
                    verbose=0)
    
    pred_i = clf.predict_proba(val_x)
    
    test_pred.append(clf.predict_proba(test_X))
    
    oof_pred[val_index]=pred_i
    


    pickle.dump(clf, open(f"{root}/fold{fold}_lgbm.pkl", 'wb'))
    
    models.append(clf)
    
print(oof_pred.shape)


#models

test_pred = np.mean(test_pred,axis=0)


CV_score = get_score(oof["label"].values,oof_pred)
print(CV_score)

th_oof_pred = np.argmax(oof_pred,axis=-1)
print(classification_report(oof["label"].values,th_oof_pred))

test_score = get_score(test["label"].values,test_pred)
print(test_score)

th_test_pred = np.argmax(test_pred,axis=-1)

print(classification_report(test["label"].values,th_test_pred))
