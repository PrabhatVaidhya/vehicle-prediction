import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv')
orig = pd.read_csv('EV_Adoption_and_Range_Anxiety_Dataset.csv')

print(f"Loaded train data: {train.shape}, original data: {orig.shape}")

def preprocess(df):
    d = df.copy()
    if 'Will_Buy_EV' in d.columns:
        d['Target'] = (d['Will_Buy_EV'] == 'Yes').astype(int)
    else:
        d['Target'] = None
        
    # Ordinal / Numeric encodings
    d['Range_Anxiety_Num'] = d['Range_Anxiety_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
    d['Home_Charging_Num'] = (d['Home_Charging_Possible'] == 'Yes').astype(int)
    d['Subsidy_Num'] = (d['Subsidy_Available'] == 'Yes').astype(int)
    
    # Feature engineering
    d['Total_Charging_Stations'] = d['Charging_Stations_Near_Home'] + d['Charging_Stations_Near_Work']
    d['Charging_Diff'] = d['Charging_Stations_Near_Work'] - d['Charging_Stations_Near_Home']
    d['Charging_Ratio'] = (d['Charging_Stations_Near_Work'] + 1) / (d['Charging_Stations_Near_Home'] + 1)
    d['Commute_per_Total_Charging'] = d['Daily_Commute_km'] / (d['Total_Charging_Stations'] + 1)
    d['Income_per_Car'] = d['Annual_Income_USD'] / (d['Number_of_Cars_Owned'] + 1)
    d['Income_per_Age'] = d['Annual_Income_USD'] / (d['Age'] + 1)
    d['Env_Concern_x_Income'] = d['Environmental_Concern_Level'] * d['Annual_Income_USD']
    d['Range_Anxiety_x_Commute'] = d['Range_Anxiety_Num'] * d['Daily_Commute_km']
    d['EV_Readiness_Score'] = d['Home_Charging_Num'] + d['Subsidy_Num'] + (d['Total_Charging_Stations'] >= 10).astype(int)
    
    # Group aggregations
    city_income_mean = d.groupby('City_Type')['Annual_Income_USD'].transform('mean')
    d['Income_vs_CityMean'] = d['Annual_Income_USD'] / (city_income_mean + 1)
    
    city_commute_mean = d.groupby('City_Type')['Daily_Commute_km'].transform('mean')
    d['Commute_vs_CityMean'] = d['Daily_Commute_km'] / (city_commute_mean + 1)
    
    cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible', 'Subsidy_Available', 'Range_Anxiety_Level']
    for c in cat_cols:
        d[c] = d[c].astype('category')
        
    return d

df_processed = preprocess(train)
features = [c for c in df_processed.columns if c not in ['id', 'Will_Buy_EV', 'Target']]
print(f"Features ({len(features)}): {features}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(df_processed))
scores = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(df_processed, df_processed['Target'])):
    X_tr, y_tr = df_processed.iloc[trn_idx][features], df_processed.iloc[trn_idx]['Target']
    X_va, y_val = df_processed.iloc[val_idx][features], df_processed.iloc[val_idx]['Target']
    
    model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    preds = model.predict_proba(X_va)[:, 1]
    oof[val_idx] = preds
    score = roc_auc_score(y_val, preds)
    scores.append(score)
    print(f"Fold {fold+1} ROC-AUC: {score:.6f}")

overall_score = roc_auc_score(df_processed['Target'], oof)
print(f"--> Overall 5-Fold OOF ROC-AUC: {overall_score:.6f}")
