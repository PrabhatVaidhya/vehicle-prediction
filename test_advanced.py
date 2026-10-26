import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("train.csv")
orig = pd.read_csv("EV_Adoption_and_Range_Anxiety_Dataset.csv")

if "Buyer_ID" in orig.columns:
    orig = orig.drop(columns=["Buyer_ID"])

train["is_orig"] = 0
orig["is_orig"] = 1
orig["id"] = -1

combined = pd.concat([train, orig], ignore_index=True)
combined["Target"] = (combined["Will_Buy_EV"] == "Yes").astype(int)

# Ordinal
range_map = {"Low": 1, "Medium": 2, "High": 3}
combined["Range_Anxiety_Num"] = combined["Range_Anxiety_Level"].map(range_map).fillna(1)
combined["Home_Charging_Num"] = (combined["Home_Charging_Possible"] == "Yes").astype(int)
combined["Subsidy_Num"] = (combined["Subsidy_Available"] == "Yes").astype(int)

# 1. High/Low Propensity Rule Masks
combined["High_Propensity_Mask"] = (
    (combined["Subsidy_Num"] == 1) & 
    (combined["Range_Anxiety_Num"] == 1) & 
    (combined["Environmental_Concern_Level"] >= 3)
).astype(int)

combined["Zero_Propensity_Mask"] = (
    (combined["Subsidy_Num"] == 0) | 
    (combined["Range_Anxiety_Num"] == 3)
).astype(int)

# 2. Key Non-Linear Combinations
combined["Income_x_Env2"] = combined["Annual_Income_USD"] * (combined["Environmental_Concern_Level"] ** 2)
combined["Income_Env_Subsidy"] = combined["Annual_Income_USD"] * combined["Environmental_Concern_Level"] * combined["Subsidy_Num"]
combined["Income_Threshold_67k"] = (combined["Annual_Income_USD"] > 67500).astype(int) * combined["Subsidy_Num"]
combined["Income_Threshold_134k"] = (combined["Annual_Income_USD"] > 134000).astype(int) * combined["Subsidy_Num"]

# 3. Charging Infrastructure
combined["Total_Charging_Stations"] = combined["Charging_Stations_Near_Home"] + combined["Charging_Stations_Near_Work"]
combined["Charging_Diff"] = combined["Charging_Stations_Near_Work"] - combined["Charging_Stations_Near_Home"]
combined["Charging_Ratio"] = (combined["Charging_Stations_Near_Work"] + 1) / (combined["Charging_Stations_Near_Home"] + 1)
combined["Commute_per_Total_Charging"] = combined["Daily_Commute_km"] / (combined["Total_Charging_Stations"] + 1)
combined["Commute_per_Home_Charging"] = combined["Daily_Commute_km"] / (combined["Charging_Stations_Near_Home"] + 1)
combined["Income_per_Car"] = combined["Annual_Income_USD"] / (combined["Number_of_Cars_Owned"] + 1)
combined["Income_per_Age"] = combined["Annual_Income_USD"] / (combined["Age"] + 1)
combined["Env_Concern_x_Income"] = combined["Environmental_Concern_Level"] * combined["Annual_Income_USD"]
combined["Range_Anxiety_x_Commute"] = combined["Range_Anxiety_Num"] * combined["Daily_Commute_km"]

# 4. Multi-way categorical combinations
combined["Sub_Range_Env"] = (
    combined["Subsidy_Available"].astype(str) + "_" + 
    combined["Range_Anxiety_Level"].astype(str) + "_" + 
    combined["Environmental_Concern_Level"].astype(str)
)
combined["Sub_Home_Range"] = (
    combined["Subsidy_Available"].astype(str) + "_" + 
    combined["Home_Charging_Possible"].astype(str) + "_" + 
    combined["Range_Anxiety_Level"].astype(str)
)
combined["City_Car"] = combined["City_Type"].astype(str) + "_" + combined["Current_Car_Type"].astype(str)

# 5. Composite Score
combined["EV_Readiness_Score"] = (
    combined["Home_Charging_Num"] * 2.0 +
    combined["Subsidy_Num"] * 3.0 +
    (combined["Total_Charging_Stations"] >= 10).astype(int) * 1.5 +
    (combined["Range_Anxiety_Num"] == 1).astype(int) * 2.0 +
    (combined["Environmental_Concern_Level"] >= 4).astype(int) * 2.0
)

cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", 
            "Range_Anxiety_Level", "Sub_Range_Env", "Sub_Home_Range", "City_Car"]
for c in cat_cols:
    combined[c] = combined[c].astype("category")

train_df = combined[combined["is_orig"] == 0].reset_index(drop=True)
orig_df = combined[combined["is_orig"] == 1].reset_index(drop=True)

drop_cols = ["id", "Will_Buy_EV", "Target", "is_orig"]
features = [c for c in train_df.columns if c not in drop_cols]
print(f"Features ({len(features)}): {features}", flush=True)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof = np.zeros(len(train_df))

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, train_df["Target"])):
    X_tr = pd.concat([train_df.iloc[trn_idx][features], orig_df[features]], ignore_index=True)
    y_tr = np.concatenate([train_df.iloc[trn_idx]["Target"].values, orig_df["Target"].values])
    X_va = train_df.iloc[val_idx][features]
    y_va = train_df.iloc[val_idx]["Target"].values
    
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
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    val_pred = model.predict_proba(X_va)[:, 1]
    oof[val_idx] = val_pred
    score = roc_auc_score(y_va, val_pred)
    print(f"Fold {fold+1} ROC-AUC: {score:.6f}", flush=True)

print(f"--> Advanced Features 3-Fold OOF ROC-AUC: {roc_auc_score(train_df['Target'], oof):.6f}", flush=True)
