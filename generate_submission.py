import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print("  Kaggle Playground Series s6e9: Predicting EV Purchases")
print("  10-Model 5-Fold Stratified CV Ensemble Pipeline")
print("=" * 65, flush=True)

# 1. Load Data
start_time = time.time()
print("\n[1/4] Loading datasets...", flush=True)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")
orig = pd.read_csv("EV_Adoption_and_Range_Anxiety_Dataset.csv")

print(f"Train: {train.shape}, Test: {test.shape}, Orig: {orig.shape}", flush=True)

# 2. Feature Engineering
print("\n[2/4] Engineering domain & interaction features...", flush=True)

def preprocess(df):
    d = df.copy()
    if "Will_Buy_EV" in d.columns:
        d["Target"] = (d["Will_Buy_EV"] == "Yes").astype(int)
    else:
        d["Target"] = None
        
    # Ordinal / Numeric encodings
    d["Range_Anxiety_Num"] = d["Range_Anxiety_Level"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(1)
    d["Home_Charging_Num"] = (d["Home_Charging_Possible"] == "Yes").astype(int)
    d["Subsidy_Num"] = (d["Subsidy_Available"] == "Yes").astype(int)
    
    # Feature engineering
    d["Total_Charging_Stations"] = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["Charging_Diff"] = d["Charging_Stations_Near_Work"] - d["Charging_Stations_Near_Home"]
    d["Charging_Ratio"] = (d["Charging_Stations_Near_Work"] + 1) / (d["Charging_Stations_Near_Home"] + 1)
    d["Commute_per_Total_Charging"] = d["Daily_Commute_km"] / (d["Total_Charging_Stations"] + 1)
    d["Commute_per_Home_Charging"] = d["Daily_Commute_km"] / (d["Charging_Stations_Near_Home"] + 1)
    d["Income_per_Car"] = d["Annual_Income_USD"] / (d["Number_of_Cars_Owned"] + 1)
    d["Income_per_Age"] = d["Annual_Income_USD"] / (d["Age"] + 1)
    d["Env_Concern_x_Income"] = d["Environmental_Concern_Level"] * d["Annual_Income_USD"]
    d["Range_Anxiety_x_Commute"] = d["Range_Anxiety_Num"] * d["Daily_Commute_km"]
    d["EV_Readiness_Score"] = (
        d["Home_Charging_Num"] * 2.0 +
        d["Subsidy_Num"] * 1.5 +
        (d["Total_Charging_Stations"] >= 10).astype(int) * 1.5 +
        (d["Range_Anxiety_Num"] == 1).astype(int) * 1.0 +
        (d["Environmental_Concern_Level"] >= 4).astype(int) * 1.0
    )
    
    # Group aggregations
    city_income_mean = d.groupby("City_Type")["Annual_Income_USD"].transform("mean")
    d["Income_vs_CityMean"] = d["Annual_Income_USD"] / (city_income_mean + 1)
    
    city_commute_mean = d.groupby("City_Type")["Daily_Commute_km"].transform("mean")
    d["Commute_vs_CityMean"] = d["Daily_Commute_km"] / (city_commute_mean + 1)
    
    # Categorical columns
    cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
    for c in cat_cols:
        d[c] = d[c].astype("category")
        
    return d

train_p = preprocess(train)
test_p = preprocess(test)
orig_p = preprocess(orig)

features = [c for c in train_p.columns if c not in ["id", "Will_Buy_EV", "Target"]]
print(f"Total features ({len(features)}): {features}", flush=True)

# 3. 5-Fold Stratified CV with Multi-Seed Ensembling
print("\n[3/4] Training 10 Models (5-Fold CV x 2 Random Seeds)...", flush=True)

N_SPLITS = 5
SEEDS = [42, 2026]
oof_total = np.zeros(len(train_p))
test_preds = np.zeros(len(test_p))

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n>>> Running Seed {seed} ({seed_idx + 1}/{len(SEEDS)}) <<<", flush=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    oof_seed = np.zeros(len(train_p))
    
    for fold, (trn_idx, val_idx) in enumerate(skf.split(train_p, train_p["Target"])):
        t_fold = time.time()
        
        # Training data augmented with original dataset
        X_tr = pd.concat([train_p.iloc[trn_idx][features], orig_p[features]], ignore_index=True)
        y_tr = np.concatenate([train_p.iloc[trn_idx]["Target"].values, orig_p["Target"].values])
        
        X_va = train_p.iloc[val_idx][features]
        y_va = train_p.iloc[val_idx]["Target"].values
        
        model = lgb.LGBMClassifier(
            n_estimators=1600,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=25,
            random_state=seed + fold,
            n_jobs=-1,
            verbose=-1
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        val_pred = model.predict_proba(X_va)[:, 1]
        oof_seed[val_idx] = val_pred
        score = roc_auc_score(y_va, val_pred)
        
        test_pred_fold = model.predict_proba(test_p[features])[:, 1]
        test_preds += test_pred_fold / (N_SPLITS * len(SEEDS))
        
        print(f"  Seed {seed} | Fold {fold + 1} ROC-AUC: {score:.6f} ({time.time() - t_fold:.1f}s)", flush=True)
        
    seed_auc = roc_auc_score(train_p["Target"], oof_seed)
    print(f"  ==> Seed {seed} Overall OOF ROC-AUC: {seed_auc:.6f}", flush=True)
    oof_total += oof_seed / len(SEEDS)

# Overall Ensemble Score
final_oof_auc = roc_auc_score(train_p["Target"], oof_total)
print("\n" + "=" * 65, flush=True)
print(f"  FINAL 10-MODEL ENSEMBLE OOF ROC-AUC: {final_oof_auc:.6f}", flush=True)
print("=" * 65, flush=True)

# 4. Generate & Verify Submission
print("\n[4/4] Writing submission.csv...", flush=True)
submission = pd.DataFrame({
    "id": test["id"].values,
    "Will_Buy_EV": test_preds
})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Saved: {submission_path}", flush=True)

# Verification checks
print("\n" + "-" * 40, flush=True)
print("  SUBMISSION FILE VERIFICATION CHECKS", flush=True)
print("-" * 40, flush=True)
print(f"1. Submission shape: {submission.shape} (Sample sub: {sample_sub.shape}) -> {'PASSED' if submission.shape == sample_sub.shape else 'FAILED'}")
print(f"2. Column names: {list(submission.columns)} (Expected: ['id', 'Will_Buy_EV']) -> {'PASSED' if list(submission.columns) == ['id', 'Will_Buy_EV'] else 'FAILED'}")
print(f"3. Null / NaN count: {submission.isnull().sum().sum()} -> {'PASSED' if submission.isnull().sum().sum() == 0 else 'FAILED'}")
print(f"4. ID matching: {(submission['id'] == sample_sub['id']).all()} -> {'PASSED' if (submission['id'] == sample_sub['id']).all() else 'FAILED'}")
print(f"5. Probability stats: min={submission['Will_Buy_EV'].min():.6f}, max={submission['Will_Buy_EV'].max():.6f}, mean={submission['Will_Buy_EV'].mean():.6f}")

print("\nFirst 10 rows of submission.csv:")
print(submission.head(10))

elapsed = time.time() - start_time
print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} mins)!", flush=True)
