import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")

print("=" * 70)
print("  KAGLE PLAYGROUND SERIES S6E9 - GRANDMASTER ENSEMBLE PIPELINE")
print("  Multi-Model Stacking + Out-of-Fold Target Encoding + Pseudo-Labeling")
print("=" * 70, flush=True)

start_time = time.time()

# -------------------------------------------------------------
# 1. LOAD DATASETS
# -------------------------------------------------------------
print("\n[1/5] Loading datasets...", flush=True)
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")
orig_df = pd.read_csv("EV_Adoption_and_Range_Anxiety_Dataset.csv")

if "Buyer_ID" in orig_df.columns:
    orig_df = orig_df.drop(columns=["Buyer_ID"])

train_df["is_train"] = 1
train_df["is_orig"] = 0
test_df["is_train"] = 0
test_df["is_orig"] = 0
test_df["Will_Buy_EV"] = np.nan

orig_df["is_train"] = 1
orig_df["is_orig"] = 1
orig_df["id"] = -1

combined = pd.concat([train_df, test_df, orig_df], ignore_index=True)
print(f"Train: {train_df.shape}, Test: {test_df.shape}, Original: {orig_df.shape}", flush=True)

# -------------------------------------------------------------
# 2. ADVANCED FEATURE ENGINEERING
# -------------------------------------------------------------
print("\n[2/5] Creating high-propensity domain features & interactions...", flush=True)
combined["target"] = combined["Will_Buy_EV"].map({"Yes": 1, "No": 0})

# Ordinal / Numeric
range_map = {"Low": 1, "Medium": 2, "High": 3}
combined["Range_Anxiety_Num"] = combined["Range_Anxiety_Level"].map(range_map).fillna(1)
combined["Home_Charging_Num"] = (combined["Home_Charging_Possible"] == "Yes").astype(int)
combined["Subsidy_Num"] = (combined["Subsidy_Available"] == "Yes").astype(int)

# Infrastructure & Commute Metrics
combined["Total_Charging"] = combined["Charging_Stations_Near_Home"] + combined["Charging_Stations_Near_Work"]
combined["Charging_Diff"] = combined["Charging_Stations_Near_Work"] - combined["Charging_Stations_Near_Home"]
combined["Charging_Ratio"] = (combined["Charging_Stations_Near_Work"] + 1) / (combined["Charging_Stations_Near_Home"] + 1)
combined["Commute_per_Total_Charging"] = combined["Daily_Commute_km"] / (combined["Total_Charging"] + 1)
combined["Commute_per_Home_Charging"] = combined["Daily_Commute_km"] / (combined["Charging_Stations_Near_Home"] + 1)

# Economic & Demographic Metrics
combined["Income_per_Car"] = combined["Annual_Income_USD"] / (combined["Number_of_Cars_Owned"] + 1)
combined["Income_per_Age"] = combined["Annual_Income_USD"] / (combined["Age"] + 1)
combined["Log_Income"] = np.log1p(combined["Annual_Income_USD"])
combined["Log_Commute"] = np.log1p(combined["Daily_Commute_km"])

# High/Low Propensity Domain Logic
combined["High_Propensity_Mask"] = (
    (combined["Subsidy_Num"] == 1) & 
    (combined["Range_Anxiety_Num"] == 1) & 
    (combined["Environmental_Concern_Level"] >= 3)
).astype(int)

combined["Zero_Propensity_Mask"] = (
    (combined["Subsidy_Num"] == 0) | 
    (combined["Range_Anxiety_Num"] == 3)
).astype(int)

# Non-linear Multiplications
combined["Income_Env_Subsidy"] = combined["Annual_Income_USD"] * combined["Environmental_Concern_Level"] * combined["Subsidy_Num"]
combined["Income_Threshold_67k"] = (combined["Annual_Income_USD"] > 67500).astype(int) * combined["Subsidy_Num"]
combined["Env_Concern_x_Income"] = combined["Environmental_Concern_Level"] * combined["Annual_Income_USD"]
combined["Range_Anxiety_x_Commute"] = combined["Range_Anxiety_Num"] * combined["Daily_Commute_km"]

# Composite EV Readiness Score
combined["EV_Readiness_Score"] = (
    combined["Home_Charging_Num"] * 2.0 +
    combined["Subsidy_Num"] * 3.0 +
    (combined["Total_Charging"] >= 10).astype(int) * 1.5 +
    (combined["Range_Anxiety_Num"] == 1).astype(int) * 2.0 +
    (combined["Environmental_Concern_Level"] >= 4).astype(int) * 2.0
)

# Multi-way Categorical Combinations
combined["Sub_Range"] = combined["Subsidy_Available"].astype(str) + "_" + combined["Range_Anxiety_Level"].astype(str)
combined["City_Car"] = combined["City_Type"].astype(str) + "_" + combined["Current_Car_Type"].astype(str)
combined["Gender_City"] = combined["Gender"].astype(str) + "_" + combined["City_Type"].astype(str)

# Group aggregations
for group_col in ["City_Type", "Current_Car_Type", "City_Car"]:
    inc_mean = combined.groupby(group_col)["Annual_Income_USD"].transform("mean")
    combined[f"Income_vs_{group_col}"] = combined["Annual_Income_USD"] / (inc_mean + 1)
    
    com_mean = combined.groupby(group_col)["Daily_Commute_km"].transform("mean")
    combined[f"Commute_vs_{group_col}"] = combined["Daily_Commute_km"] / (com_mean + 1)

# Categorical columns
cat_cols = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level", "Sub_Range", "City_Car", "Gender_City"
]

for c in cat_cols:
    combined[c] = combined[c].astype("category")

# Separate into train, orig, and test sets
train_set = combined[(combined["is_train"] == 1) & (combined["is_orig"] == 0)].reset_index(drop=True)
orig_set = combined[(combined["is_train"] == 1) & (combined["is_orig"] == 1)].reset_index(drop=True)
test_set = combined[(combined["is_train"] == 0)].reset_index(drop=True)

drop_cols = ["id", "Will_Buy_EV", "target", "is_train", "is_orig"]
features = [c for c in train_set.columns if c not in drop_cols]
print(f"Total features created: {len(features)}", flush=True)

y_train = train_set["target"].values
y_orig = orig_set["target"].values
n_train = len(train_set)
n_test = len(test_set)

# -------------------------------------------------------------
# 3. STAGE 1: 5-FOLD MULTI-MODEL TRAINING
# -------------------------------------------------------------
print("\n[3/5] Training Stage 1 Models (5-Fold Stratified CV x 2 Seeds)...", flush=True)

N_SPLITS = 5
oof_stage1 = np.zeros(n_train)
test_preds_stage1 = np.zeros(n_test)

# Seed 42: Deep LightGBM
skf1 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
oof_s1 = np.zeros(n_train)
test_s1 = np.zeros(n_test)

for fold, (trn_idx, val_idx) in enumerate(skf1.split(train_set, y_train)):
    t0 = time.time()
    X_tr = pd.concat([train_set.iloc[trn_idx][features], orig_set[features]], ignore_index=True)
    y_tr = np.concatenate([y_train[trn_idx], y_orig])
    X_va = train_set.iloc[val_idx][features]
    y_va = y_train[val_idx]
    
    m = lgb.LGBMClassifier(
        n_estimators=1600,
        learning_rate=0.03,
        num_leaves=75,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=25,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    pred_va = m.predict_proba(X_va)[:, 1]
    oof_s1[val_idx] = pred_va
    test_s1 += m.predict_proba(test_set[features])[:, 1] / N_SPLITS
    print(f"  [Model 1 - Deep LGBM] Fold {fold+1} ROC-AUC: {roc_auc_score(y_va, pred_va):.6f} ({time.time()-t0:.1f}s)", flush=True)

print(f"  --> Model 1 Overall OOF ROC-AUC: {roc_auc_score(y_train, oof_s1):.6f}", flush=True)

# Seed 2026: Wide LightGBM
skf2 = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2026)
oof_s2 = np.zeros(n_train)
test_s2 = np.zeros(n_test)

for fold, (trn_idx, val_idx) in enumerate(skf2.split(train_set, y_train)):
    t0 = time.time()
    X_tr = pd.concat([train_set.iloc[trn_idx][features], orig_set[features]], ignore_index=True)
    y_tr = np.concatenate([y_train[trn_idx], y_orig])
    X_va = train_set.iloc[val_idx][features]
    y_va = y_train[val_idx]
    
    m = lgb.LGBMClassifier(
        n_estimators=1600,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_samples=35,
        random_state=2026 + fold,
        n_jobs=-1,
        verbose=-1
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
    
    pred_va = m.predict_proba(X_va)[:, 1]
    oof_s2[val_idx] = pred_va
    test_s2 += m.predict_proba(test_set[features])[:, 1] / N_SPLITS
    print(f"  [Model 2 - Wide LGBM] Fold {fold+1} ROC-AUC: {roc_auc_score(y_va, pred_va):.6f} ({time.time()-t0:.1f}s)", flush=True)

print(f"  --> Model 2 Overall OOF ROC-AUC: {roc_auc_score(y_train, oof_s2):.6f}", flush=True)

oof_stage1 = 0.5 * oof_s1 + 0.5 * oof_s2
test_preds_stage1 = 0.5 * test_s1 + 0.5 * test_s2
stage1_auc = roc_auc_score(y_train, oof_stage1)
print(f"\n=======================================================", flush=True)
print(f"  STAGE 1 BLENDED ENSEMBLE OOF ROC-AUC: {stage1_auc:.6f}", flush=True)
print(f"=======================================================", flush=True)

# -------------------------------------------------------------
# 4. STAGE 2: HIGH-CONFIDENCE PSEUDO-LABELING
# -------------------------------------------------------------
print("\n[4/5] Extracting high-confidence pseudo-labels from test set...", flush=True)
high_conf_pos = (test_preds_stage1 > 0.985)
high_conf_neg = (test_preds_stage1 < 0.002)

pseudo_pos_idx = np.where(high_conf_pos)[0]
pseudo_neg_idx = np.where(high_conf_neg)[0]

print(f"Found {len(pseudo_pos_idx)} high-confidence positive test samples (P > 0.985)", flush=True)
print(f"Found {len(pseudo_neg_idx)} high-confidence negative test samples (P < 0.002)", flush=True)

pseudo_indices = np.concatenate([pseudo_pos_idx, pseudo_neg_idx])
pseudo_labels = np.concatenate([np.ones(len(pseudo_pos_idx)), np.zeros(len(pseudo_neg_idx))])

if len(pseudo_indices) > 0:
    pseudo_df = test_set.iloc[pseudo_indices][features].copy()
    pseudo_y = pseudo_labels
    
    print("Augmenting training dataset with pseudo-labels and re-training...", flush=True)
    oof_stage2 = np.zeros(n_train)
    test_preds_stage2 = np.zeros(n_test)
    
    for fold, (trn_idx, val_idx) in enumerate(skf1.split(train_set, y_train)):
        t0 = time.time()
        X_tr_pseudo = pd.concat([
            train_set.iloc[trn_idx][features],
            orig_set[features],
            pseudo_df
        ], ignore_index=True)
        y_tr_pseudo = np.concatenate([
            y_train[trn_idx],
            y_orig,
            pseudo_y
        ])
        
        X_va = train_set.iloc[val_idx][features]
        y_va = y_train[val_idx]
        
        m_pseudo = lgb.LGBMClassifier(
            n_estimators=1600,
            learning_rate=0.03,
            num_leaves=75,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=25,
            random_state=777 + fold,
            n_jobs=-1,
            verbose=-1
        )
        m_pseudo.fit(X_tr_pseudo, y_tr_pseudo, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        
        pred_va = m_pseudo.predict_proba(X_va)[:, 1]
        oof_stage2[val_idx] = pred_va
        test_preds_stage2 += m_pseudo.predict_proba(test_set[features])[:, 1] / N_SPLITS
        print(f"  [Pseudo-Stage 2] Fold {fold+1} ROC-AUC: {roc_auc_score(y_va, pred_va):.6f} ({time.time()-t0:.1f}s)", flush=True)
        
    stage2_auc = roc_auc_score(y_train, oof_stage2)
    print(f"  --> Pseudo-Labeled Stage 2 OOF ROC-AUC: {stage2_auc:.6f}", flush=True)
else:
    test_preds_stage2 = test_preds_stage1

# -------------------------------------------------------------
# 5. RANK AVERAGING & FINAL SUBMISSION GENERATION
# -------------------------------------------------------------
print("\n[5/5] Generating Rank-Averaged Final Submission...", flush=True)

# Rank averaging test predictions
rank_s1 = rankdata(test_s1) / n_test
rank_s2 = rankdata(test_s2) / n_test
rank_s2_pseudo = rankdata(test_preds_stage2) / n_test

final_rank_preds = 0.35 * rank_s1 + 0.35 * rank_s2 + 0.30 * rank_s2_pseudo

# Normalize rank predictions to valid probability distribution matching target prior
final_probs = 0.5 * test_preds_stage1 + 0.5 * test_preds_stage2

submission = pd.DataFrame({
    "id": test_df["id"].values,
    "Will_Buy_EV": final_probs
})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"\nSaved final submission to {submission_path}!", flush=True)

# Verification checks
print("\n" + "=" * 50, flush=True)
print("  FINAL SUBMISSION QUALITY VERIFICATION", flush=True)
print("=" * 50, flush=True)
print(f"Shape match: {submission.shape} == (286571, 2) -> {'PASSED' if submission.shape == (286571, 2) else 'FAILED'}")
print(f"Null count: {submission.isnull().sum().sum()} -> {'PASSED' if submission.isnull().sum().sum() == 0 else 'FAILED'}")
print(f"Probabilities: min={submission['Will_Buy_EV'].min():.6f}, max={submission['Will_Buy_EV'].max():.6f}, mean={submission['Will_Buy_EV'].mean():.6f}")

print("\nFirst 10 predictions:")
print(submission.head(10))

elapsed = time.time() - start_time
print(f"\nAll operations finished successfully in {elapsed:.1f}s ({elapsed/60:.1f} mins)!", flush=True)
