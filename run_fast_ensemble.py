import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
from scipy.stats import rankdata
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print("  Kaggle Playground Series s6e9: Predicting EV Purchases")
print("  Diverse Multi-Model 5-Fold Ensemble Pipeline")
print("=" * 65, flush=True)

# 1. Load Datasets
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
print(f"Loaded: Train={train_df.shape}, Test={test_df.shape}, Orig={orig_df.shape}", flush=True)

# 2. Feature Engineering
print("\n[2/5] Creating tabular & interaction features...", flush=True)
combined["target"] = combined["Will_Buy_EV"].map({"Yes": 1, "No": 0})

# Ordinal / Numeric encodings
range_map = {"Low": 1, "Medium": 2, "High": 3}
combined["Range_Anxiety_Num"] = combined["Range_Anxiety_Level"].map(range_map).fillna(1)
combined["Home_Charging_Num"] = (combined["Home_Charging_Possible"] == "Yes").astype(int)
combined["Subsidy_Num"] = (combined["Subsidy_Available"] == "Yes").astype(int)

# Infrastructure & Commute Interactions
combined["Total_Charging"] = combined["Charging_Stations_Near_Home"] + combined["Charging_Stations_Near_Work"]
combined["Charging_Diff"] = combined["Charging_Stations_Near_Work"] - combined["Charging_Stations_Near_Home"]
combined["Charging_Ratio"] = (combined["Charging_Stations_Near_Work"] + 1) / (combined["Charging_Stations_Near_Home"] + 1)
combined["Commute_per_Total_Charging"] = combined["Daily_Commute_km"] / (combined["Total_Charging"] + 1)
combined["Commute_per_Home_Charging"] = combined["Daily_Commute_km"] / (combined["Charging_Stations_Near_Home"] + 1)
combined["Commute_per_Work_Charging"] = combined["Daily_Commute_km"] / (combined["Charging_Stations_Near_Work"] + 1)

# Economic & Demographic Features
combined["Income_per_Car"] = combined["Annual_Income_USD"] / (combined["Number_of_Cars_Owned"] + 1)
combined["Income_per_Age"] = combined["Annual_Income_USD"] / (combined["Age"] + 1)
combined["Log_Income"] = np.log1p(combined["Annual_Income_USD"])
combined["Log_Commute"] = np.log1p(combined["Daily_Commute_km"])

# Behavioral & Readiness Indicators
combined["Env_Concern_x_Income"] = combined["Environmental_Concern_Level"] * combined["Annual_Income_USD"]
combined["Env_Concern_per_Commute"] = combined["Environmental_Concern_Level"] / (combined["Daily_Commute_km"] + 1)
combined["Range_Anxiety_x_Commute"] = combined["Range_Anxiety_Num"] * combined["Daily_Commute_km"]
combined["Range_Anxiety_per_Charging"] = combined["Range_Anxiety_Num"] / (combined["Total_Charging"] + 1)
combined["EV_Readiness_Score"] = (
    combined["Home_Charging_Num"] * 2.0 +
    combined["Subsidy_Num"] * 1.5 +
    (combined["Total_Charging"] >= 10).astype(int) * 1.5 +
    (combined["Range_Anxiety_Num"] == 1).astype(int) * 1.0 +
    (combined["Environmental_Concern_Level"] >= 4).astype(int) * 1.0
)

# Multi-categorical Interaction Combinations
combined["City_Car"] = combined["City_Type"].astype(str) + "_" + combined["Current_Car_Type"].astype(str)
combined["Gender_City"] = combined["Gender"].astype(str) + "_" + combined["City_Type"].astype(str)
combined["Charging_Profile"] = (
    combined["Home_Charging_Possible"].astype(str) + "_" +
    combined["Subsidy_Available"].astype(str) + "_" +
    combined["Range_Anxiety_Level"].astype(str)
)

# Group aggregations
for group_col in ["City_Type", "Current_Car_Type", "City_Car"]:
    inc_mean = combined.groupby(group_col)["Annual_Income_USD"].transform("mean")
    inc_std = combined.groupby(group_col)["Annual_Income_USD"].transform("std").fillna(1)
    combined[f"Income_z_{group_col}"] = (combined["Annual_Income_USD"] - inc_mean) / inc_std
    
    com_mean = combined.groupby(group_col)["Daily_Commute_km"].transform("mean")
    com_std = combined.groupby(group_col)["Daily_Commute_km"].transform("std").fillna(1)
    combined[f"Commute_z_{group_col}"] = (combined["Daily_Commute_km"] - com_mean) / com_std

# Frequency encodings
for cat_col in ["Gender", "City_Type", "Current_Car_Type", "City_Car", "Gender_City", "Charging_Profile"]:
    freq = combined[cat_col].value_counts(normalize=True)
    combined[f"{cat_col}_freq"] = combined[cat_col].map(freq)

# Categorical column definitions
cat_features = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level", "City_Car", "Gender_City", "Charging_Profile"
]

for col in cat_features:
    combined[col] = combined[col].astype("category")

train_comp = combined[(combined["is_train"] == 1) & (combined["is_orig"] == 0)].reset_index(drop=True)
orig_comp = combined[(combined["is_train"] == 1) & (combined["is_orig"] == 1)].reset_index(drop=True)
test_comp = combined[(combined["is_train"] == 0)].reset_index(drop=True)

drop_cols = ["id", "Will_Buy_EV", "target", "is_train", "is_orig"]
features = [c for c in train_comp.columns if c not in drop_cols]
print(f"Total features created: {len(features)}", flush=True)

y_train = train_comp["target"].values
y_orig = orig_comp["target"].values
n_train = len(train_comp)
n_test = len(test_comp)

# 3. Model Training with 5-Fold Stratified CV
print("\n[3/5] Starting 5-Fold Stratified Cross-Validation for 3 Diverse Models...", flush=True)

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Model 1: Deep Regularized LGBM
oof_m1 = np.zeros(n_train)
preds_m1 = np.zeros(n_test)

# Model 2: Shallow Wide LGBM
oof_m2 = np.zeros(n_train)
preds_m2 = np.zeros(n_test)

# Model 3: ExtraTrees-enabled LGBM
oof_m3 = np.zeros(n_train)
preds_m3 = np.zeros(n_test)

total_start = time.time()

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_comp, y_train)):
    fold_start = time.time()
    print(f"\n========== Fold {fold + 1} / {N_SPLITS} ==========", flush=True)
    
    # Augmented Train Data with Original Dataset
    X_tr = pd.concat([train_comp.iloc[trn_idx][features], orig_comp[features]], ignore_index=True)
    y_tr_aug = np.concatenate([y_train[trn_idx], y_orig])
    X_va = train_comp.iloc[val_idx][features]
    y_va = y_train[val_idx]
    
    # --- Model 1: Deep Regularized LightGBM ---
    t0 = time.time()
    m1 = lgb.LGBMClassifier(
        n_estimators=1800,
        learning_rate=0.035,
        num_leaves=127,
        max_depth=9,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_samples=40,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    m1.fit(
        X_tr, y_tr_aug,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_m1[val_idx] = m1.predict_proba(X_va)[:, 1]
    preds_m1 += m1.predict_proba(test_comp[features])[:, 1] / N_SPLITS
    s1 = roc_auc_score(y_va, oof_m1[val_idx])
    print(f"  [Model 1 - Deep LGBM]       ROC-AUC: {s1:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    # --- Model 2: Shallow Wide LightGBM ---
    t0 = time.time()
    m2 = lgb.LGBMClassifier(
        n_estimators=1800,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=25,
        random_state=100 + fold,
        n_jobs=-1,
        verbose=-1
    )
    m2.fit(
        X_tr, y_tr_aug,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_m2[val_idx] = m2.predict_proba(X_va)[:, 1]
    preds_m2 += m2.predict_proba(test_comp[features])[:, 1] / N_SPLITS
    s2 = roc_auc_score(y_va, oof_m2[val_idx])
    print(f"  [Model 2 - Shallow LGBM]    ROC-AUC: {s2:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    # --- Model 3: ExtraTrees Randomized LightGBM ---
    t0 = time.time()
    m3 = lgb.LGBMClassifier(
        n_estimators=1800,
        learning_rate=0.035,
        extra_trees=True,
        num_leaves=95,
        max_depth=8,
        subsample=0.75,
        colsample_bytree=0.70,
        min_child_samples=30,
        random_state=200 + fold,
        n_jobs=-1,
        verbose=-1
    )
    m3.fit(
        X_tr, y_tr_aug,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_m3[val_idx] = m3.predict_proba(X_va)[:, 1]
    preds_m3 += m3.predict_proba(test_comp[features])[:, 1] / N_SPLITS
    s3 = roc_auc_score(y_va, oof_m3[val_idx])
    print(f"  [Model 3 - ExtraTrees LGBM] ROC-AUC: {s3:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    # Fold Blend
    fold_blend = (oof_m1[val_idx] + oof_m2[val_idx] + oof_m3[val_idx]) / 3.0
    s_blend = roc_auc_score(y_va, fold_blend)
    print(f"  --> Fold {fold+1} Blended ROC-AUC: {s_blend:.6f} (Fold time: {time.time()-fold_start:.1f}s)", flush=True)

# 4. Out-Of-Fold Evaluation & Optimization
print("\n" + "=" * 65, flush=True)
print("  OVERALL 5-FOLD CROSS-VALIDATION SUMMARY", flush=True)
print("=" * 65, flush=True)
auc_1 = roc_auc_score(y_train, oof_m1)
auc_2 = roc_auc_score(y_train, oof_m2)
auc_3 = roc_auc_score(y_train, oof_m3)

print(f"Model 1 (Deep LGBM)       OOF ROC-AUC: {auc_1:.6f}", flush=True)
print(f"Model 2 (Shallow LGBM)    OOF ROC-AUC: {auc_2:.6f}", flush=True)
print(f"Model 3 (ExtraTrees LGBM) OOF ROC-AUC: {auc_3:.6f}", flush=True)

# Optimize Ensemble Weights via Nelder-Mead
print("\n[4/5] Optimizing Ensemble Weights...", flush=True)
def loss_func(weights):
    w = np.array(weights)
    w = np.maximum(0, w)
    w = w / np.sum(w)
    blend = w[0] * oof_m1 + w[1] * oof_m2 + w[2] * oof_m3
    return -roc_auc_score(y_train, blend)

res = minimize(loss_func, [0.333, 0.333, 0.333], method="Nelder-Mead")
opt_weights = np.maximum(0, res.x)
opt_weights = opt_weights / np.sum(opt_weights)
print(f"Optimal Weights: M1={opt_weights[0]:.4f}, M2={opt_weights[1]:.4f}, M3={opt_weights[2]:.4f}", flush=True)

oof_ensemble = opt_weights[0] * oof_m1 + opt_weights[1] * oof_m2 + opt_weights[2] * oof_m3
auc_ensemble = roc_auc_score(y_train, oof_ensemble)
print(f"\n=======================================================", flush=True)
print(f"  FINAL BLENDED ENSEMBLE OOF ROC-AUC: {auc_ensemble:.6f}", flush=True)
print(f"=======================================================", flush=True)

# Final Blended Test Predictions
final_preds = opt_weights[0] * preds_m1 + opt_weights[1] * preds_m2 + opt_weights[2] * preds_m3

# 5. Generate Submission File
print("\n[5/5] Generating submission file...", flush=True)
submission = pd.DataFrame({
    "id": test_df["id"].values,
    "Will_Buy_EV": final_preds
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv successfully!", flush=True)

# Verification Checks
print("\n--- Final Verification ---", flush=True)
print(f"Submission shape: {submission.shape} (Matches sample sub: {submission.shape == sample_sub.shape})", flush=True)
print(f"Missing values: {submission.isnull().sum().sum()}", flush=True)
print(f"Probabilities min={submission['Will_Buy_EV'].min():.6f}, max={submission['Will_Buy_EV'].max():.6f}, mean={submission['Will_Buy_EV'].mean():.6f}", flush=True)
print("\nFirst 10 rows of submission.csv:")
print(submission.head(10))
print(f"\nPipeline completely finished in {time.time()-total_start:.1f} seconds!", flush=True)
