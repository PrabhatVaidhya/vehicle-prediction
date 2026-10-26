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
import xgboost as xgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")

print("=" * 65)
print("  Kaggle Playground Series s6e9: Predicting EV Purchases")
print("  High-Performance 5-Fold Ensemble Pipeline (LGBM + XGB + CatBoost)")
print("=" * 65, flush=True)

# 1. Load Data
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

# 2. Domain Feature Engineering
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

# Behavioral & Willingness Indicators
combined["Env_Concern_x_Income"] = combined["Environmental_Concern_Level"] * combined["Annual_Income_USD"]
combined["Env_Concern_per_Commute"] = combined["Environmental_Concern_Level"] / (combined["Daily_Commute_km"] + 1)
combined["Range_Anxiety_x_Commute"] = combined["Range_Anxiety_Num"] * combined["Daily_Commute_km"]
combined["Range_Anxiety_per_Charging"] = combined["Range_Anxiety_Num"] / (combined["Total_Charging"] + 1)

# Composite EV Readiness Score
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

# Convert categoricals to integers for XGBoost & CatBoost, and category dtype for LightGBM
for col in cat_features:
    combined[col] = combined[col].astype("category")

train_comp = combined[(combined["is_train"] == 1) & (combined["is_orig"] == 0)].reset_index(drop=True)
orig_comp = combined[(combined["is_train"] == 1) & (combined["is_orig"] == 1)].reset_index(drop=True)
test_comp = combined[(combined["is_train"] == 0)].reset_index(drop=True)

drop_cols = ["id", "Will_Buy_EV", "target", "is_train", "is_orig"]
features = [c for c in train_comp.columns if c not in drop_cols]
print(f"Total features created: {len(features)}", flush=True)

# Numeric matrix for XGBoost and fast CatBoost
X_train_num = train_comp[features].copy()
X_orig_num = orig_comp[features].copy()
X_test_num = test_comp[features].copy()

for col in cat_features:
    X_train_num[col] = X_train_num[col].cat.codes
    X_orig_num[col] = X_orig_num[col].cat.codes
    X_test_num[col] = X_test_num[col].cat.codes

y_train = train_comp["target"].values
y_orig = orig_comp["target"].values
n_train = len(train_comp)
n_test = len(test_comp)

# 3. Model Training with 5-Fold Stratified CV
print("\n[3/5] Starting 5-Fold Stratified Cross-Validation...", flush=True)

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_lgb = np.zeros(n_train)
preds_lgb = np.zeros(n_test)

oof_xgb = np.zeros(n_train)
preds_xgb = np.zeros(n_test)

oof_cat = np.zeros(n_train)
preds_cat = np.zeros(n_test)

total_start = time.time()

for fold, (trn_idx, val_idx) in enumerate(skf.split(train_comp, y_train)):
    fold_start = time.time()
    print(f"\n========== Fold {fold + 1} / {N_SPLITS} ==========", flush=True)
    
    # Train/Validation datasets
    X_tr_lgb = pd.concat([train_comp.iloc[trn_idx][features], orig_comp[features]], ignore_index=True)
    y_tr_aug = np.concatenate([y_train[trn_idx], y_orig])
    X_va_lgb = train_comp.iloc[val_idx][features]
    y_va = y_train[val_idx]
    
    X_tr_num = pd.concat([X_train_num.iloc[trn_idx], X_orig_num], ignore_index=True)
    X_va_num = X_train_num.iloc[val_idx]
    
    # --- 1. LightGBM ---
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1500,
        learning_rate=0.04,
        num_leaves=95,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=42 + fold,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(
        X_tr_lgb, y_tr_aug,
        eval_set=[(X_va_lgb, y_va)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    oof_lgb[val_idx] = lgb_model.predict_proba(X_va_lgb)[:, 1]
    preds_lgb += lgb_model.predict_proba(test_comp[features])[:, 1] / N_SPLITS
    score_lgb = roc_auc_score(y_va, oof_lgb[val_idx])
    print(f"  [LightGBM] ROC-AUC: {score_lgb:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    # --- 2. XGBoost ---
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1500,
        learning_rate=0.04,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42 + fold,
        n_jobs=-1,
        eval_metric="auc",
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_tr_num, y_tr_aug,
        eval_set=[(X_va_num, y_va)],
        verbose=False
    )
    oof_xgb[val_idx] = xgb_model.predict_proba(X_va_num)[:, 1]
    preds_xgb += xgb_model.predict_proba(X_test_num)[:, 1] / N_SPLITS
    score_xgb = roc_auc_score(y_va, oof_xgb[val_idx])
    print(f"  [XGBoost]  ROC-AUC: {score_xgb:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    # --- 3. CatBoost ---
    t0 = time.time()
    cat_model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.06,
        depth=7,
        eval_metric="AUC",
        random_seed=42 + fold,
        thread_count=-1,
        verbose=False,
        early_stopping_rounds=50
    )
    cat_model.fit(
        X_tr_num, y_tr_aug,
        eval_set=(X_va_num, y_va),
        verbose=False
    )
    oof_cat[val_idx] = cat_model.predict_proba(X_va_num)[:, 1]
    preds_cat += cat_model.predict_proba(X_test_num)[:, 1] / N_SPLITS
    score_cat = roc_auc_score(y_va, oof_cat[val_idx])
    print(f"  [CatBoost] ROC-AUC: {score_cat:.6f} ({time.time()-t0:.1f}s)", flush=True)
    
    # Fold blend score
    fold_blend = (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3.0
    print(f"  --> Fold {fold+1} Ensemble ROC-AUC: {roc_auc_score(y_va, fold_blend):.6f} (Fold total: {time.time()-fold_start:.1f}s)", flush=True)

# 4. Out-Of-Fold Evaluation & Optimization
print("\n" + "=" * 65, flush=True)
print("  OVERALL 5-FOLD CROSS-VALIDATION SUMMARY", flush=True)
print("=" * 65, flush=True)
auc_lgb = roc_auc_score(y_train, oof_lgb)
auc_xgb = roc_auc_score(y_train, oof_xgb)
auc_cat = roc_auc_score(y_train, oof_cat)

print(f"LightGBM OOF ROC-AUC: {auc_lgb:.6f}", flush=True)
print(f"XGBoost  OOF ROC-AUC: {auc_xgb:.6f}", flush=True)
print(f"CatBoost OOF ROC-AUC: {auc_cat:.6f}", flush=True)

# Optimize Ensemble Weights via Nelder-Mead
print("\n[4/5] Optimizing Ensemble Weights...", flush=True)
def loss_func(weights):
    w = np.array(weights)
    w = w / np.sum(w)
    blend = w[0] * oof_lgb + w[1] * oof_xgb + w[2] * oof_cat
    return -roc_auc_score(y_train, blend)

res = minimize(loss_func, [0.333, 0.333, 0.333], method="Nelder-Mead")
opt_weights = np.maximum(0, res.x)
opt_weights = opt_weights / np.sum(opt_weights)
print(f"Optimal Weights: LGBM={opt_weights[0]:.4f}, XGB={opt_weights[1]:.4f}, Cat={opt_weights[2]:.4f}", flush=True)

oof_ensemble = opt_weights[0] * oof_lgb + opt_weights[1] * oof_xgb + opt_weights[2] * oof_cat
auc_ensemble = roc_auc_score(y_train, oof_ensemble)
print(f"\n=======================================================", flush=True)
print(f"  FINAL BLENDED ENSEMBLE OOF ROC-AUC: {auc_ensemble:.6f}", flush=True)
print(f"=======================================================", flush=True)

# Final Blended Test Predictions
final_preds = opt_weights[0] * preds_lgb + opt_weights[1] * preds_xgb + opt_weights[2] * preds_cat

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
print(f"\nPipeline finished in {time.time()-total_start:.1f}s!", flush=True)
