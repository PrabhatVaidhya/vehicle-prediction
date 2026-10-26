# Predicting Electric Vehicle Purchases (Kaggle Playground Series s6e9)

An end-to-end competitive machine learning and demand analytics pipeline forecasting electric vehicle (EV) purchase intent across ~965K consumer records.

## 🚀 Key Highlights & Verified Benchmarks
- **Dataset Scale:** 668,665 Train + 286,571 Test + 10,000 Seed records (~965K total instances).
- **Metric:** Area Under the ROC Curve (ROC-AUC).
- **Baseline OOF ROC-AUC:** `0.941578` (Raw features, single LightGBM).
- **Engineered OOF ROC-AUC:** `0.941708` (+28 domain-engineered features).
- **Ensemble OOF ROC-AUC:** `0.941913` (10-model multi-seed & pseudo-labeled ensemble).
- **Inference Latency:** `19.94 ms` per 1,000 predictions (`0.0199 ms`/sample on CPU).
- **Inference Throughput:** ~`50,150` predictions/second.

## 🛠️ Feature Engineering & Architecture
- **Infrastructure & Commute Metrics:** Charging station availability ratios near home vs. work, commute-to-charging infrastructure density.
- **Economic & Demographics:** Income-per-car, income-per-age, groupwise statistical pooling across city and vehicle types.
- **Decision & Propensity Logic:** High/zero propensity rule masks, composite EV readiness index, non-linear interactions between environmental concern and income.
- **Modeling Techniques:** Stratified 5-Fold Cross Validation, Multi-seed LightGBM/CatBoost/XGBoost ensembling, Semi-supervised high-confidence pseudo-labeling, and Rank-Averaged blending.

## 📁 Repository Structure
- `grandmaster_pipeline.py`: Full multi-stage ensemble pipeline with 41 features, pseudo-labeling, and rank averaging.
- `benchmark_cv_metrics.py`: Rigorous empirical validation suite measuring exact baseline, feature lift, and inference latency.
- `run_full_pipeline.py`: Multi-model ensemble (LightGBM + XGBoost + CatBoost) with Nelder-Mead weight optimization.
- `run_fast_ensemble.py`: High-speed multi-architecture LightGBM ensemble.
- `train_baseline.py`: Initial 5-fold stratified cross-validation baseline.
- `generate_submission.py`: 10-model 5-fold stratified ensemble inference script.

## 💻 Quick Start
```bash
# Clone the repository
git clone https://github.com/PrabhatVaidhya/<repo-name>.git
cd <repo-name>

# Install dependencies
pip install lightgbm xgboost catboost scikit-learn pandas numpy scipy

# Run benchmark metrics
python benchmark_cv_metrics.py
```
