"""
Train on REAL aggregated Indonesia DHS regional panel data.

HONEST CAVEATS:
- This is aggregated by province-year (96 records), NOT individual patient data
- Features: ANC4%, facility delivery%, SBA%, urban share%, LBW%, etc.
- Target: LBW% (low birth weight) as proxy for maternal health outcomes
- This is regional-level analysis, NOT for clinical prediction per-patient
- Use this as proof we trained on Indonesian-origin data, but NOT for production

Compare to:
  - UCI Bangladesh: 1014 individual records (used for current model)
  - Indonesia regional: 96 province-year aggregates (this script)
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

DATASET_PATH = "/Users/zelphyx/Projects/Maternin-AI/datasets/real_datasets/indonesia_regional/dhs_indonesia_regional_panel.csv"


def main():
    print("=" * 70)
    print("INDONESIA Training — DHS Regional Panel (proof of regional use)")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/5] Dataset: {len(df)} records (REAL Indonesia DHS regional)")
    print(f"  Years: {sorted(df['year'].unique())}")
    print(f"  Provinces: {df['province_name'].nunique()}")

    # Use LBW% as proxy target — high LBW = poor maternal health outcome
    # Bucket into 3 categories: rendah (<10%), sedang (10-12%), tinggi (>12%)
    df["lbw_bucket"] = pd.cut(
        df["lbw_pct"],
        bins=[0, 0.10, 0.12, 1.0],
        labels=["rendah", "sedang", "tinggi"],
        include_lowest=True,
    ).astype(str)
    print(f"\n  LBW distribution:")
    for b, n in df["lbw_bucket"].value_counts().items():
        print(f"    {b}: {n}")

    # Features (regional-level)
    feature_columns = [
        "anc4_pct", "facility_delivery_pct", "sba_pct", "urban_share_pct",
        "low_education_pct", "risky_maternal_age_pct", "birth_interval_short_pct",
        "avg_parity", "full_immun_pct",
    ]

    X = df[feature_columns].values
    le = LabelEncoder()
    le.fit(["rendah", "sedang", "tinggi"])
    y = le.transform(df["lbw_bucket"])

    # Small dataset — use LOOCV or repeated stratified split
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
    cv_accs, cv_f1s = [], []
    best_model = None
    best_acc = 0

    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                C=0.5, class_weight="balanced", max_iter=1000, random_state=42,
            )),
        ])
        pipeline.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        f1 = f1_score(y_test, pipeline.predict(X_test), average="weighted")
        cv_accs.append(acc)
        cv_f1s.append(f1)
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline

    print(f"\n[2/5] LR with 5x stratified CV:")
    print(f"  Acc: {np.mean(cv_accs):.4f} +/- {np.std(cv_accs):.4f}")
    print(f"  F1:  {np.mean(cv_f1s):.4f} +/- {np.std(cv_f1s):.4f}")

    # Try XGBoost
    xgb_accs, xgb_f1s = [], []
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        xgb_model = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            objective="multi:softprob", num_class=3,
            random_state=42, tree_method="hist",
        )
        xgb_model.fit(X_train, y_train)
        acc = accuracy_score(y_test, xgb_model.predict(X_test))
        f1 = f1_score(y_test, xgb_model.predict(X_test), average="weighted")
        xgb_accs.append(acc)
        xgb_f1s.append(f1)

    print(f"\n[3/5] XGBoost with 5x stratified CV:")
    print(f"  Acc: {np.mean(xgb_accs):.4f} +/- {np.std(xgb_accs):.4f}")
    print(f"  F1:  {np.mean(xgb_f1s):.4f} +/- {np.std(xgb_f1s):.4f}")

    # Try Random Forest
    rf_accs, rf_f1s = [], []
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        rf = RandomForestClassifier(
            n_estimators=100, max_depth=4, class_weight="balanced", random_state=42,
        )
        rf.fit(X_train, y_train)
        acc = accuracy_score(y_test, rf.predict(X_test))
        f1 = f1_score(y_test, rf.predict(X_test), average="weighted")
        rf_accs.append(acc)
        rf_f1s.append(f1)

    print(f"\n[4/5] RF with 5x stratified CV:")
    print(f"  Acc: {np.mean(rf_accs):.4f} +/- {np.std(rf_accs):.4f}")
    print(f"  F1:  {np.mean(rf_f1s):.4f} +/- {np.std(rf_f1s):.4f}")

    # Save report
    report = {
        "dataset": "DHS_Indonesia_Regional_Panel",
        "n_records": len(df),
        "n_provinces": df["province_name"].nunique(),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "features": feature_columns,
        "target": "lbw_bucket (proxy for maternal health outcome)",
        "results": {
            "logistic_regression": {
                "cv_acc_mean": round(float(np.mean(cv_accs)), 4),
                "cv_acc_std": round(float(np.std(cv_accs)), 4),
                "cv_f1_mean": round(float(np.mean(cv_f1s)), 4),
                "cv_f1_std": round(float(np.std(cv_f1s)), 4),
            },
            "xgboost": {
                "cv_acc_mean": round(float(np.mean(xgb_accs)), 4),
                "cv_acc_std": round(float(np.std(xgb_accs)), 4),
                "cv_f1_mean": round(float(np.mean(xgb_f1s)), 4),
                "cv_f1_std": round(float(np.std(xgb_f1s)), 4),
            },
            "random_forest": {
                "cv_acc_mean": round(float(np.mean(rf_accs)), 4),
                "cv_acc_std": round(float(np.std(rf_accs)), 4),
                "cv_f1_mean": round(float(np.mean(rf_f1s)), 4),
                "cv_f1_std": round(float(np.std(rf_f1s)), 4),
            },
        },
        "important_disclaimers": [
            "Aggregated province-year data, NOT individual patient data",
            "Cannot be used for clinical per-patient prediction",
            "Useful only for regional health policy analysis",
            "Production model still uses UCI Bangladesh individual data",
        ],
    }

    output_path = "/Users/zelphyx/Projects/Maternin-AI/datasets/real_datasets/indonesia_regional/training_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[5/5] Report saved: {output_path}")
    print(f"\n{'=' * 70}")
    print(f"OK Indonesia regional training done (limited usefulness).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
