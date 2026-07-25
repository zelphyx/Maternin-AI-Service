"""
Train + diagnostic XGBoost risk aggregator with proper 3-way split.

Purpose: honest overfit detection
  - 60% train / 20% val / 20% test (held-out)
  - Report train_acc, val_acc, test_acc separately
  - Identify overfit gap
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
import xgboost as xgb

DATASET_PATH = "/Users/zelphyx/Projects/Maternin-AI/datasets/maternal_health_risk/maternin_clinical_robust_50k.csv"
ARTIFACT_DIR = "/Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts"
OUTPUT_PKL = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1.pkl")
OUTPUT_META = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1_metadata.json")
SEED = 42


def main():
    print("=" * 70)
    print("MaternIn — XGBoost Aggregator Diagnostic Training (3-way split)")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/6] Dataset loaded: {len(df):,} samples")

    # Build composite label (avoids circular leakage)
    has_critical_systolic = df["systolic_bp"] >= 160
    has_critical_diastolic = df["diastolic_bp"] >= 110
    has_severe_anemia = df["hemoglobin_g_dl"] < 8.0
    has_high_age = df["age"] >= 35
    has_extreme_gestational = (
        (df["gestational_age_weeks"] < 28) | (df["gestational_age_weeks"] > 40)
    )
    critical_count = (
        has_critical_systolic.astype(int) + has_critical_diastolic.astype(int)
        + has_severe_anemia.astype(int) + has_high_age.astype(int)
        + has_extreme_gestational.astype(int)
    )
    int_label = pd.Series(0, index=df.index)
    is_merah = (critical_count >= 2) | has_critical_systolic | has_critical_diastolic
    is_kuning = (critical_count == 1) & ~is_merah
    int_label[is_merah] = 2
    int_label[is_kuning] = 1
    df["target_badge"] = int_label.map({0: "hijau", 1: "kuning", 2: "merah"})

    feature_columns = [
        "triage_lapis1_score", "preeclampsia_risk_prob", "anemia_risk_prob",
        "age", "gestational_age_weeks",
        "systolic_bp", "diastolic_bp", "hemoglobin_g_dl",
    ]

    label_encoder = LabelEncoder()
    label_encoder.fit(["hijau", "kuning", "merah"])
    df["target_badge_int"] = label_encoder.transform(df["target_badge"])

    X = df[feature_columns].values
    y = df["target_badge_int"].values
    print(f"\n[2/6] Target distribution:")
    for i, name in enumerate(label_encoder.classes_):
        count = (y == i).sum()
        print(f"    {name}: {count:,} ({count/len(y):.1%})")

    # 3-way split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval,
    )
    print(f"\n[3/6] 3-way split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,} (HELD OUT)")

    print(f"\n[4/6] Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=SEED,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Evaluate on all 3 splits
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = accuracy_score(y_val, model.predict(X_val))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    train_f1 = f1_score(y_train, model.predict(X_train), average="weighted")
    val_f1 = f1_score(y_val, model.predict(X_val), average="weighted")
    test_f1 = f1_score(y_test, model.predict(X_test), average="weighted")

    y_test_pred = model.predict(X_test)
    test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred)

    overfit_gap = train_acc - val_acc

    print(f"\n[5/6] RESULTS:")
    print(f"  {'Split':<10s} {'Accuracy':<12s} {'F1':<10s}")
    print(f"  {'-'*32}")
    print(f"  {'Train':<10s} {train_acc:<12.4f} {train_f1:<10.4f}")
    print(f"  {'Val':<10s} {val_acc:<12.4f} {val_f1:<10.4f}")
    print(f"  {'Test':<10s} {test_acc:<12.4f} {test_f1:<10.4f}")
    print()
    print(f"  Test Precision: {test_prec:.4f}")
    print(f"  Test Recall:    {test_rec:.4f}")
    print()
    print(f"  OVERFIT DIAGNOSTIC:")
    print(f"    Train - Val gap: {overfit_gap:.4f} ({overfit_gap*100:.2f} pp)")
    if overfit_gap < 0.02:
        print(f"    [GOOD] Gap < 2pp -> no overfitting")
    elif overfit_gap < 0.10:
        print(f"    [OK] Gap < 10pp -> mild overfitting")
    else:
        print(f"    [WARN] Gap >= 10pp -> significant overfitting")
    print()
    print(f"  Val - Test gap: {val_acc - test_acc:.4f} ({(val_acc - test_acc)*100:.2f} pp)")

    print(f"\n  Test Confusion Matrix:")
    print(f"           hijau  kuning  merah")
    for i, name in enumerate(label_encoder.classes_):
        print(f"    {name:6s}  {cm[i]}")

    importances = model.feature_importances_
    print(f"\n  Feature Importance:")
    for feat, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:30s} {imp:.4f} {bar}")

    # 5-fold CV on train set
    print(f"\n[6/6] 5-fold CV on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"  CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  CV F1 per fold: {[f'{s:.4f}' for s in cv_scores]}")

    print(f"\n  Saving model...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    artifact = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
    }
    joblib.dump(artifact, OUTPUT_PKL)

    metadata = {
        "model_name": "risk_aggregator_v1",
        "model_type": "XGBClassifier",
        "features": feature_columns,
        "labels": list(label_encoder.classes_),
        "dataset": "maternin_clinical_robust_50k.csv",
        "dataset_size": len(df),
        "split": "60/20/20 train/val/test",
        "split_seed": SEED,
        "metrics": {
            "train_accuracy": round(float(train_acc), 4),
            "val_accuracy": round(float(val_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "train_f1": round(float(train_f1), 4),
            "val_f1": round(float(val_f1), 4),
            "test_f1": round(float(test_f1), 4),
            "test_precision": round(float(test_prec), 4),
            "test_recall": round(float(test_rec), 4),
            "overfit_gap_train_val": round(float(overfit_gap), 4),
            "generalization_gap_val_test": round(float(val_acc - test_acc), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
        },
        "confusion_matrix_test": {
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1]),
        },
        "feature_importance": {
            feat: round(float(imp), 4)
            for feat, imp in zip(feature_columns, importances)
        },
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {OUTPUT_PKL}")
    print(f"  Saved: {OUTPUT_META}")
    print(f"\n{'=' * 70}")
    print(f"OK Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
