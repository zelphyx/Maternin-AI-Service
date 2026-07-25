"""
Train + diagnostic LR preeclampsia with proper 3-way split.

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

DATASET_PATH = "/Users/zelphyx/Projects/Maternin-AI/datasets/maternal_health_risk/maternin_clinical_robust_50k.csv"
ARTIFACT_DIR = "/Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts"
OUTPUT_PKL = os.path.join(ARTIFACT_DIR, "preeclampsia_lr_v1.pkl")
OUTPUT_META = os.path.join(ARTIFACT_DIR, "preeclampsia_lr_v1_metadata.json")
SEED = 42


def main():
    print("=" * 70)
    print("MaternIn — LR Preeclampsia Diagnostic Training (3-way split)")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/6] Dataset loaded: {len(df):,} samples")

    df["protein_urine_encoded"] = df["protein_urine"].map({
        "negatif": 0, "trace": 1, "positif_1": 2, "positif_2": 3,
        "positif_3": 4, "positif_4": 5,
        "positif_ringan": 1, "positif": 2, "positif_kuat": 4,
    }).fillna(0).astype(int)

    severe_symptom_cols = [
        "symptom_nyeri_ulu_hati", "symptom_pandangan_kabur",
        "symptom_kejang", "symptom_perdarahan",
    ]
    has_severe_symptoms = any(col in df.columns for col in severe_symptom_cols)

    is_hypertensive = (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)
    has_proteinuria = df["protein_urine_encoded"] >= 2

    if has_severe_symptoms:
        severe_indicators = pd.Series(0, index=df.index)
        for col in severe_symptom_cols:
            if col in df.columns:
                severe_indicators += df[col].map(
                    {True: 1, False: 0, "True": 1, "False": 0,
                     1: 1, 0: 0, "ya": 1, "tidak": 0,
                     "berat": 1, "hebat": 1, "parah": 1}
                ).fillna(0).astype(int)
        df["target_preeclampsia"] = (
            (is_hypertensive & has_proteinuria)
            | (df["systolic_bp"] >= 160)
            | (df["diastolic_bp"] >= 110)
            | (severe_indicators >= 1)
        ).astype(int)
    else:
        df["target_preeclampsia"] = (is_hypertensive & has_proteinuria).astype(int)

    feature_columns = [
        "systolic_bp", "diastolic_bp", "protein_urine_encoded",
        "has_preeclampsia_history", "has_hypertension_history",
        "age", "gestational_age_weeks", "bmi",
    ]
    for col in ["has_preeclampsia_history", "has_hypertension_history"]:
        df[col] = df[col].map({True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)

    X = df[feature_columns].values
    y = df["target_preeclampsia"].values
    print(f"\n[2/6] Target: {np.bincount(y)} (positive rate: {y.mean():.2%})")

    # 3-way split: 60% train / 20% val / 20% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval,
    )
    print(f"\n[3/6] 3-way split:")
    print(f"  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,} (HELD OUT until final eval)")

    print(f"\n[4/6] Training LR...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=SEED, solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)

    # Evaluate on all three splits
    train_acc = accuracy_score(y_train, pipeline.predict(X_train))
    val_acc = accuracy_score(y_val, pipeline.predict(X_val))
    test_acc = accuracy_score(y_test, pipeline.predict(X_test))

    train_f1 = f1_score(y_train, pipeline.predict(X_train))
    val_f1 = f1_score(y_val, pipeline.predict(X_val))
    test_f1 = f1_score(y_test, pipeline.predict(X_test))

    y_test_pred = pipeline.predict(X_test)
    test_prec = precision_score(y_test, y_test_pred)
    test_rec = recall_score(y_test, y_test_pred)
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
    print(f"  Test Confusion: TN={cm[0][0]:,} FP={cm[0][1]:,} FN={cm[1][0]:,} TP={cm[1][1]:,}")
    print()
    print(f"  OVERFIT DIAGNOSTIC:")
    print(f"    Train - Val gap: {overfit_gap:.4f} ({overfit_gap*100:.2f} pp)")
    if overfit_gap < 0.02:
        print(f"    [GOOD] Gap < 2pp -> no overfitting")
    elif overfit_gap < 0.10:
        print(f"    [OK] Gap < 10pp -> mild overfitting, acceptable")
    else:
        print(f"    [WARN] Gap >= 10pp -> significant overfitting")
    print()
    print(f"  Val - Test gap: {val_acc - test_acc:.4f} ({(val_acc - test_acc)*100:.2f} pp)")
    print(f"    (Small gap = model generalizes to unseen data)")

    print(f"\n[6/6] 5-fold CV on training set...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    print(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  CV F1 per fold: {[f'{s:.4f}' for s in cv_scores]}")

    print(f"\n  Saving model...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PKL)

    metadata = {
        "model_name": "preeclampsia_lr_v1",
        "model_type": "LogisticRegression",
        "features": feature_columns,
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
