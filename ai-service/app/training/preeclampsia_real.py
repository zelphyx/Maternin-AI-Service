"""
Train LR on REAL UCI Bangladesh data as proxy for preeclampsia risk.

Note: UCI dataset doesn't have preeclampsia label. We use:
  - high_risk label (binary: high vs mid+low) as proxy
  - BP features directly available
  - This is a real-world, honest test of LR model on real clinical data.
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

DATASET_PATH = "/Users/zelphyx/Projects/Maternin-AI/datasets/real_datasets/uci_maternal/Maternal Health Risk Data Set.csv"
ARTIFACT_DIR = "/Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts"
OUTPUT_PKL = os.path.join(ARTIFACT_DIR, "preeclampsia_lr_v1.pkl")
OUTPUT_META = os.path.join(ARTIFACT_DIR, "preeclampsia_lr_v1_metadata.json")
SEED = 42


def main():
    print("=" * 70)
    print("REAL DATA Training — LR Preeclampsia Proxy on UCI Bangladesh")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/5] Dataset: {len(df)} REAL UCI records")

    # Map features
    df["systolic_bp"] = df["SystolicBP"]
    df["diastolic_bp"] = df["DiastolicBP"]
    df["age"] = df["Age"]
    df["gestational_age_weeks"] = 28.0  # median
    df["protein_urine_encoded"] = 0  # not available
    df["bmi"] = 25.0  # not available — typical normal
    df["has_preeclampsia_history"] = 0  # not available
    df["has_hypertension_history"] = np.where(
        (df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90), 1, 0
    )

    # Proxy target: high risk as preeclampsia proxy
    # (high-risk in UCI correlates strongly with hypertensive disorders)
    df["target_preeclampsia"] = np.where(df["RiskLevel"] == "high risk", 1, 0)

    feature_columns = [
        "systolic_bp", "diastolic_bp", "protein_urine_encoded",
        "has_preeclampsia_history", "has_hypertension_history",
        "age", "gestational_age_weeks", "bmi",
    ]

    X = df[feature_columns].values
    y = df["target_preeclampsia"].values
    print(f"  Target: {np.bincount(y)} (positive rate: {y.mean():.2%})")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval,
    )
    print(f"\n[2/5] 3-way split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print(f"\n[3/5] Training LR...")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=1000,
            random_state=SEED, solver="lbfgs",
        )),
    ])
    pipeline.fit(X_train, y_train)

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

    print(f"\n[4/5] RESULTS (REAL DATA — UCI Bangladesh):")
    print(f"  {'Split':<10s} {'Accuracy':<12s} {'F1':<10s}")
    print(f"  {'-'*32}")
    print(f"  {'Train':<10s} {train_acc:<12.4f} {train_f1:<10.4f}")
    print(f"  {'Val':<10s} {val_acc:<12.4f} {val_f1:<10.4f}")
    print(f"  {'Test':<10s} {test_acc:<12.4f} {test_f1:<10.4f}")
    print()
    print(f"  Test Precision: {test_prec:.4f}")
    print(f"  Test Recall:    {test_rec:.4f}")
    print(f"  Test Confusion: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
    print()
    print(f"  OVERFIT DIAGNOSTIC:")
    print(f"    Train - Val gap: {overfit_gap:.4f} ({overfit_gap*100:.2f} pp)")
    print(f"  Val - Test gap: {val_acc - test_acc:.4f} ({(val_acc - test_acc)*100:.2f} pp)")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    print(f"  5-fold CV F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    print(f"\n[5/5] Saving...")
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PKL)

    metadata = {
        "model_name": "preeclampsia_lr_v1",
        "model_type": "LogisticRegression",
        "features": feature_columns,
        "dataset": "UCI_Maternal_Health_Risk_863",
        "dataset_size": len(df),
        "dataset_origin": "Bangladesh public hospitals (Ahmed et al., 2021)",
        "dataset_real_vs_synthetic": "REAL",
        "target_definition": "high_risk label (proxy for preeclampsia/hypertensive)",
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
        "caveats": [
            "Training on real UCI Bangladesh data (1014 records)",
            "Several features (protein_urine, BMI) NOT available in UCI — proxied with defaults",
            "Target is UCI 'high_risk' which is a proxy for preeclampsia/hypertensive disorders",
            "For production, retrain with labeled Indonesian preeclampsia cohort",
        ],
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {OUTPUT_PKL}")
    print(f"\n{'=' * 70}")
    print(f"OK Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
