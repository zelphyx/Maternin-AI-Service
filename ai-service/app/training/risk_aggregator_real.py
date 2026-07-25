"""
Train risk aggregator on REAL UCI Bangladesh maternal health data.

UCI Dataset 863: Maternal Health Risk Data Set
- Source: Ahmed et al. (2021), Internet of Things and Cyber-Physical Systems
- Origin: IoT-based risk monitoring in Bangladesh public hospitals
- 1014 records with clinical features + RiskLevel label (high/mid/low)
- This is REAL clinical data, NOT synthetic

Comparison:
- BEFORE: maternin_clinical_robust_50k.csv (synthetic) — got 100% acc
- AFTER: UCI real data — should be lower (~70-85%) which is REALISTIC
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

DATASET_PATH = "/Users/zelphyx/Projects/Maternin-AI/datasets/real_datasets/uci_maternal/Maternal Health Risk Data Set.csv"
ARTIFACT_DIR = "/Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts"
OUTPUT_PKL = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1.pkl")
OUTPUT_META = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1_metadata.json")
SEED = 42


def main():
    print("=" * 70)
    print("REAL DATA Training — XGBoost Aggregator on UCI Bangladesh")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/5] Dataset loaded: {len(df)} records (REAL UCI 863)")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Risk distribution:")
    for lvl, count in df["RiskLevel"].value_counts().items():
        print(f"    {lvl}: {count} ({count/len(df)*100:.1f}%)")

    # Map UCI features to our model features
    # UCI: Age, SystolicBP, DiastolicBP, BS, BodyTemp, HeartRate
    # Our: triage_lapis1_score, preeclampsia_risk_prob, anemia_risk_prob,
    #       age, gestational_age_weeks, systolic_bp, diastolic_bp, hemoglobin_g_dl
    #
    # We need to engineer proxy features from UCI columns:
    # - systolic_bp -> from SystolicBP
    # - diastolic_bp -> from DiastolicBP
    # - age -> from Age
    # - gestational_age_weeks -> derive from BS proxy (BS correlates with GA in some studies)
    #   Better: use median GA of 28 weeks (typical cohort)
    # - hemoglobin_g_dl -> not in UCI; use clinical baseline 12.0 (typical normal)
    # - triage_lapis1_score -> derive from BP features
    # - preeclampsia_risk_prob -> derive from BP thresholds
    # - anemia_risk_prob -> not derivable from UCI; use 0.0 baseline

    df["systolic_bp"] = df["SystolicBP"]
    df["diastolic_bp"] = df["DiastolicBP"]
    df["age"] = df["Age"]
    df["gestational_age_weeks"] = 28.0  # median GA (no GA info in UCI)
    df["hemoglobin_g_dl"] = 12.0  # typical normal (no Hb info in UCI)

    # Derive preeclampsia_risk_prob from BP: higher BP -> higher risk
    df["preeclampsia_risk_prob"] = np.clip(
        ((df["systolic_bp"] - 120) / 60) * 0.5 + ((df["diastolic_bp"] - 80) / 30) * 0.3,
        0.0, 1.0,
    )

    # Derive triage_lapis1_score (0-100) from BP + other risk factors
    # Same logic as production rule engine
    triage_score = np.zeros(len(df))
    triage_score += np.where(df["systolic_bp"] >= 160, 30, np.where(df["systolic_bp"] >= 140, 15, 0))
    triage_score += np.where(df["diastolic_bp"] >= 110, 30, np.where(df["diastolic_bp"] >= 90, 15, 0))
    # HeartRate abnormal: >100 or <60
    triage_score += np.where((df["HeartRate"] > 100) | (df["HeartRate"] < 60), 10, 0)
    # BodyTemp: fever >38C
    triage_score += np.where(df["BodyTemp"] > 38, 10, 0)
    # BS abnormal: >140 or <70
    triage_score += np.where((df["BS"] > 14) | (df["BS"] < 7), 10, 0)
    df["triage_lapis1_score"] = np.clip(triage_score, 0, 100)

    df["anemia_risk_prob"] = 0.0  # not derivable from UCI features

    # Map labels: high -> merah (2), mid -> kuning (1), low -> hijau (0)
    label_map = {"low risk": "hijau", "mid risk": "kuning", "high risk": "merah"}
    df["target_badge"] = df["RiskLevel"].map(label_map)

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
    print(f"\n[2/5] Target distribution:")
    for i, name in enumerate(label_encoder.classes_):
        count = (y == i).sum()
        print(f"    {name}: {count} ({count/len(y)*100:.1f}%)")

    # 3-way split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval,
    )
    print(f"\n[3/5] 3-way split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    print(f"\n[4/5] Training XGBoost on REAL data...")
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
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

    print(f"\n[5/5] RESULTS (REAL DATA — UCI Bangladesh):")
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"\n  5-fold CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

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
        "dataset": "UCI_Maternal_Health_Risk_863",
        "dataset_size": len(df),
        "dataset_origin": "Bangladesh public hospitals (Ahmed et al., 2021)",
        "dataset_real_vs_synthetic": "REAL",
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
        "caveats": [
            "Training on real UCI Bangladesh data (1014 records)",
            "Several features (hemoglobin, anemia_prob) NOT available in UCI — proxied with defaults",
            "Label uses UCI RiskLevel which differs from Indonesian clinical thresholds",
            "For production, retrain with Indonesian patient data from partner puskesmas",
        ],
    }
    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved: {OUTPUT_PKL}")
    print(f"  Saved: {OUTPUT_META}")
    print(f"\n{'=' * 70}")
    print(f"OK Done! REAL data training complete.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
