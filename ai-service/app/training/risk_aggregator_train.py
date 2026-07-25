"""
MaternIn AI — Training Script: XGBoost Risk Aggregator (Lapis 2)
=================================================================
Target metrik (PRD Section 5): Akurasi 93%, Presisi 93%, Recall 94%, F1 93%

Menerima 3 input: triage_score (Lapis 1), preeclampsia_prob, anemia_prob
Output: aggregate_score (0-100) dan risk_badge (hijau/kuning/merah)

Dataset: maternin_clinical_robust_50k.csv
Output:  app/model_artifacts/risk_aggregator_v1.pkl

Jalankan:
    cd /Users/zelphyx/Projects/Maternin-AI/ai-service
    source .venv/bin/activate
    python app/training/risk_aggregator_train.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

try:
    import xgboost as xgb
except ImportError:
    print("❌ XGBoost not installed. Run: pip install xgboost")
    exit(1)

# ── Paths ────────────────────────────────────────────────────────────────
DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "datasets",
    "maternal_health_risk", "maternin_clinical_robust_50k.csv"
)
ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")
OUTPUT_PKL = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1.pkl")
OUTPUT_META = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1_metadata.json")


def main():
    print("=" * 70)
    print("MaternIn — XGBoost Risk Aggregator Training (Lapis 2)")
    print("=" * 70)

    # ── 1. Load Dataset ──────────────────────────────────────────────
    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/5] Dataset loaded: {len(df):,} samples")

    # ── 2. Feature Engineering ───────────────────────────────────────
    # FIX: risk_badge label must come from actual clinical outcomes, NOT from
    # the same heuristic outputs that the model will receive as features.
    # Using heuristic-generated labels creates a second-order circular dependency
    # where the aggregator learns to replicate heuristic logic instead of
    # mapping features to real risk.
    #
    # Priority: use actual adverse outcome if available, else use a composite
    # that includes outcome-independent clinical criteria.
    outcome_cols = [
        "had_preclampsia_outcome", "had_severe_complication",
        "icu_admission", "maternal_mortality", "adverse_outcome_composite",
    ]
    has_outcome = any(c in df.columns for c in outcome_cols)

    if has_outcome:
        for col in outcome_cols:
            if col in df.columns:
                int_label = df[col].map(
                    {"normal": 0, "low": 0, "medium": 1, "high": 2,
                     0: 0, 1: 1, 2: 2, False: 0, True: 2,
                     "False": 0, "True": 2, "0": 0, "1": 1, "2": 2}
                ).fillna(0).astype(int)
                df["target_badge"] = int_label.map({0: "hijau", 1: "kuning", 2: "merah"})
                break
    else:
        # Fallback: composite label using outcome-independent clinical markers.
        # This is still heuristic-derived but uses a broader set of criteria
        # than just the 3 XGBoost input features.
        has_critical_systolic = df["systolic_bp"] >= 160
        has_critical_diastolic = df["diastolic_bp"] >= 110
        has_severe_anemia = df["hemoglobin_g_dl"] < 8.0
        has_high_age = df["age"] >= 35
        has_extreme_gestational = (
            (df["gestational_age_weeks"] < 28) | (df["gestational_age_weeks"] > 40)
        )

        critical_count = (
            has_critical_systolic.astype(int)
            + has_critical_diastolic.astype(int)
            + has_severe_anemia.astype(int)
            + has_high_age.astype(int)
            + has_extreme_gestational.astype(int)
        )
        # Composite thresholds: merah if >=2 critical OR any severe BP,
        # kuning if exactly 1 critical, hijau if none.
        int_label = pd.Series(0, index=df.index)  # default hijau
        is_merah = (critical_count >= 2) | has_critical_systolic | has_critical_diastolic
        is_kuning = (critical_count == 1) & ~is_merah
        int_label[is_merah] = 2
        int_label[is_kuning] = 1
        df["target_badge"] = int_label.map({0: "hijau", 1: "kuning", 2: "merah"})

    feature_columns = [
        "triage_lapis1_score",
        "preeclampsia_risk_prob",
        "anemia_risk_prob",
        "age",
        "gestational_age_weeks",
        "systolic_bp",
        "diastolic_bp",
        "hemoglobin_g_dl",
    ]

    label_encoder = LabelEncoder()
    label_encoder.fit(["hijau", "kuning", "merah"])
    df["target_badge"] = label_encoder.transform(df["target_badge"])

    X = df[feature_columns].values
    y = df["target_badge"].values

    print(f"\n[2/5] Feature engineering complete")
    print(f"  Features: {feature_columns}")
    print(f"  Target distribution:")
    for i, name in enumerate(label_encoder.classes_):
        count = (y == i).sum()
        print(f"    {name}: {count:,} ({count/len(y):.1%})")

    # ── 3. Train/Test Split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    print(f"\n[3/5] Train/Test split: {len(X_train):,} / {len(X_test):,}")

    # ── 4. Train XGBoost ─────────────────────────────────────────────
    print(f"\n[4/5] Training XGBoost Classifier...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        use_label_encoder=False,
        tree_method="hist",  # Optimal for Apple Silicon
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Evaluation Results:")
    print(f"    Accuracy:  {acc:.4f} ({acc:.2%})")
    print(f"    Precision: {prec:.4f} ({prec:.2%})")
    print(f"    Recall:    {rec:.4f} ({rec:.2%})")
    print(f"    F1-Score:  {f1:.4f} ({f1:.2%})")

    print(f"\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=list(label_encoder.classes_),
    ))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion Matrix:")
    print(f"           hijau  kuning  merah")
    for i, name in enumerate(label_encoder.classes_):
        print(f"    {name:6s}  {cm[i]}")

    # Feature importance
    importances = model.feature_importances_
    print(f"\n  Feature Importance:")
    for feat, imp in sorted(zip(feature_columns, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"    {feat:30s} {imp:.4f} {bar}")

    # 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted")
    print(f"\n  5-Fold CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # ── 5. Save Model Artifact ───────────────────────────────────────
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # Save as a bundle: model + label_encoder + feature_columns
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
        "train_size": len(X_train),
        "test_size": len(X_test),
        "split_ratio": "70:30",
        "metrics": {
            "accuracy": round(acc, 4),
            "precision_weighted": round(prec, 4),
            "recall_weighted": round(rec, 4),
            "f1_weighted": round(f1, 4),
            "cv_f1_mean": round(cv_scores.mean(), 4),
            "cv_f1_std": round(cv_scores.std(), 4),
        },
        "feature_importance": {
            feat: round(float(imp), 4)
            for feat, imp in zip(feature_columns, importances)
        },
    }

    with open(OUTPUT_META, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[5/5] Model saved!")
    print(f"  Artifact: {OUTPUT_PKL}")
    print(f"  Metadata: {OUTPUT_META}")
    print(f"  File size: {os.path.getsize(OUTPUT_PKL):,} bytes")
    print(f"\n{'=' * 70}")
    print(f"✅ XGBoost Risk Aggregator Training Complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
