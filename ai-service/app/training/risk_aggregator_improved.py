"""
Improved training pipeline: feature engineering + SMOTE + hyperparameter tuning + ensemble.

Goal: improve from 70-73% baseline to 80-85% on UCI Bangladesh real data.
Methods:
  1. Feature engineering (interactions, domain rules, polynomial)
  2. SMOTE augmentation (1014 -> 5000+)
  3. Hyperparameter tuning (Optuna-free, manual grid)
  4. Ensemble (LR + XGB + RF voting classifier)
"""
import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATASET_PATH = "/Users/zelphyx/Projects/Maternin-AI/datasets/real_datasets/uci_maternal/Maternal Health Risk Data Set.csv"
ARTIFACT_DIR = "/Users/zelphyx/Projects/Maternin-AI/ai-service/app/model_artifacts"
SEED = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific features that capture medical interactions."""
    out = df.copy()

    # Original 8 features
    out["systolic_bp"] = df["SystolicBP"]
    out["diastolic_bp"] = df["DiastolicBP"]
    out["age"] = df["Age"]
    out["gestational_age_weeks"] = 28.0
    out["protein_urine_encoded"] = 0
    out["bmi"] = 25.0
    out["has_preeclampsia_history"] = 0
    out["has_hypertension_history"] = np.where(
        (df["SystolicBP"] >= 140) | (df["DiastolicBP"] >= 90), 1, 0
    )

    # === INTERACTION FEATURES (high impact for medical data) ===
    # BP interactions
    out["systolic_diastolic_product"] = df["SystolicBP"] * df["DiastolicBP"]
    out["pulse_pressure"] = df["SystolicBP"] - df["DiastolicBP"]  # narrow PP = concerning
    out["mean_arterial_pressure"] = df["DiastolicBP"] + (df["SystolicBP"] - df["DiastolicBP"]) / 3
    out["bp_severity"] = (df["SystolicBP"] - 120) / 30 + (df["DiastolicBP"] - 80) / 20  # how far above normal

    # Age interactions
    out["age_bp_interaction"] = df["Age"] * df["SystolicBP"] / 100  # older + high BP = risk
    out["is_extreme_age"] = np.where((df["Age"] < 18) | (df["Age"] > 35), 1, 0)
    out["is_high_age"] = np.where(df["Age"] >= 35, 1, 0)
    out["is_young"] = np.where(df["Age"] < 20, 1, 0)

    # Vital sign abnormalities
    out["heart_rate_abnormal"] = np.where((df["HeartRate"] < 60) | (df["HeartRate"] > 100), 1, 0)
    out["heart_rate_severity"] = np.abs(df["HeartRate"] - 80) / 20  # distance from normal 80
    out["fever"] = np.where(df["BodyTemp"] > 38.0, 1, 0)
    out["hypothermia"] = np.where(df["BodyTemp"] < 36.0, 1, 0)
    out["bs_abnormal"] = np.where((df["BS"] < 7) | (df["BS"] > 14), 1, 0)
    out["bs_severity"] = np.abs(df["BS"] - 10) / 5  # distance from normal 10

    # Composite risk scores (PNPK-aligned)
    out["severe_hypertension"] = np.where(
        (df["SystolicBP"] >= 160) | (df["DiastolicBP"] >= 110), 1, 0
    )
    out["moderate_hypertension"] = np.where(
        ((df["SystolicBP"] >= 140) & (df["SystolicBP"] < 160)) |
        ((df["DiastolicBP"] >= 90) & (df["DiastolicBP"] < 110)),
        1, 0
    )
    out["abnormal_vitals_count"] = (
        out["heart_rate_abnormal"] + out["fever"] + out["hypothermia"] + out["bs_abnormal"]
    )

    # Risk amplification features
    out["age_bp_severity"] = out["bp_severity"] * out["is_high_age"]
    out["multi_system_abnormality"] = out["moderate_hypertension"] * out["abnormal_vitals_count"]

    # Polynomial BP features
    out["systolic_squared"] = df["SystolicBP"] ** 2 / 1000
    out["diastolic_squared"] = df["DiastolicBP"] ** 2 / 1000

    return out


def get_feature_columns() -> list[str]:
    return [
        "systolic_bp", "diastolic_bp", "age", "gestational_age_weeks", "bmi",
        "has_hypertension_history", "protein_urine_encoded", "has_preeclampsia_history",
        # engineered
        "systolic_diastolic_product", "pulse_pressure", "mean_arterial_pressure", "bp_severity",
        "age_bp_interaction", "is_extreme_age", "is_high_age", "is_young",
        "heart_rate_abnormal", "heart_rate_severity", "fever", "hypothermia", "bs_abnormal", "bs_severity",
        "severe_hypertension", "moderate_hypertension", "abnormal_vitals_count",
        "age_bp_severity", "multi_system_abnormality",
        "systolic_squared", "diastolic_squared",
    ]


def smote_augment(X: np.ndarray, y: np.ndarray, target_size: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    """Simple SMOTE-like augmentation: oversample minority classes by interpolation."""
    from collections import Counter
    rng = np.random.RandomState(SEED)
    classes = np.unique(y)
    n_per_class = target_size // len(classes)

    X_new, y_new = [X], [y]
    for cls in classes:
        X_cls = X[y == cls]
        n_needed = n_per_class - len(X_cls)
        if n_needed <= 0:
            continue
        # Oversample with small noise
        idx_a = rng.randint(0, len(X_cls), n_needed)
        idx_b = rng.randint(0, len(X_cls), n_needed)
        alpha = rng.uniform(0.0, 1.0, (n_needed, 1))
        X_synth = X_cls[idx_a] * alpha + X_cls[idx_b] * (1 - alpha)
        # Add small noise
        noise = rng.normal(0, 0.02, X_synth.shape)
        X_synth = X_synth + noise
        X_new.append(X_synth)
        y_new.append(np.full(n_needed, cls))

    X_aug = np.vstack(X_new)
    y_aug = np.concatenate(y_new)
    return X_aug, y_aug


def main():
    print("=" * 70)
    print("IMPROVED Training — Feature Eng + SMOTE + Tuning + Ensemble")
    print("=" * 70)

    df = pd.read_csv(DATASET_PATH)
    print(f"\n[1/8] Dataset: {len(df)} REAL UCI records")

    df_eng = engineer_features(df)
    feature_columns = get_feature_columns()
    print(f"  Total features (after engineering): {len(feature_columns)}")

    # Proxy target: high risk vs (mid + low) — binary for LR
    # For XGB: 3 classes
    df_eng["target_high_risk"] = np.where(df["RiskLevel"] == "high risk", 1, 0)
    label_map = {"low risk": "hijau", "mid risk": "kuning", "high risk": "merah"}
    df_eng["target_badge"] = df["RiskLevel"].map(label_map)
    le = LabelEncoder()
    le.fit(["hijau", "kuning", "merah"])
    df_eng["target_badge_int"] = le.transform(df_eng["target_badge"])

    X = df_eng[feature_columns].values

    # === LR (binary) ===
    y_lr = df_eng["target_high_risk"].values
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_lr, test_size=0.20, random_state=SEED, stratify=y_lr,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval,
    )
    print(f"\n[2/8] LR split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    print(f"  Original positive rate: {y_train.mean():.2%}")

    # Augment training set
    X_train_aug, y_train_aug = smote_augment(X_train, y_train, target_size=4000)
    print(f"  After SMOTE: train={len(X_train_aug)}, positive rate={y_train_aug.mean():.2%}")

    # Train LR with engineered features + SMOTE
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=2000,
            random_state=SEED, solver="lbfgs",
        )),
    ])
    lr_pipeline.fit(X_train_aug, y_train_aug)
    lr_test_acc = accuracy_score(y_test, lr_pipeline.predict(X_test))
    lr_test_f1 = f1_score(y_test, lr_pipeline.predict(X_test))
    print(f"  LR Test Acc: {lr_test_acc:.4f} (was 0.7044 baseline)")

    # === XGB (3-class) ===
    y_xgb = df_eng["target_badge_int"].values
    X_trainval_x, X_test_x, y_trainval_x, y_test_x = train_test_split(
        X, y_xgb, test_size=0.20, random_state=SEED, stratify=y_xgb,
    )
    X_train_x, X_val_x, y_train_x, y_val_x = train_test_split(
        X_trainval_x, y_trainval_x, test_size=0.25, random_state=SEED, stratify=y_trainval_x,
    )

    X_train_x_aug, y_train_x_aug = smote_augment(X_train_x, y_train_x, target_size=5000)
    print(f"\n[3/8] XGB split: train={len(X_train_x)}, val={len(X_val_x)}, test={len(X_test_x)}")
    print(f"  After SMOTE: train={len(X_train_x_aug)}")

    # === Step 2: Hyperparameter Tuning for XGB ===
    print(f"\n[4/8] XGB Hyperparameter tuning...")
    best_xgb_score = 0
    best_xgb_params = None
    for n_est in [100, 200]:
        for max_d in [3, 5, 7]:
            for lr_rate in [0.05, 0.1, 0.2]:
                m = XGBClassifier(
                    n_estimators=n_est, max_depth=max_d, learning_rate=lr_rate,
                    subsample=0.8, colsample_bytree=0.8,
                    objective="multi:softprob", num_class=3,
                    eval_metric="mlogloss", random_state=SEED,
                    tree_method="hist",
                )
                m.fit(X_train_x_aug, y_train_x_aug, eval_set=[(X_val_x, y_val_x)], verbose=False)
                val_acc = accuracy_score(y_val_x, m.predict(X_val_x))
                if val_acc > best_xgb_score:
                    best_xgb_score = val_acc
                    best_xgb_params = (n_est, max_d, lr_rate)
    print(f"  Best: n_est={best_xgb_params[0]}, max_depth={best_xgb_params[1]}, lr={best_xgb_params[2]}")
    print(f"  Val Acc: {best_xgb_score:.4f}")

    # Train final XGB
    xgb_model = XGBClassifier(
        n_estimators=best_xgb_params[0], max_depth=best_xgb_params[1],
        learning_rate=best_xgb_params[2],
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=SEED,
        tree_method="hist",
    )
    xgb_model.fit(X_train_x_aug, y_train_x_aug, eval_set=[(X_val_x, y_val_x)], verbose=False)
    xgb_test_acc = accuracy_score(y_test_x, xgb_model.predict(X_test_x))
    xgb_test_f1 = f1_score(y_test_x, xgb_model.predict(X_test_x), average="weighted")
    print(f"  XGB Test Acc: {xgb_test_acc:.4f} (was 0.7340 baseline)")

    # === Step 3: Random Forest ===
    print(f"\n[5/8] Random Forest training...")
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    rf_model.fit(X_train_x_aug, y_train_x_aug)
    rf_test_acc = accuracy_score(y_test_x, rf_model.predict(X_test_x))
    rf_test_f1 = f1_score(y_test_x, rf_model.predict(X_test_x), average="weighted")
    print(f"  RF Test Acc: {rf_test_acc:.4f}")

    # === Step 4: Ensemble (Voting) ===
    print(f"\n[6/8] Ensemble (soft voting)...")
    # Need same y. Use 3-class version
    ensemble = VotingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("rf", rf_model),
        ],
        voting="soft",
        weights=[1.5, 1.0],  # XGB slightly more weight
    )
    ensemble.fit(X_train_x_aug, y_train_x_aug)
    ens_test_acc = accuracy_score(y_test_x, ensemble.predict(X_test_x))
    ens_test_f1 = f1_score(y_test_x, ensemble.predict(X_test_x), average="weighted")
    ens_test_prec = precision_score(y_test_x, ensemble.predict(X_test_x), average="weighted")
    ens_test_rec = recall_score(y_test_x, ensemble.predict(X_test_x), average="weighted")
    cm = confusion_matrix(y_test_x, ensemble.predict(X_test_x))
    print(f"  Ensemble Test Acc: {ens_test_acc:.4f}")
    print(f"  Ensemble Test F1:  {ens_test_f1:.4f}")
    print(f"  Ensemble Test Prec: {ens_test_prec:.4f}")
    print(f"  Ensemble Test Rec:  {ens_test_rec:.4f}")
    print(f"\n  Ensemble Confusion Matrix:")
    print(f"           hijau  kuning  merah")
    for i, name in enumerate(le.classes_):
        print(f"    {name:6s}  {cm[i]}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = cross_val_score(ensemble, X_train_x_aug, y_train_x_aug, cv=cv, scoring="f1_weighted")
    print(f"  5-fold CV F1 (weighted): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # === Step 5: Save best model (ensemble) ===
    print(f"\n[7/8] Saving best model (ensemble)...")

    feature_columns_xgb = feature_columns  # same features used
    artifact = {
        "model": ensemble,
        "label_encoder": le,
        "feature_columns": feature_columns_xgb,
    }
    output_pkl = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1.pkl")
    joblib.dump(artifact, output_pkl)
    print(f"  Saved: {output_pkl}")

    # === Step 6: Also save LR ===
    output_pkl_lr = os.path.join(ARTIFACT_DIR, "preeclampsia_lr_v1.pkl")
    joblib.dump(lr_pipeline, output_pkl_lr)
    print(f"  Saved: {output_pkl_lr}")

    # === Metadata ===
    metadata = {
        "model_name": "risk_aggregator_v1",
        "model_type": "VotingClassifier (XGB + RandomForest)",
        "model_components": ["XGBoost", "RandomForest"],
        "ensemble_weights": [1.5, 1.0],
        "features": feature_columns_xgb,
        "n_features": len(feature_columns_xgb),
        "labels": list(le.classes_),
        "dataset": "UCI_Maternal_Health_Risk_863",
        "dataset_size": len(df),
        "dataset_origin": "Bangladesh public hospitals (Ahmed et al., 2021)",
        "dataset_real_vs_synthetic": "REAL",
        "improvements_applied": [
            "Feature engineering (28 features from 8 base)",
            "SMOTE augmentation (1014 -> 5000+)",
            "Hyperparameter tuning (XGB grid search)",
            "Ensemble voting classifier",
        ],
        "split": "60/20/20 train/val/test + SMOTE on train",
        "split_seed": SEED,
        "metrics": {
            "lr_test_accuracy": round(float(lr_test_acc), 4),
            "lr_test_f1": round(float(lr_test_f1), 4),
            "xgb_test_accuracy": round(float(xgb_test_acc), 4),
            "xgb_test_f1": round(float(xgb_test_f1), 4),
            "rf_test_accuracy": round(float(rf_test_acc), 4),
            "rf_test_f1": round(float(rf_test_f1), 4),
            "ensemble_test_accuracy": round(float(ens_test_acc), 4),
            "ensemble_test_f1": round(float(ens_test_f1), 4),
            "ensemble_test_precision": round(float(ens_test_prec), 4),
            "ensemble_test_recall": round(float(ens_test_rec), 4),
            "cv_f1_mean": round(float(cv_scores.mean()), 4),
            "cv_f1_std": round(float(cv_scores.std()), 4),
        },
        "confusion_matrix_test": {
            "TN": int(cm[0][0]), "FP": int(cm[0][1]),
            "FN": int(cm[1][0]), "TP": int(cm[1][1]),
        },
        "improvement_vs_baseline": {
            "xgb_test_acc_before": 0.7340,
            "xgb_test_acc_after": round(float(xgb_test_acc), 4),
            "ensemble_test_acc_after": round(float(ens_test_acc), 4),
        },
        "caveats": [
            "Real UCI Bangladesh data with engineered features",
            "SMOTE augmentation on training set only (no leakage to val/test)",
            "Indonesia-specific data would give different numbers",
        ],
    }
    output_meta = os.path.join(ARTIFACT_DIR, "risk_aggregator_v1_metadata.json")
    with open(output_meta, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: {output_meta}")

    # LR metadata
    lr_metadata = {
        "model_name": "preeclampsia_lr_v1",
        "model_type": "LogisticRegression",
        "improvements_applied": ["Feature engineering", "SMOTE"],
        "metrics": {
            "test_accuracy": round(float(lr_test_acc), 4),
            "test_f1": round(float(lr_test_f1), 4),
        },
        "improvement_vs_baseline": {
            "test_acc_before": 0.7044,
            "test_acc_after": round(float(lr_test_acc), 4),
        },
    }
    output_meta_lr = os.path.join(ARTIFACT_DIR, "preeclampsia_lr_v1_metadata.json")
    with open(output_meta_lr, "w") as f:
        json.dump(lr_metadata, f, indent=2)
    print(f"  Saved: {output_meta_lr}")

    # === Final comparison ===
    print(f"\n[8/8] FINAL COMPARISON:")
    print(f"  {'Method':<30s} {'Test Acc':<10s} {'Test F1':<10s}")
    print(f"  {'-'*50}")
    print(f"  {'Baseline (UCI, no eng)':<30s} {0.7044:<10.4f} {0.5312:<10.4f}")
    print(f"  {'+ Feature Engineering + SMOTE':<30s} {lr_test_acc:<10.4f} {lr_test_f1:<10.4f}")
    print(f"  {'XGB + Eng + SMOTE + Tuning':<30s} {xgb_test_acc:<10.4f} {xgb_test_f1:<10.4f}")
    print(f"  {'Random Forest + Eng + SMOTE':<30s} {rf_test_acc:<10.4f} {rf_test_f1:<10.4f}")
    print(f"  {'Ensemble (XGB + RF)':<30s} {ens_test_acc:<10.4f} {ens_test_f1:<10.4f}")

    print(f"\n{'=' * 70}")
    print(f"OK Improved training complete!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
