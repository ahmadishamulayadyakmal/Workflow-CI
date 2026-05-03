"""
modelling.py – Kriteria 3: Workflow CI dengan MLflow Project
=============================================================
Dataset  : Teen Mental Health Dataset (sudah dipreproses – Kriteria 1)
Target   : depression_risk  (multiclass: 0=low / 1=medium / 2=high)
Model    : RandomForestClassifier (autolog MLflow – Kriteria 2)
CI       : Dijalankan otomatis via GitHub Actions + MLflow Project

Hubungan antar Kriteria
-----------------------
Kriteria 1  →  menghasilkan Teen_Mental_Health_preprocessing.csv
Kriteria 2  →  modelling.py ini (autolog, tanpa hyperparameter tuning)
Kriteria 3  →  mengemas modelling.py ke dalam MLProject + GitHub Actions CI
"""

import os
import warnings

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# 1. Konfigurasi MLflow                                               #
# ------------------------------------------------------------------ #
mlflow.set_experiment("Teen-Mental-Health-Depression-Risk")

# ------------------------------------------------------------------ #
# 2. Load Dataset hasil preprocessing (Kriteria 1)                   #
#    Path relatif terhadap lokasi script ini                          #
# ------------------------------------------------------------------ #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Teen_Mental_Health_preprocessing.csv")


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] File tidak ditemukan: {path}\n"
            "Pastikan Teen_Mental_Health_preprocessing.csv ada di folder MLProject."
        )
    df = pd.read_csv(path)
    print(f"[INFO] Dataset dimuat  : {path}")
    print(f"[INFO] Shape           : {df.shape}")
    print(f"[INFO] Kolom           : {df.columns.tolist()}")
    return df


# ------------------------------------------------------------------ #
# 3. Split fitur & target (sama persis dengan Kriteria 2)             #
# ------------------------------------------------------------------ #
def prepare_data(df: pd.DataFrame):
    X = df.drop("depression_risk", axis=1)
    y = df["depression_risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[INFO] Train size      : {X_train.shape}")
    print(f"[INFO] Test  size      : {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
# 4. Training dengan MLflow autolog (Basic – Kriteria 2)              #
# ------------------------------------------------------------------ #
def train_model(X_train, X_test, y_train, y_test):
    # Aktifkan autolog scikit-learn → parameter + metrik + model tersimpan otomatis
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="RandomForest-CI"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )

        print("\n[INFO] Melatih model RandomForestClassifier ...")
        model.fit(X_train, y_train)

        # Evaluasi
        y_pred    = model.predict(X_test)
        train_acc = model.score(X_train, y_train)
        test_acc  = accuracy_score(y_test, y_pred)

        # Log metrik tambahan secara manual
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("test_accuracy",  test_acc)

        print(f"\n[HASIL] Train Accuracy : {train_acc:.4f}")
        print(f"[HASIL] Test  Accuracy : {test_acc:.4f}")
        print("\n[HASIL] Classification Report:")
        print(
            classification_report(
                y_test, y_pred,
                target_names=["low", "medium", "high"],
            )
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\n[INFO] MLflow Run ID   : {run_id}")
        print("[INFO] Artefak tersimpan di folder mlruns/")

    return model, run_id


# ------------------------------------------------------------------ #
# 5. Main                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    print("=" * 55)
    print("  WORKFLOW CI - TEEN MENTAL HEALTH DEPRESSION RISK")
    print("=" * 55)

    df                               = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = prepare_data(df)
    model, run_id                    = train_model(X_train, X_test, y_train, y_test)

    print("\n[SELESAI] Model berhasil dilatih dan di-log ke MLflow!")
