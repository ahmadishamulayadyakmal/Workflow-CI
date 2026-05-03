# MLProject – Teen Mental Health Depression Risk

MLflow Project untuk melatih ulang model klasifikasi **depresi remaja** secara otomatis
melalui GitHub Actions CI.

---

## Struktur Folder

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml          # GitHub Actions CI workflow
└── MLProject/
    ├── MLProject                  # File konfigurasi MLflow Project
    ├── conda.yaml                 # Environment dependencies
    ├── modelling.py               # Script training (dari Kriteria 2)
    ├── Teen_Mental_Health_preprocessing.csv   # Dataset preprocessed (dari Kriteria 1)
    └── teen_mental_health_preprocessing/      # Folder data split (opsional)
```

---

## Hubungan dengan Kriteria Sebelumnya

| Kriteria | Output                                      | Digunakan di Kriteria 3 |
|----------|---------------------------------------------|-------------------------|
| **1**    | `Teen_Mental_Health_preprocessing.csv`      | Dataset input training  |
| **2**    | `modelling.py` + MLflow autolog             | Script di dalam MLProject |
| **3**    | `MLProject` + GitHub Actions CI             | Re-training otomatis    |

---

## Cara Menjalankan Secara Lokal

```bash
# Clone repository
git clone https://github.com/<username>/Workflow-CI.git
cd Workflow-CI/MLProject

# Install dependencies
pip install mlflow==2.19.0 scikit-learn==1.6.0 pandas==2.2.2 numpy==2.2.1 joblib==1.4.2

# Jalankan MLflow Project
mlflow run . --env-manager=local

# Lihat hasil di MLflow UI
mlflow ui
# buka http://127.0.0.1:5000
```

---

## Trigger CI

Workflow CI (`mlflow-ci.yml`) akan otomatis berjalan ketika:

- **Push** ke branch `main`
- **Pull Request** ke branch `main`
- **Manual** via tombol *Run workflow* di tab Actions GitHub

Model akan dilatih ulang secara otomatis setiap kali salah satu trigger di atas dipantik.

---

## Dataset

- **Sumber:** Teen Mental Health Dataset
- **File:** `Teen_Mental_Health_preprocessing.csv`
- **Baris:** 2.500 sampel
- **Fitur:** 11 fitur numerik (hasil preprocessing Kriteria 1)
- **Target:** `depression_risk` — 0 (low), 1 (medium), 2 (high)
