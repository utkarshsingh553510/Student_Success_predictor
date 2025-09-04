# Student Success Predictor

Predicts whether a student will **Pass** or **Fail** using classic machine learning on simple academic & lifestyle signals.

## 🔍 What it does
- Trains a **Logistic Regression** model to classify student outcome (Pass/Fail).
- Uses features: **StudyHours, Attendance, PastScore, SleepHours, Internet** (Yes/No).
- Shows evaluation via **classification report** and **confusion matrix**.

## 📂 Recommended Repo Structure
```
student-success-predictor/
├─ data/
│  └─ student_success_dataset.csv
├─ notebooks/
│  └─ Student_Success_Predictor.ipynb
├─ src/
│  └─ train.py
├─ models/              # saved models (created after training)
├─ reports/             # metrics & plots (auto-created)
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

## 🧰 Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## 🚀 Quickstart
1) **Clone the repo**
```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

2) **Create folders**
```bash
mkdir -p data notebooks src models reports
```

3) **Add your files**
- Put your dataset at: `data/student_success_dataset.csv`
- Put your notebook at: `notebooks/Student_Success_Predictor.ipynb`

4) **Install deps**
```bash
pip install -r requirements.txt
```

5) **Train from script (optional, in addition to the notebook)**
```bash
python src/train.py --data data/student_success_dataset.csv --outdir .
```

This will:
- Train a Logistic Regression pipeline
- Save model to `models/logreg.pkl`
- Save metrics to `reports/metrics.txt`
- Save confusion matrix plot to `reports/confusion_matrix.png`

## 🧪 Features (columns)
- `Internet` – Yes/No (encoded)
- `StudyHours` – avg study hours/day
- `Attendance` – percentage (0–100)
- `PastScore` – previous exam score
- `SleepHours` – avg sleep hours/day
- **Target:** `Passed` – Yes/No (encoded)

## 📈 Example Outputs
- **Classification report** (precision/recall/F1)
- **Confusion matrix** image

> Note: Actual scores depend on your dataset.

## 🛣️ Roadmap / Ideas
- Try other models (Random Forest, SVM, XGBoost)
- Cross-validation & hyperparameter tuning
- Feature importance & SHAP explanations
- Simple web app (Flask/Streamlit) for live predictions

## 📜 License
MIT — see `LICENSE` for details.

## 🙌 Acknowledgements
Made by Utkarsh Singh.
