import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

def load_data(path):
    df = pd.read_csv(path)
    # Basic cleaning
    df = df.dropna().copy()
    # Map categorical
    if 'Internet' in df.columns:
        df['Internet'] = df['Internet'].astype(str).str.strip().str.lower().map({'yes':1,'no':0})
    if 'Passed' in df.columns:
        df['Passed'] = df['Passed'].astype(str).str.strip().str.lower().map({'yes':1,'no':0})
    return df

def main():
    parser = argparse.ArgumentParser(description="Train Student Success Predictor (Logistic Regression)")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='.', help='Repository root (where models/ & reports/ will be created)')
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, 'reports'), exist_ok=True)

    df = load_data(args.data)

    feature_cols = ['Internet', 'StudyHours', 'Attendance', 'PastScore', 'SleepHours']
    target_col = 'Passed'

    missing = [c for c in feature_cols+[target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    X = df[feature_cols]
    y = df[target_col]

    numeric_features = ['StudyHours', 'Attendance', 'PastScore', 'SleepHours']
    # Internet is already 0/1 after mapping
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_features)],
        remainder='passthrough'
    )

    clf = Pipeline(steps=[
        ('prep', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Metrics
    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    # Save metrics
    metrics_path = os.path.join(args.outdir, 'reports', 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(report + "\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n")

    # Plot confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.xticks([0,1], ['Fail','Pass'])
    plt.yticks([0,1], ['Fail','Pass'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    cm_path = os.path.join(args.outdir, 'reports', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # Save model
    model_path = os.path.join(args.outdir, 'models', 'logreg.pkl')
    joblib.dump(clf, model_path)

    print("Training complete.")
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved confusion matrix: {cm_path}")

if __name__ == '__main__':
    main()
