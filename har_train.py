import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Load data
def load_data():
    base_path = "data/UCI HAR Dataset/"
    X_train = pd.read_csv(base_path + "train/X_train.txt", delim_whitespace=True, header=None)
    y_train = pd.read_csv(base_path + "train/y_train.txt", delim_whitespace=True, header=None)
    X_test = pd.read_csv(base_path + "test/X_test.txt", delim_whitespace=True, header=None)
    y_test = pd.read_csv(base_path + "test/y_test.txt", delim_whitespace=True, header=None)
    activity_labels = pd.read_csv(base_path + "activity_labels.txt", delim_whitespace=True, header=None, index_col=0)
    y_train = y_train.replace(activity_labels.to_dict()[1])
    y_test = y_test.replace(activity_labels.to_dict()[1])
    return X_train, y_train, X_test, y_test

# Train and save model
def train_model():
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_data()
    X_train = X_train.iloc[:, :100]  # Keep only first 100 columns
    X_test = X_test.iloc[:, :100]


    print("Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train.values.ravel())

    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/har_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("Model and scaler saved to 'models/' folder.")

if __name__ == "__main__":
    train_model()
