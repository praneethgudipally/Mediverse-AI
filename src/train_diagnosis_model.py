import pandas as pandas
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_FOLDER / "data" / "diagnosis_data.csv"

MODEL_FOLDER = PROJECT_FOLDER / "models"

MODEL_FILE = MODEL_FOLDER / "diagnosis_model.pkl"


# --------------------------------------------------
# CREATE MODEL FOLDER
# --------------------------------------------------

MODEL_FOLDER.mkdir(
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

if not DATA_FILE.exists():

    print(
        "Diagnosis dataset not found."
    )

    raise SystemExit


diagnosis_data = pandas.read_csv(
    DATA_FILE
)


print(
    "Diagnosis dataset loaded successfully."
)

print(
    "Total rows:",
    len(diagnosis_data)
)


# --------------------------------------------------
# INPUT FEATURES AND TARGET
# --------------------------------------------------

feature_columns = [
    "fever",
    "cough",
    "shortness_of_breath",
    "chest_pain",
    "fatigue",
    "headache",
    "nausea",
    "vomiting"
]


target_column = "diagnosis"


input_data = diagnosis_data[
    feature_columns
]

target_data = diagnosis_data[
    target_column
]


# --------------------------------------------------
# SPLIT DATA
# --------------------------------------------------

training_inputs, testing_inputs, training_targets, testing_targets = (
    train_test_split(
        input_data,
        target_data,
        test_size=0.20,
        random_state=42
    )
)
# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

diagnosis_model = LogisticRegression(
    max_iter=2000
)


diagnosis_model.fit(
    training_inputs,
    training_targets
)


print(
    "Diagnosis model trained successfully."
)


# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

predicted_targets = diagnosis_model.predict(
    testing_inputs
)


model_accuracy = accuracy_score(
    testing_targets,
    predicted_targets
)


print(
    "Model accuracy:",
    round(model_accuracy, 2)
)


print(
    "\nClassification Report:"
)


print(
    classification_report(
        testing_targets,
        predicted_targets,
        zero_division=0
    )
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    diagnosis_model,
    MODEL_FILE
)


print(
    "Diagnosis model saved successfully at:",
    MODEL_FILE
)