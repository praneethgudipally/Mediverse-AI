import pandas as pandas

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import joblib


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parents[1]

DATA_FILE = PROJECT_FOLDER / "data" / "ehr_data.csv"

MODEL_FOLDER = PROJECT_FOLDER / "models"

MODEL_FILE = MODEL_FOLDER / "mortality_model.pkl"


# --------------------------------------------------
# CREATE MODEL FOLDER
# --------------------------------------------------

MODEL_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        "Dataset not found. Check data/ehr_data.csv"
    )


ehr_data = pandas.read_csv(DATA_FILE)


print("Dataset loaded successfully.")
print(
    f"Total rows: {len(ehr_data)}"
)


# --------------------------------------------------
# SELECT FEATURES
# --------------------------------------------------

feature_columns = [
    "age",
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "glucose_level",
    "previous_admission",
    "icu_history",
    "length_of_stay"
]


target_column = "mortality"


# --------------------------------------------------
# CHECK REQUIRED COLUMNS
# --------------------------------------------------

required_columns = feature_columns + [
    target_column
]


missing_columns = [
    column
    for column in required_columns
    if column not in ehr_data.columns
]


if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )


# --------------------------------------------------
# PREPARE INPUT AND OUTPUT
# --------------------------------------------------

input_data = ehr_data[
    feature_columns
]


output_data = ehr_data[
    target_column
]


# --------------------------------------------------
# SPLIT DATASET
# --------------------------------------------------

training_inputs, testing_inputs, training_outputs, testing_outputs = (
    train_test_split(
        input_data,
        output_data,
        test_size=0.2,
        random_state=42,
        stratify=output_data
    )
)


# --------------------------------------------------
# CREATE MACHINE LEARNING PIPELINE
# --------------------------------------------------

mortality_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------

mortality_model.fit(
    training_inputs,
    training_outputs
)


print("Mortality model trained successfully.")


# --------------------------------------------------
# EVALUATE MODEL
# --------------------------------------------------

predicted_outputs = mortality_model.predict(
    testing_inputs
)


model_accuracy = accuracy_score(
    testing_outputs,
    predicted_outputs
)


print(
    f"Model accuracy: {model_accuracy:.2f}"
)


print("\nClassification Report:")
print(
    classification_report(
        testing_outputs,
        predicted_outputs,
        zero_division=0
    )
)


# --------------------------------------------------
# SAVE MODEL
# --------------------------------------------------

joblib.dump(
    mortality_model,
    MODEL_FILE
)


print(
    f"Model saved successfully at: {MODEL_FILE}"
)