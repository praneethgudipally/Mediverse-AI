import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MediVerse AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_FILE = PROJECT_FOLDER / "data" / "ehr_data.csv"

MORTALITY_MODEL_FILE = (
    PROJECT_FOLDER / "models" / "mortality_model.pkl"
)

DIAGNOSIS_MODEL_FILE = (
    PROJECT_FOLDER / "models" / "diagnosis_model.pkl"
)


# ============================================================
# LOAD EHR DATA
# ============================================================

try:

    patient_data = pd.read_csv(DATA_FILE)

except Exception as error:

    st.error("❌ Could not load the EHR dataset.")

    st.write(
        "Expected file:"
    )

    st.code(str(DATA_FILE))

    st.exception(error)

    st.stop()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patient_id",
    "age",
    "gender",
    "heart_rate",
    "blood_pressure",
    "oxygen_level",
    "glucose_level",
    "previous_admission",
    "icu_history",
    "length_of_stay",
    "mortality"
]

missing_columns = [
    column
    for column in required_columns
    if column not in patient_data.columns
]

if missing_columns:

    st.error("❌ Required columns are missing from ehr_data.csv.")

    st.write("Missing columns:")

    for column in missing_columns:
        st.write(f"• {column}")

    st.stop()


# ============================================================
# LOAD MORTALITY MODEL
# ============================================================

mortality_model = None
mortality_model_error = None

try:

    mortality_model = joblib.load(
        MORTALITY_MODEL_FILE
    )

except Exception as error:

    mortality_model_error = error


# ============================================================
# LOAD DIAGNOSIS MODEL
# ============================================================

diagnosis_model = None
diagnosis_model_error = None

try:

    diagnosis_model = joblib.load(
        DIAGNOSIS_MODEL_FILE
    )

except Exception as error:

    diagnosis_model_error = error


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏥 MediVerse AI")

    st.caption(
        "Multimodal Clinical Intelligence"
    )

    st.divider()

    navigation = st.radio(
        "Navigate",
        [
            "Overview",
            "Patient Clinical Profile",
            "AI Clinical Assessment",
            "Analytics",
            "About MediVerse"
        ]
    )

    st.divider()

    if mortality_model is not None:

        st.success(
            "🟢 Mortality Model Online"
        )

    else:

        st.warning(
            "🟡 Mortality Model Unavailable"
        )

    if diagnosis_model is not None:

        st.success(
            "🟢 Diagnosis Model Online"
        )

    else:

        st.warning(
            "🟡 Diagnosis Model Unavailable"
        )

    st.divider()

    st.caption(
        "Academic Research Prototype"
    )


# ============================================================
# COMMON CALCULATIONS
# ============================================================

total_patients = len(patient_data)

average_age = patient_data["age"].mean()

average_length_of_stay = (
    patient_data["length_of_stay"].mean()
)

mortality_cases = int(
    patient_data["mortality"].sum()
)


# ============================================================
# OVERVIEW
# ============================================================

if navigation == "Overview":

    st.title("🏥 MediVerse AI")

    st.subheader(
        "Multimodal Clinical Intelligence"
    )

    st.write(
        "An AI-assisted Electronic Health Record "
        "analysis platform designed to support "
        "clinical outcome prediction."
    )

    st.divider()

    st.header(
        "📊 Clinical Intelligence Overview"
    )

    st.write(
        "Current prototype uses structured EHR data "
        "to demonstrate patient analysis and "
        "machine-learning prediction."
    )

    st.divider()

    # --------------------------------------------------------
    # KPI SECTION
    # --------------------------------------------------------

    st.subheader(
        "🏥 Hospital Dataset Summary"
    )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    with metric_1:

        st.metric(
            "👥 Total Patients",
            total_patients
        )

    with metric_2:

        st.metric(
            "🎂 Average Age",
            f"{average_age:.1f} years"
        )

    with metric_3:

        st.metric(
            "🛏️ Average Hospital Stay",
            f"{average_length_of_stay:.1f} days"
        )

    with metric_4:

        st.metric(
            "⚠️ Mortality Cases",
            mortality_cases
        )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM CAPABILITIES
    # --------------------------------------------------------

    st.header(
        "🧬 Clinical AI Capabilities"
    )

    capability_1, capability_2, capability_3 = st.columns(3)

    with capability_1:

        st.subheader(
            "🩺 Patient Analysis"
        )

        st.write(
            "Analyzes patient demographics, "
            "vital signs, medical history and "
            "hospitalization information."
        )

    with capability_2:

        st.subheader(
            "🤖 AI Prediction"
        )

        st.write(
            "Uses machine-learning models to "
            "generate experimental clinical "
            "predictions."
        )

    with capability_3:

        st.subheader(
            "📈 Clinical Analytics"
        )

        st.write(
            "Provides statistical summaries "
            "and visual exploration of EHR data."
        )

    st.divider()

    # --------------------------------------------------------
    # MODEL STATUS
    # --------------------------------------------------------

    st.header(
        "🤖 AI Model Status"
    )

    status_1, status_2 = st.columns(2)

    with status_1:

        if mortality_model is not None:

            st.success(
                "🟢 Mortality Risk Model — ONLINE"
            )

            st.caption(
                "Structured EHR mortality prediction"
            )

        else:

            st.error(
                "🔴 Mortality Risk Model — NOT FOUND"
            )

    with status_2:

        if diagnosis_model is not None:

            st.success(
                "🟢 Clinical Pattern Model — ONLINE"
            )

            st.caption(
                "Symptom-based classification"
            )

        else:

            st.error(
                "🔴 Clinical Pattern Model — NOT FOUND"
            )

    st.divider()

    # --------------------------------------------------------
    # PATIENT TABLE
    # --------------------------------------------------------

    st.header(
        "👥 Patient Records"
    )

    st.dataframe(
        patient_data,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # MORTALITY DISTRIBUTION
    # --------------------------------------------------------

    st.header(
        "⚠️ Mortality Distribution"
    )

    mortality_distribution = (
        patient_data["mortality"]
        .value_counts()
        .sort_index()
    )

    mortality_distribution.index = [
        "Survived"
        if value == 0
        else "Mortality"
        for value in mortality_distribution.index
    ]

    st.bar_chart(
        mortality_distribution
    )


# ============================================================
# PATIENT CLINICAL PROFILE
# ============================================================

elif navigation == "Patient Clinical Profile":

    st.title(
        "🧬 Patient Clinical Profile"
    )

    st.write(
        "Explore the complete structured clinical "
        "profile of an individual patient."
    )

    st.divider()

    # --------------------------------------------------------
    # PATIENT SELECTION
    # --------------------------------------------------------

    patient_ids = (
        patient_data["patient_id"]
        .astype(str)
        .tolist()
    )

    selected_patient_id = st.selectbox(
        "🔎 Select Patient",
        patient_ids
    )

    selected_patient = patient_data[
        patient_data["patient_id"].astype(str)
        == selected_patient_id
    ].iloc[0]

    st.divider()

    # --------------------------------------------------------
    # PATIENT INFORMATION
    # --------------------------------------------------------

    st.header(
        "👤 Patient Information"
    )

    information_1, information_2, information_3, information_4 = st.columns(4)

    with information_1:

        st.metric(
            "Patient ID",
            str(selected_patient["patient_id"])
        )

    with information_2:

        st.metric(
            "Age",
            f"{selected_patient['age']} years"
        )

    with information_3:

        st.metric(
            "Gender",
            str(selected_patient["gender"])
        )

    with information_4:

        st.metric(
            "Hospital Stay",
            f"{selected_patient['length_of_stay']} days"
        )

    st.divider()

    # --------------------------------------------------------
    # ANATOMICAL INTELLIGENCE
    # --------------------------------------------------------

    anatomy_column, indicators_column = st.columns(
        [1, 2]
    )

    with anatomy_column:

        st.subheader(
            "🧬 Patient Anatomical Intelligence"
        )

        st.markdown(
            "# 🧍"
        )

        st.info(
            "Human anatomy visualization"
        )

        st.caption(
            "Prototype visualization linking "
            "clinical indicators to the patient profile."
        )

    # --------------------------------------------------------
    # CLINICAL INDICATORS
    # --------------------------------------------------------

    with indicators_column:

        st.subheader(
            "❤️ Clinical Indicators"
        )

        vital_1, vital_2 = st.columns(2)

        with vital_1:

            st.metric(
                "❤️ Heart Rate",
                f"{selected_patient['heart_rate']} bpm"
            )

            st.metric(
                "🫁 Oxygen Level",
                f"{selected_patient['oxygen_level']}%"
            )

            st.metric(
                "🍬 Glucose Level",
                str(selected_patient["glucose_level"])
            )

        with vital_2:

            st.metric(
                "🩸 Blood Pressure",
                str(selected_patient["blood_pressure"])
            )

            st.metric(
                "🛏️ Length of Stay",
                f"{selected_patient['length_of_stay']} days"
            )

            if selected_patient["icu_history"] == 1:

                st.metric(
                    "🏥 ICU History",
                    "Yes"
                )

            else:

                st.metric(
                    "🏥 ICU History",
                    "No"
                )

    st.divider()

    # --------------------------------------------------------
    # PATIENT HISTORY
    # --------------------------------------------------------

    st.header(
        "📋 Patient History"
    )

    history_1, history_2 = st.columns(2)

    with history_1:

        if selected_patient["previous_admission"] == 1:

            st.warning(
                "⚠️ Previous hospital admission recorded."
            )

        else:

            st.success(
                "✅ No previous hospital admission recorded."
            )

    with history_2:

        if selected_patient["icu_history"] == 1:

            st.warning(
                "⚠️ Previous ICU history recorded."
            )

        else:

            st.success(
                "✅ No previous ICU history recorded."
            )

    st.divider()

    # --------------------------------------------------------
    # RECORDED OUTCOME
    # --------------------------------------------------------

    st.header(
        "⚕️ Recorded Clinical Outcome"
    )

    if selected_patient["mortality"] == 1:

        st.error(
            "Mortality outcome recorded for this patient."
        )

    else:

        st.success(
            "Patient survival outcome recorded."
        )

    st.caption(
        "This is the recorded dataset outcome, "
        "not a medical diagnosis."
    )


# ============================================================
# AI CLINICAL ASSESSMENT
# ============================================================

elif navigation == "AI Clinical Assessment":

    st.title(
        "🤖 AI Clinical Assessment"
    )

    st.write(
        "Enter patient symptoms and clinical measurements "
        "to generate experimental machine-learning predictions."
    )

    st.divider()

    # --------------------------------------------------------
    # SECTION 1 — SYMPTOMS
    # --------------------------------------------------------

    st.header(
        "🩺 Symptom Assessment"
    )

    st.write(
        "Select the symptoms currently observed "
        "in the patient."
    )

    symptom_column_1, symptom_column_2, symptom_column_3, symptom_column_4 = st.columns(4)

    with symptom_column_1:

        fever = st.checkbox(
            "🌡️ Fever"
        )

    with symptom_column_2:

        cough = st.checkbox(
            "🫁 Cough"
        )

    with symptom_column_3:

        shortness_of_breath = st.checkbox(
            "😮‍💨 Shortness of Breath"
        )

    with symptom_column_4:

        chest_pain = st.checkbox(
            "❤️ Chest Pain"
        )

    symptom_column_5, symptom_column_6, symptom_column_7, symptom_column_8 = st.columns(4)

    with symptom_column_5:

        fatigue = st.checkbox(
            "😴 Fatigue"
        )

    with symptom_column_6:

        headache = st.checkbox(
            "🤕 Headache"
        )

    with symptom_column_7:

        nausea = st.checkbox(
            "🤢 Nausea"
        )

    with symptom_column_8:

        vomiting = st.checkbox(
            "🤮 Vomiting"
        )

    st.divider()

    # --------------------------------------------------------
    # DIAGNOSIS MODEL
    # --------------------------------------------------------

    st.header(
        "🧠 Clinical AI Assessment"
    )

    if diagnosis_model is None:

        st.error(
            "❌ Diagnosis model is not available."
        )

        st.write(
            "Expected model:"
        )

        st.code(
            str(DIAGNOSIS_MODEL_FILE)
        )

        if diagnosis_model_error is not None:

            with st.expander(
                "Show technical error"
            ):

                st.exception(
                    diagnosis_model_error
                )

    else:

        analyze_button = st.button(
            "🔍 Analyze Patient",
            use_container_width=True
        )

        if analyze_button:

            diagnosis_features = pd.DataFrame(
                [[
                    int(fever),
                    int(cough),
                    int(shortness_of_breath),
                    int(chest_pain),
                    int(fatigue),
                    int(headache),
                    int(nausea),
                    int(vomiting)
                ]],
                columns=[
                    "fever",
                    "cough",
                    "shortness_of_breath",
                    "chest_pain",
                    "fatigue",
                    "headache",
                    "nausea",
                    "vomiting"
                ]
            )

            try:

                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                diagnosis_prediction = (
                    diagnosis_model
                    .predict(
                        diagnosis_features
                    )[0]
                )

                # --------------------------------------------
                # CONFIDENCE
                # --------------------------------------------

                diagnosis_confidence = None

                if hasattr(
                    diagnosis_model,
                    "predict_proba"
                ):

                    diagnosis_probabilities = (
                        diagnosis_model
                        .predict_proba(
                            diagnosis_features
                        )[0]
                    )

                    diagnosis_confidence = (
                        max(
                            diagnosis_probabilities
                        ) * 100
                    )

                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                st.success(
                    "✅ Clinical assessment completed."
                )

                st.divider()

                result_column_1, result_column_2 = st.columns(2)

                with result_column_1:

                    st.subheader(
                        "🧠 Predicted Clinical Pattern"
                    )

                    st.markdown(
                        f"### {diagnosis_prediction}"
                    )

                    st.write(
                        "The model identified this category "
                        "as the highest-probability classification "
                        "for the selected symptoms."
                    )

                with result_column_2:

                    st.subheader(
                        "📊 Model Confidence"
                    )

                    if diagnosis_confidence is not None:

                        st.metric(
                            "Confidence",
                            f"{diagnosis_confidence:.2f}%"
                        )

                        st.progress(
                            min(
                                diagnosis_confidence / 100,
                                1.0
                            )
                        )

                    else:

                        st.info(
                            "This model does not provide "
                            "probability estimates."
                        )

                st.divider()

                # --------------------------------------------
                # SELECTED SYMPTOMS
                # --------------------------------------------

                st.subheader(
                    "📋 Selected Symptoms"
                )

                selected_symptoms = []

                if fever:
                    selected_symptoms.append(
                        "Fever"
                    )

                if cough:
                    selected_symptoms.append(
                        "Cough"
                    )

                if shortness_of_breath:
                    selected_symptoms.append(
                        "Shortness of Breath"
                    )

                if chest_pain:
                    selected_symptoms.append(
                        "Chest Pain"
                    )

                if fatigue:
                    selected_symptoms.append(
                        "Fatigue"
                    )

                if headache:
                    selected_symptoms.append(
                        "Headache"
                    )

                if nausea:
                    selected_symptoms.append(
                        "Nausea"
                    )

                if vomiting:
                    selected_symptoms.append(
                        "Vomiting"
                    )

                if len(selected_symptoms) == 0:

                    st.info(
                        "No symptoms were selected."
                    )

                else:

                    for symptom in selected_symptoms:

                        st.write(
                            f"• {symptom}"
                        )

                st.divider()

                st.warning(
                    "⚕️ This is an experimental AI output "
                    "from the current academic prototype. "
                    "It is not a medical diagnosis."
                )

            except Exception as error:

                st.error(
                    "❌ The diagnosis model could not process "
                    "the selected symptoms."
                )

                st.write(
                    "This usually means that the features "
                    "used by the model during training do not "
                    "exactly match the current input columns."
                )

                with st.expander(
                    "Show technical error"
                ):

                    st.exception(
                        error
                    )

    st.divider()

    # ========================================================
    # MORTALITY RISK MODEL
    # ========================================================

    st.header(
        "⚠️ Mortality Risk Model"
    )

    st.write(
        "Enter structured EHR measurements to estimate "
        "the mortality probability generated by the "
        "current experimental model."
    )

    mortality_input_column_1, mortality_input_column_2 = st.columns(2)

    with mortality_input_column_1:

        mortality_age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=45,
            step=1
        )

        mortality_heart_rate = st.number_input(
            "Heart Rate (bpm)",
            min_value=20,
            max_value=250,
            value=80,
            step=1
        )

        mortality_blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=30,
            max_value=300,
            value=120,
            step=1
        )

        mortality_oxygen = st.number_input(
            "Oxygen Level (%)",
            min_value=50,
            max_value=100,
            value=98,
            step=1
        )

    with mortality_input_column_2:

        mortality_glucose = st.number_input(
            "Glucose Level",
            min_value=20,
            max_value=600,
            value=100,
            step=1
        )

        mortality_previous_admission = st.selectbox(
            "Previous Admission",
            [
                "No",
                "Yes"
            ]
        )

        mortality_icu_history = st.selectbox(
            "ICU History",
            [
                "No",
                "Yes"
            ]
        )

        mortality_length_of_stay = st.number_input(
            "Length of Stay (days)",
            min_value=0,
            max_value=365,
            value=3,
            step=1
        )

    st.divider()

    if mortality_model is None:

        st.error(
            "❌ Mortality model is not available."
        )

        st.write(
            "Expected model:"
        )

        st.code(
            str(MORTALITY_MODEL_FILE)
        )

        if mortality_model_error is not None:

            with st.expander(
                "Show technical error"
            ):

                st.exception(
                    mortality_model_error
                )

    else:

        predict_button = st.button(
            "⚠️ Predict Mortality Risk",
            use_container_width=True
        )

        if predict_button:

            mortality_features = pd.DataFrame(
                [[
                    mortality_age,
                    mortality_heart_rate,
                    mortality_blood_pressure,
                    mortality_oxygen,
                    mortality_glucose,
                    1
                    if mortality_previous_admission == "Yes"
                    else 0,
                    1
                    if mortality_icu_history == "Yes"
                    else 0,
                    mortality_length_of_stay
                ]],
                columns=[
                    "age",
                    "heart_rate",
                    "blood_pressure",
                    "oxygen_level",
                    "glucose_level",
                    "previous_admission",
                    "icu_history",
                    "length_of_stay"
                ]
            )

            try:

                # --------------------------------------------
                # MODEL PREDICTION
                # --------------------------------------------

                mortality_prediction = (
                    mortality_model
                    .predict(
                        mortality_features
                    )[0]
                )

                # --------------------------------------------
                # PROBABILITY
                # --------------------------------------------

                mortality_probability = None

                if hasattr(
                    mortality_model,
                    "predict_proba"
                ):

                    probability_values = (
                        mortality_model
                        .predict_proba(
                            mortality_features
                        )[0]
                    )

                    # Find probability corresponding
                    # to class 1 if available.
                    if hasattr(
                        mortality_model,
                        "classes_"
                    ):

                        model_classes = list(
                            mortality_model.classes_
                        )

                        if 1 in model_classes:

                            mortality_class_index = (
                                model_classes.index(1)
                            )

                            mortality_probability = (
                                probability_values[
                                    mortality_class_index
                                ] * 100
                            )

                        else:

                            mortality_probability = (
                                max(
                                    probability_values
                                ) * 100
                            )

                    else:

                        mortality_probability = (
                            max(
                                probability_values
                            ) * 100
                        )

                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                st.success(
                    "✅ Mortality risk analysis completed."
                )

                st.divider()

                result_column_1, result_column_2 = st.columns(2)

                with result_column_1:

                    st.subheader(
                        "⚕️ Risk Classification"
                    )

                    if mortality_prediction == 1:

                        st.error(
                            "Elevated Risk Classification"
                        )

                        st.write(
                            "The current model classified "
                            "this input as mortality-risk positive."
                        )

                    else:

                        st.success(
                            "Lower Risk Classification"
                        )

                        st.write(
                            "The current model classified "
                            "this input as mortality-risk negative."
                        )

                with result_column_2:

                    st.subheader(
                        "📊 Estimated Probability"
                    )

                    if mortality_probability is not None:

                        st.metric(
                            "Mortality Probability",
                            f"{mortality_probability:.2f}%"
                        )

                        st.progress(
                            min(
                                mortality_probability / 100,
                                1.0
                            )
                        )

                    else:

                        st.info(
                            "Probability is not available "
                            "for this model."
                        )

                st.divider()

                # --------------------------------------------
                # INPUT SUMMARY
                # --------------------------------------------

                st.subheader(
                    "📋 Clinical Data Used"
                )

                input_summary_column_1, input_summary_column_2 = st.columns(2)

                with input_summary_column_1:

                    st.write(
                        f"**Age:** {mortality_age}"
                    )

                    st.write(
                        f"**Heart Rate:** "
                        f"{mortality_heart_rate} bpm"
                    )

                    st.write(
                        f"**Blood Pressure:** "
                        f"{mortality_blood_pressure}"
                    )

                    st.write(
                        f"**Oxygen Level:** "
                        f"{mortality_oxygen}%"
                    )

                with input_summary_column_2:

                    st.write(
                        f"**Glucose:** "
                        f"{mortality_glucose}"
                    )

                    st.write(
                        f"**Previous Admission:** "
                        f"{mortality_previous_admission}"
                    )

                    st.write(
                        f"**ICU History:** "
                        f"{mortality_icu_history}"
                    )

                    st.write(
                        f"**Length of Stay:** "
                        f"{mortality_length_of_stay} days"
                    )

                st.divider()

                st.warning(
                    "⚕️ Experimental output from the current "
                    "prototype model. The model was developed "
                    "using a small demonstration dataset and "
                    "must not be used for clinical decisions."
                )

            except Exception as error:

                st.error(
                    "❌ The mortality model could not process "
                    "the entered clinical data."
                )

                st.write(
                    "The model may have been trained with "
                    "different feature names or data types."
                )

                with st.expander(
                    "Show technical error"
                ):

                    st.exception(
                        error
                    )


# ============================================================
# ANALYTICS
# ============================================================

elif navigation == "Analytics":

    st.title(
        "📈 Clinical Analytics"
    )

    st.write(
        "Explore patterns within the structured "
        "Electronic Health Record dataset."
    )

    st.divider()

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    st.header(
        "🎂 Patient Age Distribution"
    )

    age_distribution = (
        patient_data["age"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        age_distribution
    )

    st.divider()

    # --------------------------------------------------------
    # GENDER
    # --------------------------------------------------------

    st.header(
        "👥 Gender Distribution"
    )

    gender_distribution = (
        patient_data["gender"]
        .value_counts()
    )

    st.bar_chart(
        gender_distribution
    )

    st.divider()

    # --------------------------------------------------------
    # LENGTH OF STAY
    # --------------------------------------------------------

    st.header(
        "🛏️ Hospital Length of Stay"
    )

    stay_distribution = (
        patient_data["length_of_stay"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        stay_distribution
    )

    st.divider()

    # --------------------------------------------------------
    # ICU HISTORY
    # --------------------------------------------------------

    st.header(
        "🏥 ICU History"
    )

    icu_distribution = (
        patient_data["icu_history"]
        .value_counts()
        .sort_index()
    )

    icu_distribution.index = [
        "No ICU History"
        if value == 0
        else "Previous ICU History"
        for value in icu_distribution.index
    ]

    st.bar_chart(
        icu_distribution
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.header(
        "📊 Dataset Information"
    )

    analytics_1, analytics_2, analytics_3 = st.columns(3)

    with analytics_1:

        st.metric(
            "Patients",
            len(patient_data)
        )

    with analytics_2:

        st.metric(
            "Features",
            len(patient_data.columns)
        )

    with analytics_3:

        st.metric(
            "Mortality Cases",
            int(
                patient_data["mortality"].sum()
            )
        )

    st.info(
        "Current analytics are based on the structured "
        "demonstration EHR dataset. Future development "
        "will integrate larger clinical datasets and "
        "unstructured clinical notes."
    )


# ============================================================
# ABOUT MEDIVERSE
# ============================================================

elif navigation == "About MediVerse":

    st.title(
        "🧬 About MediVerse AI"
    )

    st.subheader(
        "Multimodal EHR Prediction System "
        "Using Deep Learning for Clinical Outcome Prediction"
    )

    st.write(
        "MediVerse AI is an academic research prototype "
        "focused on using Artificial Intelligence and "
        "Electronic Health Records for clinical outcome prediction."
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.header(
        "🎯 Project Objective"
    )

    st.write(
        "The system is designed to analyze structured "
        "patient information and, in future development, "
        "combine it with unstructured clinical text."
    )

    st.write(
        "The goal is to provide AI-assisted insights "
        "that can support healthcare professionals "
        "in understanding patient risk patterns."
    )

    st.divider()

    # --------------------------------------------------------
    # CURRENT SYSTEM
    # --------------------------------------------------------

    st.header(
        "🤖 Current Prototype"
    )

    current_1, current_2 = st.columns(2)

    with current_1:

        st.subheader(
            "Structured EHR Analysis"
        )

        st.write(
            "• Patient demographics"
        )

        st.write(
            "• Heart rate"
        )

        st.write(
            "• Blood pressure"
        )

        st.write(
            "• Oxygen level"
        )

        st.write(
            "• Glucose"
        )

        st.write(
            "• Previous admission"
        )

        st.write(
            "• ICU history"
        )

        st.write(
            "• Length of stay"
        )

    with current_2:

        st.subheader(
            "Machine Learning"
        )

        st.write(
            "• Clinical pattern classification"
        )

        st.write(
            "• Mortality risk prediction"
        )

        st.write(
            "• Probability estimation"
        )

        st.write(
            "• Patient-level analysis"
        )

        st.write(
            "• Interactive prediction dashboard"
        )

    st.divider()

    # --------------------------------------------------------
    # FUTURE MULTIMODAL SYSTEM
    # --------------------------------------------------------

    st.header(
        "🚀 Future Multimodal Architecture"
    )

    st.write(
        "The planned MediVerse AI architecture "
        "will combine multiple clinical data modalities:"
    )

    st.write(
        "1. Structured Electronic Health Records"
    )

    st.write(
        "2. Clinical Notes"
    )

    st.write(
        "3. Natural Language Processing"
    )

    st.write(
        "4. Clinical Language Models"
    )

    st.write(
        "5. Deep Learning"
    )

    st.write(
        "6. Multimodal Feature Fusion"
    )

    st.write(
        "7. Clinical Outcome Prediction"
    )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.header(
        "💻 Technology Stack"
    )

    technology_1, technology_2, technology_3 = st.columns(3)

    with technology_1:

        st.subheader(
            "🐍 Python"
        )

        st.write(
            "Core programming and ML workflow."
        )

        st.subheader(
            "🐼 Pandas / NumPy"
        )

        st.write(
            "Data processing and analysis."
        )

    with technology_2:

        st.subheader(
            "🤖 Scikit-learn"
        )

        st.write(
            "Machine-learning models and prediction."
        )

        st.subheader(
            "🧠 Deep Learning"
        )

        st.write(
            "Planned neural-network based modelling."
        )

    with technology_3:

        st.subheader(
            "📊 Streamlit"
        )

        st.write(
            "Interactive clinical dashboard."
        )

        st.subheader(
            "📝 Clinical NLP"
        )

        st.write(
            "Planned analysis of clinical notes."
        )

    st.divider()

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.warning(
        "⚕️ RESEARCH PROTOTYPE — NOT A MEDICAL DIAGNOSIS"
    )

    st.write(
        "MediVerse AI is an academic research prototype. "
        "Its predictions are experimental and should not "
        "replace professional medical judgment."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🏥 MediVerse AI | Multimodal Clinical Intelligence | "
    "Academic Research Prototype"
)