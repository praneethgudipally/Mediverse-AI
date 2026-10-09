\# 🏥 MediVerse AI



\### Multimodal EHR Prediction System for Clinical Outcome Prediction



<p align="center">

&#x20; <strong>AI-powered clinical decision support using Electronic Health Records</strong>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white" alt="Python"/>

&#x20; <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge\&logo=streamlit\&logoColor=white" alt="Streamlit"/>

&#x20; <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge\&logo=scikitlearn\&logoColor=white" alt="Scikit-learn"/>

&#x20; <img src="https://img.shields.io/badge/Healthcare-AI-008F95?style=for-the-badge" alt="Healthcare AI"/>

&#x20; <img src="https://img.shields.io/badge/Project-Research%20Prototype-5271FF?style=for-the-badge" alt="Research prototype"/>

</p>



\---



\## 📌 Overview



\*\*MediVerse AI\*\* is a healthcare-focused AI/ML research project designed to explore how patient Electronic Health Record (EHR) data and clinical symptoms can be used to support clinical outcome prediction.



The project combines a Python-based machine learning pipeline with an interactive Streamlit dashboard. The current prototype implements two classification models: a mortality outcome model based on structured clinical features and a symptom-pattern classification model based on selected patient symptoms.



The long-term goal is to develop a multimodal prediction system that combines structured EHR information with unstructured clinical notes to investigate clinical outcome prediction.



> \*\*Current status:\*\* Working prototype with two Logistic Regression models. Real MIMIC-IV integration, clinical-text processing, ClinicalBERT, deep neural networks, and multimodal feature fusion are planned development stages, not completed components.



\## 🎯 Problem Statement



Electronic Health Records contain valuable patient information, including demographics, vital signs, medical history, laboratory measurements, and clinical notes. Analyzing this information can be complex and time-consuming.



MediVerse AI explores the use of machine learning to process selected patient features, identify patterns in the available data, and present model-generated predictions through an accessible clinical dashboard.



\## 💡 Project Objectives



\* Develop an interactive healthcare-oriented dashboard for patient information.

\* Process structured EHR features for machine learning.

\* Build a prototype model for mortality outcome classification.

\* Build a symptom-based classification model.

\* Display model predictions and estimated class probabilities.

\* Investigate the integration of real, deidentified clinical data.

\* Extend the architecture toward multimodal learning using structured EHR data and clinical text.



\## 🏗️ System Architecture



\### 1. Current Working Architecture



```text

&#x20;            PATIENT DATA / USER INPUT

&#x20;                      |

&#x20;            +---------+----------+

&#x20;            |                    |

&#x20;            v                    v

&#x20;    STRUCTURED EHR         SYMPTOM INPUT

&#x20;    Clinical features      Selected symptoms

&#x20;            |                    |

&#x20;            v                    v

&#x20;    EHR preprocessing     Binary encoding

&#x20;            |                    |

&#x20;            v                    v

&#x20;    MORTALITY MODEL       SYMPTOM CLASSIFIER

&#x20;    Logistic Regression   Logistic Regression

&#x20;            |                    |

&#x20;            v                    v

&#x20;    Outcome prediction    Pattern classification

&#x20;            |                    |

&#x20;            +----------+---------+

&#x20;                       |

&#x20;                       v

&#x20;             STREAMLIT DASHBOARD

&#x20;                       |

&#x20;                       v

&#x20;           Visualize results and

&#x20;           support clinical research

```



\### 2. Proposed Multimodal Architecture



```text

&#x20;            MIMIC-IV EHR DATA

&#x20;                      |

&#x20;                      v

&#x20;             Data preprocessing

&#x20;                      |

&#x20;                      v

&#x20;            Structured EHR encoder

&#x20;                      |

&#x20;                      +------------------+

&#x20;                                         |

&#x20;                                         v

&#x20;                                  FEATURE FUSION

&#x20;                                         ^

&#x20;                                         |

&#x20;            MIMIC-IV-Note                 |

&#x20;            Clinical text                 |

&#x20;                      |                  |

&#x20;                      v                  |

&#x20;             Text preprocessing          |

&#x20;                      |                  |

&#x20;                      v                  |

&#x20;             ClinicalBERT encoder -------+

&#x20;                                         |

&#x20;                                         v

&#x20;                               PREDICTION MODEL

&#x20;                               DNN / suitable ML model

&#x20;                                         |

&#x20;                                         v

&#x20;                             CLINICAL OUTCOME PREDICTIONS

&#x20;                                         |

&#x20;                                         v

&#x20;                                MEDIVERSE AI UI

```



\*The second diagram represents the proposed research direction. The final architecture and algorithms will be determined during implementation and evaluation.\*



\## 🤖 Current Machine Learning Models



\### Model 1 — Mortality Outcome Prediction



\*\*Algorithm:\*\* Logistic Regression



The model uses structured clinical features to predict the mortality class represented in the training dataset.



\*\*Input features:\*\*



\* Age

\* Heart rate

\* Blood pressure

\* Oxygen level

\* Glucose level

\* Previous admission

\* ICU history

\* Length of hospital stay



\*\*Output:\*\* Predicted mortality class and, where used, model-estimated class probabilities.



\### Model 2 — Symptom-Pattern Classification



\*\*Algorithm:\*\* Logistic Regression



The model processes selected symptoms encoded as binary features.



| Symptom             | Feature representation  |

| ------------------- | ----------------------- |

| Fever               | 1 = present, 0 = absent |

| Cough               | 1 = present, 0 = absent |

| Shortness of breath | 1 = present, 0 = absent |

| Chest pain          | 1 = present, 0 = absent |

| Fatigue             | 1 = present, 0 = absent |

| Headache            | 1 = present, 0 = absent |

| Nausea              | 1 = present, 0 = absent |

| Vomiting            | 1 = present, 0 = absent |



The model predicts one of the predefined symptom-pattern categories, such as respiratory-like, cardiac-like, gastrointestinal-like, general infection-like, or general symptom pattern.



The displayed probability is a model estimate and must not be interpreted as medical certainty.



\## 💻 Technology Stack



| Technology          | Purpose                                    |

| ------------------- | ------------------------------------------ |

| Python              | Core application and model development     |

| Streamlit           | Interactive web dashboard                  |

| Pandas              | Dataset loading and manipulation           |

| NumPy               | Numerical operations, where needed         |

| Scikit-learn        | Machine learning algorithms and evaluation |

| Logistic Regression | Current classification models              |

| Joblib              | Saving and loading trained model artifacts |

| CSV                 | Current prototype dataset storage          |

| Git and GitHub      | Version control and project collaboration  |



\## 📂 Project Structure



```text

Mediverse-AI/

│

├── app/

│   └── dashboard.py

│

├── data/

│   ├── ehr\_data.csv

│   └── diagnosis\_data.csv

│

├── models/

│   ├── mortality\_model.pkl

│   └── diagnosis\_model.pkl

│

├── src/

│   ├── train\_model.py

│   └── train\_diagnosis\_model.py

│

├── .gitignore

└── README.md

```



\### Directory explanation



\* \*\*`app/`\*\* — Streamlit dashboard and user interface.

\* \*\*`data/`\*\* — Small synthetic datasets used by the current prototype.

\* \*\*`models/`\*\* — Serialized trained classification models.

\* \*\*`src/`\*\* — Model training scripts.

\* \*\*`.gitignore`\*\* — Excludes virtual environments, local secrets, caches, and designated raw-data directories.

\* \*\*`README.md`\*\* — Project documentation and setup instructions.



\## ⚙️ Installation and Setup



\### Prerequisites



\* Python installed on your system.

\* Git.

\* A terminal such as PowerShell, Command Prompt, or a Unix shell.



\### Step 1 — Clone the repository



```bash

git clone https://github.com/praneethgudipally/Mediverse-AI.git

cd Mediverse-AI

```



\### Step 2 — Create a virtual environment



\*\*Windows PowerShell:\*\*



```powershell

python -m venv venv

.\\venv\\Scripts\\Activate.ps1

```



If PowerShell blocks environment activation, use Command Prompt and run:



```cmd

venv\\Scripts\\activate.bat

```



\### Step 3 — Install dependencies



```bash

python -m pip install streamlit pandas numpy scikit-learn joblib

```



\### Step 4 — Launch the dashboard



```bash

streamlit run app/dashboard.py

```



Streamlit will provide a local URL, typically:



```text

http://localhost:8501

```



Open the URL in your browser to interact with the application.



\*\*Note:\*\* The serialized model files were trained in a particular Python/scikit-learn environment. If loading them fails because of version incompatibility, install a compatible scikit-learn version or retrain the models using the supplied training scripts.



\## 📊 Dataset and Model Evaluation



The current prototype uses small \*\*synthetic datasets\*\* for development, experimentation, and UI integration.



The initial development results were:



\* Mortality model: accuracy of 1.00 on a very small held-out test split.

\* Symptom-pattern model: accuracy of approximately 0.50 during initial evaluation.



These results are preliminary and are not evidence of clinical performance. The small synthetic datasets are not representative of real patient populations, and the reported accuracy values should not be compared directly with results from validated clinical models.



Future evaluation should use appropriate data splitting, class-sensitive metrics such as precision, recall and F1-score, and additional metrics appropriate to each prediction task.



\## 🧬 Planned Multimodal Development



The proposed research direction is to integrate:



\*\*Structured clinical data — MIMIC-IV\*\*



Potential inputs include demographics, vital signs, laboratory measurements, hospital admissions, and ICU information.



\*\*Clinical text — MIMIC-IV-Note\*\*



Clinical notes can provide contextual information that is not represented directly by structured fields.



\*\*Potential model components\*\*



\* Data cleaning and feature engineering.

\* Clinical-text preprocessing and ClinicalBERT-based representations.

\* Structured-feature representation using suitable machine learning or deep learning models.

\* Feature fusion to combine structured and textual information.

\* Evaluation of selected clinical outcome prediction tasks.



These components will be implemented and evaluated incrementally. Their inclusion here describes the research roadmap, not the current prototype.



\### Planned clinical outcomes



Depending on data availability, study design, and evaluation results, future work may investigate:



\* Mortality prediction.

\* Hospital readmission prediction.

\* ICU admission prediction.

\* Length-of-stay prediction.

\* Other defined clinical outcome prediction tasks.



\## 🛣️ Development Roadmap



\* \[x] Create the project structure.

\* \[x] Load and explore prototype EHR data.

\* \[x] Train the initial Logistic Regression mortality model.

\* \[x] Train the symptom-pattern classification model.

\* \[x] Save and load trained model artifacts.

\* \[x] Integrate model predictions into the Streamlit application.

\* \[ ] Refine the clinical dashboard and add project screenshots.

\* \[ ] Improve reproducibility and document dependency versions.

\* \[ ] Obtain required authorization for restricted clinical data access.

\* \[ ] Build and validate a MIMIC-IV preprocessing pipeline.

\* \[ ] Integrate an authorized clinical-notes dataset.

\* \[ ] Experiment with ClinicalBERT and structured-data representations.

\* \[ ] Implement and evaluate multimodal feature fusion.

\* \[ ] Conduct rigorous model evaluation and error analysis.



\## 🔒 Data Privacy and Responsible Use



MediVerse AI is an educational and research prototype.



\* The current prototype uses synthetic data.

\* Restricted clinical datasets must only be accessed and processed in accordance with their access requirements and data-use agreements.

\* Raw MIMIC-IV and other restricted clinical data must not be uploaded to this public repository.

\* Personal health information, credentials, access tokens, and secrets must not be committed to Git.

\* Predictions must not be treated as confirmed diagnoses or used as the sole basis for clinical decisions.

\* Any future real-world clinical use would require appropriate validation, privacy safeguards, and regulatory and institutional review as applicable.



\## 🔮 Future Scope



The long-term objective is to develop a more comprehensive clinical decision-support research platform through:



\* Integration of real, deidentified clinical data.

\* Multimodal learning using structured EHR features and clinical text.

\* Comparison of classical machine learning and deep learning approaches.

\* Improved interpretability and error analysis.

\* Clear presentation of model limitations and uncertainty.

\* A polished, user-friendly clinical dashboard.



\## 👨‍💻 Project Information



\*\*Project:\*\* MediVerse AI — Multimodal EHR Prediction System Using Deep Learning for Clinical Outcome Prediction



\*\*Project type:\*\* Final-year academic research prototype



\*\*Repository:\*\* \[github.com/praneethgudipally/Mediverse-AI](https://github.com/praneethgudipally/Mediverse-AI)



\## ⚠️ Disclaimer



MediVerse AI is intended for educational and research purposes only. It is not a clinically validated medical device, does not provide a confirmed diagnosis, and must not be used to make patient-care decisions.



\---



<p align="center">

&#x20; <strong>Building toward multimodal AI for clinical research.</strong>

</p>



