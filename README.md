# AI-Powered HVAC and Electrical Fault Diagnosis Assistant

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](HVAC_FAULT_DETECTION.ipynb)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-EC6B35)](https://xgboost.ai/)
[![Kaggle](https://img.shields.io/badge/Data-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/pjay01/solar-panel-fault-detection)

An educational machine-learning and retrieval-augmented generation (RAG)
project for detecting electrical operating conditions from photovoltaic and
three-phase sensor measurements. The notebook trains an XGBoost multiclass
classifier, evaluates its predictions, and connects the predicted label to a
Mistral-powered assistant that retrieves supporting material from a technical
reference PDF.

> The repository name retains the original HVAC project title, while the
> current dataset and label definitions focus specifically on photovoltaic and
> three-phase electrical faults.

## Project at a glance

| Item | Details |
|---|---|
| Dataset | `pjay01/solar-panel-fault-detection` on Kaggle |
| Data file | `modified_labels.csv` |
| Download size | 499,586,067 bytes (approximately 499.6 MB / 476.4 MiB) |
| Loaded dataset | 2,163,480 rows and 15 original columns |
| Prediction target | 8 classes: `F0` through `F7` |
| Training split | 1,730,784 records (80%) |
| Test split | 432,696 records (20%) |
| Classifier | XGBoost multiclass classifier |
| Recorded train accuracy | 80.42% |
| Recorded test accuracy | 80.30% |
| Recorded macro / weighted F1 | 0.79 / 0.80 |
| Explanation layer | LangChain, Chroma, MiniLM embeddings, and Mistral AI |

The dataset dimensions and metrics above come from the stored outputs of the
included notebook. Re-running the project may produce different results if the
dataset, package versions, or model parameters change.

## Fault classes handled

The following mapping is defined by the notebook:

| Label | Operating condition | What it represents |
|---|---|---|
| `F0` | Normal condition | The system is operating without a labelled fault. |
| `F1` | Line-to-ground fault (LG) | One phase conductor comes into contact with ground. |
| `F2` | Line-to-line fault (LL) | Two phase conductors form an unintended short circuit. |
| `F3` | Double-line-to-ground fault (LLG) | Two phases are involved in a fault path to ground. |
| `F4` | Three-phase fault (LLL) | All three phases are shorted together; this is a severe symmetrical fault. |
| `F5` | Open-circuit fault | A broken conductor, disconnected path, or loose connection interrupts current flow. |
| `F6` | Short-circuit fault | A very low-resistance path causes excessive current. |
| `F7` | Arc fault | Damaged or loose wiring creates electrical arcing and an unsafe, unstable current path. |

`F0` provides the normal baseline; `F1`-`F7` represent fault conditions that
the classifier attempts to distinguish from the measured electrical signals.

## Dataset and input features

KaggleHub automatically downloads the public
[`modified_labels.csv`](https://www.kaggle.com/datasets/pjay01/solar-panel-fault-detection)
file. No Google Drive mount is required.

The 15 original columns are grouped as follows:

- **Time:** `Time`.
- **PV and DC measurements:** `Ipv`, `Vpv`, and `Vdc`.
- **Three-phase currents:** `ia`, `ib`, and `ic`.
- **Three-phase voltages:** `va`, `vb`, and `vc`.
- **Aggregate or fault-related signals:** `Iabc`, `If`, `Vabc`, and `Vf`.
- **Target:** `Label`, containing `F0`-`F7`.

The notebook also creates domain-inspired features:

- `v_rms`: three-phase root-mean-square voltage.
- `Ppv`: photovoltaic-side power calculated as `Ipv × Vpv`.
- `Xa`, `Xb`, `Xc`, and `Xf`: voltage-to-current, impedance-like ratios.
- `X0`, `X1`, and `X2`: magnitudes of zero-, positive-, and negative-sequence
  symmetrical components.

After feature engineering, the working table contains 24 columns. One-hot
encoding the eight target labels produces a 31-column intermediate table; the
model itself receives the 23 non-label features.

## End-to-end workflow

```text
Public Kaggle CSV
        ↓
Exploratory data analysis
        ↓
Electrical feature engineering
        ↓
One-hot encoding and standard scaling
        ↓
Stratified 80/20 train-test split
        ↓
XGBoost training and evaluation
        ↓
Predicted fault label (F0-F7)
        ↓
Technical PDF retrieval with MiniLM + Chroma
        ↓
Mistral explanation and follow-up Q&A
```

## Machine-learning stage

The classifier uses an XGBoost `multi:softmax` objective with eight output
classes. The saved notebook run reports:

- training accuracy: **80.42%**;
- test accuracy: **80.30%**;
- macro-average precision, recall, and F1: **0.81, 0.79, and 0.79**;
- weighted-average precision, recall, and F1: **0.81, 0.80, and 0.80**.

The notebook also prints the complete confusion matrix and per-class
classification report, then saves the trained estimator as
`xgboost_fault_model.pkl`.

## RAG explanation stage

The second part of the notebook is a prototype explanation assistant:

1. Load `Three_Phase_Power_System_Fault_Reference.pdf`.
2. Split the document into 1,200-character chunks with 250-character overlap.
3. Embed the chunks with `sentence-transformers/all-MiniLM-L6-v2`.
4. Store and search them in a persistent Chroma vector database.
5. Retrieve the five most relevant chunks for a question.
6. Ask `mistral-large-latest` to explain the predicted fault using only the
   retrieved context.
7. Display the prediction and support follow-up questions with `ipywidgets`.

This separates the two responsibilities: **XGBoost predicts the class**, while
**RAG retrieves evidence and Mistral explains the result**.

## Outputs

- Dataset summaries, distributions, correlations, and electrical checks.
- Engineered and scaled model features.
- Training/test sizes and evaluation metrics.
- Confusion matrix and per-class classification report.
- Saved `xgboost_fault_model.pkl` model.
- Predicted fault label and sensor-value summary.
- Retrieved, context-grounded fault explanation.
- Interactive follow-up question-and-answer interface.

## Repository structure

```text
HVAC-Fault-Detection/
├── HVAC_FAULT_DETECTION.ipynb  # Structured analysis, ML, and RAG workflow
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
└── .gitignore                  # Secrets and generated artifacts excluded
```

## Run the project

### Google Colab

1. Open `HVAC_FAULT_DETECTION.ipynb` in Google Colab.
2. Run the cells from top to bottom.
3. KaggleHub downloads `modified_labels.csv` automatically.
4. Upload `Three_Phase_Power_System_Fault_Reference.pdf` before the RAG stage.
5. Enter your Mistral API key only when the notebook securely prompts for it.

### Local Jupyter environment

```bash
git clone https://github.com/arkman0011/HVAC-Fault-Detection.git
cd HVAC-Fault-Detection
python -m pip install -r requirements.txt
jupyter notebook HVAC_FAULT_DETECTION.ipynb
```

The CSV is approximately 500 MB, so allow enough download time, memory, and
disk space when running the notebook.

## Security and responsible use

- Never write a Kaggle or Mistral API key directly in the notebook.
- `.env`, model files, notebook checkpoints, and the local Chroma database are
  excluded through `.gitignore`.
- This is an educational prototype, not a certified protection, maintenance,
  or safety-control system.
- A predicted class should be confirmed using appropriate electrical testing
  and qualified engineering judgment.

## Technologies

Python, Jupyter, pandas, NumPy, scikit-learn, XGBoost, Seaborn, Matplotlib,
KaggleHub, LangChain, Chroma, Sentence Transformers, Mistral AI, and
ipywidgets.

## Data source

Pjay01, **Solar Panel Fault Detection**, Kaggle, version 1, licensed under the
MIT License: <https://www.kaggle.com/datasets/pjay01/solar-panel-fault-detection>
