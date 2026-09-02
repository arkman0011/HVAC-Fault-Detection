# AI-Powered HVAC Fault Diagnosis Assistant

This project combines machine learning and retrieval-augmented generation to
classify electrical fault conditions and explain the predicted result in clear,
technical language.

## Workflow

1. Download the public Kaggle dataset and inspect the labelled sensor data.
2. Perform exploratory data analysis and electrical feature engineering.
3. Encode and scale the features, then create stratified training and test sets.
4. Train and evaluate an XGBoost classifier for eight fault classes (`F0`-`F7`).
5. Save the trained model and connect its prediction to a RAG assistant.
6. Retrieve supporting information from a power-system reference PDF with
   MiniLM embeddings and Chroma.
7. Use Mistral AI to produce a grounded explanation and support follow-up
   questions through notebook widgets.

## Inputs

- [`pjay01/solar-panel-fault-detection`](https://www.kaggle.com/datasets/pjay01/solar-panel-fault-detection):
  public Kaggle dataset containing `modified_labels.csv` with labelled
  electrical sensor measurements.
- `Three_Phase_Power_System_Fault_Reference.pdf`: reference material used by
  the RAG explanation stage.
- User-entered voltage, current, and symmetrical-component measurements.

KaggleHub downloads the CSV automatically. The reference PDF is not included;
place it in the notebook working directory or update its path.

## Outputs

- Exploratory plots and summary statistics.
- Train/test accuracy, confusion matrix, and classification report.
- `xgboost_fault_model.pkl`: saved fault-classification model.
- Predicted fault label with a retrieved, context-grounded explanation.
- Interactive follow-up question and answer interface.

## Run the notebook

The notebook can run in Google Colab or another Jupyter environment. It uses
KaggleHub instead of mounting Google Drive.

1. Open `HVAC_FAULT_DETECTION.ipynb` in Google Colab.
2. Allow KaggleHub to download `modified_labels.csv` from the public dataset.
3. Upload the reference PDF and update its path if necessary.
4. Run the cells from top to bottom.
5. Enter the Mistral API key only when prompted. Never write it directly in the
   notebook or commit it to GitHub.

## Main technologies

Python, pandas, NumPy, scikit-learn, XGBoost, Seaborn, Matplotlib, LangChain,
Chroma, Sentence Transformers, Mistral AI, and ipywidgets.
