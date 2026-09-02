# AI-Powered HVAC Fault Diagnosis Assistant

This project combines machine learning and retrieval-augmented generation to
classify electrical fault conditions and explain the predicted result in clear,
technical language.

## Workflow

1. Load and inspect the labelled sensor dataset.
2. Perform exploratory data analysis and electrical feature engineering.
3. Encode and scale the features, then create stratified training and test sets.
4. Train and evaluate an XGBoost classifier for eight fault classes (`F0`-`F7`).
5. Save the trained model and connect its prediction to a RAG assistant.
6. Retrieve supporting information from a power-system reference PDF with
   MiniLM embeddings and Chroma.
7. Use Mistral AI to produce a grounded explanation and support follow-up
   questions through notebook widgets.

## Inputs

- `modified_labels.csv`: labelled electrical sensor measurements.
- `Three_Phase_Power_System_Fault_Reference.pdf`: reference material used by
  the RAG explanation stage.
- User-entered voltage, current, and symmetrical-component measurements.

The dataset and reference PDF are not included. Place them in the paths used by
the notebook or update those paths for your Google Drive environment.

## Outputs

- Exploratory plots and summary statistics.
- Train/test accuracy, confusion matrix, and classification report.
- `xgboost_fault_model.pkl`: saved fault-classification model.
- Predicted fault label with a retrieved, context-grounded explanation.
- Interactive follow-up question and answer interface.

## Run the notebook

The notebook was designed for Google Colab because it mounts Google Drive and
uses notebook installation commands.

1. Open `HVAC_FAULT_DETECTION.ipynb` in Google Colab.
2. Upload the CSV and reference PDF to Google Drive.
3. Update their paths if necessary.
4. Run the cells from top to bottom.
5. Enter the Mistral API key only when prompted. Never write it directly in the
   notebook or commit it to GitHub.

## Main technologies

Python, pandas, NumPy, scikit-learn, XGBoost, Seaborn, Matplotlib, LangChain,
Chroma, Sentence Transformers, Mistral AI, and ipywidgets.
