# Smart Irrigation Prediction System

An end-to-end machine learning project to estimate water requirement for irrigation using three regression models:
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

The project includes:
- Model training notebooks
- Model comparison notebook
- Flask web app with smart motor ON/OFF rule
- Report-ready analytics graphs (comparison, residuals, category-wise errors)
- Correlation heatmap generation

## 1. Project Structure

- app.py: Flask app with UI + prediction API
- DATASET - Sheet1.csv: input dataset
- train_model.ipynb: Random Forest training workflow
- train_dt.ipynb: Linear Regression workflow and report graphs
- train_xgb.ipynb: XGBoost training workflow
- compare_models.ipynb: metric comparison of all models
- encoders.pkl: categorical encoders
- lr_model.pkl: trained Linear Regression model
- rf_model.pkl: trained Random Forest model
- xgb_model.pkl: trained XGBoost model
- heatmaps/: generated heatmap images
- requirements.txt: Python dependencies
- Procfile, runtime.txt: deployment configuration

## 2. Problem Statement

Given these inputs:
- CROP TYPE
- SOIL TYPE
- REGION
- TEMPERATURE
- WEATHER CONDITION

predict:
- WATER REQUIREMENT

Then apply motor control logic:
- Motor ON when prediction > 7.5
- Motor OFF when prediction <= 7.5

## 3. Tech Stack

- Python 3.11
- Flask
- Pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib
- Gunicorn (deployment)

## 4. Setup and Run

## 4.1 Local setup

1. Open terminal in smart_irrigation folder
2. Create and activate virtual environment
3. Install dependencies
4. Run app

Example commands:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
python app.py
```

App runs at:
- http://127.0.0.1:5000

## 4.2 Production style run

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

Procfile already contains:
- web: gunicorn app:app --bind 0.0.0.0:$PORT

## 5. Model Training Workflow

## 5.1 train_dt.ipynb (Linear Regression)

- Builds one-hot encoded pipeline
- Trains and saves lr_model.pkl
- Generates:
  - Top Feature Impact graph
  - Report graphs section (model comparison, diagnostics)

## 5.2 train_model.ipynb (Random Forest)

- Encodes categorical variables
- Trains Random Forest
- Saves rf_model.pkl and encoders.pkl
- Generates heatmap and feature importance

## 5.3 train_xgb.ipynb (XGBoost)

- Encodes categorical variables
- Trains XGBoost
- Saves xgb_model.pkl
- Generates heatmap and feature importance

## 5.4 compare_models.ipynb

- Loads trained models
- Evaluates R2, RMSE, MAE
- Compares model quality side by side

## 6. Flask Application Flow

app.py handles:
- Loading dataset, encoders, and 3 trained models
- Normalization of categorical text values
- Temperature bucket conversion (10-20, 20-30, 30-40, 40-50)
- Running all available models for each request
- Selecting best model based on computed metrics
- Returning full predictions + selected model + metrics
- UI rendering with model comparison graph and motor control card

Motor threshold constant:
- MOTOR_ON_THRESHOLD = 7.5

## 7. API Details

Endpoint:
- POST /predict

Required JSON fields:
- CROP TYPE
- SOIL TYPE
- REGION
- TEMPERATURE
- WEATHER CONDITION

Example request:

```json
{
  "CROP TYPE": "BANANA",
  "SOIL TYPE": "HUMID",
  "REGION": "HUMID",
  "TEMPERATURE": 25,
  "WEATHER CONDITION": "RAINY"
}
```

Example response includes:
- predictions for all models
- final_prediction
- selected_model
- model_metrics (R2, RMSE, MSE)
- model_load_errors (if any)

## 8. Graphs for Report

The project now supports report-friendly visualizations.

## 8.1 Correlation Heatmap

Generated in notebooks and saved to:
- heatmaps/DATASET - Sheet1_heatmap.png

Use this to explain feature relationships in EDA section.

## 8.2 Feature Impact Graphs

- Linear Regression coefficients (top important features)
- Random Forest feature importance
- XGBoost feature importance

Use this for interpretability section.

## 8.3 Model Comparison Graph

Shows side-by-side:
- R2 (higher better)
- RMSE (lower better)
- MAE (lower better)

Use this for model selection justification.

## 8.4 Diagnostics (Best Model)

- Actual vs Predicted scatter
- Residual plot
- Residual distribution histogram
- Crop-wise MAE bar chart

Use this for error analysis and reliability discussion.

## 9. UI Features

The web dashboard includes:
- Input form for all required fields
- Prediction cards for all 3 models
- Final selected prediction display
- Model comparison metric graph in result section
- Auto motor status (ON/OFF) using threshold rule
- Recent predictions history

## 10. Common Errors and Fixes

## 10.1 NameError: lr_model / rf_model / xgb_model not defined

Cause:
- Notebook cell executed out of order

Fix:
- Run training cells first, then plotting cells
- Plotting cells now include fallback loading from saved pkl files

## 10.2 Missing model artifact error in app

Cause:
- Model pkl files not present

Fix:
- Run training notebooks and ensure these files exist:
  - lr_model.pkl
  - rf_model.pkl
  - xgb_model.pkl
  - encoders.pkl

## 10.3 Unsupported value for field

Cause:
- Input category not in trained encoder classes

Fix:
- Use values from dropdown in UI
- Retrain models if new categories must be supported

## 10.4 Negative prediction values

Cause:
- Linear Regression is unconstrained and can output below zero

Fix options:
- Use selected best model output (already applied in app)
- Optional post-processing: clip prediction at zero for display

## 11. Reproducibility Tips

- Keep Python version aligned with runtime.txt
- Use requirements.txt versions
- Avoid mixing artifacts generated from very different sklearn/xgboost versions
- Retrain and regenerate pkl files after major dependency updates

## 12. Future Improvements

- Add model retraining script with CLI
- Save graph outputs automatically in reports/ folder
- Add confidence intervals or prediction uncertainty
- Add category coverage checks before inference
- Add CI pipeline for notebook-to-artifact validation

## 13. Credits

Built as a practical Smart Irrigation ML workflow integrating:
- data exploration
- multi-model training
- model selection
- explainability graphs
- deployable web prediction interface
