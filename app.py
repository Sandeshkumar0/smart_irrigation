from flask import Flask, jsonify, render_template_string, request
from pathlib import Path
import pandas as pd
import pickle
import warnings
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.exceptions import InconsistentVersionWarning
import sklearn.compose._column_transformer as _ct

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "DATASET - Sheet1.csv"
ENCODER_PATH = BASE_DIR / "encoders.pkl"
MODEL_SOURCE = "train_model.ipynb"
TARGET_COL = "WATER REQUIREMENT"
MOTOR_ON_THRESHOLD = 7.5

MODEL_FILES = {
    "Linear Regression": BASE_DIR / "lr_model.pkl",
    "Random Forest": BASE_DIR / "rf_model.pkl",
    "XGBoost": BASE_DIR / "xgb_model.pkl",
}

if not Path(ENCODER_PATH).exists():
    raise FileNotFoundError(
        f"Missing encoder artifact: {ENCODER_PATH}. Run {MODEL_SOURCE} first."
    )

missing_models = [path for path in MODEL_FILES.values() if not Path(path).exists()]
if missing_models:
    raise FileNotFoundError(
        f"Missing model artifacts: {', '.join(missing_models)}. Train models first."
    )

# Compatibility shim for older sklearn pickles (e.g., 1.6.x) loaded on newer sklearn.
if not hasattr(_ct, "_RemainderColsList"):
    class _RemainderColsList(list):
        pass

    _ct._RemainderColsList = _RemainderColsList

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)

models = {}
model_load_errors = {}
for name, path in MODEL_FILES.items():
    try:
        with open(path, "rb") as f:
            models[name] = pickle.load(f)
    except Exception as exc:
        model_load_errors[name] = str(exc)

if not models:
    raise RuntimeError(
        "No models could be loaded. Please train models again in the current environment."
    )

df = pd.read_csv(DATA_PATH)

FIELDS = ["CROP TYPE", "SOIL TYPE", "REGION", "TEMPERATURE", "WEATHER CONDITION"]


def normalize_text(value: str) -> str:
    return str(value).strip().upper()


def temperature_to_bucket(value: float) -> str:
    if value < 10 or value > 50:
        raise ValueError("Temperature must be between 10 and 50")
    if value < 20:
        return "10-20"
    if value < 30:
        return "20-30"
    if value < 40:
        return "30-40"
    return "40-50"


def get_choices() -> dict:
    choices = {}
    for col in FIELDS:
        if col == "TEMPERATURE":
            continue
        if col in encoders:
            choices[col] = sorted(encoders[col].classes_.tolist())
        else:
            choices[col] = sorted(
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .unique()
                .tolist()
            )
    return choices


def safe_temperature_to_bucket(value) -> str | None:
  try:
    if pd.isna(value):
      return None

    if isinstance(value, str):
      label = normalize_text(value)
      if label in set(encoders["TEMPERATURE"].classes_):
        return label

    return temperature_to_bucket(float(value))
  except (TypeError, ValueError):
    return None


def build_model_inputs(source_df: pd.DataFrame):
  normalized_df = pd.DataFrame(index=source_df.index)
  encoded_df = pd.DataFrame(index=source_df.index)
  valid_mask = pd.Series(True, index=source_df.index)

  for col in FIELDS:
    if col == "TEMPERATURE":
      normalized_col = source_df[col].apply(safe_temperature_to_bucket)
    else:
      normalized_col = source_df[col].apply(
        lambda x: normalize_text(x) if pd.notna(x) else None
      )

    classes = set(encoders[col].classes_)
    valid_col = normalized_col.isin(classes)
    valid_mask &= valid_col

    normalized_df[col] = normalized_col

    encode_map = {label: idx for idx, label in enumerate(encoders[col].classes_)}
    encoded_df[col] = normalized_col.map(encode_map)

  y_true = pd.to_numeric(source_df[TARGET_COL], errors="coerce")
  valid_mask &= y_true.notna()

  return (
    normalized_df.loc[valid_mask, FIELDS],
    encoded_df.loc[valid_mask, FIELDS],
    y_true.loc[valid_mask],
  )


def evaluate_models(source_df: pd.DataFrame):
    X_raw_eval, X_encoded_eval, y_eval = build_model_inputs(source_df)
    metrics = {}
    errors = {}

    if y_eval.empty:
      return metrics, {"_global": "No valid rows available to evaluate models."}

    for name, model in models.items():
        try:
            if name == "Linear Regression":
                y_pred = model.predict(X_raw_eval)
            else:
                y_pred = model.predict(X_encoded_eval)

            mse = mean_squared_error(y_eval, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_eval, y_pred)
            metrics[name] = {
                "r2": float(r2),
                "rmse": float(rmse),
                "mse": float(mse),
            }
        except Exception as exc:
            errors[name] = str(exc)

    return metrics, errors


def choose_best_model(model_names):
    available = [name for name in model_names if name in model_metrics]
    if not available:
        return None

    # Ranking priority: higher R2, then lower RMSE, then lower MSE.
    available.sort(
        key=lambda n: (
            -model_metrics[n]["r2"],
            model_metrics[n]["rmse"],
            model_metrics[n]["mse"],
        )
    )
    return available[0]


model_metrics, model_metric_errors = evaluate_models(df)


def _build_analytics_cache():
    import random

    fi = {}
    for name, model in models.items():
        raw = model
        if hasattr(raw, "steps"):
            raw = raw.steps[-1][1]
        if hasattr(raw, "feature_importances_"):
            fi[name] = raw.feature_importances_.tolist()

    X_raw_full, X_enc_full, y_full = build_model_inputs(df)
    sample_idx = list(y_full.index)
    if len(sample_idx) > 200:
        sample_idx = random.sample(sample_idx, 200)

    X_raw_s = X_raw_full.loc[sample_idx]
    X_enc_s = X_enc_full.loc[sample_idx]
    y_s     = y_full.loc[sample_idx]

    batch_preds = {}
    for name, model in models.items():
        try:
            if name == "Linear Regression":
                batch_preds[name] = model.predict(X_raw_s).tolist()
            else:
                batch_preds[name] = model.predict(X_enc_s).tolist()
        except Exception:
            pass

    actuals = y_s.tolist()
    avp_rows = []
    residuals_lists = {name: [] for name in models}
    for i, actual in enumerate(actuals):
        row = {"actual": float(actual)}
        for name, preds in batch_preds.items():
            pred = float(preds[i])
            row[name] = pred
            residuals_lists[name].append(float(actual) - pred)
        avp_rows.append(row)

    return {
        "model_metrics": model_metrics,
        "feature_importances": fi,
        "actual_vs_predicted": avp_rows,
        "residuals": residuals_lists,
    }


analytics_cache = _build_analytics_cache()


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Smart Irrigation - Dashboard</title>
  <style>
    :root {
      --bg-page: #eef6fb;
      --bg-card: #ffffff;
      --accent-dark: #1a3a4a;
      --accent-mid: #2196a8;
      --accent-light: #56c0a8;
      --text-main: #1a2e38;
      --text-muted: #6a8a98;
      --danger: #e74c3c;
      --border: #cfe8ef;
      --light-panel: #dff3f0;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg-page);
      color: var(--text-main);
    }

    .top-nav {
      position: sticky; top: 0; z-index: 100;
      background: linear-gradient(90deg, #1a3a4a 0%, #1b4a3a 100%);
      display: flex; align-items: center; gap: 4px;
      padding: 0 20px;
      height: 54px;
      box-shadow: 0 2px 12px rgba(26,58,74,0.22);
    }
    .top-nav .nav-brand {
      font-size: 17px; font-weight: 700; color: #a8e6d4;
      letter-spacing: 0.5px; margin-right: auto;
      display: flex; align-items: center; gap: 8px;
    }
    .top-nav a {
      color: #a8e6d4; text-decoration: none;
      padding: 7px 16px; border-radius: 8px;
      font-size: 14px; font-weight: 600;
      transition: background 0.18s;
    }
    .top-nav a:hover { background: rgba(255,255,255,0.10); }
    .top-nav a.active { background: var(--accent-mid); color: #fff; }

    .container {
      max-width: 1100px;
      margin: 20px auto;
      padding: 0 12px 18px;
    }

    .hero {
      background: linear-gradient(135deg, #1a3a4a 0%, #1b5a48 100%);
      color: #e8f7f2;
      border-radius: 16px;
      padding: 18px 20px;
      margin-bottom: 14px;
    }

    .hero h1 {
      margin: 0;
      font-size: 34px;
    }

    .hero p {
      margin: 6px 0 0;
      color: #a8d4c8;
      font-size: 14px;
    }

    .grid {
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 12px;
    }

    .stack {
      display: grid;
      gap: 12px;
    }

    .card {
      background: var(--bg-card);
      border-radius: 16px;
      border: 1px solid var(--border);
      padding: 18px;
      transition: 0.2s ease;
    }

    .card:hover {
      transform: translateY(-3px);
      border-color: var(--accent-light);
      box-shadow: 0 4px 16px rgba(33,150,168,0.10);
    }

    h2 {
      margin: 0 0 10px;
      color: var(--accent-dark);
      font-size: 26px;
    }

    .muted { color: var(--text-muted); font-size: 13px; }

    .form-grid {
      margin-top: 12px;
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .field label {
      display: block;
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 5px;
      font-weight: 600;
    }

    select, input[type="number"] {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 9px 10px;
      font-size: 15px;
      background: #fff;
      color: var(--text-main);
    }

    button {
      margin-top: 10px;
      width: 100%;
      border: none;
      border-radius: 10px;
      padding: 10px 12px;
      background: linear-gradient(90deg, var(--accent-mid) 0%, var(--accent-light) 100%);
      color: #fff;
      font-size: 15px;
      cursor: pointer;
      transition: 0.2s ease;
      font-weight: 600;
    }

    button:hover { filter: brightness(1.07); transform: translateY(-1px); }

    .result {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: #f0fafc;
      display: none;
    }

    .result.show { display: block; }

    .result.error {
      border-color: #f3c7bf;
      background: #fff4f2;
      color: #8e2e2e;
    }

    .big-value {
      font-size: 42px;
      color: var(--accent-dark);
      font-weight: 700;
      margin: 4px 0;
    }

    .model-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
      margin-top: 10px;
    }

    .model-box {
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px;
      font-size: 13px;
      background: #f5fbfd;
    }

    .model-box b {
      display: block;
      margin-bottom: 4px;
      color: var(--accent-dark);
      font-size: 12px;
    }

    .comparison-wrap {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px;
      background: #ffffff;
    }

    .comparison-title {
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 8px;
      font-weight: 600;
    }

    .metric-block {
      margin-bottom: 10px;
    }

    .metric-label {
      font-size: 12px;
      color: var(--accent-dark);
      font-weight: 700;
      margin-bottom: 4px;
    }

    .bar-row {
      display: grid;
      grid-template-columns: 120px 1fr 64px;
      gap: 8px;
      align-items: center;
      margin-bottom: 4px;
      font-size: 12px;
    }

    .bar-name {
      color: var(--text-main);
    }

    .bar-track {
      width: 100%;
      height: 8px;
      border-radius: 999px;
      background: #d9eef5;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      min-width: 2px;
    }

    .bar-fill.lr { background: #5fa8d3; }
    .bar-fill.rf { background: #56c0a8; }
    .bar-fill.xgb { background: #f5a623; }

    .bar-value {
      text-align: right;
      color: var(--text-muted);
      font-variant-numeric: tabular-nums;
    }

    .motor-dot {
      width: 78px;
      height: 78px;
      border-radius: 50%;
      margin: 8px auto;
      background: #b8d4dc;
      border: 8px solid #ddf0f5;
      transition: 0.2s ease;
    }

    .motor-dot.on {
      background: var(--accent-mid);
      border-color: #c2e8f0;
    }

    .motor-title {
      text-align: center;
      font-size: 28px;
      color: var(--accent-dark);
      margin: 5px 0;
      font-weight: 700;
    }

    .motor-reason {
      text-align: center;
      font-size: 13px;
      color: var(--text-muted);
    }

    .action-card {
      background: var(--light-panel);
      border: 1px solid #a8dcd5;
    }

    .action-main {
      font-size: 30px;
      font-weight: 700;
      color: var(--accent-dark);
      margin: 8px 0;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .chips span {
      border: 1px solid #90cfe0;
      background: #eaf7fb;
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 12px;
      color: #1b5a6a;
    }

    .history-table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 6px;
      font-size: 13px;
    }

    .history-table td {
      border-bottom: 1px solid var(--border);
      padding: 8px 2px;
    }

    .empty {
      color: var(--text-muted);
      font-size: 13px;
      margin-top: 8px;
    }

    @media (max-width: 900px) {
      .grid { grid-template-columns: 1fr; }
    }

    @media (max-width: 680px) {
      .form-grid { grid-template-columns: 1fr; }
      .model-grid { grid-template-columns: 1fr; }
      h2 { font-size: 22px; }
      .big-value { font-size: 34px; }
    }
  </style>
</head>
<body>
  <nav class="top-nav">
    <div class="nav-brand">🌱 Smart Irrigation</div>
    <a href="/" class="active">🏠 Dashboard</a>
    <a href="/analytics">📊 Analytics</a>
  </nav>
  <div class="container">
    <section class="hero">
      <h1>Irrigation Intelligence</h1>
      <p>Simple dashboard using your trained models (Linear Regression, Random Forest, XGBoost).</p>
    </section>

    <main class="grid">
      <section class="card">
        <h2>Smart Prediction</h2>
        <p class="muted">Input values select karo, fir 3 models se prediction milega.</p>

        <div class="form-grid">
          <div class="field">
            <label for="CROP TYPE">CROP TYPE</label>
            <select id="CROP TYPE" name="CROP TYPE">
              {% for opt in choices['CROP TYPE'] %}
              <option value="{{ opt }}">{{ opt }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field">
            <label for="SOIL TYPE">SOIL TYPE</label>
            <select id="SOIL TYPE" name="SOIL TYPE">
              {% for opt in choices['SOIL TYPE'] %}
              <option value="{{ opt }}">{{ opt }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field">
            <label for="REGION">REGION</label>
            <select id="REGION" name="REGION">
              {% for opt in choices['REGION'] %}
              <option value="{{ opt }}">{{ opt }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field">
            <label for="WEATHER CONDITION">WEATHER CONDITION</label>
            <select id="WEATHER CONDITION" name="WEATHER CONDITION">
              {% for opt in choices['WEATHER CONDITION'] %}
              <option value="{{ opt }}">{{ opt }}</option>
              {% endfor %}
            </select>
          </div>

          <div class="field" style="grid-column: 1 / -1;">
            <label for="TEMPERATURE">TEMPERATURE (10 to 50)</label>
            <input id="TEMPERATURE" name="TEMPERATURE" type="number" min="10" max="50" step="0.1" value="30" />
          </div>
        </div>

        <button id="predictBtn" type="button">Predict From 3 Models</button>

        <div id="result" class="result">
          <div class="muted">Final Prediction (XGBoost)</div>
          <div id="finalValue" class="big-value">-</div>
          <div class="model-grid">
            <div class="model-box"><b>Linear Regression</b><span id="lrVal">-</span></div>
            <div class="model-box"><b>Random Forest</b><span id="rfVal">-</span></div>
            <div class="model-box"><b>XGBoost</b><span id="xgbVal">-</span></div>
          </div>
        </div>
      </section>

      <div class="stack">
        <section class="card">
          <h2>Auto Motor Control</h2>
          <div id="motorDot" class="motor-dot"></div>
          <div id="motorTitle" class="motor-title">Motor OFF</div>
          <div id="motorReason" class="motor-reason">Rule: Motor ON only when prediction is above {{ motor_threshold }}.</div>
        </section>

        <section class="card action-card">
          <h2>Today's Action Plan</h2>
          <div id="actionMain" class="action-main">Wait and Monitor</div>
        </section>

        <section class="card">
          <h2>Recent Predictions (Past Data)</h2>
          <table class="history-table" id="historyTable"></table>
          <div id="historyEmpty" class="empty">No past predictions yet.</div>
        </section>
      </div>
    </main>
  </div>

  <script>
    const resultBox = document.getElementById('result');
    const predictBtn = document.getElementById('predictBtn');

    const motorDot = document.getElementById('motorDot');
    const motorTitle = document.getElementById('motorTitle');
    const motorReason = document.getElementById('motorReason');

    const actionMain = document.getElementById('actionMain');

    const historyTable = document.getElementById('historyTable');
    const historyEmpty = document.getElementById('historyEmpty');
    const storageKey = 'irrigation_past_predictions_v1';
    const MOTOR_ON_THRESHOLD = {{ motor_threshold | tojson }};

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    function showLoading() {
      resultBox.className = 'result show';
      resultBox.innerHTML = '<div class="muted">Predicting...</div>';
    }

    function getModelClass(modelName) {
      if (modelName === 'Linear Regression') return 'lr';
      if (modelName === 'Random Forest') return 'rf';
      return 'xgb';
    }

    function buildMetricSection(modelMetrics, metricKey, label, higherIsBetter) {
      const modelNames = ['Linear Regression', 'Random Forest', 'XGBoost'];
      const rows = modelNames
        .map((name) => ({ name, value: Number(modelMetrics[name] && modelMetrics[name][metricKey]) }))
        .filter((item) => Number.isFinite(item.value));

      if (rows.length === 0) return '';

      const values = rows.map((item) => item.value);
      const maxValue = Math.max(...values);
      const minValue = Math.min(...values);

      const barRows = rows.map((item) => {
        let ratio = 0;
        if (higherIsBetter) {
          ratio = maxValue > 0 ? item.value / maxValue : 0;
        } else {
          ratio = item.value > 0 ? minValue / item.value : 1;
        }

        const width = Math.max(2, Math.min(100, ratio * 100));
        return (
          '<div class="bar-row">' +
            '<div class="bar-name">' + item.name + '</div>' +
            '<div class="bar-track"><div class="bar-fill ' + getModelClass(item.name) + '" style="width:' + width.toFixed(1) + '%"></div></div>' +
            '<div class="bar-value">' + item.value.toFixed(3) + '</div>' +
          '</div>'
        );
      }).join('');

      return (
        '<div class="metric-block">' +
          '<div class="metric-label">' + label + '</div>' +
          barRows +
        '</div>'
      );
    }

    function buildComparisonGraph(modelMetrics) {
      if (!modelMetrics || Object.keys(modelMetrics).length === 0) {
        return '<div class="muted" style="margin-top:8px;">Model comparison metrics not available.</div>';
      }

      const r2Section = buildMetricSection(modelMetrics, 'r2', 'R2 (higher is better)', true);
      const rmseSection = buildMetricSection(modelMetrics, 'rmse', 'RMSE (lower is better)', false);
      const mseSection = buildMetricSection(modelMetrics, 'mse', 'MSE (lower is better)', false);

      if (!r2Section && !rmseSection && !mseSection) {
        return '<div class="muted" style="margin-top:8px;">Model comparison metrics not available.</div>';
      }

      return (
        '<div class="comparison-wrap">' +
          '<div class="comparison-title">Model Comparison Graph (from training metrics)</div>' +
          r2Section + rmseSection + mseSection +
        '</div>'
      );
    }

    function showPrediction(data) {
      const lr = data.predictions['Linear Regression'];
      const rf = data.predictions['Random Forest'];
      const xgb = data.predictions['XGBoost'];
      const xgbError = data.model_load_errors ? data.model_load_errors['XGBoost'] : null;
      const modelMetrics = data.model_metrics || {};
      const selectedMetrics = modelMetrics[data.selected_model] || null;
      const comparisonGraph = buildComparisonGraph(modelMetrics);
      const xgbErrorNote = (xgb === undefined && xgbError)
        ? '<div class="muted" style="margin-top:8px;color:#8e2e2e;">XGBoost unavailable: ' + escapeHtml(xgbError) + '</div>'
        : '';
      const selectedMetricNote = selectedMetrics
        ? '<div class="muted" style="margin-top:8px;">Best by metrics - R2: ' + Number(selectedMetrics.r2).toFixed(4) + ', RMSE: ' + Number(selectedMetrics.rmse).toFixed(4) + ', MSE: ' + Number(selectedMetrics.mse).toFixed(4) + '</div>'
        : '';
      resultBox.className = 'result show';
      resultBox.innerHTML =
        '<div class="muted">Final Prediction (' + data.selected_model + ')</div>' +
        '<div class="big-value">' + Number(data.final_prediction).toFixed(3) + '</div>' +
        '<div class="model-grid">' +
          '<div class="model-box"><b>Linear Regression</b><span>' + (lr !== undefined ? Number(lr).toFixed(3) : 'N/A') + '</span></div>' +
          '<div class="model-box"><b>Random Forest</b><span>' + (rf !== undefined ? Number(rf).toFixed(3) : 'N/A') + '</span></div>' +
          '<div class="model-box"><b>XGBoost</b><span>' + (xgb !== undefined ? Number(xgb).toFixed(3) : 'N/A') + '</span></div>' +
        '</div>' +
        comparisonGraph +
        selectedMetricNote +
        xgbErrorNote;
    }

    function showError(message) {
      resultBox.className = 'result show error';
      resultBox.textContent = 'Error: ' + message;
    }

    function setMotorFromValue(value) {
      const on = value > MOTOR_ON_THRESHOLD;
      motorDot.classList.toggle('on', on);
      motorTitle.textContent = on ? 'Motor ON' : 'Motor OFF';
      motorReason.textContent = on
        ? 'Prediction ' + value.toFixed(3) + ' is above ' + MOTOR_ON_THRESHOLD.toFixed(1) + ', so motor is ON.'
        : 'Prediction ' + value.toFixed(3) + ' is ' + MOTOR_ON_THRESHOLD.toFixed(1) + ' or below, so motor is OFF.';
    }

    function updateActionPlan(value) {
      let text = 'Skip Motor For Now';

      if (value > MOTOR_ON_THRESHOLD) {
        text = 'Start Motor';
      }

      actionMain.textContent = text;
    }

    function readHistory() {
      try {
        return JSON.parse(localStorage.getItem(storageKey) || '[]');
      } catch (e) {
        return [];
      }
    }

    function saveHistory(rows) {
      localStorage.setItem(storageKey, JSON.stringify(rows));
    }

    function pushHistory(value) {
      const rows = readHistory();
      const stamp = new Date().toLocaleString();
      rows.unshift({ time: stamp, value: value });
      const sliced = rows.slice(0, 8);
      saveHistory(sliced);
      renderHistory();
    }

    function renderHistory() {
      const rows = readHistory();
      historyTable.innerHTML = '';
      if (rows.length === 0) {
        historyEmpty.style.display = 'block';
        return;
      }

      historyEmpty.style.display = 'none';
      rows.forEach((row) => {
        const tr = document.createElement('tr');
        tr.innerHTML = '<td>' + row.time + '</td><td><b>' + Number(row.value).toFixed(3) + '</b></td>';
        historyTable.appendChild(tr);
      });
    }

    predictBtn.addEventListener('click', async () => {
      const payload = {
        'CROP TYPE': document.getElementById('CROP TYPE').value,
        'SOIL TYPE': document.getElementById('SOIL TYPE').value,
        'REGION': document.getElementById('REGION').value,
        'TEMPERATURE': document.getElementById('TEMPERATURE').value,
        'WEATHER CONDITION': document.getElementById('WEATHER CONDITION').value
      };

      showLoading();
      predictBtn.disabled = true;
      predictBtn.textContent = 'Predicting...';

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (!res.ok) throw new Error(data.error || 'Prediction failed');

        showPrediction(data);

        setMotorFromValue(data.final_prediction);
        updateActionPlan(data.final_prediction);
        pushHistory(data.final_prediction);
      } catch (err) {
        showError(err.message);
      } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Predict From 3 Models';
      }
    });

    renderHistory();
  </script>
</body>
</html>
"""


ANALYTICS_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Smart Irrigation – Model Analytics</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #0d1e2a;
      --surface: #122333;
      --surface2: #1a3040;
      --card: #142234;
      --border: #1e3d56;
      --green: #56c0a8;
      --green-bright: #7dd8c0;
      --green-dim: #1a4a42;
      --blue: #4fafd8;
      --blue-bright: #7ec8e8;
      --orange: #f5a623;
      --red: #e05252;
      --text: #dff0f8;
      --muted: #6a9ab8;
      --accent-lr: #4fafd8;
      --accent-rf: #56c0a8;
      --accent-xgb: #f5a623;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }

    /* ── Navbar ── */
    .top-nav {
      position: sticky; top: 0; z-index: 100;
      background: linear-gradient(90deg, #091824 0%, #0d2018 100%);
      display: flex; align-items: center; gap: 4px;
      padding: 0 24px;
      height: 56px;
      border-bottom: 1px solid var(--border);
      box-shadow: 0 2px 16px rgba(0,0,0,0.5);
    }
    .nav-brand {
      font-size: 17px; font-weight: 800; color: var(--green-bright);
      margin-right: auto; letter-spacing: 0.5px;
    }
    .top-nav a {
      color: var(--muted); text-decoration: none;
      padding: 8px 18px; border-radius: 8px;
      font-size: 14px; font-weight: 600;
      transition: all 0.18s;
    }
    .top-nav a:hover { background: var(--surface2); color: var(--text); }
    .top-nav a.active { background: linear-gradient(90deg, #1a4a42, #174260); color: var(--green-bright); }

    /* ── Layout ── */
    .page { max-width: 1180px; margin: 0 auto; padding: 36px 20px 60px; }

    .page-header { margin-bottom: 40px; }
    .page-header h1 {
      font-size: 36px; font-weight: 800;
      background: linear-gradient(135deg, var(--green-bright) 0%, var(--blue-bright) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .page-header p { color: var(--muted); margin-top: 8px; font-size: 15px; line-height: 1.6; }

    /* ── Score Strip ── */
    .score-strip {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 16px;
      margin-bottom: 40px;
    }
    .score-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 22px 24px;
      position: relative;
      overflow: hidden;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    .score-card:hover { transform: translateY(-3px); box-shadow: 0 6px 28px rgba(79,175,216,0.12); }
    .score-card::before {
      content: '';
      position: absolute; top: 0; left: 0; right: 0; height: 3px;
    }
    .score-card.lr::before { background: var(--accent-lr); }
    .score-card.rf::before { background: var(--accent-rf); }
    .score-card.xgb::before { background: var(--accent-xgb); }
    .score-card .model-name { font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
    .score-card .r2-big { font-size: 42px; font-weight: 800; line-height: 1; }
    .score-card.lr .r2-big { color: var(--accent-lr); }
    .score-card.rf .r2-big { color: var(--accent-rf); }
    .score-card.xgb .r2-big { color: var(--accent-xgb); }
    .score-card .score-sub { font-size: 12px; color: var(--muted); margin-top: 6px; }
    .score-card .badge {
      position: absolute; top: 14px; right: 14px;
      background: rgba(86,192,168,0.18); color: var(--green-bright);
      font-size: 11px; font-weight: 700; padding: 3px 10px; border-radius: 99px;
      border: 1px solid rgba(86,192,168,0.3);
    }

    /* ── Section ── */
    .section { margin-bottom: 44px; }
    .section-header {
      display: flex; align-items: flex-end; gap: 14px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 14px;
    }
    .section-header .icon {
      width: 40px; height: 40px; border-radius: 12px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px; background: var(--surface2);
      flex-shrink: 0;
    }
    .section-header h2 { font-size: 20px; font-weight: 700; color: var(--text); }
    .section-header p { font-size: 13px; color: var(--muted); margin-top: 3px; line-height: 1.5; }

    /* ── Cards ── */
    .chart-card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 24px;
      transition: box-shadow 0.2s;
    }
    .chart-card:hover { box-shadow: 0 4px 32px rgba(79,175,216,0.10); }
    .chart-card .card-title {
      font-size: 14px; font-weight: 700; color: var(--muted);
      letter-spacing: 0.5px; text-transform: uppercase;
      margin-bottom: 18px;
    }

    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .three-col { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }

    /* ── Callout box ── */
    .callout {
      margin-top: 20px;
      background: rgba(79,175,216,0.07);
      border: 1px solid rgba(86,192,168,0.22);
      border-left: 4px solid var(--green);
      border-radius: 12px;
      padding: 16px 20px;
      font-size: 14px;
      color: #b8d8e8;
      line-height: 1.7;
    }
    .callout strong { color: var(--green-bright); }

    /* ── Legend pill ── */
    .legend-row {
      display: flex; gap: 18px; margin-bottom: 14px; flex-wrap: wrap;
    }
    .legend-pill {
      display: flex; align-items: center; gap: 7px;
      font-size: 12px; font-weight: 600; color: var(--muted);
    }
    .legend-pill .dot {
      width: 12px; height: 12px; border-radius: 3px;
    }

    /* ── Loading overlay ── */
    #loadingOverlay {
      position: fixed; inset: 0;
      background: var(--bg);
      display: flex; flex-direction: column; align-items: center; justify-content: center;
      gap: 18px; z-index: 9999;
      transition: opacity 0.4s;
    }
    #loadingOverlay.hidden { opacity: 0; pointer-events: none; }
    .spinner {
      width: 44px; height: 44px; border-radius: 50%;
      border: 4px solid var(--border);
      border-top-color: var(--blue-bright);
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    #loadingOverlay p { color: var(--muted); font-size: 15px; }

    @media (max-width: 820px) {
      .two-col, .three-col { grid-template-columns: 1fr; }
      .score-strip { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>

<div id="loadingOverlay">
  <div class="spinner"></div>
  <p>Loading model analytics…</p>
</div>

<nav class="top-nav">
  <div class="nav-brand">🌱 Smart Irrigation</div>
  <a href="/">🏠 Dashboard</a>
  <a href="/analytics" class="active">📊 Analytics</a>
</nav>

<div class="page">
  <div class="page-header">
    <h1>Model Performance Analytics</h1>
    <p>A visual breakdown of how <strong>Linear Regression</strong>, <strong>Random Forest</strong>, and <strong>XGBoost</strong>
       perform on the Smart Irrigation dataset — and why the ensemble models win.</p>
  </div>

  <!-- ═══ Score Strip ═══ -->
  <div class="score-strip">
    <div class="score-card lr" id="scoreCardLR">
      <div class="model-name">Linear Regression</div>
      <div class="r2-big" id="r2LR">--</div>
      <div class="score-sub">R² Score &nbsp;|&nbsp; RMSE: <span id="rmseLR">--</span></div>
    </div>
    <div class="score-card rf" id="scoreCardRF">
      <div class="model-name">Random Forest</div>
      <div class="r2-big" id="r2RF">--</div>
      <div class="score-sub">R² Score &nbsp;|&nbsp; RMSE: <span id="rmseRF">--</span></div>
    </div>
    <div class="score-card xgb" id="scoreCardXGB">
      <div class="model-name">XGBoost</div>
      <div class="r2-big" id="r2XGB">--</div>
      <div class="score-sub">R² Score &nbsp;|&nbsp; RMSE: <span id="rmseXGB">--</span></div>
    </div>
  </div>

  <!-- ═══ Section 1: Metric Comparison ═══ -->
  <div class="section">
    <div class="section-header">
      <div class="icon">📈</div>
      <div>
        <h2>Model Accuracy Comparison</h2>
        <p>Side-by-side comparison of R², RMSE and MSE across all three models.</p>
      </div>
    </div>
    <div class="two-col">
      <div class="chart-card">
        <div class="card-title">R² Score (higher = better)</div>
        <canvas id="chartR2" height="220"></canvas>
      </div>
      <div class="chart-card">
        <div class="card-title">RMSE &amp; MSE (lower = better)</div>
        <canvas id="chartRMSE" height="220"></canvas>
      </div>
    </div>
    <div class="callout">
      <strong>Why does this matter?</strong> R² tells us how well the model explains variance in water requirements.
      An R² of 1.0 is perfect — Linear Regression typically scores near 0 on non-linear data,
      while Random Forest and XGBoost capture complex <em>crop × soil × weather</em> interactions,
      yielding a much higher R² and lower prediction error.
    </div>
  </div>

  <!-- ═══ Section 2: Feature Importance ═══ -->
  <div class="section">
    <div class="section-header">
      <div class="icon">🧩</div>
      <div>
        <h2>Feature Importance</h2>
        <p>Which input features drive the prediction the most? (Available only for tree-based models.)</p>
      </div>
    </div>
    <div class="two-col">
      <div class="chart-card">
        <div class="card-title">Random Forest — Feature Importance</div>
        <canvas id="chartFIRF" height="220"></canvas>
      </div>
      <div class="chart-card">
        <div class="card-title">XGBoost — Feature Importance</div>
        <canvas id="chartFIXGB" height="220"></canvas>
      </div>
    </div>
    <div class="callout">
      <strong>Insight:</strong> Both Random Forest and XGBoost agree on which features matter most.
      Linear Regression cannot capture non-linear feature interactions and therefore lacks
      meaningful feature importance — another reason the tree models generalise better here.
    </div>
  </div>

  <!-- ═══ Section 3: Actual vs Predicted ═══ -->
  <div class="section">
    <div class="section-header">
      <div class="icon">🎯</div>
      <div>
        <h2>Actual vs. Predicted</h2>
        <p>Scatter plot of true target values against each model's prediction. Closer to the diagonal = better.</p>
      </div>
    </div>
    <div class="legend-row">
      <div class="legend-pill"><div class="dot" style="background:#5fa8d3"></div>Linear Regression</div>
      <div class="legend-pill"><div class="dot" style="background:#4caf50"></div>Random Forest</div>
      <div class="legend-pill"><div class="dot" style="background:#f5a623"></div>XGBoost</div>
    </div>
    <div class="chart-card">
      <div class="card-title">Actual vs Predicted — All Models (sampled)</div>
      <canvas id="chartScatter" height="300"></canvas>
    </div>
    <div class="callout">
      <strong>Reading the chart:</strong> The grey diagonal line represents a perfect predictor (predicted = actual).
      Points tightly clustered around the diagonal indicate low error. Notice how Random Forest and XGBoost
      hug the diagonal much more closely than Linear Regression.
    </div>
  </div>

  <!-- ═══ Section 4: Residuals ═══ -->
  <div class="section">
    <div class="section-header">
      <div class="icon">📉</div>
      <div>
        <h2>Residuals Distribution</h2>
        <p>How errors are distributed — a well-behaved model centres tightly around zero.</p>
      </div>
    </div>
    <div class="three-col">
      <div class="chart-card">
        <div class="card-title" style="color:#5fa8d3">Linear Regression</div>
        <canvas id="chartResLR" height="200"></canvas>
      </div>
      <div class="chart-card">
        <div class="card-title" style="color:#4caf50">Random Forest</div>
        <canvas id="chartResRF" height="200"></canvas>
      </div>
      <div class="chart-card">
        <div class="card-title" style="color:#f5a623">XGBoost</div>
        <canvas id="chartResXGB" height="200"></canvas>
      </div>
    </div>
    <div class="callout">
      <strong>What to look for:</strong> A narrow, symmetric histogram centred at 0 means the model makes small,
      unbiased errors. Linear Regression often shows a wide, flat spread, while ensemble models produce
      a tight bell-shaped distribution — proof of their superior precision.
    </div>
  </div>

</div>

<script>
(async function() {
  const overlay = document.getElementById('loadingOverlay');

  // ── Fetch Analytics Data ──────────────────────────────────────────────────
  // Data injected server-side — no fetch needed, instant load
  data = __ANALYTICS_DATA__;


  const { model_metrics, feature_importances, actual_vs_predicted, residuals } = data;
  const MODELS = ['Linear Regression', 'Random Forest', 'XGBoost'];
  const COLORS = { 'Linear Regression': '#5fa8d3', 'Random Forest': '#4caf50', 'XGBoost': '#f5a623' };
  const FIELDS = __FIELDS_JSON__;


  // ── Score Strip ───────────────────────────────────────────────────────────
  const scorePairs = [
    ['LR', 'Linear Regression'], ['RF', 'Random Forest'], ['XGB', 'XGBoost']
  ];
  let best = null;
  scorePairs.forEach(([id, name]) => {
    const m = model_metrics[name];
    if (!m) return;
    document.getElementById('r2' + id).textContent = m.r2.toFixed(4);
    document.getElementById('rmse' + id).textContent = m.rmse.toFixed(4);
    if (!best || m.r2 > model_metrics[best].r2) best = name;
  });
  if (best) {
    const badgeId = best === 'Linear Regression' ? 'scoreCardLR' : best === 'Random Forest' ? 'scoreCardRF' : 'scoreCardXGB';
    const badge = document.createElement('div');
    badge.className = 'badge'; badge.textContent = '⭐ Best Model';
    document.getElementById(badgeId).appendChild(badge);
  }

  // ── Shared Chart.js Defaults ──────────────────────────────────────────────
  Chart.defaults.color = '#7a9a72';
  Chart.defaults.font.family = 'Inter, sans-serif';
  const gridColor = 'rgba(255,255,255,0.05)';
  const tickColor = '#5a7a52';

  function darkAxes(extra) {
    return {
      x: { grid: { color: gridColor }, ticks: { color: tickColor }, ...((extra && extra.x) || {}) },
      y: { grid: { color: gridColor }, ticks: { color: tickColor }, ...((extra && extra.y) || {}) }
    };
  }

  // ── Chart 1: R² Bars ──────────────────────────────────────────────────────
  new Chart(document.getElementById('chartR2'), {
    type: 'bar',
    data: {
      labels: MODELS,
      datasets: [{
        label: 'R² Score',
        data: MODELS.map(m => model_metrics[m] ? model_metrics[m].r2 : 0),
        backgroundColor: MODELS.map(m => COLORS[m]),
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }, tooltip: {
        callbacks: { label: ctx => ' R²: ' + ctx.raw.toFixed(5) }
      }},
      scales: darkAxes({ y: { min: 0, max: 1 } }),
    }
  });

  // ── Chart 2: RMSE + MSE Grouped Bars ─────────────────────────────────────
  new Chart(document.getElementById('chartRMSE'), {
    type: 'bar',
    data: {
      labels: MODELS,
      datasets: [
        {
          label: 'RMSE',
          data: MODELS.map(m => model_metrics[m] ? model_metrics[m].rmse : 0),
          backgroundColor: MODELS.map(m => COLORS[m] + 'cc'),
          borderRadius: 6, borderSkipped: false,
        },
        {
          label: 'MSE',
          data: MODELS.map(m => model_metrics[m] ? model_metrics[m].mse : 0),
          backgroundColor: MODELS.map(m => COLORS[m] + '55'),
          borderRadius: 6, borderSkipped: false,
        },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#7a9a72', boxWidth: 12, borderRadius: 4 } } },
      scales: darkAxes(),
    }
  });

  // ── Chart 3 & 4: Feature Importance ──────────────────────────────────────
  function fiChart(canvasId, fiData, color) {
    if (!fiData) { return; }
    const sorted = FIELDS.map((f, i) => ({ name: f, val: fiData[i] || 0 })).sort((a,b) => b.val - a.val);
    new Chart(document.getElementById(canvasId), {
      type: 'bar',
      data: {
        labels: sorted.map(x => x.name.replace('WEATHER CONDITION', 'WEATHER')),
        datasets: [{
          label: 'Importance',
          data: sorted.map(x => x.val),
          backgroundColor: color + 'cc',
          borderColor: color,
          borderWidth: 1,
          borderRadius: 6,
          borderSkipped: false,
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        plugins: { legend: { display: false }, tooltip: {
          callbacks: { label: ctx => ' Importance: ' + ctx.raw.toFixed(4) }
        }},
        scales: darkAxes(),
      }
    });
  }
  fiChart('chartFIRF', feature_importances['Random Forest'], '#4caf50');
  fiChart('chartFIXGB', feature_importances['XGBoost'], '#f5a623');

  // ── Chart 5: Actual vs Predicted Scatter ──────────────────────────────────
  const avp = actual_vs_predicted;
  const allActuals = avp.map(r => r.actual);
  const minA = Math.min(...allActuals), maxA = Math.max(...allActuals);
  new Chart(document.getElementById('chartScatter'), {
    type: 'scatter',
    data: {
      datasets: [
        ...MODELS.map(name => ({
          label: name,
          data: avp.map(r => ({ x: r.actual, y: r[name] !== undefined ? r[name] : null })).filter(p => p.y !== null),
          backgroundColor: COLORS[name] + '88',
          pointRadius: 4,
          pointHoverRadius: 6,
        })),
        {
          label: 'Perfect Fit',
          data: [{ x: minA, y: minA }, { x: maxA, y: maxA }],
          type: 'line',
          borderColor: 'rgba(255,255,255,0.2)',
          borderDash: [6, 4],
          borderWidth: 2,
          pointRadius: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#7a9a72', boxWidth: 12 } } },
      scales: {
        x: { title: { display: true, text: 'Actual', color: '#7a9a72' }, grid: { color: gridColor }, ticks: { color: tickColor } },
        y: { title: { display: true, text: 'Predicted', color: '#7a9a72' }, grid: { color: gridColor }, ticks: { color: tickColor } },
      }
    }
  });

  // ── Charts 6a-c: Residuals Histograms ────────────────────────────────────
  function histChart(canvasId, resArray, color) {
    const BINS = 12;
    const mn = Math.min(...resArray), mx = Math.max(...resArray);
    const step = (mx - mn) / BINS || 1;
    const counts = new Array(BINS).fill(0);
    const labels = [];
    for (let i = 0; i < BINS; i++) {
      labels.push((mn + i * step).toFixed(2));
    }
    resArray.forEach(v => {
      let idx = Math.floor((v - mn) / step);
      if (idx === BINS) idx = BINS - 1;
      counts[idx]++;
    });
    new Chart(document.getElementById(canvasId), {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Count',
          data: counts,
          backgroundColor: color + '99',
          borderColor: color,
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'Residual (actual − predicted)', color: '#7a9a72' }, grid: { color: gridColor }, ticks: { color: tickColor } },
          y: { title: { display: true, text: 'Count', color: '#7a9a72' }, grid: { color: gridColor }, ticks: { color: tickColor } },
        }
      }
    });
  }
  histChart('chartResLR', residuals['Linear Regression'] || [], '#5fa8d3');
  histChart('chartResRF', residuals['Random Forest'] || [], '#4caf50');
  histChart('chartResXGB', residuals['XGBoost'] || [], '#f5a623');

  // ── Done ──────────────────────────────────────────────────────────────────
  overlay.classList.add('hidden');
  setTimeout(() => overlay.style.display = 'none', 500);
})();
</script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
  return render_template_string(
    HTML_PAGE,
    choices=get_choices(),
    model_source=MODEL_SOURCE,
    motor_threshold=MOTOR_ON_THRESHOLD,
  )


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    normalized_row = {}
    encoded_row = {}
    for col in FIELDS:
        if col not in payload:
            return jsonify({"error": f"Missing field: {col}"}), 400

        if col == "TEMPERATURE":
            try:
                value = temperature_to_bucket(float(payload[col]))
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
        else:
            value = normalize_text(payload[col])

        if value not in set(encoders[col].classes_):
            return jsonify({"error": f"Unsupported value for {col}: {value}"}), 400

        normalized_row[col] = value
        encoded_row[col] = encoders[col].transform([value])[0]

    X_raw = pd.DataFrame([normalized_row], columns=FIELDS)
    X_encoded = pd.DataFrame([encoded_row], columns=FIELDS)

    # Model-specific input:
    # - Linear Regression model is a Pipeline expecting raw categorical values.
    # - Random Forest and XGBoost models expect encoded numeric values.
    all_predictions = {}
    if "Linear Regression" in models:
        all_predictions["Linear Regression"] = float(models["Linear Regression"].predict(X_raw)[0])
    if "Random Forest" in models:
        all_predictions["Random Forest"] = float(models["Random Forest"].predict(X_encoded)[0])
    if "XGBoost" in models:
        all_predictions["XGBoost"] = float(models["XGBoost"].predict(X_encoded)[0])

    if not all_predictions:
        return jsonify({"error": "No prediction model available"}), 500

    selected_model = choose_best_model(all_predictions.keys())
    if selected_model is None:
      # Fallback if metrics could not be computed for some reason.
      selected_model = next(iter(all_predictions.keys()))

    final_prediction = all_predictions[selected_model]

    return jsonify(
        {
            "predictions": all_predictions,
            "final_prediction": final_prediction,
            "selected_model": selected_model,
        "model_metrics": model_metrics,
        "model_metric_errors": model_metric_errors,
            "model_load_errors": model_load_errors,
        }
    )


@app.route("/analytics", methods=["GET"])
def analytics():
    import json
    # Embed data directly into the page as inline JS — avoids Jinja2 mangling JS/CSS curly braces
    data_json = json.dumps(analytics_cache)
    fields_json = json.dumps(FIELDS)
    page = ANALYTICS_PAGE.replace("__ANALYTICS_DATA__", data_json).replace("__FIELDS_JSON__", fields_json)
    return page


@app.route("/analytics-data", methods=["GET"])
def analytics_data():
    return jsonify(analytics_cache)



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
