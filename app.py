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


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Smart Irrigation - Simple Dashboard</title>
  <style>
    :root {
      --bg-page: #f2f6ee;
      --bg-card: #ffffff;
      --green-dark: #1a3a1a;
      --green-mid: #2e7d32;
      --green-light: #76c442;
      --text-main: #1a2e1a;
      --text-muted: #7a8a75;
      --danger: #e74c3c;
      --border: #e2ebe0;
      --light-panel: #e8f5df;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg-page);
      color: var(--text-main);
    }

    .container {
      max-width: 1100px;
      margin: 20px auto;
      padding: 0 12px 18px;
    }

    .hero {
      background: var(--green-dark);
      color: #f0f7ec;
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
      color: #c1d3bc;
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
      border-color: var(--green-light);
    }

    h2 {
      margin: 0 0 10px;
      color: var(--green-dark);
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
      background: var(--green-mid);
      color: #fff;
      font-size: 15px;
      cursor: pointer;
      transition: 0.2s ease;
    }

    button:hover { filter: brightness(1.04); }

    .result {
      margin-top: 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px;
      background: #f6fbf4;
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
      color: var(--green-dark);
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
      background: #fff;
    }

    .model-box b {
      display: block;
      margin-bottom: 4px;
      color: var(--green-dark);
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
      color: var(--green-dark);
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
      background: #e5efe1;
      overflow: hidden;
    }

    .bar-fill {
      height: 100%;
      border-radius: 999px;
      min-width: 2px;
    }

    .bar-fill.lr { background: #5fa8d3; }
    .bar-fill.rf { background: #76c442; }
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
      background: #c4cec2;
      border: 8px solid #e8ede8;
      transition: 0.2s ease;
    }

    .motor-dot.on {
      background: var(--green-mid);
      border-color: #d5ebd1;
    }

    .motor-title {
      text-align: center;
      font-size: 28px;
      color: var(--green-dark);
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
      border: 1px solid #cde6bc;
    }

    .action-main {
      font-size: 30px;
      font-weight: 700;
      color: var(--green-dark);
      margin: 8px 0;
    }

    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 8px;
    }

    .chips span {
      border: 1px solid #b8d8a7;
      background: #f5ffef;
      border-radius: 999px;
      padding: 5px 8px;
      font-size: 12px;
      color: #345d2f;
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


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
