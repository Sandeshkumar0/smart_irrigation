from flask import Flask, jsonify, render_template_string, request
from pathlib import Path
import pandas as pd
import pickle

app = Flask(__name__)

DATA_PATH = "DATASET - Sheet1.csv"
MODEL_PATH = "rf_model.pkl"
ENCODER_PATH = "encoders.pkl"
MODEL_SOURCE = "train_model.ipynb"
TARGET_COL = "WATER REQUIREMENT"

# Load artifacts once at startup
if not Path(MODEL_PATH).exists() or not Path(ENCODER_PATH).exists():
    raise FileNotFoundError(
        f"Missing model artifacts. Run {MODEL_SOURCE} first to generate "
        f"{MODEL_PATH} and {ENCODER_PATH}."
    )

model = pickle.load(open(MODEL_PATH, "rb"))
encoders = pickle.load(open(ENCODER_PATH, "rb"))
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
            # fallback in case encoder missing (should not happen here)
            choices[col] = sorted(df[col].dropna().astype(str).str.strip().str.upper().unique().tolist())
    return choices


HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Smart Irrigation Predictor</title>
  <style>
    :root {
      --bg: #f7f2e9;
      --card: #fffef9;
      --ink: #18332a;
      --muted: #5f786c;
      --primary: #2c8a57;
      --primary-2: #1f6a44;
      --ring: #bde4cb;
      --line: #dbe2d3;
      --radius: 20px;
      --shadow: 0 18px 40px rgba(22, 46, 37, 0.16);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", Tahoma, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(900px 420px at -8% -12%, #d8ecd5 0%, transparent 62%),
        radial-gradient(1000px 460px at 108% 10%, #f5e5c9 0%, transparent 60%),
        linear-gradient(160deg, rgba(255,255,255,0.4) 0%, rgba(255,255,255,0) 42%),
        var(--bg);
      min-height: 100vh;
      display: grid;
      place-items: center;
      padding: 24px;
    }

    .card {
      width: min(940px, 100%);
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
      backdrop-filter: blur(2px);
    }

    .head {
      padding: 24px 24px 22px;
      background:
        radial-gradient(420px 120px at 12% -15%, rgba(255,255,255,0.24), transparent 70%),
        linear-gradient(125deg, #1f6f3f, #2f8f4e);
      color: #f4fff4;
    }

    .head h1 {
      margin: 0;
      font-size: clamp(1.25rem, 2.8vw, 1.95rem);
      letter-spacing: 0.3px;
    }

    .head p {
      margin: 7px 0 0;
      opacity: 0.92;
      font-size: 0.95rem;
    }

    .model-pill {
      display: inline-flex;
      align-items: center;
      margin-top: 10px;
      font-size: 0.8rem;
      border: 1px solid rgba(255, 255, 255, 0.35);
      border-radius: 999px;
      padding: 5px 10px;
      background: rgba(255, 255, 255, 0.13);
      letter-spacing: 0.2px;
    }

    form {
      padding: 22px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px 16px;
    }

    .field {
      display: grid;
      gap: 7px;
    }

    .field label {
      font-size: 0.78rem;
      color: var(--muted);
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
    }

    select {
      appearance: none;
      border: 1px solid #c9d6c8;
      background:
        linear-gradient(180deg, #ffffff 0%, #f9fdf7 100%);
      border-radius: 12px;
      padding: 11px 12px;
      font-size: 0.96rem;
      color: var(--ink);
      transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.08s ease;
    }

    input[type="number"] {
      border: 1px solid #c9d6c8;
      background:
        linear-gradient(180deg, #ffffff 0%, #f9fdf7 100%);
      border-radius: 12px;
      padding: 11px 12px;
      font-size: 0.96rem;
      color: var(--ink);
      transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.08s ease;
    }

    select:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 4px var(--ring);
      transform: translateY(-1px);
    }

    input[type="number"]:focus {
      outline: none;
      border-color: var(--primary);
      box-shadow: 0 0 0 4px var(--ring);
      transform: translateY(-1px);
    }

    .actions {
      grid-column: 1 / -1;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 4px;
    }

    button {
      border: none;
      border-radius: 12px;
      background: linear-gradient(135deg, var(--primary), #2e9c61);
      color: white;
      padding: 12px 18px;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      letter-spacing: 0.2px;
      box-shadow: 0 8px 18px rgba(44, 138, 87, 0.28);
      transition: transform 0.08s ease, box-shadow 0.2s ease, filter 0.2s ease;
    }

    button:hover {
      filter: saturate(1.05);
      box-shadow: 0 10px 20px rgba(31, 106, 68, 0.34);
    }
    button:active { transform: translateY(1px); }

    .result {
      margin: 2px 22px 22px;
      border-radius: 14px;
      border: 1px solid #cfe2cd;
      padding: 14px 16px;
      background: linear-gradient(180deg, #f8fff5, #f3fdf0);
      display: none;
    }

    .result.show { display: block; }

    .value {
      font-size: clamp(1.4rem, 3.4vw, 2rem);
      color: #1f6f3f;
      font-weight: 700;
      margin-top: 6px;
      letter-spacing: 0.2px;
    }

    .error {
      color: #a42020;
      background: #fff3f1;
      border-color: #f4c7c1;
    }

    @media (max-width: 760px) {
      form { grid-template-columns: 1fr; }
      .card { border-radius: 16px; }
      .head { padding: 20px; }
      form { padding: 18px; }
      .result { margin: 2px 18px 18px; }
    }
  </style>
</head>
<body>
  <main class="card">
    <div class="head">
      <h1>Smart Irrigation Predictor</h1>
    </div>

    <form id="predictForm">
      {% for field, options in choices.items() %}
      <div class="field">
        <label for="{{ field }}">{{ field }}</label>
        <select id="{{ field }}" name="{{ field }}" required>
          {% for opt in options %}
          <option value="{{ opt }}">{{ opt }}</option>
          {% endfor %}
        </select>
      </div>
      {% endfor %}
      <div class="field">
        <label for="TEMPERATURE">TEMPERATURE (Exact)</label>
        <input
          id="TEMPERATURE"
          name="TEMPERATURE"
          type="number"
          min="10"
          max="50"
          step="0.1"
          placeholder="e.g. 27.4"
          required
        />
      </div>

      <div class="actions">
        <button type="submit">Predict Water Requirement</button>
      </div>
    </form>

    <section id="result" class="result" aria-live="polite"></section>
  </main>

  <script>
    const form = document.getElementById('predictForm');
    const resultBox = document.getElementById('result');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();

      const data = {};
      new FormData(form).forEach((value, key) => {
        data[key] = value;
      });

      resultBox.className = 'result';
      resultBox.innerHTML = 'Calculating...';
      resultBox.classList.add('show');

      try {
        const res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        const payload = await res.json();

        if (!res.ok) {
          throw new Error(payload.error || 'Prediction failed');
        }

        resultBox.innerHTML = `
          <div>Predicted Water Requirement</div>
          <div class="value">${payload.prediction.toFixed(3)} units</div>
        `;
      } catch (err) {
        resultBox.classList.add('error');
        resultBox.innerHTML = `<strong>Error:</strong> ${err.message}`;
      }
    });
  </script>
</body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_PAGE, choices=get_choices(), model_source=MODEL_SOURCE)


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    row = {}
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

        row[col] = encoders[col].transform([value])[0]

    X = pd.DataFrame([row], columns=FIELDS)
    prediction = float(model.predict(X)[0])
    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
