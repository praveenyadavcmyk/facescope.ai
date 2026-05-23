
import os
import re
import pickle
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Download NLTK data (silent if already present) ──────────
nltk.download("stopwords", quiet=True)

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Flask App ─────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")

# ── Load Model & Vectorizer ───────────────────────────────────
MODEL_PATH      = os.path.join("model", "best_model.pkl")
VECTORIZER_PATH = os.path.join("model", "tfidf_vectorizer.pkl")

def load_artifacts():
    """Load pickled model and TF-IDF vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        logger.warning(
            "Model files not found. Run `python train_model.py` first."
        )
        return None, None
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    logger.info("✅ Model and vectorizer loaded successfully.")
    return model, vectorizer


model, vectorizer = load_artifacts()

# ── Text Preprocessing (mirrors train_model.py) ──────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """Clean and normalize raw news text before vectorization."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)        # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML
    text = re.sub(r"[^a-z\s]", "", text)                 # keep letters only
    tokens = text.split()
    tokens = [
        stemmer.stem(w)
        for w in tokens
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict whether news text is FAKE or REAL.

    Request JSON:
        { "text": "<news article text>" }

    Response JSON (success):
        {
          "prediction": "FAKE" | "REAL",
          "confidence": 0.0–100.0,
          "label": 0 | 1,
          "fake_probability": 0.0–1.0,
          "real_probability": 0.0–1.0
        }

    Response JSON (error):
        { "error": "<message>" }
    """
    # ── Input validation ─────────────────────────────────────
    if not request.is_json:
        return jsonify({"error": "Request must be JSON."}), 400

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing field: 'text'"}), 400

    raw_text = data["text"].strip()
    if len(raw_text) < 10:
        return jsonify({"error": "Text too short. Please provide more content."}), 400
    if len(raw_text) > 100_000:
        return jsonify({"error": "Text too long (max 100,000 characters)."}), 400

    # ── Model readiness check ────────────────────────────────
    if model is None or vectorizer is None:
        return jsonify({
            "error": "Model not loaded. Run `python train_model.py` first."
        }), 503

    # ── Preprocess & predict ─────────────────────────────────
    try:
        clean = preprocess_text(raw_text)
        if not clean:
            return jsonify({"error": "Text could not be processed. Try different content."}), 400

        X = vectorizer.transform([clean])
        label      = int(model.predict(X)[0])         # 0=Fake, 1=Real
        proba      = model.predict_proba(X)[0]         # [P(fake), P(real)]

        fake_prob  = float(proba[0])
        real_prob  = float(proba[1])
        prediction = "REAL" if label == 1 else "FAKE"
        confidence = round(max(fake_prob, real_prob) * 100, 2)

        logger.info(
            "Prediction: %s  Confidence: %.1f%%  "
            "(fake=%.3f real=%.3f)  text_len=%d",
            prediction, confidence, fake_prob, real_prob, len(raw_text)
        )

        return jsonify({
            "prediction":       prediction,
            "confidence":       confidence,
            "label":            label,
            "fake_probability": round(fake_prob, 4),
            "real_probability": round(real_prob, 4),
        })

    except Exception as e:
        logger.exception("Prediction error: %s", e)
        return jsonify({"error": "Internal server error during prediction."}), 500


@app.route("/health")
def health():
    """Simple health-check endpoint for deployment/monitoring."""
    return jsonify({
        "status":       "ok",
        "model_loaded": model is not None,
    })


@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files (charts, CSS, JS)."""
    return send_from_directory("static", filename)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    logger.info("🚀 Starting Fake News Detector onhttp://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)



