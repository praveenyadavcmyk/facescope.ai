
import os
import re
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


# ─────────────────────────────────────────────────────────────
#  STEP 1: Load Dataset
# ─────────────────────────────────────────────────────────────

def load_dataset():
    """
    Loads Fake.csv and True.csv from the dataset/ folder.
    Expected format from Kaggle 'Fake and Real News Dataset':
      - Fake.csv  → columns: title, text, subject, date
      - True.csv  → columns: title, text, subject, date
    """
    fake_path = "dataset/Fake.csv"
    true_path = "dataset/True.csv"

    # If dataset files exist, load them
    if os.path.exists(fake_path) and os.path.exists(true_path):
        print("✅ Loading dataset from CSV files...")
        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        # Label the data: 0 = Fake, 1 = Real
        fake_df["label"] = 0
        true_df["label"] = 1

        # Combine title + text for richer features
        fake_df["content"] = fake_df["title"].fillna("") + " " + fake_df["text"].fillna("")
        true_df["content"] = true_df["title"].fillna("") + " " + true_df["text"].fillna("")

        df = pd.concat([fake_df[["content", "label"]], true_df[["content", "label"]]], ignore_index=True)

    else:
        # ── Fallback: Generate synthetic dataset for demonstration ──
        print("⚠️  Dataset CSVs not found. Generating synthetic data for demo...")
        print("   (Download Fake.csv & True.csv from Kaggle and place in dataset/)")
        df = generate_synthetic_dataset()

    print(f"📊 Dataset shape: {df.shape}")
    print(f"   Fake news samples : {(df['label'] == 0).sum()}")
    print(f"   Real news samples : {(df['label'] == 1).sum()}")
    return df


def generate_synthetic_dataset():
    """Creates a small synthetic dataset so the project runs without Kaggle files."""
    import random
    random.seed(42)

    fake_templates = [
        "SHOCKING: {} claims {} is secretly {}. Sources say the truth will {}.",
        "BREAKING: {} exposed as {} in massive {} scandal that mainstream media {}.",
        "You won't believe what {} did to {} — the {} they don't want you to know.",
        "EXCLUSIVE: {} confirms {} plot to {} — share before this gets {}.",
        "Scientists HATE him: {} discovers {} trick that {} the entire {}.",
    ]
    real_templates = [
        "{} announced new {} policy aimed at improving {} across the country.",
        "The {} released its annual report showing {} growth in {} sector.",
        "Researchers at {} published findings on {} in the journal {}.",
        "{} held a press conference addressing {} concerns over {}.",
        "According to official data, {} has increased {} by {} percent this year.",
    ]

    fake_words = ["government","elite","deep state","globalists","scientists","pharma",
                  "hidden","suppressed","exposed","destroy","silence","control"]
    real_words = ["Congress","Federal Reserve","study","university","economy",
                  "policy","report","committee","officials","statement","data"]

    rows = []
    for _ in range(2000):
        tmpl = random.choice(fake_templates)
        words = random.choices(fake_words, k=4)
        rows.append({"content": tmpl.format(*words), "label": 0})
    for _ in range(2000):
        tmpl = random.choice(real_templates)
        words = random.choices(real_words, k=3)
        rows.append({"content": tmpl.format(*words), "label": 1})

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
#  STEP 2: Text Preprocessing
# ─────────────────────────────────────────────────────────────

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    """
    Cleans raw news text:
      - Lowercase
      - Remove URLs, HTML tags, special characters
      - Tokenize & remove stopwords
      - Stem words (optional but improves generalization)
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords and stem
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]

    return " ".join(tokens)


# ─────────────────────────────────────────────────────────────
#  STEP 3: Dataset Insights (Visualization)
# ─────────────────────────────────────────────────────────────

def plot_insights(df: pd.DataFrame):
    """Saves class distribution and word-count plots to static/."""
    os.makedirs("static", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d0d0d")
    for ax in axes:
        ax.set_facecolor("#1a1a1a")

    # Class distribution
    counts = df["label"].value_counts()
    bars = axes[0].bar(["Fake", "Real"], [counts[0], counts[1]],
                       color=["#ff4d4d", "#00e5a0"], edgecolor="none", width=0.5)
    axes[0].set_title("Class Distribution", color="white", pad=12)
    axes[0].tick_params(colors="white")
    axes[0].spines[:].set_visible(False)
    for bar, count in zip(bars, [counts[0], counts[1]]):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     str(count), ha="center", color="white", fontsize=11)

    # Word count distribution
    df["word_count"] = df["content"].str.split().str.len()
    axes[1].hist(df[df["label"]==0]["word_count"].clip(0, 500), bins=40,
                 alpha=0.7, color="#ff4d4d", label="Fake")
    axes[1].hist(df[df["label"]==1]["word_count"].clip(0, 500), bins=40,
                 alpha=0.7, color="#00e5a0", label="Real")
    axes[1].set_title("Word Count Distribution", color="white", pad=12)
    axes[1].tick_params(colors="white")
    axes[1].legend(facecolor="#1a1a1a", labelcolor="white")
    axes[1].spines[:].set_visible(False)

    plt.tight_layout()
    plt.savefig("static/insights.png", dpi=120, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close()
    print("📈 Insights chart saved → static/insights.png")


# ─────────────────────────────────────────────────────────────
#  STEP 4: Train & Evaluate Models
# ─────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_test, y_test):
    """Prints a full classification report for a model."""
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Fake','Real'])}")

    return {"name": name, "model": model, "accuracy": acc, "f1": f1,
            "y_pred": y_pred}


def plot_confusion_matrix(results, y_test):
    """Saves side-by-side confusion matrices for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d0d0d")

    for ax, res in zip(axes, results):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="RdYlGn",
                    xticklabels=["Fake","Real"], yticklabels=["Fake","Real"],
                    ax=ax, cbar=False,
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f'{res["name"]} — Confusion Matrix',
                     color="white", pad=12)
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.set_xlabel("Predicted", color="#aaa")
        ax.set_ylabel("Actual", color="#aaa")

    plt.tight_layout()
    plt.savefig("static/confusion_matrices.png", dpi=120,
                bbox_inches="tight", facecolor="#0d0d0d")
    plt.close()
    print("📊 Confusion matrices saved → static/confusion_matrices.png")


# ─────────────────────────────────────────────────────────────
#  STEP 5: Save Best Model
# ─────────────────────────────────────────────────────────────

def save_model(model, vectorizer):
    """Pickles the model and vectorizer into model/ folder."""
    os.makedirs("model", exist_ok=True)
    with open("model/best_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    print("💾 Model saved     → model/best_model.pkl")
    print("💾 Vectorizer saved → model/tfidf_vectorizer.pkl")


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("   FAKE NEWS DETECTION — MODEL TRAINING PIPELINE")
    print("="*55)

    # 1. Load
    df = load_dataset()

    # 2. Visualize insights
    plot_insights(df)

    # 3. Preprocess
    print("\n🔄 Preprocessing text (this may take a moment)...")
    df["clean_text"] = df["content"].apply(preprocess_text)
    df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
    print(f"✅ Preprocessing complete. Samples after cleaning: {len(df)}")

    # 4. TF-IDF Vectorization
    print("\n🔢 Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=50_000,   # top 50k terms
        ngram_range=(1, 2),    # unigrams + bigrams
        sublinear_tf=True,     # log-scale TF
        min_df=2               # ignore very rare terms
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values
    print(f"✅ Feature matrix shape: {X.shape}")

    # 5. Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 6. Train models
    print("\n🤖 Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000, C=1.0, solver="lbfgs", n_jobs=-1, random_state=42
    )
    lr_model.fit(X_train, y_train)

    print("🤖 Training Naive Bayes (Multinomial)...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)

    # 7. Evaluate
    print("\n📋 MODEL EVALUATION RESULTS")
    results = [
        evaluate_model("Logistic Regression", lr_model, X_test, y_test),
        evaluate_model("Naive Bayes", nb_model, X_test, y_test),
    ]

    # Confusion matrices
    plot_confusion_matrix(results, y_test)

    # 8. Pick best model by F1
    best = max(results, key=lambda r: r["f1"])
    print(f"\n🏆 Best model: {best['name']} (F1={best['f1']:.4f})")

    # 9. Save
    save_model(best["model"], vectorizer)

    # Also save model name for reference
    with open("model/model_info.txt", "w") as f:
        f.write(f"best_model={best['name']}\n")
        f.write(f"accuracy={best['accuracy']:.4f}\n")
        f.write(f"f1_score={best['f1']:.4f}\n")

    print("\n✅ Training pipeline complete!\n")


if __name__ == "__main__":
    main()
