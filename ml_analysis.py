"""
==============================================================================
AI Receptionist — ML Analysis Pipeline
==============================================================================
Covers all required deliverables:

  Step 1  — Data Loading & Preprocessing
  Step 2  — TF-IDF Feature Extraction
  Step 3  — Naive Bayes Classifier + Confusion Matrix
  Step 4  — Dimensionality Reduction with SVD (LSA)
  Step 5  — Logistic Regression with SVD Features + Confusion Matrix
  Step 6  — Dimensionality Reduction with PCA
  Step 7  — Logistic Regression with PCA Features + Confusion Matrix
  Step 8  — Side-by-side Visual Comparison of All Three Models

Data source: CallSession and ConversationMessage records from the
             AI receptionist's Django database (conversation transcripts
             labelled by the auto-categorisation that already runs after
             every call).  When the database is empty / unavailable the
             module falls back to a realistic synthetic dataset generated
             from the domain vocabulary of this salon-booking system.

Usage (standalone, no Django needed):
    python ml_pipeline/ml_analysis.py

Usage (inside Django shell):
    from ml_pipeline.ml_analysis import run_pipeline
    run_pipeline(use_db=True)
==============================================================================
"""

from __future__ import annotations

import os
import sys
import warnings
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe inside Django
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Output directory (sits next to this file)
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# STEP 1 — Data Loading & Preprocessing
# ===========================================================================

CATEGORY_LABELS = {
    "make_appointment": "Book Appointment",
    "cancel_appointment": "Cancel Appointment",
    "reschedule_appointment": "Reschedule Appointment",
    "ask_question": "General Inquiry",
    "unknown": "Unknown",
}

# Realistic synthetic transcripts drawn from the actual agent instructions,
# service names, and tool responses found in this codebase.
SYNTHETIC_DATA: list[dict] = [
    # ── Book Appointment ────────────────────────────────────────────────────
    {"text": "Hi I would like to book a haircut appointment for tomorrow afternoon please", "label": "make_appointment"},
    {"text": "Can I schedule a manicure for Saturday morning", "label": "make_appointment"},
    {"text": "I need to book a pedicure and a facial this Friday", "label": "make_appointment"},
    {"text": "I want to make an appointment for a hair coloring session next week", "label": "make_appointment"},
    {"text": "Please book me a massage for two o clock tomorrow", "label": "make_appointment"},
    {"text": "Hi can you book a hair styling appointment for me on Monday", "label": "make_appointment"},
    {"text": "I would like to schedule a waxing appointment as soon as possible", "label": "make_appointment"},
    {"text": "Can you set up an appointment for a full body massage on Thursday", "label": "make_appointment"},
    {"text": "I want to reserve a slot for a hair treatment and blowout", "label": "make_appointment"},
    {"text": "Please arrange an appointment for a nail art session this weekend", "label": "make_appointment"},
    {"text": "Book me in for a trim and blow dry at three pm please", "label": "make_appointment"},
    {"text": "I need a slot for a deep conditioning treatment next Tuesday", "label": "make_appointment"},
    {"text": "Can I get an appointment for eyebrow shaping and tinting", "label": "make_appointment"},
    {"text": "Schedule a hot stone massage for Friday evening please", "label": "make_appointment"},
    {"text": "I would like to book with Sarah for a colour and cut on Wednesday", "label": "make_appointment"},
    {"text": "Can you fit me in for a gel manicure tomorrow morning", "label": "make_appointment"},
    {"text": "I want to come in for a facial and a lash lift if you have availability", "label": "make_appointment"},
    {"text": "Please book an appointment for a keratin treatment next Saturday", "label": "make_appointment"},
    {"text": "Hi I am calling to make a booking for a balayage highlights appointment", "label": "make_appointment"},
    {"text": "I would like to schedule a couples massage for this Sunday afternoon", "label": "make_appointment"},

    # ── Cancel Appointment ──────────────────────────────────────────────────
    {"text": "I need to cancel my appointment for tomorrow it is an emergency", "label": "cancel_appointment"},
    {"text": "Can you cancel my haircut booking this Friday please", "label": "cancel_appointment"},
    {"text": "I want to cancel my manicure appointment scheduled for Saturday morning", "label": "cancel_appointment"},
    {"text": "Please cancel the pedicure I booked for next week I cannot make it", "label": "cancel_appointment"},
    {"text": "Hi I need to call off my massage appointment booked for Thursday", "label": "cancel_appointment"},
    {"text": "Cancel my booking for the facial treatment please", "label": "cancel_appointment"},
    {"text": "I will not be able to come in for my hair coloring appointment can you cancel it", "label": "cancel_appointment"},
    {"text": "I want to remove my appointment for the hair treatment next Tuesday", "label": "cancel_appointment"},
    {"text": "Please cancel my upcoming waxing session I need to reschedule", "label": "cancel_appointment"},
    {"text": "Hi I need to cancel my appointment booked under my number", "label": "cancel_appointment"},
    {"text": "Cancel my three o clock appointment for today please", "label": "cancel_appointment"},
    {"text": "I cannot make it in tomorrow can you remove my booking", "label": "cancel_appointment"},
    {"text": "I need to cancel the lash tint appointment I made last week", "label": "cancel_appointment"},
    {"text": "Please remove my gel nail appointment for this Friday afternoon", "label": "cancel_appointment"},
    {"text": "I want to cancel all my upcoming appointments at the salon", "label": "cancel_appointment"},

    # ── Reschedule Appointment ──────────────────────────────────────────────
    {"text": "I need to reschedule my appointment from Friday to Monday please", "label": "reschedule_appointment"},
    {"text": "Can I move my haircut appointment to a different day this week", "label": "reschedule_appointment"},
    {"text": "I want to change the time of my manicure from morning to afternoon", "label": "reschedule_appointment"},
    {"text": "Please reschedule my pedicure appointment to next Thursday instead", "label": "reschedule_appointment"},
    {"text": "Can you move my massage booking to Saturday morning if there is availability", "label": "reschedule_appointment"},
    {"text": "I need to shift my hair coloring appointment one day earlier", "label": "reschedule_appointment"},
    {"text": "I would like to change my facial appointment to a different time slot", "label": "reschedule_appointment"},
    {"text": "Can I push my appointment back by two hours please", "label": "reschedule_appointment"},
    {"text": "I need to move my nail appointment I have a conflict at work", "label": "reschedule_appointment"},
    {"text": "Please move my booking from two o clock to four o clock", "label": "reschedule_appointment"},
    {"text": "I want to reschedule my waxing appointment to next week", "label": "reschedule_appointment"},
    {"text": "Can you find me an alternative slot for my hair treatment", "label": "reschedule_appointment"},
    {"text": "I need to change my appointment to a different date entirely", "label": "reschedule_appointment"},
    {"text": "Please move my eyebrow appointment from Monday to Wednesday", "label": "reschedule_appointment"},
    {"text": "I would like to reschedule my deep conditioning appointment to Friday", "label": "reschedule_appointment"},

    # ── General Inquiry / FAQ ───────────────────────────────────────────────
    {"text": "What are your business hours for the salon this week", "label": "ask_question"},
    {"text": "Do you offer hair coloring services and what is the price", "label": "ask_question"},
    {"text": "Where is the salon located and how do I get there", "label": "ask_question"},
    {"text": "What services do you offer for nails and how long do they take", "label": "ask_question"},
    {"text": "Is the salon open on Sundays and bank holidays", "label": "ask_question"},
    {"text": "How much does a full set of acrylic nails cost at your salon", "label": "ask_question"},
    {"text": "Do you have any parking near the salon", "label": "ask_question"},
    {"text": "What is the phone number for the main salon branch", "label": "ask_question"},
    {"text": "Do you accept walk in customers or is it appointment only", "label": "ask_question"},
    {"text": "Can you tell me about your loyalty program and rewards", "label": "ask_question"},
    {"text": "What is your cancellation policy and how much notice do you need", "label": "ask_question"},
    {"text": "Do you have gift cards or vouchers available for purchase", "label": "ask_question"},
    {"text": "How long does a balayage usually take at your salon", "label": "ask_question"},
    {"text": "What brands of hair products do you use during treatments", "label": "ask_question"},
    {"text": "Do you offer any student or senior discounts on services", "label": "ask_question"},
    {"text": "Is there a waiting list for popular appointments like balayage", "label": "ask_question"},
    {"text": "What time does the salon close on Saturdays", "label": "ask_question"},
    {"text": "Can I bring my child with me to my appointment", "label": "ask_question"},
    {"text": "Do you offer beard trimming and grooming services for men", "label": "ask_question"},
    {"text": "How much is a basic cut and blow dry at your salon", "label": "ask_question"},
]


def load_data(use_db: bool = False) -> pd.DataFrame:
    """
    Load call transcript data.

    When use_db=True, pulls ConversationMessage + CallSession data from
    the Django ORM.  Falls back to the synthetic dataset automatically if
    Django is not set up or the database is empty.

    Returns a DataFrame with columns: text, label
    """
    print("\n" + "=" * 70)
    print("STEP 1 — Data Loading & Preprocessing")
    print("=" * 70)

    if use_db:
        try:
            from receptionist.models import CallSession, ConversationMessage

            sessions = CallSession.objects.filter(
                category__isnull=False
            ).exclude(category="").exclude(conversation_transcript=[])

            records = []
            for session in sessions:
                transcript = session.conversation_transcript or []
                # Join all user turns into a single document
                user_text = " ".join(
                    msg.get("text", "") or msg.get("content", "")
                    for msg in transcript
                    if msg.get("role") == "user"
                )
                if user_text.strip() and session.category:
                    records.append({"text": user_text.strip(), "label": session.category})

            if len(records) >= 20:
                df = pd.DataFrame(records)
                print(f"  ✔  Loaded {len(df)} records from Django database.")
            else:
                print(
                    f"  ⚠  Only {len(records)} usable DB records found. "
                    "Falling back to synthetic dataset."
                )
                df = pd.DataFrame(SYNTHETIC_DATA)
        except Exception as exc:
            print(f"  ⚠  DB access failed ({exc}). Using synthetic dataset.")
            df = pd.DataFrame(SYNTHETIC_DATA)
    else:
        df = pd.DataFrame(SYNTHETIC_DATA)
        print(f"  ✔  Loaded synthetic dataset: {len(df)} samples.")

    # ── Basic preprocessing ─────────────────────────────────────────────────
    df["text"] = (
        df["text"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 5]

    # Friendly label names for display
    df["label_display"] = df["label"].map(CATEGORY_LABELS).fillna(df["label"])

    print(f"\n  Dataset shape : {df.shape}")
    print(f"  Unique labels : {df['label'].nunique()}")
    print("\n  Class distribution:")
    dist = df["label_display"].value_counts()
    for name, count in dist.items():
        bar = "█" * count
        print(f"    {name:<28} {count:>3}  {bar}")

    # ── Train / Test split ──────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    print(f"\n  Train size : {len(X_train)}  |  Test size : {len(X_test)}")

    # ── Visualise class distribution ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    dist.plot(kind="bar", ax=ax, color=colors[: len(dist)], edgecolor="white", width=0.6)
    ax.set_title("Class Distribution — Call Intent Categories", fontsize=13, fontweight="bold")
    ax.set_xlabel("Intent Category", fontsize=11)
    ax.set_ylabel("Number of Samples", fontsize=11)
    ax.tick_params(axis="x", rotation=20)
    for p in ax.patches:
        ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    _save("step1_class_distribution.png")

    return df, X_train, X_test, y_train, y_test


# ===========================================================================
# STEP 2 — TF-IDF Feature Extraction
# ===========================================================================

def extract_tfidf(X_train, X_test):
    """Fit TF-IDF on training data, transform both splits."""
    print("\n" + "=" * 70)
    print("STEP 2 — TF-IDF Feature Extraction")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),         # unigrams + bigrams
        sublinear_tf=True,          # apply log normalization to TF
        min_df=1,
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    vocab = vectorizer.get_feature_names_out()
    print(f"\n  Vocabulary size  : {len(vocab)} terms")
    print(f"  TF-IDF matrix    : {X_train_tfidf.shape[0]} docs × {X_train_tfidf.shape[1]} features")
    print(f"  Matrix sparsity  : {1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.1%}")

    # Top terms by mean TF-IDF weight
    mean_weights = np.asarray(X_train_tfidf.mean(axis=0)).ravel()
    top_idx = mean_weights.argsort()[-20:][::-1]
    top_terms = [(vocab[i], mean_weights[i]) for i in top_idx]

    print("\n  Top 20 terms by mean TF-IDF weight:")
    for term, weight in top_terms:
        bar = "▓" * int(weight * 300)
        print(f"    {term:<30} {weight:.4f}  {bar}")

    # ── Visualise TF-IDF matrix (heatmap of first 30 docs × 30 terms) ──────
    dense_sample = X_train_tfidf[:30, top_idx[:30]].toarray()
    term_labels = [vocab[i] for i in top_idx[:30]]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    sns.heatmap(
        dense_sample,
        ax=axes[0],
        cmap="YlOrRd",
        xticklabels=term_labels,
        yticklabels=[f"Doc {i+1}" for i in range(30)],
        cbar_kws={"label": "TF-IDF Weight"},
    )
    axes[0].set_title("TF-IDF Matrix — First 30 Docs × Top 30 Terms", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=7)

    # Bar chart of top terms
    terms_plot, weights_plot = zip(*top_terms)
    axes[1].barh(range(len(terms_plot)), weights_plot, color="#4C72B0", edgecolor="white")
    axes[1].set_yticks(range(len(terms_plot)))
    axes[1].set_yticklabels(terms_plot, fontsize=8)
    axes[1].set_xlabel("Mean TF-IDF Weight")
    axes[1].set_title("Top 20 Terms by Mean TF-IDF Weight", fontsize=11, fontweight="bold")
    axes[1].invert_yaxis()

    plt.tight_layout()
    _save("step2_tfidf_features.png")

    return X_train_tfidf, X_test_tfidf, vectorizer


# ===========================================================================
# STEP 3 — Baseline Model: Naive Bayes + Confusion Matrix
# ===========================================================================

def train_naive_bayes(X_train_tfidf, X_test_tfidf, y_train, y_test):
    """Train Multinomial Naive Bayes on TF-IDF features."""
    print("\n" + "=" * 70)
    print("STEP 3 — Baseline Model: Naive Bayes with TF-IDF")
    print("=" * 70)

    clf = MultinomialNB(alpha=0.5)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    labels = sorted(set(y_test))
    label_names = [CATEGORY_LABELS.get(l, l) for l in labels]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\n  Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_names, zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    _print_cm_analysis(cm, label_names, model_name="Naive Bayes")
    fig = _plot_confusion_matrix(cm, label_names, "Naive Bayes (TF-IDF)", "#4C72B0")
    _save("step3_nb_confusion_matrix.png")

    return clf, y_pred, cm, accuracy


# ===========================================================================
# STEP 4 — Dimensionality Reduction with SVD (LSA)
# ===========================================================================

def apply_svd(X_train_tfidf, X_test_tfidf, n_components: int = 50):
    """Apply Truncated SVD (Latent Semantic Analysis) to TF-IDF matrix."""
    print("\n" + "=" * 70)
    print("STEP 4 — Dimensionality Reduction with SVD (LSA)")
    print("=" * 70)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train_tfidf)
    X_test_svd = svd.transform(X_test_tfidf)

    explained = svd.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"\n  Original features  : {X_train_tfidf.shape[1]}")
    print(f"  SVD components     : {n_components}")
    print(f"  Variance explained : {cumulative[-1]:.2%}")
    print(
        f"\n  First 5 singular values capture "
        f"{cumulative[4]:.1%} of the variance."
    )
    print(
        textwrap.dedent("""
  How SVD captures semantic relationships:
  ─────────────────────────────────────────
  SVD decomposes the TF-IDF matrix (docs × terms) into three matrices:
      U  (docs  × k)   — document-concept space
      Σ  (k × k)       — concept strengths (singular values)
      Vᵀ (k × terms)   — term-concept space

  In our receptionist dataset this means:
  • Concept 1 might group "book", "schedule", "appointment" → booking intent
  • Concept 2 might group "cancel", "remove", "call off"   → cancellation intent
  • Concept 3 might group "hours", "price", "location"     → FAQ intent

  Words never seen together in training that share a latent concept will
  receive similar representations — a major advantage over raw TF-IDF.
  """)
    )

    # ── Explained variance plot ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].bar(
        range(1, n_components + 1), explained * 100,
        color="#55A868", edgecolor="white", alpha=0.85,
    )
    axes[0].set_xlabel("SVD Component")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].set_title("SVD — Explained Variance per Component", fontweight="bold")
    axes[0].set_xlim(0, n_components + 1)

    axes[1].plot(range(1, n_components + 1), cumulative * 100,
                 color="#C44E52", linewidth=2, marker="o", markersize=3)
    axes[1].axhline(80, color="grey", linestyle="--", linewidth=1, label="80% threshold")
    axes[1].axhline(90, color="orange", linestyle="--", linewidth=1, label="90% threshold")
    comp_80 = np.argmax(cumulative >= 0.80) + 1
    comp_90 = np.argmax(cumulative >= 0.90) + 1
    axes[1].axvline(comp_80, color="grey", linestyle=":", linewidth=1)
    axes[1].axvline(comp_90, color="orange", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Explained Variance (%)")
    axes[1].set_title("SVD — Cumulative Explained Variance", fontweight="bold")
    axes[1].legend()
    axes[1].annotate(f"{comp_80} comps\n→ 80%", xy=(comp_80, 80),
                     xytext=(comp_80 + 3, 75), fontsize=8, color="grey",
                     arrowprops=dict(arrowstyle="->", color="grey"))

    plt.suptitle("Step 4 — Truncated SVD (Latent Semantic Analysis)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save("step4_svd_variance.png")

    return X_train_svd, X_test_svd, svd


# ===========================================================================
# STEP 5 — Logistic Regression with SVD Features
# ===========================================================================

def train_lr_svd(X_train_svd, X_test_svd, y_train, y_test):
    """Train Logistic Regression on SVD-reduced features."""
    print("\n" + "=" * 70)
    print("STEP 5 — Logistic Regression with SVD Features")
    print("=" * 70)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
    clf.fit(X_train_svd, y_train)
    y_pred = clf.predict(X_test_svd)

    labels = sorted(set(y_test))
    label_names = [CATEGORY_LABELS.get(l, l) for l in labels]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\n  Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_names, zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    _print_cm_analysis(cm, label_names, model_name="Logistic Regression (SVD)")
    _plot_confusion_matrix(cm, label_names, "Logistic Regression (SVD)", "#55A868")
    _save("step5_lr_svd_confusion_matrix.png")

    print(
        textwrap.dedent("""
  Effect of SVD dimensionality reduction on performance:
  ───────────────────────────────────────────────────────
  • Reduces noise: removing low-variance components filters out
    idiosyncratic word-frequency noise from training transcripts.
  • Faster inference: 50 floats vs 500 sparse dimensions.
  • Better generalisation on short texts (like voice transcripts)
    because semantically similar caller phrases map to the same
    latent concept even when different surface words are used.
  • Trade-off: some fine-grained distinguishing features are lost,
    which may hurt precision on closely related intents (e.g.
    cancel vs reschedule).
  """)
    )

    return clf, y_pred, cm, accuracy


# ===========================================================================
# STEP 6 — Dimensionality Reduction with PCA
# ===========================================================================

def apply_pca(X_train_tfidf, X_test_tfidf, n_components: int = 50):
    """Apply PCA to (standardised) TF-IDF features."""
    print("\n" + "=" * 70)
    print("STEP 6 — Dimensionality Reduction with PCA")
    print("=" * 70)

    # PCA requires a dense matrix and standardised features
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_dense)
    X_test_scaled = scaler.transform(X_test_dense)

    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    print(f"\n  Original features (dense) : {X_train_dense.shape[1]}")
    print(f"  PCA components            : {n_components}")
    print(f"  Variance explained        : {cumulative[-1]:.2%}")
    print(
        textwrap.dedent("""
  PCA vs SVD for text data:
  ──────────────────────────
  PCA maximises variance in the standardised feature space.  Because it
  requires a dense matrix it is heavier in memory, but it can capture
  variance directions missed by Truncated SVD (which works directly on
  the sparse raw TF-IDF values without standardisation).

  For text:
  • SVD (LSA) is generally preferred — it handles sparsity natively.
  • PCA after StandardScaler can outperform SVD when feature scales
    vary widely, as scaling equalises the influence of rare but
    important terms.
  • On short voice transcripts with a small vocabulary both methods
    tend to converge to similar performance.
  """)
    )

    return X_train_pca, X_test_pca, pca, scaler


# ===========================================================================
# STEP 7 — Logistic Regression with PCA Features
# ===========================================================================

def train_lr_pca(X_train_pca, X_test_pca, y_train, y_test):
    """Train Logistic Regression on PCA-reduced features."""
    print("\n" + "=" * 70)
    print("STEP 7 — Logistic Regression with PCA Features")
    print("=" * 70)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, solver="lbfgs")
    clf.fit(X_train_pca, y_train)
    y_pred = clf.predict(X_test_pca)

    labels = sorted(set(y_test))
    label_names = [CATEGORY_LABELS.get(l, l) for l in labels]

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy : {accuracy:.4f} ({accuracy*100:.1f}%)")
    print("\n  Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=label_names, zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    _print_cm_analysis(cm, label_names, model_name="Logistic Regression (PCA)")
    _plot_confusion_matrix(cm, label_names, "Logistic Regression (PCA)", "#C44E52")
    _save("step7_lr_pca_confusion_matrix.png")

    print(
        textwrap.dedent("""
  Which dimensionality reduction works better for this text data?
  ───────────────────────────────────────────────────────────────
  • SVD (LSA) typically wins on sparse text matrices because it avoids
    the dense materialisation step and works with the natural co-occurrence
    structure of the TF-IDF space.
  • PCA after scaling can recover when rare but discriminative terms
    would otherwise be drowned out by high-frequency ones.
  • On this dataset (short caller transcripts, ~4 classes) both methods
    deliver competitive accuracy with Logistic Regression.  The
    side-by-side comparison in Step 8 reveals which edges each method
    gets right and wrong.
  """)
    )

    return clf, y_pred, cm, accuracy


# ===========================================================================
# STEP 8 — Visual Comparison of All Three Confusion Matrices
# ===========================================================================

def compare_all_models(
    cm_nb, cm_svd, cm_pca,
    acc_nb, acc_svd, acc_pca,
    y_test,
):
    """Side-by-side plot of all three confusion matrices + accuracy bar."""
    print("\n" + "=" * 70)
    print("STEP 8 — Visual Comparison of All Three Models")
    print("=" * 70)

    labels = sorted(set(y_test))
    label_names = [CATEGORY_LABELS.get(l, l) for l in labels]

    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    model_data = [
        ("Naive Bayes\n(TF-IDF)", cm_nb, acc_nb, "#4C72B0"),
        ("Logistic Regression\n(SVD / LSA)", cm_svd, acc_svd, "#55A868"),
        ("Logistic Regression\n(PCA)", cm_pca, acc_pca, "#C44E52"),
    ]

    for col, (title, cm, acc, color) in enumerate(model_data):
        ax = fig.add_subplot(gs[0, col])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
        ax.set_title(f"{title}\nAccuracy: {acc*100:.1f}%", fontsize=11, fontweight="bold", color=color)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Predicted Label", fontsize=8)
        ax.set_ylabel("True Label", fontsize=8)

    # Accuracy bar chart
    ax_bar = fig.add_subplot(gs[1, :])
    models = ["Naive Bayes\n(TF-IDF)", "LR + SVD", "LR + PCA"]
    accs = [acc_nb * 100, acc_svd * 100, acc_pca * 100]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    bars = ax_bar.bar(models, accs, color=colors, edgecolor="white", width=0.45)
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel("Accuracy (%)", fontsize=12)
    ax_bar.set_title("Model Accuracy Comparison", fontsize=13, fontweight="bold")
    for bar, acc in zip(bars, accs):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{acc:.1f}%",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    fig.suptitle(
        "Step 8 — Side-by-Side Model Comparison\n"
        "AI Receptionist Intent Classification Pipeline",
        fontsize=14, fontweight="bold", y=0.98,
    )

    _save("step8_model_comparison.png")

    # Print summary table
    print("\n  ╔══════════════════════════════════════════════╗")
    print("  ║            Model Accuracy Summary            ║")
    print("  ╠══════════════════════════════════════════════╣")
    for name, acc in zip(["Naive Bayes (TF-IDF)", "LR + SVD", "LR + PCA"], [acc_nb, acc_svd, acc_pca]):
        bar = "█" * int(acc * 30)
        print(f"  ║  {name:<22} {acc*100:>5.1f}%  {bar:<20}  ║")
    print("  ╚══════════════════════════════════════════════╝")

    best = max(zip([acc_nb, acc_svd, acc_pca], ["Naive Bayes (TF-IDF)", "LR + SVD", "LR + PCA"]))
    print(f"\n  Best model: {best[1]}  ({best[0]*100:.1f}% accuracy)")


# ===========================================================================
# INTERNAL HELPERS
# ===========================================================================

def _save(filename: str) -> None:
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close("all")
    print(f"  📊 Saved: {path}")


def _print_cm_analysis(cm: np.ndarray, label_names: list[str], model_name: str) -> None:
    """Print TP / FP / FN / TN for each class."""
    print(f"\n  Confusion Matrix Analysis — {model_name}:")
    print(f"  {'Class':<28} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
    print("  " + "-" * 50)
    n = len(label_names)
    for i, name in enumerate(label_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        print(f"  {name:<28} {tp:>5} {fp:>5} {fn:>5} {tn:>5}")


def _plot_confusion_matrix(
    cm: np.ndarray,
    label_names: list[str],
    title: str,
    color: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=True, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {title}", fontsize=12, fontweight="bold", color=color)
    ax.tick_params(axis="x", rotation=25, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    return fig


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def run_pipeline(use_db: bool = False) -> None:
    """Execute the full 8-step ML pipeline."""
    print("\n" + "█" * 70)
    print("  AI RECEPTIONIST — ML ANALYSIS PIPELINE")
    print("  Steps 1-8 | Intent Classification from Call Transcripts")
    print("█" * 70)

    # Step 1
    df, X_train, X_test, y_train, y_test = load_data(use_db=use_db)

    # Step 2
    X_train_tfidf, X_test_tfidf, vectorizer = extract_tfidf(X_train, X_test)

    # Step 3
    nb_clf, nb_pred, cm_nb, acc_nb = train_naive_bayes(
        X_train_tfidf, X_test_tfidf, y_train, y_test
    )

    # Step 4
    X_train_svd, X_test_svd, svd = apply_svd(X_train_tfidf, X_test_tfidf, n_components=50)

    # Step 5
    lr_svd_clf, lr_svd_pred, cm_svd, acc_svd = train_lr_svd(
        X_train_svd, X_test_svd, y_train, y_test
    )

    # Step 6
    X_train_pca, X_test_pca, pca, scaler = apply_pca(
        X_train_tfidf, X_test_tfidf, n_components=50
    )

    # Step 7
    lr_pca_clf, lr_pca_pred, cm_pca, acc_pca = train_lr_pca(
        X_train_pca, X_test_pca, y_train, y_test
    )

    # Step 8
    compare_all_models(cm_nb, cm_svd, cm_pca, acc_nb, acc_svd, acc_pca, y_test)

    print("\n" + "█" * 70)
    print(f"  Pipeline complete.  All outputs saved to: {OUTPUT_DIR}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    # Run without Django by default.
    # Pass --db flag to pull live data from the Django database.
    use_db = "--db" in sys.argv
    run_pipeline(use_db=use_db)
