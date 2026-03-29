"""
train_model.py — Improved ML Training Script
═══════════════════════════════════════════════════════════════
Feature Engineering:
  • Word-level TF-IDF  (1,2)-grams,  max 5000 features
  • Char-level TF-IDF  (2,5)-grams,  max 3000 features  → better generalisation
  • 15 Linguistic / Stylometric features                  → catches fake patterns
Full sklearn Pipeline saved as a single model.pkl
═══════════════════════════════════════════════════════════════
"""

import os
import joblib

import pandas as pd

from sklearn.calibration      import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model     import LogisticRegression
from sklearn.metrics          import (accuracy_score, classification_report,
                                       confusion_matrix)
from sklearn.model_selection  import (StratifiedKFold, cross_val_score,
                                       train_test_split)
from sklearn.pipeline         import FeatureUnion, Pipeline
from sklearn.preprocessing    import MaxAbsScaler
from sklearn.svm              import LinearSVC
from sklearn.ensemble         import VotingClassifier
from nltk.corpus              import stopwords

# Import shared transformers so the saved pickle can be deserialised by app.py
from features import TextCleaner, LinguisticFeatures  # noqa: F401

import nltk
from nltk.corpus import stopwords

for _r in ['stopwords', 'punkt', 'punkt_tab',
           'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
           'vader_lexicon']:
    try:    nltk.data.find(f'corpora/{_r}')
    except LookupError: nltk.download(_r, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
DATA_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'data', 'sample_dataset.csv')




# ─── Pipeline Builder ────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    word_branch = Pipeline([
        ('clean', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            max_features=5000, ngram_range=(1, 2),
            sublinear_tf=True, min_df=2, strip_accents='unicode'
        ))
    ])

    char_branch = Pipeline([
        ('clean', TextCleaner()),
        ('tfidf', TfidfVectorizer(
            max_features=3000, ngram_range=(2, 5),
            analyzer='char_wb', sublinear_tf=True, min_df=3
        ))
    ])

    feature_union = FeatureUnion([
        ('word_tfidf', word_branch),   # semantic / topic patterns
        ('char_tfidf', char_branch),   # sub-word / style patterns
        ('linguistic', LinguisticFeatures()),  # explicit fake signals
    ])

    # Calibrated SVC gives better probability estimates for confidence bars
    svc = CalibratedClassifierCV(
        LinearSVC(max_iter=3000, class_weight='balanced', C=0.8, random_state=42)
    )
    lr = LogisticRegression(
        max_iter=2000, class_weight='balanced', C=0.6,
        solver='lbfgs', random_state=42
    )

    # Soft-voting ensemble: average probabilities
    clf = VotingClassifier(
        estimators=[('lr', lr), ('svc', svc)],
        voting='soft'
    )

    return Pipeline([
        ('features', feature_union),
        ('scaler',   MaxAbsScaler()),
        ('clf',      clf),
    ])


# ─── Training Entrypoint ─────────────────────────────────────────────────────
def train():
    print("=" * 68)
    print("  Fake Review Detection — Improved Pipeline Training")
    print("  Features: Word TF-IDF + Char TF-IDF + 15 Linguistic Features")
    print("  Model:    VotingClassifier (LogReg + Calibrated LinearSVC)")
    print("=" * 68)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df[['review', 'label']].dropna()
    df['label'] = df['label'].str.strip().str.lower()

    lmap = {'genuine': 1, 'fake': 0, 'real': 1, '1': 1, '0': 0}
    df['label'] = df['label'].map(lmap)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df = df[df['review'].str.len() > 10].drop_duplicates(subset='review')

    print(f"\n[INFO] Dataset : {len(df)} reviews  "
          f"(Genuine={df['label'].sum()}, Fake={(df['label']==0).sum()})")

    X, y = df['review'].tolist(), df['label'].tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Split   : Train={len(X_tr)} | Test={len(X_te)}")

    print("\n[INFO] Building pipeline & training …")
    pipeline = build_pipeline()
    pipeline.fit(X_tr, y_tr)

    # ── Metrics ──────────────────────────────────────────────────────────────
    y_pred = pipeline.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    print("\n[INFO] 5-fold cross-validation …")
    cv = cross_val_score(
        pipeline, X, y,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring='f1_weighted', n_jobs=-1
    )

    print("\n" + "=" * 68)
    print(f"  Hold-out Accuracy  : {acc * 100:.2f}%")
    print(f"  CV F1 (5-fold)     : {cv.mean() * 100:.2f}% ± {cv.std() * 100:.2f}%")
    print("=" * 68)

    cm = confusion_matrix(y_te, y_pred)
    print(f"\nConfusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=['Fake', 'Genuine']))

    os.makedirs(MODELS_DIR, exist_ok=True)
    out = os.path.join(MODELS_DIR, 'model.pkl')
    joblib.dump(pipeline, out, compress=3)
    print(f"[SAVED] Pipeline → {out}")
    print("\n✓ Done. Run: python app.py")
    print("=" * 68)


if __name__ == '__main__':
    train()
