"""
features.py — Shared custom sklearn transformers
Must be importable by BOTH train_model.py and app.py
so joblib can deserialise the saved pipeline correctly.
"""

import re
import string

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords

for _r in ['stopwords']:
    try:    nltk.data.find(f'corpora/{_r}')
    except LookupError: nltk.download(_r, quiet=True)

STOP_WORDS = set(stopwords.words('english'))

# ── Regex patterns ────────────────────────────────────────────────────────────
RE_HYPERBOLE = re.compile(
    r'\b(perfect|amazing|fantastic|incredible|wonderful|outstanding|superb|'
    r'brilliant|magnificent|spectacular|phenomenal|extraordinary|greatest|'
    r'life.changing|best.ever|mind.blowing|unbelievable|exceptional|glorious)\b', re.I)
RE_NEG_EXTREME = re.compile(
    r'\b(terrible|horrible|awful|worst|useless|garbage|trash|disgusting|'
    r'pathetic|scam|fraud|fake|ridiculous|disappointing)\b', re.I)
RE_CTA = re.compile(
    r'\b(buy now|purchase now|order now|everyone should buy|must buy|'
    r'you need this|tell everyone|get this now|highly recommend everyone|'
    r"run don't walk|don't hesitate)\b", re.I)
RE_PERSONAL   = re.compile(r"\b(I|my|me|we|our|myself|I've|I'm|I'd)\b")
RE_TEMPORAL   = re.compile(
    r'\b(yesterday|last week|last month|months?|weeks?|years?|days?|'
    r'ago|recently|after|since|so far|first day|first week)\b', re.I)
RE_SPECIFIC   = re.compile(
    r'\b(battery|screen|display|quality|performance|speed|size|weight|'
    r'price|cost|shipping|delivery|service|packaging|instructions|setup|'
    r'installation|camera|lens|sound|audio|bluetooth|wifi|warranty|'
    r'refund|return|durability|build|design|ergonomic|comfort|material)\b', re.I)
RE_MULTI_EXCL = re.compile(r'!{2,}')
RE_ALL_CAPS   = re.compile(r'\b[A-Z]{3,}\b')


def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'http\S+|www\S+', ' ', t)
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(w for w in t.split() if w not in STOP_WORDS and len(w) > 2)


class TextCleaner(BaseEstimator, TransformerMixin):
    """Lowercase → strip HTML/URLs → remove punctuation/digits/stopwords."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [clean_text(t) for t in X]


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """
    Extract 15 hand-crafted stylometric / linguistic features from raw text.

    Fake signals   (higher = more likely fake):
      exclamation density, caps ratio, all-caps word ratio,
      word repetition, hyperbole count, CTA phrases, multi-exclamation.

    Genuine signals (higher = more likely genuine):
      text length, lexical diversity, personal pronoun count,
      temporal references, specific product-feature mentions, avg word length.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text  = str(text)
            words = text.split()
            n_w   = max(len(words), 1)
            n_c   = max(len(text), 1)
            uniq  = len(set(w.lower() for w in words))
            upper = sum(1 for c in text if c.isupper())
            caps  = len(RE_ALL_CAPS.findall(text))

            freq = {}
            for w in words:
                freq[w.lower()] = freq.get(w.lower(), 0) + 1
            max_rep = max(freq.values()) if freq else 1

            rows.append([
                n_c,                                     #  1 text length
                n_w,                                     #  2 word count
                n_c / n_w,                               #  3 avg word length
                uniq / n_w,                              #  4 lexical diversity (TTR)
                text.count('!') / n_c * 100,             #  5 exclamation density
                upper / n_c * 100,                       #  6 caps ratio
                caps / n_w * 100,                        #  7 all-caps word ratio
                (max_rep - 1) / n_w * 100,               #  8 top-word repetition
                len(RE_HYPERBOLE.findall(text)),          #  9 hyperbole word count
                len(RE_NEG_EXTREME.findall(text)),        # 10 extreme-negative count
                len(RE_CTA.findall(text)),                # 11 call-to-action phrases
                len(RE_PERSONAL.findall(text)),           # 12 personal pronoun count
                len(RE_TEMPORAL.findall(text)),           # 13 temporal reference count
                len(RE_SPECIFIC.findall(text)),           # 14 specific-feature mentions
                len(RE_MULTI_EXCL.findall(text)),         # 15 multi-exclamation count
            ])
        return sp.csr_matrix(np.array(rows, dtype=np.float32))
