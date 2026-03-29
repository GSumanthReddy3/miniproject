"""
app.py — ReviewGuard AI
Flask Backend: ML + NLP + Auth + DB
Results are displayed on a SEPARATE /results/<id> page.
"""

import os, re, json, string, joblib, logging, traceback
import features  # noqa: F401 — must be imported so joblib can deserialise the pipeline
from datetime import datetime
from collections import Counter

import nltk, numpy as np, pandas as pd, requests
from bs4 import BeautifulSoup
from flask import (Flask, render_template, request, jsonify,
                   session, redirect, url_for, flash)
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text as sql_text
from werkzeug.security import generate_password_hash, check_password_hash
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ─── NLTK ────────────────────────────────────────────────────────────────────
# ─── NLTK ────────────────────────────────────────────────────────────────────
for _r in ['stopwords', 'punkt', 'punkt_tab',
           'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng',
           'vader_lexicon']:
    try:
        if _r in ['stopwords', 'vader_lexicon']:
            nltk.data.find(f'corpora/{_r}')
        else:
            nltk.data.find(f'tokenizers/{_r}')
    except LookupError:
        nltk.download(_r, quiet=True)

# ─── App Config ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'model')
DB_PATH    = os.path.join(BASE_DIR, 'reviewdb.sqlite')

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rg-ai-secret-2025-$ecure!')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

db = SQLAlchemy(app)
logging.basicConfig(level=logging.INFO, format='%(levelname)s — %(message)s')
logger = logging.getLogger(__name__)

STOP_WORDS = set(stopwords.words('english'))
vader      = SentimentIntensityAnalyzer()


# ─── Database Models ──────────────────────────────────────────────────────────
class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    analyses      = db.relationship('Analysis', backref='user', lazy=True)


class Analysis(db.Model):
    __tablename__ = 'analyses'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    input_type    = db.Column(db.String(20))
    total         = db.Column(db.Integer)
    fake_count    = db.Column(db.Integer)
    genuine_count = db.Column(db.Integer)
    pros_json     = db.Column(db.Text)
    cons_json     = db.Column(db.Text)
    result_json   = db.Column(db.Text)       # full result dict for results page
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    reviews       = db.relationship('ReviewRecord', backref='analysis', lazy=True)


class ReviewRecord(db.Model):
    __tablename__ = 'review_records'
    id            = db.Column(db.Integer, primary_key=True)
    analysis_id   = db.Column(db.Integer, db.ForeignKey('analyses.id'))
    review_text   = db.Column(db.Text)
    prediction    = db.Column(db.String(20))
    confidence    = db.Column(db.Float)


# ─── Model / Pipeline Loader ──────────────────────────────────────────────────
_pipeline = None

def load_model():
    global _pipeline
    if _pipeline is None:
        path = os.path.join(MODELS_DIR, 'model.pkl')
        if not os.path.exists(path):
            raise FileNotFoundError(
                "ML model not found. Please run:  python train_model.py"
            )
        _pipeline = joblib.load(path)
        logger.info("✓ Pipeline loaded successfully")
    return _pipeline


# ─── NLP Functions ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r'<[^>]+>', ' ', t)
    t = re.sub(r'http\S+|www\S+', ' ', t)
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(w for w in t.split() if w not in STOP_WORDS and len(w) > 2)


def predict_review(text: str):
    """Run inference through the full pipeline (raw text input)."""
    model = load_model()
    text  = str(text).strip()
    if len(text) < 3:
        return 'unknown', 0.0, [50.0, 50.0]
    pred  = model.predict([text])[0]
    proba = model.predict_proba([text])[0].tolist()
    label = 'genuine' if pred == 1 else 'fake'
    conf  = round(max(proba) * 100, 2)
    return label, conf, [round(p * 100, 2) for p in proba]


def extract_insights(reviews: list) -> dict:
    """
    NLP: NLTK POS tagging + VADER sentiment to generate explanation sentences.
    Transforms raw text into meaningful product advantages and disadvantages.
    """
    if not reviews:
        return {'advantages': [], 'disadvantages': []}

    feature_adjectives = {'pro': {}, 'con': {}}

    for rev in reviews:
        try:
            sents = sent_tokenize(str(rev))
        except Exception:
            sents = str(rev).split('.')

        for sent in sents:
            if len(sent.strip()) < 10:
                continue
            
            sent_score = vader.polarity_scores(sent)['compound']
            if abs(sent_score) < 0.1:
                continue
                
            try:
                tokens = word_tokenize(sent.lower())
                tagged = pos_tag(tokens)
            except Exception:
                continue

            for i, (w, tag) in enumerate(tagged):
                if tag in ('NN', 'NNS') and len(w) > 3 and w not in STOP_WORDS and w.isalpha():
                    adj = None
                    if i > 0 and tagged[i-1][1] in ('JJ', 'JJR', 'JJS'):
                        adj = tagged[i-1][0]
                    elif i < len(tagged) - 2 and tagged[i+1][0] in ('is', 'was', 'are', 'were', 'looks', 'feels', 'seems') and tagged[i+2][1] in ('JJ', 'JJR', 'JJS'):
                        adj = tagged[i+2][0]
                        
                    if adj and len(adj) > 2 and adj not in STOP_WORDS:
                        adj_score = vader.polarity_scores(adj)['compound']
                        
                        if adj_score > 0.1 or (adj_score == 0 and sent_score >= 0.15):
                            group = 'pro'
                        elif adj_score < -0.1 or (adj_score == 0 and sent_score <= -0.15):
                            group = 'con'
                        else:
                            continue
                            
                        # Keep original capitalization for the noun if possible, but lower is safer for grouping
                        if w not in feature_adjectives[group]:
                            feature_adjectives[group][w] = Counter()
                        feature_adjectives[group][w][adj] += 1

    def generate_sentences(group_dict, limit=10, is_pro=True):
        pro_templates = [
            "Users frequently noted that the {feature} is {adj}.",
            "The {feature} was commonly described as {adj}.",
            "Many reviewers found the {feature} to be {adj}.",
            "The {feature} is considered {adj}."
        ]
        con_templates = [
            "Users frequently complained that the {feature} is {adj}.",
            "The {feature} was commonly described as {adj}.",
            "Many reviewers found the {feature} to be {adj}.",
            "Reviewers noted the {feature} is {adj}."
        ]
        templates = pro_templates if is_pro else con_templates
        
        sorted_features = sorted(group_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)
        
        sentences = []
        for i, (feature, adjs) in enumerate(sorted_features):
            if not adjs: continue
            top_adjs = [a[0] for a in adjs.most_common(2)]
            adj_str = " and ".join(top_adjs)
            template = templates[i % len(templates)]
            sentences.append(template.format(feature=feature.replace('_', ' '), adj=adj_str))
            if len(sentences) >= limit:
                break
        return sentences

    return {
        'advantages': generate_sentences(feature_adjectives['pro'], limit=10, is_pro=True),
        'disadvantages': generate_sentences(feature_adjectives['con'], limit=10, is_pro=False)
    }


def analyze_trends(df: pd.DataFrame):
    """Group reviews by month. Returns (trend_list, trend_summary)."""
    if df is None or df.empty:
        return [], None
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if not date_cols:
        return [], None
    try:
        df = df.copy()
        df['_dt'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
        df = df.dropna(subset=['_dt'])
        if df.empty:
            return [], None
        df['_mo'] = df['_dt'].dt.to_period('M')
        monthly   = df.groupby('_mo').size().reset_index(name='cnt')
        monthly   = monthly.sort_values('_mo')

        trend_list = []
        for _, row in monthly.iterrows():
            trend_list.append({
                "month": str(row['_mo']),
                "reviews": int(row['cnt'])
            })
            
        trend_summary = None
        if len(trend_list) > 1:
            first_mo = trend_list[0]['month']
            last_mo = trend_list[-1]['month']
            first_val = trend_list[0]['reviews']
            last_val = trend_list[-1]['reviews']
            
            if last_val > first_val:
                trend_summary = f"Review activity increased from {first_mo} to {last_mo}, indicating growing product popularity."
            elif last_val < first_val:
                trend_summary = f"Review activity decreased from {first_mo} to {last_mo}, suggesting a drop in product popularity."
            else:
                trend_summary = f"Review activity remained stable between {first_mo} and {last_mo}."
        elif len(trend_list) == 1:
            trend_summary = f"All {trend_list[0]['reviews']} reviews were submitted in {trend_list[0]['month']}."

        return trend_list, trend_summary
    except Exception as e:
        logger.warning(f"Trend analysis error: {e}")
        return [], None


def scrape_reviews_from_url(url: str) -> list:
    headers = {'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 Chrome/123.0.0.0 Safari/537.36'
    )}
    resp = requests.get(url, headers=headers, timeout=12)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
        tag.decompose()
    selectors = ['[data-hook="review-body"]', '.review-text', '.review-content',
                 '.comment-body', '.user-review', '.review', 'article', 'blockquote']
    reviews = []
    for sel in selectors:
        els = soup.select(sel)
        if els:
            for el in els[:30]:
                t = el.get_text(separator=' ', strip=True)
                if len(t) > 30:
                    reviews.append(t)
            if reviews:
                break
    if not reviews:
        for p in soup.find_all('p'):
            t = p.get_text(strip=True)
            if len(t) > 50:
                reviews.append(t)
    return reviews[:50]


def process_reviews(reviews: list, source_df=None) -> dict:
    """Core pipeline: predict → filter genuine → NLP → trends → response."""
    if not reviews:
        return {'error': 'No reviews provided'}

    results        = []
    genuine_texts  = []
    fake_count     = 0
    genuine_count  = 0

    for text in reviews:
        t = str(text).strip()
        if not t:
            continue
        label, conf, proba = predict_review(t)
        results.append({
            'text':         t[:500],
            'prediction':   label,
            'confidence':   conf,
            'fake_prob':    proba[0],
            'genuine_prob': proba[1]
        })
        if label == 'genuine':
            genuine_count += 1
            genuine_texts.append(t)
        else:
            fake_count += 1

    if not results:
        return {'error': 'No valid reviews after processing'}

    total       = fake_count + genuine_count
    fake_pct    = round(fake_count    / total * 100, 1) if total else 0
    genuine_pct = round(genuine_count / total * 100, 1) if total else 0

    insights = extract_insights(genuine_texts)

    # Trend (only if source_df has dates)
    trend_data = [] 
    trend_summary = None
    if source_df is not None and len(source_df) == len(results):
        source_df = source_df.copy()
        source_df['prediction'] = [r['prediction'] for r in results]
        t_list, t_sum = analyze_trends(source_df)
        if t_list:
            trend_data = t_list
            trend_summary = t_sum

    # Overall sentiment on genuine reviews
    sentiment = 'neutral'
    if genuine_texts:
        avg = np.mean([vader.polarity_scores(t)['compound']
                       for t in genuine_texts[:20]])
        sentiment = 'positive' if avg >= 0.05 else 'negative' if avg <= -0.05 else 'neutral'
        
    # Generate requested summary sentence
    summary = f"Total of {total} reviews processed. Overall sentiment of genuine reviews is {sentiment}."
    
    # Calculate confidence proxy (average confidence of fake detector predictions)
    avg_conf = np.mean([r['confidence'] for r in results]) if results else 0.0

    major_advantage = insights['advantages'][0] if insights['advantages'] else "No significant advantages extracted."
    major_disadvantage = insights['disadvantages'][0] if insights['disadvantages'] else "No significant disadvantages extracted."

    recommendation = "Insufficient Data"
    if insights['advantages'] or insights['disadvantages']:
        if len(insights['advantages']) > len(insights['disadvantages']):
            recommendation = "Recommended to Buy"
        else:
            recommendation = "Consider Alternatives"

    return {
        'advantages':         insights['advantages'],
        'disadvantages':      insights['disadvantages'],
        'major_advantage':    major_advantage,
        'major_disadvantage': major_disadvantage,
        'recommendation':     recommendation,
        'trend':              trend_data,
        'trend_summary':      trend_summary,
        'confidence':         round(avg_conf, 1),
        'summary':            summary,
        
        # Keep these original keys around so dashboard features (Authenticity rings, tables) still function
        'total':              total,
        'fake_count':         fake_count,
        'genuine_count':      genuine_count,
        'fake_percentage':    fake_pct,
        'genuine_percentage': genuine_pct,
        'overall_sentiment':  sentiment,
        'genuine_samples':    [r['text'] for r in results if r['prediction'] == 'genuine'][:5],
        'reviews':            results,
    }


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    user = None
    if 'user_id' in session:
        user = User.query.get(session['user_id'])
    return render_template('index.html', user=user)


@app.route('/predict', methods=['POST'])
def predict():
    """Accept form POST → analyse → save to DB → redirect to /results/<id>."""
    try:
        load_model()
    except FileNotFoundError as e:
        flash(str(e), 'error')
        return redirect(url_for('index'))

    input_type = request.form.get('input_type', 'text')
    reviews, source_df = [], None

    try:
        # ── Text ──────────────────────────────────────────────────────────
        if input_type == 'text':
            raw = request.form.get('review_text', '').strip()
            if len(raw) < 10:
                flash('Please enter at least one review (minimum 10 characters).', 'error')
                return redirect(url_for('index'))
            lines = [l.strip() for l in raw.split('\n') if len(l.strip()) > 10]
            reviews = lines if lines else [raw]

        # ── CSV ───────────────────────────────────────────────────────────
        elif input_type == 'csv':
            f = request.files.get('csv_file')
            if not f or f.filename == '':
                flash('Please select a CSV file.', 'error')
                return redirect(url_for('index'))
            if not f.filename.lower().endswith('.csv'):
                flash('File must be a .csv', 'error')
                return redirect(url_for('index'))
            try:
                source_df = pd.read_csv(f)
                source_df.columns = [c.strip().lower() for c in source_df.columns]
            except Exception as e:
                flash(f'Cannot parse CSV: {e}', 'error')
                return redirect(url_for('index'))

            rc = next((c for c in source_df.columns
                       if c in ('review', 'reviews', 'text', 'comment', 'body')), None)
            if rc is None:
                text_cols = source_df.select_dtypes(include='object').columns.tolist()
                if not text_cols:
                    flash('No text column found in CSV. Expected column: review', 'error')
                    return redirect(url_for('index'))
                rc = text_cols[0]

            reviews = source_df[rc].dropna().astype(str).tolist()
            if not reviews:
                flash('CSV has no review content.', 'error')
                return redirect(url_for('index'))

        # ── URL ───────────────────────────────────────────────────────────
        elif input_type == 'url':
            url = request.form.get('review_url', '').strip()
            if not url:
                flash('Please enter a URL.', 'error')
                return redirect(url_for('index'))
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            try:
                reviews = scrape_reviews_from_url(url)
            except Exception as e:
                flash(f'URL scraping failed: {e}', 'error')
                return redirect(url_for('index'))
            if not reviews:
                flash('No reviews found on that page. The site may use JS rendering.', 'error')
                return redirect(url_for('index'))

        else:
            flash('Unknown input type.', 'error')
            return redirect(url_for('index'))

        # ── Pipeline ──────────────────────────────────────────────────────
        result = process_reviews(reviews, source_df=source_df)
        if 'error' in result:
            flash(result['error'], 'error')
            return redirect(url_for('index'))

        # ── Persist ───────────────────────────────────────────────────────
        analysis = Analysis(
            user_id       = session.get('user_id'),
            input_type    = input_type,
            total         = result['total'],
            fake_count    = result['fake_count'],
            genuine_count = result['genuine_count'],
            pros_json     = json.dumps(result['advantages']),
            cons_json     = json.dumps(result['disadvantages']),
            result_json   = json.dumps(result),
        )
        db.session.add(analysis)
        db.session.flush()

        for rev in result['reviews'][:150]:
            db.session.add(ReviewRecord(
                analysis_id = analysis.id,
                review_text = rev['text'][:500],
                prediction  = rev['prediction'],
                confidence  = rev['confidence'],
            ))

        db.session.commit()
        return redirect(url_for('results_page', analysis_id=analysis.id))

    except Exception as e:
        db.session.rollback()
        logger.error(traceback.format_exc())
        flash(f'Analysis failed: {str(e)}', 'error')
        return redirect(url_for('index'))


@app.route('/results/<int:analysis_id>')
def results_page(analysis_id):
    """Render the dedicated results dashboard page."""
    analysis = Analysis.query.get_or_404(analysis_id)
    result   = json.loads(analysis.result_json or '{}')
    return render_template('results.html', result=result, analysis=analysis)


# ─── Auth Routes ─────────────────────────────────────────────────────────────
@app.route('/register', methods=['POST'])
def register():
    data     = request.get_json() or {}
    username = data.get('username', '').strip()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not all([username, email, password]):
        return jsonify({'error': 'All fields required'}), 400
    if len(password) < 6:
        return jsonify({'error': 'Password must be ≥ 6 characters'}), 400
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return jsonify({'error': 'Invalid email'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'Username already taken'}), 409
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    u = User(username=username, email=email,
              password_hash=generate_password_hash(password))
    db.session.add(u)
    db.session.commit()
    session['user_id']  = u.id
    session['username'] = u.username
    return jsonify({'success': True, 'username': u.username})


@app.route('/login', methods=['POST'])
def login():
    data     = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')

    if not all([username, password]):
        return jsonify({'error': 'Username and password required'}), 400
    u = User.query.filter_by(username=username).first()
    if not u or not check_password_hash(u.password_hash, password):
        return jsonify({'error': 'Invalid credentials'}), 401

    session['user_id']  = u.id
    session['username'] = u.username
    return jsonify({'success': True, 'username': u.username})


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


@app.route('/history')
def history():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    rows = (Analysis.query
            .filter_by(user_id=session['user_id'])
            .order_by(Analysis.created_at.desc())
            .limit(20).all())
    return jsonify([{
        'id':           r.id,
        'input_type':   r.input_type,
        'total':        r.total,
        'fake_count':   r.fake_count,
        'genuine_count': r.genuine_count,
        'created_at':   r.created_at.strftime('%Y-%m-%d %H:%M'),
    } for r in rows])


@app.route('/session-status')
def session_status():
    if 'user_id' in session:
        return jsonify({'logged_in': True, 'username': session.get('username')})
    return jsonify({'logged_in': False})


@app.route('/reviews/<int:analysis_id>')
def reviews_page(analysis_id):
    analysis = Analysis.query.get(analysis_id)
    if not analysis:
        flash('Analysis not found.', 'error')
        return redirect(url_for('index'))
    return render_template('reviews.html', analysis=analysis)


# ─── Startup ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Add result_json column if upgrading from older schema
        try:
            db.session.execute(
                sql_text('ALTER TABLE analyses ADD COLUMN result_json TEXT')
            )
            db.session.commit()
            logger.info("✓ Migrated: added result_json column")
        except Exception:
            pass   # column already exists
        logger.info("✓ Database ready")
        try:
            load_model()
        except FileNotFoundError:
            logger.warning("⚠  Model not found — run: python train_model.py")

    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
