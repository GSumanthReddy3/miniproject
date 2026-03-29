import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import json

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

vader = SentimentIntensityAnalyzer()
STOP_WORDS = set(nltk.corpus.stopwords.words('english'))

def extract_insights(reviews: list) -> dict:
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
            score = vader.polarity_scores(sent)['compound']
            if abs(score) < 0.15:
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
                        group = 'pro' if score >= 0.15 else 'con'
                        if w not in feature_adjectives[group]:
                            feature_adjectives[group][w] = Counter()
                        feature_adjectives[group][w][adj] += 1

    def generate_sentences(group_dict, limit=10, is_pro=True):
        templates = [
            "Users frequently noted that the {feature} is {adj}.",
            "The {feature} was commonly described as {adj}.",
            "Many reviewers found the {feature} to be {adj}.",
            "The {feature} is considered {adj}."
        ]
        if not is_pro:
            templates = [
                "Users frequently complained that the {feature} is {adj}.",
                "The {feature} was commonly described as {adj}.",
                "Many reviewers found the {feature} to be {adj}.",
                "Reviewers noted the {feature} is {adj}."
            ]
            
        sorted_features = sorted(group_dict.items(), key=lambda x: sum(x[1].values()), reverse=True)
        
        sentences = []
        for i, (feature, adjs) in enumerate(sorted_features):
            if not adjs: continue
            top_adjs = [a[0] for a in adjs.most_common(2)]
            adj_str = " and ".join(top_adjs)
            template = templates[i % len(templates)]
            sentences.append(template.format(feature=feature, adj=adj_str))
            if len(sentences) >= limit:
                break
        return sentences

    return {
        'advantages': generate_sentences(feature_adjectives['pro'], limit=10, is_pro=True),
        'disadvantages': generate_sentences(feature_adjectives['con'], limit=10, is_pro=False)
    }

test_reviews = [
    "The battery is extremely long-lasting and reliable.",
    "Display is very crisp and beautiful.",
    "The camera is terrible in low light.",
    "Phone feels very premium.",
    "Great screen, but the charging speed is slow."
]

print(json.dumps(extract_insights(test_reviews), indent=2))
