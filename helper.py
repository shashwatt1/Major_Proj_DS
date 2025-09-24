import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
import emoji
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

HAS_TRANSFORMERS = True
try:
    from transformers import pipeline
except Exception:
    HAS_TRANSFORMERS = False

HAS_BERTOPIC = True
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
except Exception:
    HAS_BERTOPIC = False

HAS_SHAP = True
try:
    import shap
except Exception:
    HAS_SHAP = False

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except Exception:
    try:
        nltk.download('vader_lexicon', quiet=True)
    except Exception:
        pass

sia = SentimentIntensityAnalyzer()
_transformer_pipe = None

def get_transformer_pipeline(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    global _transformer_pipe, HAS_TRANSFORMERS
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers not installed")
    if _transformer_pipe is None:
        _transformer_pipe = pipeline("sentiment-analysis", model=model_name)
    return _transformer_pipe

def vader_sentiment(msg: str):
    if not isinstance(msg, str):
        return {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}
    return sia.polarity_scores(msg)

def transformer_sentiment(msg: str):
    if not HAS_TRANSFORMERS:
        return {'label': 'TRANSFORMER_NOT_AVAILABLE', 'score': 0.0}
    pipe = get_transformer_pipeline()
    out = pipe(msg[:512])[0]
    return {'label': out['label'], 'score': float(out['score'])}

def add_sentiment_columns(df: pd.DataFrame, method='vader'):
    df = df.copy()
    if method == 'vader':
        df['vader_scores'] = df['message'].apply(lambda x: vader_sentiment(x))
        df['vader_compound'] = df['vader_scores'].apply(lambda x: x.get('compound', 0.0))
        df['vader_sentiment'] = df['vader_compound'].apply(lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral'))
    elif method == 'transformer':
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        df['transformer_pred'] = df['message'].apply(lambda x: transformer_sentiment(x))
        df['transformer_label'] = df['transformer_pred'].apply(lambda x: x.get('label'))
        df['transformer_score'] = df['transformer_pred'].apply(lambda x: x.get('score'))
    elif method == 'both':
        df = add_sentiment_columns(df, 'vader')
        try:
            df = add_sentiment_columns(df, 'transformer')
        except Exception:
            pass
    else:
        raise ValueError("method must be 'vader', 'transformer' or 'both'")
    return df

def compare_sentiment_methods(df: pd.DataFrame, label_col='label', sample_size=400, random_state=42):
    if df.shape[0] > sample_size:
        sample = df.sample(sample_size, random_state=random_state).reset_index(drop=True)
    else:
        sample = df.copy().reset_index(drop=True)
    y_true = sample[label_col].astype(str).str.lower().replace({'positive':'positive','negative':'negative','neutral':'neutral','pos':'positive','neg':'negative'})
    sample['vader_compound'] = sample['message'].apply(lambda x: sia.polarity_scores(x)['compound'])
    sample['vader_pred'] = sample['vader_compound'].apply(lambda x: 'positive' if x>0.05 else ('negative' if x<-0.05 else 'neutral'))
    results = {}
    vader_acc = accuracy_score(y_true, sample['vader_pred'])
    vader_prf = precision_recall_fscore_support(y_true, sample['vader_pred'], average='weighted', zero_division=0)
    results['vader'] = {'accuracy': vader_acc, 'precision': vader_prf[0], 'recall': vader_prf[1], 'f1': vader_prf[2]}
    if HAS_TRANSFORMERS:
        pipe = get_transformer_pipeline()
        def tpred(x):
            try:
                out = pipe(x[:512])[0]
                return out['label'].lower()
            except Exception:
                return 'neutral'
        sample['trans_pred'] = sample['message'].apply(tpred)
        trans_acc = accuracy_score(y_true, sample['trans_pred'])
        trans_prf = precision_recall_fscore_support(y_true, sample['trans_pred'], average='weighted', zero_division=0)
        results['transformer'] = {'accuracy': trans_acc, 'precision': trans_prf[0], 'recall': trans_prf[1], 'f1': trans_prf[2]}
    return results, sample

def generate_wordcloud(text, stopwords=None, width=800, height=400):
    if stopwords is None:
        stopwords = set()
    # normalize
    txt = text.lower()
    # remove non-ascii artifacts
    txt = re.sub(r'[^\x00-\x7F]+', ' ', txt)
    # replace punctuation with space
    txt = re.sub(r'[_\-\.\,\:\;\(\)\[\]\/\\\|\"\'\<\>\=\+\*\&\^\%\$]', ' ', txt)
    # split into tokens
    tokens = txt.split()
    # additional junk tokens to always remove
    junk_set = {
        "omitted","deleted","sticker","image","video","audio","media","file","document",
        "pdf","ppt","pptx","gif","attachment","attached","forwarded","forwarded_message",
        "forwarded_msg","group_created","added","removed","left","invite","invited","join",
        "joined","left_the_group","admin","participant","message","messages","http","https",
        "www","url","poll","option","vote","votes","reply","replied"
    }
    # build final tokens list
    final_tokens = []
    for t in tokens:
        if t in junk_set:
            continue
        if t in stopwords:
            continue
        if len(t) <= 2:
            continue
        if any(ch.isdigit() for ch in t):
            continue
        if t.startswith('@') or t.startswith('#') or t.startswith('http') or t.startswith('www'):
            continue
        # remove small stray punctuation leftover
        t = re.sub(r'^[^a-zA-Z]+|[^a-zA-Z]+$', '', t)
        if t == '' or len(t) <= 2:
            continue
        final_tokens.append(t)
    if not final_tokens:
        final_text = ""
    else:
        final_text = " ".join(final_tokens)
    wc = WordCloud(width=width, height=height, stopwords=set(), background_color='white').generate(final_text)
    return wc


def top_emojis(df: pd.DataFrame, n=20):
    all_emoji = []
    pattern = None
    try:
        pattern = emoji.get_emoji_regexp()
    except Exception:
        pattern = None
    for m in df['message'].astype(str):
        if pattern:
            all_emoji.extend(pattern.findall(m))
        else:
            try:
                data = getattr(emoji, "EMOJI_DATA", None)
                if isinstance(data, dict) and data:
                    all_emoji.extend([ch for ch in m if ch in data])
                    continue
            except Exception:
                pass
            try:
                uni = getattr(emoji, "UNICODE_EMOJI_ENGLISH", None) or getattr(emoji, "UNICODE_EMOJI", None)
                if isinstance(uni, dict):
                    all_emoji.extend([ch for ch in m if ch in uni])
                    continue
            except Exception:
                pass
            all_emoji.extend([ch for ch in m if ord(ch) > 10000])
    counts = Counter(all_emoji)
    top = counts.most_common(n)
    return pd.DataFrame(top, columns=['emoji', 'count'])


def topic_modeling_bertopic(documents, n_topics=None, embedding_model='all-MiniLM-L6-v2'):
    if not HAS_BERTOPIC:
        raise ImportError("BERTopic or sentence-transformers not installed")
    embed_model = SentenceTransformer(embedding_model)
    embeddings = embed_model.encode(documents, show_progress_bar=False)
    topic_model = BERTopic(verbose=False)
    topics, probs = topic_model.fit_transform(documents, embeddings)
    info = topic_model.get_topic_info()
    return topic_model, topics, info

def anomaly_detection_on_timeseries(series_vals, contamination=0.01):
    from sklearn.ensemble import IsolationForest
    vals = np.array(series_vals).reshape(-1,1)
    iso = IsolationForest(contamination=contamination, random_state=42)
    pred = iso.fit_predict(vals)
    return pred == -1

def shap_explain_transformer(message, model=None, tokenizer=None):
    if not HAS_SHAP:
        raise ImportError("shap not installed")
    raise NotImplementedError("Please call shap_explain_transformer with a model and tokenizer and adapt to your env.")
