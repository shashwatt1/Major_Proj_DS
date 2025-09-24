import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import helper
import preprocessor
import os
import re
from datetime import datetime

st.set_page_config(page_title="WhatsApp Chat Analyzer (Major Project)", layout="wide")

@st.cache_data(show_spinner=False)
def load_stopwords(path="stop_hinglish.txt"):
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8', errors='replace') as fh:
        words = [w.strip() for w in fh if w.strip()]
    return set(words)

@st.cache_data(show_spinner=False)
def preprocess_file(file_bytes):
    if isinstance(file_bytes, bytes):
        content = file_bytes.decode('utf-8', errors='replace')
        df = preprocessor.parse_chat_from_lines(content.splitlines())
    else:
        df = preprocessor.parse_chat_from_file(file_bytes)
    return df

SAMPLE_TEXT = """12/07/2024, 2:02 PM - Alice: Hey! How are you?
12/07/2024, 2:03 PM - Bob: I'm good :) What about you?
12/07/2024, 2:05 PM - Alice: Doing great! Let's meet tomorrow.
12/07/2024, 2:06 PM - Bob: Sure. <Media omitted>
12/07/2024, 2:07 PM - System: Alice added Charlie
12/07/2024, 2:08 PM - Charlie: Hi everyone!"""

def show_overview(df):
    st.subheader("Overview")
    col1, col2, col3, col4 = st.columns(4)
    total_msgs = len(df)
    total_users = df['user'].nunique()
    total_media = df['media_type'].notna().sum()
    total_links = df['link_count'].sum()
    col1.metric("Messages", total_msgs)
    col2.metric("Users", total_users)
    col3.metric("Media Items", total_media)
    col4.metric("Links", int(total_links))

def show_activity_heatmap(df):
    st.subheader("Activity Heatmap (Hour vs Weekday)")
    pivot = df.groupby(['weekday','hour']).size().reset_index(name='count')
    weekdays = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    heat = pd.DataFrame(0, index=weekdays, columns=range(24))
    for _, row in pivot.iterrows():
        w = row['weekday']
        h = int(row['hour'])
        if w in heat.index:
            heat.at[w,h] = row['count']
    fig, ax = plt.subplots(figsize=(12,4))
    sns.heatmap(heat, ax=ax)
    st.pyplot(fig)

def show_sentiment_timeline(df, method='vader'):
    st.subheader("Sentiment Timeline")
    if method == 'vader':
        if 'vader_compound' not in df.columns:
            df = helper.add_sentiment_columns(df, method='vader')
    elif method == 'transformer':
        if not helper.HAS_TRANSFORMERS:
            st.warning("Transformer not available in this environment.")
            return
        if 'transformer_label' not in df.columns:
            df = helper.add_sentiment_columns(df, method='transformer')
    df_daily = df.copy()
    df_daily['date'] = pd.to_datetime(df_daily['date'])
    if method == 'vader':
        df_daily = df_daily.groupby('date')['vader_compound'].mean().reset_index()
        fig, ax = plt.subplots()
        ax.plot(df_daily['date'], df_daily['vader_compound'])
        ax.set_ylabel("Avg VADER Compound")
        st.pyplot(fig)
    else:
        df_daily['trans_num'] = df_daily['transformer_label'].map({'POSITIVE':1,'NEGATIVE':-1,'NEUTRAL':0})
        df_daily = df_daily.groupby('date')['trans_num'].mean().reset_index()
        fig, ax = plt.subplots()
        ax.plot(df_daily['date'], df_daily['trans_num'])
        ax.set_ylabel("Avg Transformer Sentiment (num)")
        st.pyplot(fig)

def show_wordcloud(df, stopwords):
    st.subheader("Wordcloud")
    filtered_df = df.copy()
    if 'is_system_message' in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df['is_system_message']]
    if 'media_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['media_type'].isna()]
    all_text = " ".join(filtered_df['message'].astype(str).tolist())
    junk_list = ["omitted", "deleted", "sticker", "image", "video", "audio", "message","gif", "file", "document", "poll", "option", "vote"]
    for junk in junk_list:
        pattern = r'\b' + re.escape(junk) + r'\b'
        all_text = re.sub(pattern, ' ', all_text, flags=re.IGNORECASE)
    all_text = re.sub(r'[^\x00-\x7F]+', ' ', all_text)
    for ch in ["â", "ÿ", "©", "ð", "â€", "â€\x9d"]:
        all_text = all_text.replace(ch, " ")
    all_text = all_text.lower()
    all_text = re.sub(r'\s+', ' ', all_text).strip()
    wc = helper.generate_wordcloud(all_text, stopwords=stopwords)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def show_top_emojis(df):
    st.subheader("Top Emojis")
    emo_df = helper.top_emojis(df, n=20)
    if emo_df.empty:
        st.info("No emojis found.")
        return
    st.dataframe(emo_df)

def show_topic_modeling(df):
    st.subheader("Topic Modeling (BERTopic)")
    if not helper.HAS_BERTOPIC:
        st.warning("BERTopic / sentence-transformers not installed. Install them to use topic modeling.")
        return
    docs = df['message'].astype(str).tolist()
    with st.spinner("Fitting BERTopic..."):
        try:
            topic_model, topics, info = helper.topic_modeling_bertopic(docs)
            st.write(info.head(20))
            for tid in info['Topic'].unique()[:10]:
                if tid == -1: continue
                t = topic_model.get_topic(tid)
                st.markdown(f"**Topic {tid}**: " + ", ".join([w for w,_ in t]))
        except Exception as e:
            st.error(f"BERTopic error: {e}")

def main():
    st.title("WhatsApp Chat Analyzer — Major Project Upgrade")
    st.markdown("Upload your WhatsApp exported `.txt` file. This upgraded version includes robust parsing, PII redaction, transformer sentiment (optional), topic modeling (optional), and deployment readiness.")
    sidebar = st.sidebar
    sidebar.header("Options")
    sentiment_method = sidebar.selectbox("Sentiment method", options=['vader', 'transformer', 'both'])
    stop_choice = sidebar.selectbox("Stopwords set", options=['stop_hinglish.txt', 'none'])
    date_filter = sidebar.checkbox("Enable date filter", value=False)
    sample_data_btn = sidebar.button("Load sample data")
    stopwords = set()
    if stop_choice != 'none':
        stopwords = load_stopwords(stop_choice)
    uploaded_file = st.file_uploader("Upload WhatsApp chat export (txt)", type=['txt'])
    df = None
    if sample_data_btn:
        df = preprocessor.parse_chat_from_lines(SAMPLE_TEXT.splitlines())
        st.success("Sample data loaded.")
    elif uploaded_file is not None:
        try:
            df = preprocess_file(uploaded_file.read())
            st.success("File parsed successfully.")
        except Exception as e:
            st.error(f"Failed to parse file: {e}")
            st.stop()
    else:
        st.info("Upload a WhatsApp `.txt` export or click 'Load sample data' to continue.")
        st.stop()
    if date_filter:
        min_date = pd.to_datetime(df['date'].min())
        max_date = pd.to_datetime(df['date'].max())
        dr = st.date_input("Select date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if isinstance(dr, (list, tuple)) and len(dr) == 2:
            df = df[(pd.to_datetime(df['date']) >= pd.to_datetime(dr[0])) & (pd.to_datetime(df['date']) <= pd.to_datetime(dr[1]))]
    if sentiment_method == 'vader':
        df = helper.add_sentiment_columns(df, method='vader')
    elif sentiment_method == 'transformer':
        if not helper.HAS_TRANSFORMERS:
            st.warning("Transformer not available. Falling back to VADER.")
            df = helper.add_sentiment_columns(df, method='vader')
        else:
            df = helper.add_sentiment_columns(df, method='transformer')
    else:
        df = helper.add_sentiment_columns(df, method='both')
    st.sidebar.markdown("### Export & Download")
    if st.sidebar.button("Download filtered CSV"):
        csv = df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button("Download CSV", csv, file_name="chat_filtered.csv", mime="text/csv")
    show_overview(df)
    st.markdown("---")
    left, right = st.columns([2,1])
    with left:
        show_activity_heatmap(df)
        st.markdown("---")
        show_sentiment_timeline(df, method='vader' if sentiment_method=='vader' else 'transformer')
        st.markdown("---")
        show_wordcloud(df, stopwords=stopwords)
    with right:
        st.markdown("### Users")
        st.dataframe(df.groupby('user').size().sort_values(ascending=False).reset_index(name='count'))
        st.markdown("---")
        show_top_emojis(df)
        st.markdown("---")
        st.markdown("### Topic Modeling")
        show_topic_modeling(df)
    st.markdown("---")
    st.subheader("Conversation Metrics")
    if 'response_time_seconds' in df.columns:
        rt = df['response_time_seconds'].dropna()
        if not rt.empty:
            st.write("Average response time (seconds):", int(rt.mean()))
            st.write("Median response time (seconds):", int(rt.median()))
        else:
            st.info("No response time data (maybe single-user or sequential messages)")
    st.markdown("## Raw Data (first 200 rows)")
    st.dataframe(df.head(200))

if __name__ == "__main__":
    main()
