# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y build-essential git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
# Install core dependencies first to speed up caching
RUN pip install pandas numpy streamlit seaborn matplotlib urlextract wordcloud emoji nltk chardet scikit-learn

# Optional heavy deps
RUN pip install transformers torch sentence-transformers bertopic shap

EXPOSE 8501
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.headless=true"]
