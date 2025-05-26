FROM jupyter/datascience-notebook

WORKDIR /home/jovyan/work

COPY requirements.txt /tmp/

# Install system dependencies including poppler-utils for pdftocairo (needed for Qiskit LaTeX circuit drawings)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}

RUN pip install --no-cache-dir -r /tmp/requirements.txt
RUN pip install -U notebook-as-pdf
RUN pyppeteer-install

# Download spaCy Spanish language model and NLTK resources
# RUN python -m spacy download es_core_news_sm
# RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

