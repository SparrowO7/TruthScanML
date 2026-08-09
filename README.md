# AI-Powered Fake News Detection

An internship-ready fake-news detection project that preserves a FastAPI
reference API and adds a user-facing Streamlit application. Both interfaces
use the same saved TF-IDF vectorizer and Logistic Regression model.

> This is an educational decision-support tool. A model prediction is not a
> substitute for professional fact-checking or credible primary sources.

## Features

- **Offline Prediction** — paste news text and classify it locally as Real or
  Fake using the saved ML pipeline.
- **Online Verification** — enter a headline; the app searches public news
  results, extracts readable articles, and reports source-model consensus.
- **Prediction History** — stores compact local records in SQLite; no database
  server or API key is required.
- **FastAPI reference API** — retained unchanged for backend demonstration.

## Project structure

```text
api/                         # Existing FastAPI reference implementation
model/                       # Existing saved model and TF-IDF vectorizer
streamlit_app/
├── app.py                   # Streamlit home page
├── pages/                   # Offline, Online, History, About pages
├── services/                # Read-only ML inference and online verification
├── utils/                   # Matching text preprocessing
├── database/                # SQLite history helper
└── data/                    # Generated local database (ignored by Git)
```

## Run locally

Use the existing project virtual environment:

```cmd
cd C:\Users\Harsh Tiwari\TestClone
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run streamlit_app\app.py
```

The Streamlit application opens in the browser. Restart it after dependency or
source changes.

## FastAPI reference

The FastAPI code remains available and is not required for Streamlit to work.
From the repository root:

```cmd
venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

The prediction endpoint is `POST /predict` with this body:

```json
{"text": "News text to analyze"}
```

## Online-verification design

Online Verification does not use an LLM, paid API, or API key:

```text
Headline → public news search → article extraction → existing ML model
```

Search providers and news websites can rate-limit requests, block automation,
or return inaccessible pages. Therefore online results are supporting signals,
not factual proof. Offline Prediction remains independent of internet access.

## SQLite history

SQLite is included with Python. The database is automatically created at:

```text
streamlit_app/data/prediction_history.db
```

It stores time, mode, compact input preview, prediction, confidence, and
online-source counts. It is ignored by Git and remains local to the machine or
deployment instance running the app.

## Deploy to Streamlit Community Cloud

1. Commit and push the `streamlit-upgrade` branch to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose
   **Create app**.
3. Select the GitHub repository and branch.
4. Set the entrypoint path to `streamlit_app/app.py`.
5. In Advanced settings, choose Python 3.13 to match local development.
6. Deploy and inspect the build logs.

`requirements.txt` is at repository root and `.streamlit/config.toml` provides
the app theme. Streamlit Cloud's filesystem is ephemeral, so SQLite history may
reset on redeploy or restart; that is expected for this local-history feature.

## Model integrity

The Streamlit app loads the existing files below in read-only mode. It never
re-trains, serializes, or overwrites them:

- `model/fake_news_model.pkl`
- `model/tfidf_vectorizer.pkl`
