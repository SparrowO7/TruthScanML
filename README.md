# AI-Powered Fake News Detection

An internship-ready fake-news detection project that preserves a FastAPI
reference API and adds a user-facing Streamlit application. Both interfaces
use the same saved TF-IDF vectorizer and Logistic Regression model.

> This is an educational decision-support tool. A model prediction is not a
> substitute for professional fact-checking or credible primary sources.

## Features

- **Offline Prediction** — paste news text and classify it locally as Real or
  Fake using the saved ML pipeline.
- **Online Verification** — enter a headline; the app searches free news
  sources, extracts readable articles, weighs evidence, and returns a
  **Real / Fake / Inconclusive** consensus. Fully free — no paid APIs and no
  generative-AI verdicts.
- **Prediction History** — stores compact local records in SQLite; no database
  server required.
- **FastAPI reference API** — retained unchanged for backend demonstration.

## Project structure

```text
api/                         # Existing FastAPI reference implementation
model/                       # Existing saved model and TF-IDF vectorizer
streamlit_app/
├── app.py                   # Streamlit home page
├── pages/                   # Offline, Online, History, About pages
├── services/                # ML inference, search, evidence & consensus engine
│   ├── online_verification.py   # Search chain + stance + consensus
│   ├── factcheck.py             # Optional Google Fact Check (ClaimReview)
│   ├── wikipedia.py             # Conservative death-claim cross-check
│   └── relevance.py             # Claim↔title relevance scoring
├── ui/                      # Shared animated glass theme
├── utils/                   # Matching text preprocessing + HTTP retry
├── database/                # SQLite history helper
└── data/                    # Generated local database (ignored by Git)
```

## Run locally

Use the existing project virtual environment:

```cmd
cd C:\Users\Harsh Tiwari\Desktop\TC
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

Online Verification is completely free: it uses no LLM, no paid API, and no
API key by default.

```text
                 NEWS CLAIM
                     │
        ┌────────────▼──────────────────┐
        │ Search Layer (keyless)        │
        │ DDG → Bing → Google → GDELT   │
        └────────────┬──────────────────┘
                     ▼
             Relevant Articles (entity/event relevance filter)
                     │
        ┌────────────▼─────────────┐
        │ Evidence Engine          │
        │ • Stance (SUPPORTS/      │
        │   CONTRADICTS/NEUTRAL)   │
        │ • Negation guard         │
        │ • Credibility weighting  │
        │ • Freshness scoring      │
        │ • Parallel fetching      │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ Fact Check Database      │
        │ (ClaimReview, optional)  │
        └────────────┬─────────────┘
                     ▼
        ┌──────────────────────────┐
        │ Special Checks           │
        │ • Death (incl. Wikipedia)│
        │ • Location contradiction │
        │ • Scientific claims      │
        │ • Hindi/Hinglish terms   │
        └────────────┬─────────────┘
                     ▼
              FINAL CONSENSUS
          ↙        ↓         ↘
       REAL      FAKE     INCONCLUSIVE
```

The system does not depend on a generative AI for its verdict. It searches
multiple independent sources, extracts evidence, evaluates source credibility,
freshness and stance, optionally consults existing professional fact-checks,
and then produces a weighted consensus. If evidence is insufficient or
conflicting, it returns **Inconclusive** instead of forcing a call.

Search providers and news websites can rate-limit requests, block automation,
or return inaccessible pages. Online results are therefore supporting signals,
not factual proof. Offline Prediction remains independent of internet access.

### Optional: Google Fact Check Tools (free)

Professional fact-check verdicts (Alt News, BOOM, Vishvas News, PolitiFact,
Snopes, …) are queried through Google's free Fact Check Tools API. This layer
is optional: without a key the app skips it and everything else still works.

1. Create a free API key: <https://developers.google.com/fact-check/tools/api>
2. Copy the tracked example file to the ignored secrets file:

   ```cmd
   copy .streamlit\secrets.example.toml .streamlit\secrets.toml
   ```

3. Edit `.streamlit/secrets.toml`:

   ```toml
   FACT_CHECK_API_KEY = "your-free-key"
   ```

For Streamlit Community Cloud, open the app's **Settings → Secrets** and add
the same line. Never commit or paste real keys into source code, GitHub,
screenshots, or chats.

### Wikipedia cross-check (automatic)

For death claims ("X died / X mar gaye / X ka dehant"), the app performs one
conservative Wikipedia check: an explicit death statement in the lead supports
the claim, a present-tense biography contradicts it, and anything else stays
neutral. Wikipedia is only ever one evidence source.

## SQLite history

SQLite is included with Python. The database is automatically created at:

```text
streamlit_app/data/prediction_history.db
```

It stores time, mode, compact input preview, prediction, confidence, and
online-source counts. It is ignored by Git and remains local to the machine or
deployment instance running the app.

## Deploy to Streamlit Community Cloud

1. Commit and push the `main` branch to GitHub.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and choose
   **Create app**.
3. Select the GitHub repository and branch.
4. Set the entrypoint path to `streamlit_app/app.py`.
5. In Advanced settings, choose Python 3.13 to match local development.
6. Deploy and inspect the build logs.

`requirements.txt` is at repository root and `.streamlit/config.toml` provides
the dark glass app theme. Streamlit Cloud's filesystem is ephemeral, so SQLite
history may reset on redeploy or restart; that is expected for this
local-history feature.

## Model integrity

The Streamlit app loads the existing files below in read-only mode. It never
re-trains, serializes, or overwrites them:

- `model/fake_news_model.pkl`
- `model/tfidf_vectorizer.pkl`
