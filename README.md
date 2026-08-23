# AI-Powered Fake News Detection — TruthScanML

An internship-ready fake-news detection and news-verification project that preserves a FastAPI reference API and adds a user-facing Streamlit application. Both interfaces use the same saved TF-IDF vectorizer and Logistic Regression model.

> **Educational decision-support tool:** TruthScanML is not a substitute for professional fact-checking, official statements, or credible primary sources. Online verification results are evidence-based signals and may be inconclusive when reliable evidence is insufficient.

## Features

- **Offline Prediction** — classify pasted news text locally with the saved TF-IDF + Logistic Regression pipeline.
- **Online Verification** — searches multiple free news sources, filters relevant results, extracts readable articles in parallel, evaluates stance, credibility and freshness, applies special contradiction rules, optionally checks professional fact-checks, and returns **Real / Fake / Inconclusive**.
- **Local NLI Assist** — compares claims with evidence passages using a local Natural Language Inference model. NLI is a supporting signal, not the final judge.
- **Google Fact Check integration (optional)** — queries the ClaimReview database when a key is configured.
- **Wikipedia death-claim cross-check** — conservative extra evidence for death-related claims.
- **Groq AI second opinion (optional)** — an explicitly separate AI cross-check through Groq's free tier; it does not feed the final weighted verdict.
- **Hindi / Hinglish support** — recognises terms such as `mar gaya`, `wafat`, `inteqal`, `dehant`, `girftar`, `istifa`, `pabandi`, `jhooth`, `farzi`, and related patterns.
- **INCONCLUSIVE-first design** — weak, conflicting, or insufficient evidence does not get forcibly labelled Real or Fake.
- **Prediction History** — compact local SQLite history.
- **FastAPI reference API** — retained for backend demonstration.
- **Premium Liquid Glass UI** — shared animated glass theme with translucent surfaces, layered highlights, refraction-inspired effects, verdict styling, and light/dark support.

## Project structure

```text
api/                              # FastAPI reference implementation
model/                            # Saved model + TF-IDF vectorizer
streamlit_app/
├── app.py                        # Streamlit home page
├── pages/                        # Offline, Online, History, About
├── services/
│   ├── online_verification.py    # Search, stance, evidence, consensus
│   ├── factcheck.py              # Optional Google Fact Check / ClaimReview
│   ├── wikipedia.py              # Conservative death cross-check
│   ├── nli_stance.py             # Local NLI assist
│   ├── groq_crosscheck.py        # Optional Groq second opinion
│   └── relevance.py              # Claim/source relevance
├── ui/
│   └── theme.py                  # Shared Liquid Glass theme
├── utils/                        # Text processing + HTTP retry
├── database/                     # SQLite history helper
└── data/                         # Generated local DB (ignored by Git)
```

## Run locally

```cmd
cd C:\Users\Harsh Tiwari\Desktop\TC
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m streamlit run streamlit_app\app.py
```

Or use the convenience launcher:

```cmd
run_app.bat
```

## FastAPI reference

From the repository root:

```cmd
venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Endpoint:

```text
POST /predict
```

Example body:

```json
{"text": "News text to analyze"}
```

## Online Verification architecture

```text
                         NEWS CLAIM
                              │
             ┌────────────────▼────────────────┐
             │ Search layer (free / keyless)  │
             │ DDG → Bing News → Google News   │
             │ → GDELT                          │
             └────────────────┬────────────────┘
                              ▼
                   Relevant source filtering
                  (entity + event relevance)
                              │
             ┌────────────────▼────────────────┐
             │ Article analysis                │
             │ • parallel fetching (6 threads) │
             │ • extraction fallback chain     │
             │ • retry + backoff               │
             │ • extraction cache              │
             └────────────────┬────────────────┘
                              ▼
             ┌─────────────────────────────────┐
             │ Evidence / stance engine        │
             │ • SUPPORTS / CONTRADICTS /      │
             │   NEUTRAL                        │
             │ • negation guard                │
             │ • credibility                    │
             │ • freshness                      │
             │ • death/location/science rules  │
             │ • Hindi/Hinglish patterns        │
             └────────────────┬────────────────┘
                              ▼
             ┌─────────────────────────────────┐
             │ Independent cross-checks        │
             │ • Google Fact Check (optional)  │
             │ • Wikipedia (death claims)      │
             └────────────────┬────────────────┘
                              ▼
                     Local NLI assist (optional)
                              │
                              ▼
                     Weighted final consensus
                       ↙       ↓        ↘
                    REAL     FAKE   INCONCLUSIVE
                              │
                              └──── Optional Groq
                                    second opinion
                                    (separate from verdict)
```

### Search layer

Four free sources are used:

```text
DuckDuckGo → Bing News RSS → Google News RSS → GDELT
```

The implementation can rotate the preferred provider and falls back to the remaining providers when a source fails or returns no useful results. Duplicate URLs are removed and claim/search caching reduces repeated requests.

### Relevance filter

The relevance layer evaluates:

- entity match — who/what the claim is about
- event match — what happened (death, arrest, resignation, launch, ban, etc.)
- Hindi/Hinglish variants
- claim-type-specific thresholds
- stronger matching when entity + event alignment is very strong

### Article analysis

- up to 6 articles can be fetched in parallel
- readable text extraction follows a fallback chain (`<article>` → `<main>` → paragraph extraction)
- failed pages are retried with backoff and then skipped
- article extraction is cached
- blocked/inaccessible sources do not automatically invalidate the remaining evidence

### Stance engine

Each relevant source receives:

```text
SUPPORTS
CONTRADICTS
NEUTRAL
```

Checks include location contradictions, conservative death-claim rules, negation handling, science-specific rules, debunk/confirmation patterns, title equivalence, trusted coverage, and optional local NLI assistance.

### Weighted evidence

Typical credibility tiers:

| Source type | Weight |
|---|---:|
| Professional fact-checkers | 2.0× |
| Trusted mainstream / specialist outlets | 1.5× |
| Normal sources | 1.0× |
| Repeated same-domain source | heavily downweighted |
| Known satire domains | excluded |

Freshness reduces the influence of older reporting when a current claim requires current evidence.

### NLI assist

The local NLI model compares a claim with an evidence passage:

```text
ENTAILMENT
CONTRADICTION
NEUTRAL
```

Raw logits are converted to proper probabilities before display/thresholding so confidence remains in the valid `0–100%` range. NLI is only an additional signal and does not own the final verdict.

### Final consensus

```text
Strong support
    + sufficient independent stance evidence
            → REAL

Strong contradiction
    + sufficient independent stance evidence
            → FAKE

Weak / conflicting / insufficient evidence
            → INCONCLUSIVE
```

Snippet-only/fast verification uses a lower confidence ceiling because full article content was not analysed.

## Optional: Google Fact Check Tools

When configured, the app queries Google's Fact Check Tools / ClaimReview database.

Create local secrets:

```cmd
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
```

Then add:

```toml
FACT_CHECK_API_KEY = "your-key"
```

For Streamlit Community Cloud, add the same secret under **Settings → Secrets**.

**Never commit real API keys to GitHub, source files, screenshots, or chats.**

Without a key, the rest of the verification pipeline still works.

## Optional: Groq AI second opinion

After the normal evidence verdict, the user can optionally run an independent AI cross-check through Groq's free tier.

Default model:

```toml
GROQ_API_KEY = "gsk-..."
GROQ_MODEL = "openai/gpt-oss-120b"
```

> **Groq is a second opinion only. It does not change the weighted TruthScanML verdict.**

A short-lived cache reduces repeated calls for the same claim.

## Wikipedia cross-check

For death-related claims:

- an explicit death statement can support the claim
- a clear present-tense biography can provide contradiction evidence
- missing/ambiguous information remains neutral

Wikipedia is only one evidence source and is never treated as sole proof.

## SQLite history

The local database is created at:

```text
streamlit_app/data/prediction_history.db
```

It stores compact local records such as time, mode, input preview, prediction/verdict, confidence, and online-source counts.

The database is ignored by Git. Streamlit Cloud storage is ephemeral, so history may reset after restart/redeploy.

## Deploy to Streamlit Community Cloud

1. Commit and push `main` to GitHub.
2. Create/select the app in Streamlit Community Cloud.
3. Use branch `main`.
4. Set entrypoint to:

```text
streamlit_app/app.py
```

5. Use the Python version compatible with the tested `requirements.txt`/deployment environment.
6. Add optional secrets under **Settings → Secrets**:

```toml
FACT_CHECK_API_KEY = "..."
GROQ_API_KEY = "..."
GROQ_MODEL = "openai/gpt-oss-120b"
```

7. Deploy and inspect build/runtime logs.

### Deployment notes

- The NLI model may make the first cloud verification slower while model artifacts load.
- Search providers may rate-limit or block datacenter traffic.
- Individual article sites may block automated extraction.
- Such failures are treated as partial evidence loss where possible.
- If online verification fails, Offline Prediction remains independent.

## Security

- Never commit `.streamlit/secrets.toml`.
- Never place real keys in `secrets.example.toml`.
- Rotate keys that were accidentally exposed in chats, screenshots, logs, or repositories.
- Keep optional external AI services separate from the core evidence pipeline.

## Model integrity

The Streamlit app loads these saved artifacts in read-only mode:

```text
model/fake_news_model.pkl
model/tfidf_vectorizer.pkl
```

It does not retrain, serialize, or overwrite them. The FastAPI reference API uses the same saved model/vectorizer.

## Verification philosophy

TruthScanML is designed to answer:

> **“What evidence can I find, and how strong is it?”**

rather than:

> **“Can one model declare this headline true or false?”**

That is why the system combines multiple search providers, relevance filtering, article-level evidence, credibility and freshness weighting, professional fact-check lookup, conservative death-claim checks, NLI as an assistive signal, optional Groq cross-checking, and a first-class **INCONCLUSIVE** result.

The goal is a transparent, explainable and practical news-verification workflow suitable for internship demonstration, academic viva, and further development.
