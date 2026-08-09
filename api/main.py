from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

# Load model using relative paths (no user folder issues)
model = joblib.load("./model/fake_news_model.pkl")
vectorizer = joblib.load("./model/tfidf_vectorizer.pkl")

app = FastAPI()

# Text cleaner (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class InputText(BaseModel):
    text: str

@app.post("/predict")
def predict(input: InputText):
    cleaned = clean_text(input.text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    result = "FAKE NEWS ❌" if pred == 1 else "REAL NEWS ✔️"
    return {"result": result}
