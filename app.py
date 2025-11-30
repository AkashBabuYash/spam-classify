from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sklearn.preprocessing import normalize

# Load model and vectorizer
model = joblib.load("spam_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

app = FastAPI()

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    text = request.text
    input_vec = vectorizer.transform([text])
    input_scaled = normalize(input_vec, norm='l2')
    prediction = model.predict(input_scaled)[0]
    result = "Ham (Not Spam)" if prediction == 0 else "Spam"
    return {"prediction": result}
