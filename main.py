from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from io import StringIO
import os

# Define model name
model_name = "DiKshansHAgrawAl12/IntentClassification"

# Load model and tokenizer once at startup
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create pipeline
nlp_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], # Adjust to your frontend URL (Vite default)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/predict")
def predict_text(question: str = Query(..., description="Text input for prediction")):
    try:
        result = nlp_pipeline(question)
        predicted_label = result[0]["label"]
        return {"input_text": question, "prediction": predicted_label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Allowed file extensions
ALLOWED_EXTENSIONS = {"csv", "json", "txt"}

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        # Check file extension
        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type. Only CSV, JSON, and TXT files are allowed.")
        
        # Read file content
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode("utf-8")))
        
        if "text" not in df.columns:
            raise HTTPException(status_code=400, detail="CSV file must contain a 'text' column")
        
        # Perform prediction on each row
        predictions = []
        for text in df["text"].dropna():
            result = nlp_pipeline(text)
            predictions.append({"input_text": text, "prediction": result[0]["label"]})
        
        return {"results": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))