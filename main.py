from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from io import StringIO
import os
import requests

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "https://your-frontend.vercel.app"  # Replace with your frontend URL after deployment
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hugging Face API configuration
API_URL = "https://api-inference.huggingface.co/models/DiKshansHAgrawAl12/IntentClassification"
API_TOKEN = "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"

def query_huggingface(text):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json().get("error", "API error"))
    result = response.json()
    # Adjust based on expected response structure (e.g., [{"label": "Order Inquiry", "score": 0.95}])
    return result[0]["label"] if isinstance(result, list) and result else "Unknown"

@app.get("/predict")
def predict_text(question: str = Query(..., description="Text input for prediction")):
    try:
        predicted_label = query_huggingface(question)
        return {"input_text": question, "prediction": predicted_label}
    except HTTPException as e:
        raise e
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
            predicted_label = query_huggingface(text)
            predictions.append({"input_text": text, "prediction": predicted_label})
        
        return {"results": predictions}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
