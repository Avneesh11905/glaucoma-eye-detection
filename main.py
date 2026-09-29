import cv2
import numpy as np
import pickle
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, HTTPException

from ml_features import preprocess_image, extract_one_image

# Global state for ML model components
ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model during startup
    model_path = Path(__file__).resolve().parent / 'glaucoma_final_model.pkl'
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
        
    ml['model'] = bundle['model']
    ml['scaler'] = bundle['scaler']
    ml['pca'] = bundle['pca']
    ml['le'] = bundle['label_encoder']
    ml['top_idx'] = bundle['top_feature_indices']
    
    yield  # Application is running
    
    # Clean up resources on shutdown
    ml.clear()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Glaucoma Detection API", lifespan=lifespan)

# Add CORS Middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://glaucoma.vkpatel.in", "http://localhost:5173"], # Added localhost for local dev testing
    allow_credentials=True,
    allow_methods=["*"], # Allows GET, POST, OPTIONS, etc.
    allow_headers=["*"], # Allows all headers
)

@app.get("/")
def health_check():
    """Health check endpoint required by most hosting platforms (Render/Railway)."""
    return {"status": "ok", "message": "Glaucoma Detection API is running"}

@app.post("/predict")
def predict_endpoint(file: UploadFile):
    # Read synchronously (file.file accesses the underlying standard python file object)
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file format")
        
    try:
        # Load and process
        img = cv2.resize(img, (224, 224))
        proc = preprocess_image(img)

        # Extract features
        feats = np.array(extract_one_image(proc)).reshape(1, -1)
        scaled = ml['scaler'].transform(feats)
        top = scaled[:, ml['top_idx']]
        reduced = ml['pca'].transform(top)

        # Predict
        pred = ml['model'].predict(reduced)[0]
        probs = ml['model'].predict_proba(reduced)[0]
        label = ml['le'].inverse_transform([pred])[0]
        
        prob_dict = {str(cls): float(prob) for cls, prob in zip(ml['le'].classes_, probs)}
        
        return {
            "prediction": str(label),
            "probabilities": prob_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
