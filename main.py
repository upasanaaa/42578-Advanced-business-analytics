from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
from io import BytesIO
from model import FaceClassifier
import os

app = FastAPI()

# CORS setup to allow frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can specify your frontend URL here
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when API starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_weights/face_detector.pth"

model = FaceClassifier().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float = 0.5):
    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob_real = torch.sigmoid(output).item()
            prob_fake = 1 - prob_real

        verdict = "REAL" if prob_real >= threshold else "FAKE"
        message = {
            "filename": file.filename,
            "real_prob": prob_real,
            "fake_prob": prob_fake,
            "verdict": verdict
        }
        return JSONResponse(content=message)
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
