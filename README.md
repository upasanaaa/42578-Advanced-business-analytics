# Fake Face Detector

A simple tool that detects AI-generated faces using ResNet50.

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate

# Install all dependencies
pip install -r requirements-gpu.txt

# Install Node.js on Windows
```

## Folder Structure
```
FakeFaceDetector/
├── model.py              # Model architecture (ResNet50-based)
├── train.py              # Training script
├── test.py               # Image prediction
├── model_weights/        # Trained model storage
├── data/                 # Training data
│   ├── real/             # Real face images
│   └── fake/             # AI-generated face images
├── requirements.txt      # Simple dependencies
├── requirements-gpu.txt  # Simple dependencies
└── README.md             # Project instructions
```

## How to Run

### 1. Prepare your data
```bash
mkdir -p data/real data/fake model_weights
```
- Place real face images in `data/real/`
- Place AI-generated face images in `data/fake/`

### 2. Train the model
```bash
python train.py
```

### 3. Test an image
```bash
python test.py path/to/your/image.jpg
```

### 4. Start the backend
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 5. Start the frontend
```bash
unzip my-react-app.zip
cd my-react-app
npm run dev
```

## Notes
- This model uses ResNet50 for image classification
- Works with any RGB image format
- No special face detection required - just center the face in the image
- Uses GPU if available for faster processing

