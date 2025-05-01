#  Fake Face Detector

A deep learning-powered tool that detects AI-generated (fake) faces using a ResNet50-based image classifier.

---

## 🔧 Environment Setup

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    ```
    - **Activate on Windows:**
      ```bash
      venv\Scripts\activate
      ```
    - **Activate on macOS/Linux:**
      ```bash
      source venv/bin/activate
      ```

2. **Install dependencies:**
    # To run in gpu:
    ```bash
    pip install -r requirements-gpu.txt
    ```
    
    # To run in cpu:
    ```bash
    pip install -r requirements.txt
    ```

3. **(Optional) Install Node.js**
    [Download Node.js](https://nodejs.org/)

---

##  Project Structure

```
FakeFaceDetector/
├── model.py              # ResNet50 model definition
├── train.py              # Training script
├── validation.py         # Single image prediction script
├── test.py               # Batch testing script with metrics
├── main.py               # FastAPI backend
├── my-react-app          # Frontend React code
├── model_weights/        # Directory for saved model weights
├── my-react-app/          # Frontend React code
├── data/                 # Dataset directory
│   ├── train_images/     # Training dataset
│   │   ├── real/         # Real face images
│   │   └── fake/         # AI-generated face images
│   └── test_images/      # Testing dataset
│       ├── real/         # Real face images
│       └── fake/         # AI-generated face images
├── requirements.txt      # Standard dependencies
├── requirements-gpu.txt  # GPU-specific dependencies
└── README.md             # Project documentation
```

---

##  How to Use

### 1. Prepare Your Dataset

```bash
mkdir -p data/train_images/real data/train_images/fake data/test_images/real data/test_images/fake model_weights
```

- Place real face images in `data/train_images/real/` and `data/test_images/real/`
- Place AI-generated face images in `data/train_images/fake/` and `data/test_images/fake/`

---

### 2. Train the Model

```bash
python train.py
```

---

### 3. Test the Model

#### Single Image Testing 
```bash
python validation.py path/to/your/image.jpg
```

#### Batch Testing with Metrics
```bash
python test.py
```

---

### 4. Start the Backend (FastAPI)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Once you see the message:

```
INFO:     Application startup complete.
```

in the terminal, go to step 5. 

If you want to see back end go to below link:

👉 **[http://localhost:8000/docs#/default/predict_predict_post](http://localhost:8000/docs#/default/predict_predict_post)**

to test the prediction endpoint using the FastAPI interactive documentation.

---

### 5. Start the Frontend (React)

```bash
unzip my-react-app.zip (can be done manually)
cd my-react-app
npm install (if needed)
npm run dev
```

- Open the displayed local URL (e.g., http://localhost:5173/) to use the web app. For testing, crop the face from an image, ensure the face is centered, save it locally, upload the cropped image, and then click 'Upload and Detect'.

---

##  Notes

- Model uses **ResNet50** for binary image classification.
- Works with any **RGB image** (ensure the face is centered).
- Utilizes **GPU** if available for faster inference.
- No face detection step required — just centered, cropped face images.

