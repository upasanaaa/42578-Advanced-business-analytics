import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import FaceClassifier
import time

# Load model
@st.cache_resource

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceClassifier().to(device)
    model.load_state_dict(torch.load("model_weights/face_detector.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# App title
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🖼\ufe0f Face Authenticity Checker</h1>
""", unsafe_allow_html=True)

st.write("""
### Upload a face image below and let's detect if it's **REAL** or **AI-generated**!
""")

# File uploader
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("\n")
    predict_btn = st.button("✨ Predict Now!")

    if predict_btn:
        with st.spinner('Analyzing the image... 🤖'):
            time.sleep(1)
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                prob_real = torch.sigmoid(output).item()
                prob_fake = 1 - prob_real

        st.success('Prediction complete!')
        st.write("---")
        st.subheader("Prediction Results:")

        st.progress(int(prob_real * 100))
        st.write(f"**Real Face Probability:** `{prob_real*100:.2f}%`")
        st.progress(int(prob_fake * 100))
        st.write(f"**Fake Face Probability:** `{prob_fake*100:.2f}%`")

        if prob_real >= 0.5:
            st.balloons()
            st.success("✅ This looks like a **REAL** face!")
        else:
            st.snow()
            st.error("❌ This might be an **AI-generated** or **manipulated** face!")
