import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import os
from model import FaceClassifier

def predict_image(image_path, model_path="model_weights/face_detector.pth"):
    # Check if files exist
    if not os.path.exists(image_path):
        print(f"Error: Image not found at '{image_path}'")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define image transformations (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load model
    model = FaceClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    try:
        # Load and process image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            output = model(image_tensor)
            prob_real = torch.sigmoid(output).item()
            prob_fake = 1 - prob_real
        
        # Display results
        print("\n" + "="*50)
        print(f"Image: {os.path.basename(image_path)}")
        print("="*50)
        print(f"Real probability: {prob_real:.4f} ({prob_real*100:.1f}%)")
        print(f"Fake probability: {prob_fake:.4f} ({prob_fake*100:.1f}%)")
        
        if prob_real >= 0.5:
            print("\n✅ VERDICT: REAL FACE")
        else:
            print("\n❌ VERDICT: AI-GENERATED FACE")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser(description="Detect if a face image is real or AI-generated")
    parser.add_argument("image", help="Path to the image file to analyze")
    parser.add_argument("--model", default="model_weights/face_detector.pth", 
                        help="Path to trained model (default: model_weights/face_detector.pth)")
    
    args = parser.parse_args()
    predict_image(args.image, args.model)