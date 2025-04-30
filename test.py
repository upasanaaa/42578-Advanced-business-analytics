import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from model import FaceClassifier, FaceDataset

def test_model():
    # Fixed paths from project structure
    TEST_DIR = "data/test_images"
    MODEL_PATH = "model_weights/face_detector.pth"  # Updated path to match your structure
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
        return
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    print("Loading test dataset...")
    test_dataset = FaceDataset(TEST_DIR, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"Dataset loaded with {len(test_dataset)} images")
    
    # Count real and fake samples
    real_count = sum(1 for label in test_dataset.labels if label == 1)
    fake_count = len(test_dataset) - real_count
    print(f"Dataset composition: {real_count} real faces, {fake_count} fake faces")
    
    # Load model
    print("Loading model...")
    model = FaceClassifier().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully!")
    
    # Evaluate model
    all_preds = []
    all_labels = []
    
    print("Evaluating model...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print("\n" + "="*50)
    print("Test Results:")
    print("="*50)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print("\nConfusion Matrix:")
    print(f"┌───────────────┬─────────────────┬─────────────────┐")
    print(f"│               │ Predicted Fake  │ Predicted Real  │")
    print(f"├───────────────┼─────────────────┼─────────────────┤")
    print(f"│ Actual Fake   │ {tn:^15} │ {fp:^15} │")
    print(f"├───────────────┼─────────────────┼─────────────────┤")
    print(f"│ Actual Real   │ {fn:^15} │ {tp:^15} │")
    print(f"└───────────────┴─────────────────┴─────────────────┘")
    
    print("\nDetailed Metrics:")
    print(f"True Negatives (Correctly identified fake): {tn}")
    print(f"False Positives (Fake identified as real): {fp}")
    print(f"False Negatives (Real identified as fake): {fn}")
    print(f"True Positives (Correctly identified real): {tp}")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"Specificity: {specificity:.4f}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    test_model()