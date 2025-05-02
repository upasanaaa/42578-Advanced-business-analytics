import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Dataset
from tqdm import tqdm
import os
import numpy as np
from model import FaceClassifier, FaceDataset

DATA_PATH = "data/train_images"
MODEL_PATH = "model_weights/face_detector.pth"
BATCH_SIZE = 24
EPOCHS = 20
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 1e-5

#device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#validation transform
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#focal loss function
def focal_loss(outputs, targets, alpha=0.25, gamma=2.0):
    bce_loss = nn.functional.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
    pt = torch.exp(-bce_loss)  # prevents nans when probability 0
    focal_loss = alpha * (1-pt)**gamma * bce_loss
    return focal_loss.mean()

#create dataset
print("Loading dataset...")
dataset = FaceDataset(DATA_PATH, transform=transform)
print(f"Dataset loaded with {len(dataset)} images")

#check if we have enough data
if len(dataset) < 10:
    print("⚠️ Warning: Very small dataset detected.")
    if len(dataset) == 0:
        print("❌ No images found. Please add images to data/real and data/fake folders.")
        exit()

#count real and fake samples
real_count = sum(1 for label in dataset.labels if label == 1)
fake_count = len(dataset) - real_count
print(f"Dataset composition: {real_count} real faces, {fake_count} fake faces")

#split into training and validation sets
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

#create weighted sampler for the TRAINING dataset only
train_labels = [dataset.labels[i] for i in train_dataset.indices]
train_class_weights = [1.0/fake_count if label == 0 else 1.0/real_count for label in train_labels]
train_sampler = WeightedRandomSampler(
    weights=train_class_weights,
    num_samples=len(train_dataset),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

model = FaceClassifier().to(device)

#loss function and optimizer
criterion = focal_loss
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("Starting training...")
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(EPOCHS):
    #training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        #zero the parameter gradients
        optimizer.zero_grad()
        
        #forward + backward + optimize
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        #gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        #statistics
        running_loss += loss.item()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        #update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    #validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    #confusion matrix tracking
    conf_matrix = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
            
            #update confusion matrix
            for i in range(len(labels)):
                pred_val = predictions[i].item()
                true_val = labels[i].item()
                if pred_val == 1 and true_val == 1:
                    conf_matrix["TP"] += 1
                elif pred_val == 0 and true_val == 0:
                    conf_matrix["TN"] += 1
                elif pred_val == 1 and true_val == 0:
                    conf_matrix["FP"] += 1
                elif pred_val == 0 and true_val == 1:
                    conf_matrix["FN"] += 1
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    #calculate precision and recall
    precision = conf_matrix["TP"] / (conf_matrix["TP"] + conf_matrix["FP"]) if (conf_matrix["TP"] + conf_matrix["FP"]) > 0 else 0
    recall = conf_matrix["TP"] / (conf_matrix["TP"] + conf_matrix["FN"]) if (conf_matrix["TP"] + conf_matrix["FN"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    #print epoch results
    print(f"Epoch {epoch+1}/{EPOCHS}:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"  Confusion Matrix: TP={conf_matrix['TP']}, TN={conf_matrix['TN']}, FP={conf_matrix['FP']}, FN={conf_matrix['FN']}")
    
    #learning rate scheduler step
    scheduler.step(val_loss)
    
    #save model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  Model saved to {MODEL_PATH} (improved validation loss)")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

print("\nTraining complete!")