import torch
import torch.nn as nn
import torchvision.models as models
import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class FaceClassifier(nn.Module):
    def __init__(self):
        super(FaceClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Replace the final fully connected layer
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Binary classification (real or fake)
        )
    
    def forward(self, x):
        return self.model(x)

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        
        # Load all samples from real/fake folders
        for label, subdir in enumerate(['real', 'fake']):
            folder = os.path.join(root_dir, subdir)
            if not os.path.exists(folder):
                continue
                
            for fname in os.listdir(folder):
                if fname.endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(folder, fname)
                    self.samples.append(path)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        label = self.labels[idx]
        
        # Load and process image
        try:
            image = Image.open(path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = torch.tensor([label], dtype=torch.float32)
            return image, label
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a blank image in case of error
            blank = torch.zeros((3, 224, 224))
            return blank, torch.tensor([0], dtype=torch.float32)