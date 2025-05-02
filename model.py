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

        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        #extract features
        num_features = self.model.fc.in_features
        self.model.fc = nn.Identity()  # Remove FC layer
        
        #ddd spatial attention to focus on facial features
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        #improved classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1)  #binary: Real or Fake
        )
    
    def forward(self, x):
        #extract features from backbone
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features = self.model.layer4(x)
        
        #apply attention
        attention = self.attention(features)
        attended_features = features * attention
        
        #global average pooling
        x = self.model.avgpool(attended_features)
        x = torch.flatten(x, 1)
        
        return self.classifier(x)

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        self.labels = []
        
        #load all samples from real/fake folders
        for label, subdir in enumerate(['fake', 'real']):  #0=fake, 1=real
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
        
        #load and process image
        try:
            image = Image.open(path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
                
            label = torch.tensor([label], dtype=torch.float32)
            return image, label
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            #return a blank image in case of error
            blank = torch.zeros((3, 224, 224))
            return blank, torch.tensor([0], dtype=torch.float32)