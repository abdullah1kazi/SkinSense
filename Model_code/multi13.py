import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
import copy
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torch import nn, optim
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random

# File paths and classes
data_dir = '/home/014350037/SkinSense/MultiTrain'
csv_file = '/home/014350037/SkinSense/train.csv'

# Diagnoses
benign_diagnoses = ['Nevus', 'Eborrheic Keratosis', 'Pigmented Benign Keratosis', 'Solar Lentigo', 'Dermatofribroma',
                    'Vascular Lesion', 'Lentigo NOS', 'Lichenoid Keratosis', 'Lentigo Simplex', 'AIMP', 'Angioma',
                    'Neurofibroma', 'Scar', 'Verucca', 'Acrochordon', 'Angiofibroma', 'Fibrous Papule',
                    'Cafe-Au-Lait Macule', 'Angiokeratoma', 'Clear Cell Acanthoma']

malignant_diagnoses = ['Melanoma', 'Basal Cell Carcinoma', 'Actinic Keratosis', 'Squamous Cell Carcinoma',
                       'Melanoma Metastasis', 'Atypical Melanocytic Proliferation', 'Atypical Spitz Tumor']

diagnoses_to_indices = {diagnosis: i for i, diagnosis in enumerate(benign_diagnoses + malignant_diagnoses)}

def get_class_index(labels, diagnoses_to_indices):
    # Convert multi-labels to class labels
    return torch.argmax(torch.tensor([diagnoses_to_indices[label] for label in labels]), dim=-1)

class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, balance_classes=True):
        """
        Args:
            csv_file (string): Path to the csv file with image paths and labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            balance_classes (bool, optional): If True, balance the number of samples for each class.
        """
        self.dataframe = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.balance_classes = balance_classes

        if balance_classes:
            self.balance_dataset()

    def balance_dataset(self):
        # Count the number of samples in each class
        class_counts = self.dataframe.iloc[:, 1:].sum().to_dict()

        # Find the class with the maximum samples
        max_samples = max(class_counts.values())

        # For each class, replicate images to match the number of samples in the class with the maximum samples
        for diagnosis, count in class_counts.items():
            if count < max_samples:
                # Get indices of images from the current class
                indices = self.dataframe.index[self.dataframe[diagnosis] == 1].tolist()

                # Randomly select images to replicate
                replicate_indices = random.choices(indices, k=max_samples - count)

                # Replicate the selected images and add them to the dataset
                for idx in replicate_indices:
                    row = self.dataframe.iloc[idx]
                    self.dataframe = self.dataframe.append(row, ignore_index=True)

    def __len__(self):
        return len(self.dataframe)

    def augment_image(self, image):
        # Randomly apply data augmentation transformations
        augmentations = [
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        ]
        return random.choice(augmentations)(image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)

        binary_labels = self.dataframe.iloc[idx, 1:3].values.astype(np.float32)  # Benign and Malignant
        multi_labels = self.dataframe.iloc[idx, 3:].values.astype(np.float32)    # Specific diagnoses

        # Generate class labels based on binary and multi-label columns
        binary_label = np.argmax(binary_labels)  # Get 0 or 1 based on the maximum value
        multi_label = np.argmax(multi_labels)    # Get the index of the maximum value

        class_label = binary_label * 7 + multi_label  # Combine binary and multi-label information

        # Data augmentation only for classes with fewer images (e.g., classes with index >= num_benign)
        if class_label >= num_benign:
            image = self.augment_image(image)

        if self.transform:
            image = self.transform(image)

        return image, class_label

class SkinSense(nn.Module):
    def __init__(self, backbone_model, binary_classifier, multi_classifier):
        super(SkinSense, self).__init__()
        self.resnet101 = backbone_model
        self.binary_classifier = binary_classifier
        self.multi_classifier = multi_classifier

    def forward(self, x):
        features = self.resnet101(x)
        features = features.view(features.size(0), -1)  
        binary_outputs = self.binary_classifier(features)
        multi_outputs = self.multi_classifier(features)
        return binary_outputs, multi_outputs

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomAdjustSharpness(0.5),
    transforms.GaussianBlur((5, 5), sigma=(0.1, 2.0)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the datasets
train_dataset = SkinLesionDataset(csv_file, data_dir, transform)

# Use CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a pre-trained ResNet50
model = models.resnet101(pretrained=False)
model.load_state_dict(torch.load('/home/014350037/SkinSense/BinaryMulti13/resnet101-5d3b4d8f.pth'))
model.fc = Identity()

# Freeze the layers
for param in model.parameters():
    param.requires_grad = False

# Binary classification for benign or malignant
binary_classifier = nn.Sequential(nn.Linear(2048, 512), 
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, 1))

# Multi-class classification for specific diagnoses
num_benign = len(benign_diagnoses)
num_malignant = len(malignant_diagnoses)

multi_classifier = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, num_benign + num_malignant))

model = SkinSense(model, binary_classifier, multi_classifier)

# Transfer model to GPU if available
model.to(device)

binary_criterion = nn.BCEWithLogitsLoss()
multi_criterion = nn.CrossEntropyLoss()

# KFold Cross-validation
n_epochs = 50  # Number of epochs
n_splits = 5  # Number of splits for cross-validation
patience = 10  # Patience for early stopping

kfold = KFold(n_splits=n_splits, shuffle=True)

best_model_wts = copy.deepcopy(model.state_dict())
best_loss = float('inf')
counter = 0  # Counter for early stopping

# optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.009)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

print("Starting training...")
for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):
 
    print(f'FOLD {fold}')
    print('--------------------------------')

    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=train_subsampler, num_workers=2)
    valloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=val_subsampler, num_workers=2)

    model.train()  # Set model to training mode
    print("Training Mode")
    for epoch in range(n_epochs):
        running_loss = 0.0
        val_running_loss = 0.0
        all_binary_labels = []
        all_binary_preds = []
        all_multi_labels = []
        all_multi_preds = []

        for images, (binary_labels, multi_labels) in trainloader:
            # Training phase
            images = images.to(device)

            binary_labels = binary_labels[:, 1].unsqueeze(1).to(device)
            multi_labels = multi_labels.to(device)

            optimizer.zero_grad()
            binary_outputs, multi_outputs = model(images)
            binary_loss = binary_criterion(binary_outputs, binary_labels.float())
            multi_loss = multi_criterion(multi_outputs, multi_labels.long()) if binary_labels.argmax() != 0 else 0

            # Combine the losses
            loss = binary_loss + multi_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            
        model.eval()  # Set model to validation mode
        print("Validation Mode")
        with torch.no_grad():
            for images, (binary_labels, multi_labels) in valloader:
                # Validation phase
                images = images.to(device)
                binary_labels = binary_labels[:, 1].unsqueeze(1).to(device)
                multi_labels = multi_labels.to(device)
                binary_outputs, multi_outputs = model(images)

                _, predicted_binary = torch.max(binary_outputs, 1) 
                _, predicted_multi = torch.max(multi_outputs, 1)

                binary_labels_categorical = binary_labels.argmax(dim=1)
                all_binary_labels.extend(binary_labels_categorical.detach().cpu().numpy())
                all_binary_preds.extend(predicted_binary.detach().cpu().numpy())
                all_multi_labels.extend(multi_labels.detach().cpu().numpy())
                all_multi_preds.extend(predicted_multi.detach().cpu().numpy())

                binary_loss = binary_criterion(binary_outputs, binary_labels.float())
                multi_loss = multi_criterion(multi_outputs, multi_labels.long()) if binary_labels.argmax() != 0 else 0

                val_loss = binary_loss + multi_loss

                val_running_loss += val_loss.item() * images.size(0)

        # Calculate average losses
        epoch_loss = running_loss / len(trainloader.dataset)
        val_epoch_loss = val_running_loss / len(valloader.dataset)

        # Reduce learning rate when a metric has stopped improving
        scheduler.step(val_epoch_loss)
        
        print('Epoch [{}/{}], Loss: {:.4f}, Val Loss: {:.4f}'.format(
            epoch+1, n_epochs, epoch_loss, val_epoch_loss))
        
        # save model if validation loss has decreased
        if val_epoch_loss <= best_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            best_loss,
            val_epoch_loss))
            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'/home/014350037/SkinSense/BinaryMulti13/checkpoints13/model_{fold}_{epoch}.pth')
            counter = 0
        # if the validation loss didn't improve, increment the counter
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

# load best model weights
model.load_state_dict(best_model_wts)

# save the final model
torch.save(model.state_dict(), '/home/014350037/SkinSense/BinaryMulti13/final_model13.pth')

binary_accuracy = accuracy_score(all_binary_labels, all_binary_preds)
binary_conf_matrix = confusion_matrix(all_binary_labels, all_binary_preds)

print("Binary Classifier Accuracy: {:.4f}".format(binary_accuracy))
print("Binary Classifier Confusion Matrix: \n", binary_conf_matrix)

# For multi-label classifier
multi_accuracy = accuracy_score(all_multi_labels, all_multi_preds)
multi_conf_matrix = confusion_matrix(all_multi_labels, all_multi_preds)

print("Multi-label Classifier Accuracy: {:.4f}".format(multi_accuracy))
print("Multi-label Classifier Confusion Matrix: \n", multi_conf_matrix)