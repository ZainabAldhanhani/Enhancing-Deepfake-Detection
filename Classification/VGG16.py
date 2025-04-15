import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Check device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 1. Custom Dataset Definition
# -----------------------------
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame with columns "file_name" and "label"
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, "file_name"]
        image = Image.open(img_path).convert("RGB")
        label = int(self.dataframe.loc[idx, "label"])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


def main(dataset_path):
    # -----------------------------
    # 2. Load and Split the Dataset
    # -----------------------------
    # Base directory for your Kaggle dataset
    base_dir = dataset_path
    csv_path = os.path.join(base_dir, "train.csv")

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Drop the unnecessary index column if it exists
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convert relative file paths to absolute paths
    df["file_name"] = df["file_name"].apply(lambda x: os.path.join(base_dir, x))
    # Ensure labels are integers (0 or 1)
    df["label"] = df["label"].astype(int)

    print("Total dataset size:", df.shape)

    # Split the dataframe into training (80%) and testing (20%) sets
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    print("Train dataframe size:", train_df.shape)
    print("Test dataframe size:", test_df.shape)

    # -----------------------------
    # 3. Define Data Transforms & DataLoaders
    # -----------------------------
    IMG_SIZE = 224
    BATCH_SIZE = 32

    # Transforms for training (with augmentation) and testing (only resizing and normalization)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Using ImageNet means
                            std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    # Create dataset instances
    train_dataset = CustomImageDataset(train_df, transform=train_transform)
    test_dataset = CustomImageDataset(test_df, transform=test_transform)

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # -----------------------------
    # 4. Define the VGG16 Model
    # -----------------------------
    # Load the pretrained VGG16 model
    model = models.vgg16(pretrained=True)

    # Freeze the feature layers (optional)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace the classifier with a custom classifier for binary classification
    num_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)  # Single output neuron for binary classification
    )

    model = model.to(device)

    # -----------------------------
    # 5. Define Loss Function and Optimizer
    # -----------------------------
    # Use BCEWithLogitsLoss (which applies sigmoid internally)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # -----------------------------
    # 6. Training Loop
    # -----------------------------
    num_epochs = 20

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            # Reshape labels to (batch_size, 1) and convert to float
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # -----------------------------
    # 7. Evaluation on the Test Set
    # -----------------------------
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)
            # Convert probabilities to binary predictions (threshold = 0.5)
            preds = (preds > 0.5).int().cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Compute evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    # Plot the confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Class 0", "Class 1"],
                yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classification Using VGG16.")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
