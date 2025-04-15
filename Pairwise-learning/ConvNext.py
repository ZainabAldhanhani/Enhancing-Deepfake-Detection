import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Check Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================
# 1. Define the Siamese Network (ConvNeXt)
# ============================
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        convnext = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            convnext.features,
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )

        # Fully connected layer to map features to embeddings
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),  # convnext_base output channels = 1024
            nn.ReLU(),
            nn.Linear(512, 256)    # Embedding dimension = 256
        )

    def forward(self, x1, x2):
        f1 = self.feature_extractor(x1).view(x1.size(0), -1)
        f2 = self.feature_extractor(x2).view(x2.size(0), -1)

        f1 = self.fc(f1)
        f2 = self.fc(f2)
        return f1, f2

# ============================
# 2. Define Contrastive Loss
# ============================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, f1, f2, label):
        distance = torch.nn.functional.pairwise_distance(f1, f2)
        loss = (label * distance.pow(2) + (1 - label) * (self.margin - distance).clamp(min=0).pow(2)).mean()
        return loss

# ============================
# 3. Custom Dataset for Image Pairs
# ============================
class SiameseDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.real_images = self.dataframe[self.dataframe["label"] == 1]["file_name"].tolist()
        self.ai_images = self.dataframe[self.dataframe["label"] == 0]["file_name"].tolist()

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                img1_path, img2_path = np.random.choice(self.real_images, 2)
            else:
                img1_path, img2_path = np.random.choice(self.ai_images, 2)
            label = 1
        else:
            img1_path = np.random.choice(self.real_images)
            img2_path = np.random.choice(self.ai_images)
            label = 0

        assert os.path.exists(img1_path), f"File not found: {img1_path}"
        assert os.path.exists(img2_path), f"File not found: {img2_path}"

        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

def main(dataset_path):
    # ============================
    # 4. Load & Split Dataset
    # ============================
    base_dir = dataset_path
    csv_path = os.path.join(base_dir, "train.csv")

    df = pd.read_csv(csv_path)
    df["file_name"] = df["file_name"].apply(lambda x: os.path.join(base_dir, x))
    df["label"] = df["label"].astype(int)

    assert all(os.path.exists(f) for f in df["file_name"]), "Some image paths are incorrect!"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    # ============================
    # 5. Data Transforms & DataLoaders
    # ============================
    IMG_SIZE = 224
    BATCH_SIZE = 8

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SiameseDataset(train_df, transform=transform)
    test_dataset = SiameseDataset(test_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ============================
    # 6. Train the Siamese Network
    # ============================
    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)

            optimizer.zero_grad()
            f1, f2 = model(img1, img2)
            loss = criterion(f1, f2, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")

    # ============================
    # 7. Evaluation with Confusion Matrix
    # ============================
    def evaluate(model, test_loader, threshold=0.5):
        model.eval()
        all_labels, all_preds = [], []

        with torch.no_grad():
            for img1, img2, label in test_loader:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                f1, f2 = model(img1, img2)
                distance = torch.nn.functional.pairwise_distance(f1, f2)
                predictions = (distance < threshold).float()
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        return accuracy, precision, recall, f1

    evaluate(model, test_loader, threshold=0.5)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classification Using ConNext.")
    
    # Define parameters
    parser.add_argument("--dataset_path", type=str, required=True, help="Put your  dataset path")

    # Parse arguments
    args = parser.parse_args()
    main(args.dataset_path)
