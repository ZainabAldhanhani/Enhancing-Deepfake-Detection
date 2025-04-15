import sys
import random
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Label map based on your CSV format
LABEL_MAP = {0: "real", 1: "fake"}

def load_labels(label_path):
    if not label_path.exists():
        print(f"❌ Error: Label file '{label_path}' not found.")
        sys.exit(1)
    
    df = pd.read_csv(label_path)
    
    # Validate required columns
    if not {"file_name", "label"}.issubset(df.columns):
        print("❌ Error: CSV must contain 'file_name' and 'label' columns.")
        sys.exit(1)
    
    # Map numeric labels to text (0 → real, 1 → fake)
    df["label"] = df["label"].map(LABEL_MAP)
    df["file_name"] = df["file_name"].apply(lambda x: Path(x).name)  # Keep only filename
    return df

def count_labels(df):
    print("📊 Image counts by label:")
    counts = df["label"].value_counts()
    for cls in LABEL_MAP.values():
        count = counts.get(cls, 0)
        print(f"  • {cls}: {count} images")

def show_sample_images(df, image_dir, n=3):
    print(f"\n🖼 Showing {n} random samples from each class:")
    fig, axes = plt.subplots(len(LABEL_MAP), n, figsize=(4 * n, 4 * len(LABEL_MAP)))

    for i, cls in enumerate(LABEL_MAP.values()):
        subset = df[df["label"] == cls]
        samples = subset.sample(min(n, len(subset)))
        for j, (_, row) in enumerate(samples.iterrows()):
            img_path = image_dir / row["file_name"]
            if not img_path.exists():
                print(f"  ⚠️ Warning: File '{img_path.name}' not found.")
                continue
            img = Image.open(img_path)
            axes[i][j].imshow(img)
            axes[i][j].set_title(f"{cls} - {img_path.name}")
            axes[i][j].axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python preview_dataset.py <path_to/train_data>")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.exists():
        print(f"❌ Error: Provided path '{image_dir}' does not exist.")
        sys.exit(1)

    # CSV is assumed to be in the same parent directory as train_data
    label_path = image_dir.parent / "train.csv"
    df = load_labels(label_path)

    print("🔍 Dataset Preview Script\n")
    count_labels(df)
    show_sample_images(df, image_dir)
    
