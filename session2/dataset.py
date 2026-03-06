import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pandas as pd
from PIL import Image
import os


# Example custom Dataset class
class ChineseMNISTDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Assuming filenames are structured, e.g., input_1_1_1.jpg
        row = self.data.iloc[idx]
        img_name = f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        label = row['code'] - 1  # Adjust label to be 0-indexed

        if self.transform:
            image = self.transform(image)

        return image, label

# Example usage with torchvision transforms
# transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])

# Dataloders initialization
