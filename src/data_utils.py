import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Get project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_dataset(folder_paths, img_size=256, batch_size=32, shuffle=True):
    """
    folder_paths: list of relative directory strings relative to BASE_DIR
    Example: ['data/processed', 'data/augmented']
    """

    # Make absolute paths
    abs_paths = [os.path.join(BASE_DIR, p) for p in folder_paths]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    # Custom dataset class for flat image folders
    class FlatImageDataset(Dataset):
        def __init__(self, images_dirs, transform=None):
            self.transform = transform
            self.image_paths = []

            # Gather all image file paths from each folder
            for images_dir in images_dirs:
                for f in os.listdir(images_dir):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        self.image_paths.append(os.path.join(images_dir, f))

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img

    # Create dataset
    dataset = FlatImageDataset(abs_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return dataloader
