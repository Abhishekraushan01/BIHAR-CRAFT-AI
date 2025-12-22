import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Get the absolute path to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_dataset(data_dir, img_size=256, batch_size=32, shuffle=True):
    # Make sure the path is absolute
    abs_data_dir = os.path.join(BASE_DIR, data_dir)

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
        def __init__(self, images_dir, transform=None):
            self.images_dir = images_dir
            self.transform = transform
            self.files = [
                f for f in os.listdir(images_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.images_dir, self.files[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img

    dataset = FlatImageDataset(abs_data_dir, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0
    )
    return dataloader
