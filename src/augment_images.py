import os
import random
from PIL import Image, ImageEnhance

# Folders
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define absolute data paths
RAW_DIR = os.path.join(BASE_DIR, "data", "processed")   # change to 'processed' if augmenting processed images
AUG_DIR = os.path.join(BASE_DIR, "data", "augmented")
# Make output directory
os.makedirs(AUG_DIR, exist_ok=True)

# Number of augmentations per original
N_AUG = 5

def random_rotation(image):
    angle = random.choice([0, 90, 180, 270])
    return image.rotate(angle)

def random_flip(image):
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return image

def random_color(image):
    enhancer = ImageEnhance.Color(image)
    factor = random.uniform(0.5, 1.5)  # change color saturation
    return enhancer.enhance(factor)

def random_brightness(image):
    enhancer = ImageEnhance.Brightness(image)
    factor = random.uniform(0.7, 1.3)  # brightness variation
    return enhancer.enhance(factor)

def augment_image(img_path, index):
    img = Image.open(img_path).convert("RGB")

    img = random_rotation(img)
    img = random_flip(img)
    img = random_color(img)
    img = random_brightness(img)

    filename = os.path.basename(img_path).split(".")[0]
    out_path = os.path.join(AUG_DIR, f"{filename}_aug{index}.jpg")
    img.save(out_path)

# Loop over images in RAW_DIR
for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        full_path = os.path.join(RAW_DIR, filename)

        for i in range(1, N_AUG + 1):
            augment_image(full_path, i)

print("Augmentation complete!")
