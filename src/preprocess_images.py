import os
from PIL import Image


# Base project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
OUT_DIR = os.path.join(BASE_DIR, "data", "processed")
TARGET_SIZE = (256, 256)

os.makedirs(OUT_DIR, exist_ok=True)

for filename in os.listdir(RAW_DIR):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        img_path = os.path.join(RAW_DIR, filename)
        img = Image.open(img_path).convert("RGB")  # load image
        img = img.resize(TARGET_SIZE)              # resize
        out_path = os.path.join(OUT_DIR, filename)
        img.save(out_path)                         # save to processed folder

print("All images resized to", TARGET_SIZE)
