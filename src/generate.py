import os
import torch
import torchvision.utils as vutils
from dcgan_train import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Build absolute path to the saved model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "models", "dcgan.pth")

print("Loading model from:", model_path)

# Load generator
netG = Generator()
netG.load_state_dict(torch.load(model_path, map_location=device))
netG.to(device)
netG.eval()

# Generate samples
noise = torch.randn(64, 100, 1, 1, device=device)
with torch.no_grad():
    fake = netG(noise)

# Save final output grid
out_path = os.path.join(BASE_DIR, "outputs", "dcgan_samples", "final_generated.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
vutils.save_image(fake, out_path, normalize=True)

print("Generated images saved at:", out_path)
