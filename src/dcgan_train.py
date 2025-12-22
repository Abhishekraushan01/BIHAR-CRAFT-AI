import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from data_utils import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define Generator
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# --- Define Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # flatten to [batch_size, ...]
            nn.Linear(ndf*8 * 16 * 16, 1),     # adapt 16*16 if your input image is 256×256
            nn.Sigmoid()                       # single probability per image
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    # Project directory base
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Dataset path
    processed_data_path = os.path.join(BASE_DIR, "data", "processed")

    print("Loading images from:", processed_data_path)

    # Load dataset — set num_workers to 0 on Windows if you run into multiprocessing errors
    dataloader = get_dataset(processed_data_path)

    # Build networks
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Loss & optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    nz = 100
    epochs = 50
    for epoch in range(epochs):
        for i, real_images in enumerate(dataloader, 0):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator
            netD.zero_grad()
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            errD_real = criterion(netD(real_images).view(-1), label)
            errD_real.backward()

            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)
            label.fill_(0)
            errD_fake = criterion(netD(fake_images.detach()).view(-1), label)
            errD_fake.backward()
            optimizerD.step()

            # Train Generator
            netG.zero_grad()
            label.fill_(1)
            errG = criterion(netD(fake_images).view(-1), label)
            errG.backward()
            optimizerG.step()

        print(f"Epoch [{epoch+1}/{epochs}] LossD: {errD_real+errD_fake:.4f}, LossG: {errG:.4f}")

        # Save progress images
        vutils.save_image(
            fake_images,
            os.path.join(BASE_DIR, "outputs", "dcgan_samples", f"epoch_{epoch+1}.png"),
            normalize=True
        )

    # Save final model
    torch.save(netG.state_dict(), os.path.join(BASE_DIR, "models", "dcgan.pth"))
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
