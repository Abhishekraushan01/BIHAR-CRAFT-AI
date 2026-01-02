import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from data_utils import get_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            # input noise z of shape (nz, 1, 1)
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),

            # now (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            # now (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            # now (ngf*4) x 16 x 16
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            # now (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            # now (ngf) x 64 x 64
            nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf // 2),
            nn.ReLU(True),

            # now (ngf//2) x 128 x 128
            nn.ConvTranspose2d(ngf // 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()  # output in [-1, 1] range
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
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False), # 16->8
            nn.BatchNorm2d(ndf*16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),                      # flatten to [batch_size, ...]
            nn.Linear(ndf*16 * 8 * 8, 1),     # adapt 16*16 if your input image is 256×256
            nn.Sigmoid()                       # single probability per image
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def main():
    # Project directory base
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Paths to folders relative to BASE_DIR
    dataset_folders = ["data/processed", "data/augmented"]

    print("Loading images from:", dataset_folders)

    # Use your updated get_dataset → it accepts a list of folders
    dataloader = get_dataset(dataset_folders, img_size=256, batch_size=64, shuffle=True)


    # Build networks
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Loss & optimizers
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    G_losses = []
    D_losses = []

    nz = 100
    epochs = 100
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    os.makedirs(os.path.join(BASE_DIR, "outputs/dcgan_samples"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # ===== Resume from checkpoint (if exists) =====
    start_epoch = 0
    resume_path = os.path.join(BASE_DIR, "models", "dcgan_epoch_10.pth")  # change to your checkpoint
    if os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        netG.load_state_dict(checkpoint["netG"])
        netD.load_state_dict(checkpoint["netD"])
        optimizerG.load_state_dict(checkpoint["optimizerG"])
        optimizerD.load_state_dict(checkpoint["optimizerD"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, epochs):
        for i, real_images in enumerate(dataloader, 0):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            # Train Discriminator on real
            netD.zero_grad()
            label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            output_real = netD(real_images).view(-1)
            errD_real = criterion(output_real, label)
            errD_real.backward()

            # Train Discriminator on fake
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake_images = netG(noise)

            label.fill_(0)
            output_fake_for_D = netD(fake_images.detach()).view(-1)  # new forward pass for D
            errD_fake = criterion(output_fake_for_D, label)
            errD_fake.backward()
            errD = errD_real + errD_fake

            optimizerD.step()

            # Train Generator
            netG.zero_grad()

            label.fill_(1)
            output_fake_for_G = netD(fake_images).view(-1)      # fresh forward pass for D
            errG = criterion(output_fake_for_G, label)
            errG.backward()
            optimizerG.step()

            if i % 200 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{i}/{len(dataloader)}] "
                    f"LossD: {errD.item():.4f} "
                    f"LossG: {errG.item():.4f}"
                )

            D_losses.append((errD_real + errD_fake).item())
            G_losses.append(errG.item())

        # Save progress images
        with torch.no_grad():
            fake = netG(fixed_noise)
            vutils.save_image(
             fake,
             os.path.join(BASE_DIR, "outputs", "dcgan_samples", f"epoch_{epoch+1}.png"),
             normalize=True
         )
        # Save losses
        loss_path = os.path.join(BASE_DIR, "outputs", "training_losses.pkl")
        with open(loss_path, "wb") as f:
            pickle.dump(
                {
                    "D_losses": D_losses,
                    "G_losses": G_losses,
                },
                f
            )


        # ===== Save checkpoint every 10 epochs =====
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "netG": netG.state_dict(),
                    "netD": netD.state_dict(),
                    "optimizerG": optimizerG.state_dict(),
                    "optimizerD": optimizerD.state_dict(),
                },
                os.path.join(BASE_DIR, "models", f"dcgan_epoch_{epoch+1}.pth")
            )


    # ===== Final save AFTER training =====
    torch.save(
        {
            "epoch": epochs,
            "netG": netG.state_dict(),
            "netD": netD.state_dict(),
            "optimizerG": optimizerG.state_dict(),
            "optimizerD": optimizerD.state_dict(),
        },
        os.path.join(BASE_DIR, "models", "dcgan_final.pth")
    )
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
