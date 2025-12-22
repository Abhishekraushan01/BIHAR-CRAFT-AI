#generate samples after training
import torch
import torchvision.utils as vutils
from dcgan_train import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

netG = Generator()
netG.load_state_dict(torch.load("../models/dcgan.pth"))
netG.to(device)

noise = torch.randn(64, 100, 1, 1, device=device)
fake = netG(noise)
vutils.save_image(fake, "../outputs/dcgan_samples/final_generated.png", normalize=True)
print("Generated images saved!")
