import os
from dotenv import load_dotenv
import random

import wandb
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torch import nn
from torch.cuda import amp
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.utils as vutils

from dataloader import slice_loader, LungDataset_2D
# from GAN3D import Generator, Discriminator
from DCGAN import Generator, Discriminator

import SimpleITK as sitk

sitk.ProcessObject_SetGlobalWarningDisplay(False)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def wandb_config():
    config = wandb.config
    # ENV
    config.data_path = os.getenv("VIDA_PATH")
    config.in_file = "ENV18PM_ProjSubjList_cleaned_IN.in"
    config.test_results_dir = "RESULTS"
    config.name = "DCGAN2D"
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.model = "DCGAN"

    config.lrG = 0.0002
    config.lrD = 0.0002
    config.batch = 128
    config.n_case = 64

    config.img_size = 64
    config.z_size = 100

    config.save = False
    config.debug = False
    if config.debug:
        config.epochs = 1
        config.project = "debug"
    else:
        config.epochs = 30
        config.project = "generative_model"
    return config


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_accuracy(y_true, y_prob):
    y_prob = y_prob > 0.5
    return (y_true == y_prob).sum().item() / y_true.size(0)


if __name__ == "__main__":
    load_dotenv()
    seed_everything()
    config = wandb_config()
    wandb.init(project=config.project)

    # Data
    df = pd.read_csv(os.path.join(config.data_path, config.in_file), sep="\t")
    df = df[: config.n_case]
    slices = slice_loader(df)
    Lung2D_ds = LungDataset_2D(df, slices, img_size=config.img_size)
    data_loader = DataLoader(
        Lung2D_ds, batch_size=config.batch, shuffle=True, num_workers=16
    )
    real_label = 1
    fake_label = 0

    # Model #
    # Generator
    netG = Generator().to(config.device)
    netG.apply(weights_init)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=config.lrG, betas=(0.5, 0.999))

    # Discriminator
    netD = Discriminator(in_channels=1).to(config.device)
    netD.apply(weights_init)
    optimizerD = torch.optim.Adam(netG.parameters(), lr=config.lrD, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(config.batch, config.z_size, 1, 1, device=config.device)

    img_list = []
    G_losses = []
    D_losses = []
    D_accs = []
    print("Start Training. . .")
    iter = 0
    for epoch in range(config.epochs):
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader), total=iters)
        for i, batch in pbar:
            netD.zero_grad()
            real_img = batch["image"].to(config.device, dtype=torch.float)
            batch_size = real_img.shape[0]
            label = torch.full(
                (batch_size,), real_label, dtype=torch.float, device=config.device
            )
            output = netD(real_img).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()
            acc_real = get_accuracy(label, output)

            noise = torch.randn(batch_size, config.z_size, 1, 1, device=config.device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output, label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            acc_fake = get_accuracy(label, output)
            accD = (acc_fake + acc_real) / 2
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake.detach()).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            D_accs.append(accD)
            pbar.set_description(
                f"D loss:{lossD.item():.3f}, D acc:{accD:.3f}, G loss: {lossG.item():.3f}"
            )

            wandb.log(
                {
                    "iteration": iter,
                    "D loss": lossD.item(),
                    "D acc": accD,
                    "G loss": lossG.item(),
                }
            )

            iter += 1
            # break
        # evaluate
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid_fake = vutils.make_grid(fake,normalize=True)
        plt.figure(figsize=(15, 15))
        plt.axis("off")
        plt.title(f"Fake Images at Epoch {epoch}")
        plt.imshow(grid_fake.permute(1, 2, 0))
        wandb.log({"plot": plt})
        plt.close()
