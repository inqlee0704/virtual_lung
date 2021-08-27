import os
from dotenv import load_dotenv
import time
import random

import wandb
import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,CosineAnnealingLR, ReduceLROnPlateau

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from torch import nn
from torch.cuda import amp
import torch
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
import torchvision.utils as vutils

from dataloader import LungDataset_3D
from GAN3D import Generator, Discriminator

import SimpleITK as sitk
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def wandb_config():
    config = wandb.config
    # ENV
    config.data_path = os.getenv('VIDA_PATH')
    config.in_file = 'ENV18PM_ProjSubjList_cleaned_IN.in'
    config.test_results_dir = "RESULTS"
    config.name = 'GAN3D'
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config.model = 'vanila_GAN3D'

    config.learning_rate = 0.0001
    config.lrG = 0.0025
    config.lrD = 0.00001
    config.batch = 64
    config.n_case = 0 
    config.save = False
    config.debug = False
    if config.debug:
        config.epochs = 1
        config.project = 'debug'
    else:
        config.epochs = 30
        config.project = 'generative_model'
    return config

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
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
    df = pd.read_csv(os.path.join(config.data_path,config.in_file),sep='\t')
#    df = df[:config.n_case]
    Lung3D_ds = LungDataset_3D(df)
    data_loader = DataLoader(Lung3D_ds,
                                batch_size=config.batch,
                                shuffle=True,
                                num_workers=4)

    # batch = next(iter(data_loader))
    real_label = 1
    fake_label = 0

    # Model #
    # Generator
    netG = Generator().to(config.device)
    netG.apply(weights_init)
    # print(netG)
    optimizerG = torch.optim.Adam(netG.parameters(),
                             lr=config.lrG,
                             betas=(0.5,0.999))
    
    # Discriminator
    netD = Discriminator(in_channels=1).to(config.device)
    netD.apply(weights_init)
    # print(netD)
    optimizerD = torch.optim.Adam(netG.parameters(),
                             lr=config.lrD,
                             betas=(0.5,0.999))
    criterion = nn.BCELoss()
    noise = torch.randn(config.batch, 200,1,1,device=config.device)

    fixed_noise = torch.randn(config.batch, 200,device=config.device)

    img_list = []
    G_losses = []
    D_losses = []
    D_accs = []

    print('Start Training. . .')
    iter = 0
    for epoch in range(config.epochs):
        iters = len(data_loader)
        pbar = tqdm(enumerate(data_loader),total=iters)
        for i, batch in pbar:
            netD.zero_grad()
            real_img = batch['image'].to(config.device, dtype=torch.float)
            batch_size = real_img.shape[0]
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=config.device)
            output = netD(real_img).view(-1)
            lossD_real = criterion(output, label)
            lossD_real.backward()
            D_x = output.mean().item()
            acc_real = get_accuracy(label,output)

            noise = torch.randn(batch_size, 200,device=config.device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            lossD_fake = criterion(output,label)
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            acc_fake = get_accuracy(label,output)
            accD = (acc_fake + acc_real)/2
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            output = netD(fake.detach()).view(-1)
            lossG = criterion(output, label)
            lossG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            G_losses.append(lossG.item())
            D_losses.append(lossD.item())
            D_accs.append(accD)
            pbar.set_description(f'D loss:{lossD.item():.3f}, D acc:{accD:.3f}, G loss: {lossG.item():.3f}') 
            
            wandb.log({'iteration': iter,
                'D loss': lossD.item(), 'D acc': accD,
                'G loss': lossG.item()})

            iter += 1
            # break
        # evaluate
        # grid = vutils.make_grid(batch['image'][0,:,:,:,:].permute(3,0,1,2))
        # plt.figure(figsize=(15,15))
        # plt.subplot(1,2,1)
        # plt.axis('off')
        # plt.title('Real Images')
        # plt.imshow(grid.permute(1,2,0))
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        grid_fake = vutils.make_grid(fake[0,:,:,:,:].permute(3,0,1,2))
        plt.figure(figsize=(15,15))
        # plt.subplot(1,2,2)
        plt.axis('off')
        plt.title(f'Fake Images at Epoch {epoch}')
        plt.imshow(grid_fake.permute(1,2,0))

        wandb.log({'plot':plt})
        plt.close()


        # plt.figure(figsize=(15,15))
        # plt.subplot(1,2,1)
        # plt.axis('off')
        # plt.title('Real Images')
        # plt.imshow(vutils.make_grid(images, padding=5))
        # wandb.log({'plot':plt})

        # plt.subplot(1,2,2)
        # plt.axis("off")
        # plt.title("Fake Images")
        # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        # plt.show()
