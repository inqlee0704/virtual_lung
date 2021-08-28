import torch
import torch.nn as nn
from torchsummary import summary
# Generator Code

class Generator(nn.Module):
    def __init__(self, in_channels=1, img_size=64):
        super(Generator, self).__init__()
        z = 100
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # ConvTranspose2d(in_c, out_c, kernel_size, stride, padding)
            nn.ConvTranspose2d(z, img_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.ReLU(True),
            # state size. (img_size*8) x 4 x 4
            nn.ConvTranspose2d(img_size * 8, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.ReLU(True),
            # state size. (img_size*4) x 8 x 8
            nn.ConvTranspose2d(img_size * 4, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.ReLU(True),
            # state size. (img_size*2) x 16 x 16
            nn.ConvTranspose2d(img_size * 2, img_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size),
            nn.ReLU(True),
            # state size. (img_size) x 32 x 32
            nn.ConvTranspose2d(img_size, in_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (in_channels) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, img_size=64):
        super(Discriminator, self).__init__()
        # img_size = 64
        self.main = nn.Sequential(
            # input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, img_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size) x 32 x 32
            nn.Conv2d(img_size, img_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size*2) x 16 x 16
            nn.Conv2d(img_size * 2, img_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size*4) x 8 x 8
            nn.Conv2d(img_size * 4, img_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(img_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (img_size*8) x 4 x 4
            nn.Conv2d(img_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


def test_gan3d():
    model_generator = Generator()
    batch = 1
    z = 100
    noise = torch.randn(batch, z, 1, 1)
    generated_img = model_generator(noise)
    print("Generator output shape", generated_img.shape)
    model_discriminator = Discriminator()
    out = model_discriminator(generated_img)
    print("Discriminator output", out)
    summary(model_generator.to('cuda'), (100,1,1))
    summary(model_discriminator.to('cuda'), (1, 64, 64))

# this is for test
# test_gan3d()