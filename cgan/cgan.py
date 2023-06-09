import argparse
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

mnistDir = os.path.join("..", "data", "mnist")  # change directory as needed
os.makedirs("images", exist_ok=True)
cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200,
                    help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64,
                    help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: LR")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order")
parser.add_argument("--n_cpu", type=int, default=8,
                    help="number of cpu threads to be used by this model")
parser.add_argument("--latent_dim", type=int, default=100,
                    help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=10,
                    help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=32,
                    help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1,
                    help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400,
                    help="interval between image sampling")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Basically adding the labels to the images
        generateInput = torch.cat((self.label_emb(labels), noise), -1)
        Image = self.model(generateInput)
        Image = Image.view(Image.size(0), *img_shape)
        return Image


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input ( Similar to above)
        D_In = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(D_In)
        return validity


# Loss functions
adversarialLoss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarialLoss.cuda()

# Configure data loader
os.makedirs(mnistDir, exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        mnistDir,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(),
             transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizerGenerator = torch.optim.Adam(
    generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizerDiscriminator = torch.optim.Adam(
    discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(
        0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    generatedImages = generator(z, labels)
    save_image(generatedImages.data, "images/%d.png" %
               batches_done, nrow=n_row, normalize=True)

# For training purposes
for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        BatchSize = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(BatchSize, 1).fill_(
            1.0), requires_grad=False)
        fake = Variable(FloatTensor(BatchSize, 1).fill_(0.0),
                        requires_grad=False)

        # Configure input
        realImages = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # Generator

        optimizerGenerator.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(
            0, 1, (BatchSize, opt.latent_dim))))
        gen_labels = Variable(LongTensor(
            np.random.randint(0, opt.n_classes, BatchSize)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        gradientLoss = adversarialLoss(validity, valid)

        gradientLoss.backward()
        optimizerGenerator.step()

        #  Train Discriminator
        optimizerDiscriminator.zero_grad()

        # Loss of the real images
        validityReal = discriminator(realImages, labels)
        DRealLoss = adversarialLoss(validityReal, valid)

        # Loss for fake images
        validityFake = discriminator(gen_imgs.detach(), gen_labels)
        DFakeLoss = adversarialLoss(validityFake, fake)

        # Total discriminator loss
        DiscLoss = (DRealLoss + DFakeLoss) / 2

        DiscLoss.backward()
        optimizerDiscriminator.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
              (epoch, opt.n_epochs, i, len(dataloader), DiscLoss.item(), gradientLoss.item()))

        bacthesDone = epoch * len(dataloader) + i
        if bacthesDone % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=bacthesDone)
            torch.save({'epoch': epoch, 'loss': DiscLoss.item()}, './model.pth')
