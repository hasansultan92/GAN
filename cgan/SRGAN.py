import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils
from torch.utils.data import DataLoader
#from my_dataset import MyDataset
# Define the generator network


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.deconv1 = nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(
            256, 128, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.bn11 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=9, padding=4)

    # Simple convolution
    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.relu(self.bn3(self.conv3(x)))
        x = nn.functional.relu(self.bn4(self.conv4(x)))
        x = nn.functional.relu(self.bn5(self.conv5(x)))
        x = nn.functional.relu(self.bn6(self.conv6(x)))
        x = nn.functional.relu(self.bn7(self.conv7(x)))
        x = nn.functional.relu(self.bn8(self.conv8(x)))
        x = nn.functional.relu(self.bn9(self.deconv1(x)))
        x = nn.functional.relu(self.bn10(self.deconv2(x)))
        x = nn.functional.relu(self.bn11(self.deconv3(x)))
        x = self.deconv4(x)
        return x

# Define the discriminator network
# Simple convolution


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512 * 6 * 6, 1024)
        self.fc2 = nn.Linear(1024, 1)


def forward(self, x):
    x = nn.functional.relu(self.conv1(x))
    x = nn.functional.relu(self.bn1(self.conv2(x)))
    x = nn.functional.relu(self.bn2(self.conv3(x)))
    x = nn.functional.relu(self.bn3(self.conv4(x)))
    x = nn.functional.relu(self.bn4(self.conv5(x)))
    x = nn.functional.relu(self.bn5(self.conv6(x)))
    x = nn.functional.relu(self.bn6(self.conv7(x)))
    x = nn.functional.relu(self.bn7(self.conv8(x)))
    x = x.view(-1, 512 * 6 * 6)
    x = nn.functional.relu(self.fc1(x))
    x = torch.sigmoid(self.fc2(x))
    return x


def train_gan(generator, discriminator, criterion, optimizer_g, optimizer_d, dataloader, num_epochs, device):
    for epoch in range(num_epochs):
        print(epoch)
        for i, (lr_imgs, hr_imgs) in enumerate(dataloader):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            real_labels = Variable(torch.ones(hr_imgs.size(0), 1)).to(device)
            fake_labels = Variable(torch.zeros(hr_imgs.size(0), 1)).to(device)
            #print(fake_labels, real_labels)
        # Train the generator
        optimizer_g.zero_grad()
        fake_hr_imgs = generator(lr_imgs)
        outputs = discriminator(fake_hr_imgs)
        loss_g = criterion(outputs, real_labels)
        loss_g.backward()
        optimizer_g.step()

        # Train the discriminator
        optimizer_d.zero_grad()
        outputs_real = discriminator(hr_imgs)
        loss_d_real = criterion(outputs_real, real_labels)
        outputs_fake = discriminator(fake_hr_imgs.detach())
        loss_d_fake = criterion(outputs_fake, fake_labels)
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        optimizer_d.step()

        # Print the loss
        if i % 100 == 0:
            print("Epoch [{}/{}], Step [{}/{}], Loss G: {:.4f} Loss D: {:.4f}".format(
                epoch+1, num_epochs, i+1, len(dataloader), loss_g.item(), loss_d.item()))
        # Evaluate the generator on the validation set
        if (epoch+1) % 5 == 0:
            generator.eval()
            total_psnr = 0.0
            with torch.no_grad():
                for lr_imgs_val, hr_imgs_val in dataloader:
                    lr_imgs_val = lr_imgs_val.to(device)
                    hr_imgs_val = hr_imgs_val.to(device)
                    fake_hr_imgs_val = generator(lr_imgs_val)
                    psnr_val = PSNR(fake_hr_imgs_val, hr_imgs_val)
                    total_psnr += psnr_val.item()
            avg_psnr = total_psnr / len(dataloader_val)
            print("Average PSNR on validation set: {:.4f}".format(avg_psnr))
            generator.train()

    return generator, discriminator


def PSNR(img1, img2):
    mse = nn.functional.mse_loss(img1, img2)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


class MyDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        print(data_dir)
        self.image_filenames = os.listdir(self.data_dir)

    def __getitem__(self, index):
        # Load the low-resolution and high-resolution images
        lr_img = Image.open(os.path.join(
            self.data_dir, self.image_filenames[index], "lr.png"))
        hr_img = Image.open(os.path.join(
            self.data_dir, self.image_filenames[index], "hr.png"))

        # Resize the images to the desired input/output size
        lr_img = lr_img.resize((64, 64))
        hr_img = hr_img.resize((256, 256))

        # Convert the images to tensors
        lr_img = transforms.ToTensor()(lr_img)
        hr_img = transforms.ToTensor()(hr_img)

        return lr_img, hr_img

    def __len__(self):
        return len(self.image_filenames)


if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = datasets.MNIST(root='./data', train=True,
                      transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = datasets.MNIST(root='./data', train=False,
                    transform=transform, download=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

print(val_dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)
generator = Generator()
discriminator = Discriminator()


criterion = nn.BCELoss()
optimizer_g = torch.optim.Adam(
    generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(
    discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

generator_trained, discriminator_trained = train_gan(
    generator, discriminator, criterion, optimizer_g, optimizer_d, train_loader, num_epochs=10, device=device)
torch.save(generator_trained.state_dict(), "./SRGAN.pt")
