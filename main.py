from dataset import CelebaDataset
from torch.utils.data import DataLoader
from utils import show_images, show_tensor_image, forward_diffusion_sample
from config import IMG_SIZE, BATCH_SIZE, T, LR
from torchvision import transforms 
import torch
import matplotlib.pyplot as plt
from unet import SimpleUnet

# show_images(data)

data_transforms = [
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # Scales data into [0,1] 
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
]
data_transform = transforms.Compose(data_transforms)

dataset = CelebaDataset(transform=data_transform)
dataloader= DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)


# Simulate forward diffusion

model = SimpleUnet()