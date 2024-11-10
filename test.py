import torch.nn.functional as F
from utils import forward_diffusion_sample, sample_plot_image
from model import Unet
import torch
from dataset import CelebaDataset
from config import IMG_SIZE, BATCH_SIZE, T, GRADIENT_ACCUMULATION, LR
from torchvision import transforms 
from torch.utils.data import DataLoader
from tqdm import tqdm
import time


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')  # Enable high precision for float32 matmul operations

def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)

def get_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = CelebaDataset(transform=data_transform)
    train_dataloader= DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    return train_dataloader, None

def train():
    model = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        flash_attn=False,
    ).to(device=device)

    print('Number of parameters: ', sum(p.numel() for p in model.parameters()))

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    epochs = 5000

    train_data, test_data = get_dataset()

    for epoch in tqdm(range(epochs)):
        epoch_start_time = time.time()
        for step, batch in enumerate(tqdm(train_data)):
            if step%GRADIENT_ACCUMULATION==0:
                optimizer.zero_grad()
            t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
            loss = get_loss(model, batch, t)
            loss.backward()    
            if (step+1)%GRADIENT_ACCUMULATION==0 or (step+1)==len(train_data):
                optimizer.step()
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch + 1}/{epochs} runs at {(epoch_duration/len(train_data)):.2f} seconds / batch")
        if epoch%5 == 0:
            print(f"\nEpoch {epoch + 1} achieves a loss of: ",loss.item())
            sample_plot_image(model, device=device, name=f"sample_{epoch}.png")
        if epoch%100 == 0:
            torch.save(model, f"checkpoints/model_{epoch}.pt")

if __name__=="__main__":
    train()