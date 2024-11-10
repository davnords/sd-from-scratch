from torch.utils.data import Dataset
from PIL import Image   

class CelebaDataset(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
    
    def __len__(self):
        return 30000

    def __getitem__(self, index):
        img_url = f"../datasets/CelebAMask-HQ/CelebA-HQ-img/{index}.jpg"
        image = Image.open(img_url)
        if self.transform:
            image = self.transform(image)
        return image