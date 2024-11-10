import torch
from utils import sample_plot_image

device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('checkpoints/model_300.pt').to(device)
# model.load_state_dict(torch.load('checkpoints/model_300.pt')).to(device)


img = sample_plot_image(model, device=device, return_image=True)
print(img.shape)