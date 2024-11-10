from flask import Flask
from flask import request
from unet import SimpleUnet
import torch
from utils import sample_plot_image
from PIL import Image
import io
from flask import send_file

app = Flask(__name__)
device = device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = torch.load('checkpoints/model_300.pt').to(device)
# model.load_state_dict(torch.load('checkpoints/model_300.pt')).to(device)

@app.route('/generate', methods=['GET'])
def generate():
    text_prompt = request.args.get('prompt')
    img = sample_plot_image(model, device=device, return_image=True)

     # Convert tensor to PIL image
     
    img = img.squeeze()
    img = img.permute(1, 2, 0)
    img = (img * 255).clamp(0, 255).byte()
    img = Image.fromarray(img.cpu().numpy())

    # Save image to a bytes buffer
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    buffered.seek(0)  # Rewind the buffer to start

    # Send as a binary file response
    return send_file(buffered, mimetype='image/png')