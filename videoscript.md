# Videoscript

## Inspiration

I took inspiration from this great video (diffusion from scratch) and Andrej Karpathy's videos on training neural nets. I felt that for interested viewers we could take it one step further to really learn how researchers or professionals train, monitor and deploy models using PyTorch. 

Timestamps will be provided.

## What you will learn

This will be great for learning more advanced PyTorch and seeing the power of going from empty model to deployed. In learning how to train a model we will touch on topics such as: 
- Coding a Unet
- Building an attention mechanism
- Processing image data
- Monitoring training using wandb
- Speeding up training using torch.compile
- Parallelize training on a GPU cluster using torch run
- Structuring a deep learning project (with argparse and checkpoints etc...)
- Creating a simple web ui to make inference on the model 

## Step 1
The first part will be to train a model that as closely resembles the original DDPM paper as possible (CelebaHQ dataset etc..)
## Step 2 
Step 2 is to add CLIP embeddings to the training 