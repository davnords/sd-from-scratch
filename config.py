# T = 300
IMG_SIZE = 64
BATCH_SIZE = 32 # same as DDPM (if accounted for gradient accumulation)
GRADIENT_ACCUMULATION = 4
LR = 2e-4

# DDPM trains 800k steps -> 800 -> c. 3413 epochs of Celeb data

BETA_1 = 1e-4 # DDPM
BETA_T = 1e-2 # DDPM
# LINEAR SCHEDULER -> DDPM
# RANDOM HORIZONTAL FLIPS -> DDPM
# LR of 2e-4, 2e-5 for 256x256 -> DDPM
# Data scaled to -1, 1
T = 1000 # DDPM
