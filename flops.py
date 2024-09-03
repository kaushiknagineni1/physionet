import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchprofile import profile_macs

if torch.cuda.is_available():
    print("cuda")
else:
    print("cpu")

# Check if A100 GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ViT model
model = vit_b_16(pretrained=True).to(device)

# Create a dummy input tensor with batch size 1 and the correct input size
dummy_input = torch.randn(1, 3, 224, 224).to(device)  # Adjust size based on model requirements

# Calculate the FLOPs (Multiply-Accumulate Operations, MACs)
macs = profile_macs(model, dummy_input)

# Convert MACs to FLOPs (1 MAC = 2 FLOPs)
flops = 2 * macs

print(f'Number of FLOPs: {flops}')
