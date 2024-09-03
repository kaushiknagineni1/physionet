import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchprofile import profile_macs
import numpy as np
import time

# Check if A100 GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ViT model
model = vit_b_16(pretrained=True).to(device)
model.eval()  # Set the model to evaluation mode

# Function to get embeddings
def get_embeddings(model, input_tensor):
    with torch.no_grad():
        embeddings = model(input_tensor)
    return embeddings

# Create a dummy input tensor with batch size 1 and the correct input size
batch_size = 16  # Adjust as necessary
dummy_input = torch.randn(batch_size, 3, 224, 224).to(device)  # Adjust size based on model requirements

# Profile the time for a single batch
start_time = time.time()
embeddings = get_embeddings(model, dummy_input)
end_time = time.time()

# Calculate time taken for a single batch
batch_time = end_time - start_time
print(f'Time taken for one batch: {batch_time:.4f} seconds')

# Total number of images
total_images = 377110  # based on physionet dataset

# Calculate total time
num_batches = total_images // batch_size + (1 if total_images % batch_size != 0 else 0)
total_time = num_batches * batch_time
print(f'Estimated total time: {total_time:.2f} seconds ({total_time / 3600:.2f} hours)')

# Save embeddings to disk (for demonstration, saving one batch)
embeddings_cpu = embeddings.cpu().numpy()
np.save('embeddings_batch.npy', embeddings_cpu)
print('Embeddings for one batch saved to disk.')
