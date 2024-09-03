import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import vit_b_16
from PIL import Image
import numpy as np
import time
import os
    
# Function to load image paths from a file with error handling
def load_image_paths(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []  # Return an empty list to prevent further processing

# Function to load and preprocess an image with error handling
def load_and_preprocess_image(image_path, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    except (IOError, OSError) as e:
        print(f"Error loading image '{image_path}': {e}")
        return None  # Return None to indicate failure

# Function to get embeddings
def get_embeddings(model, input_tensor):
    with torch.no_grad():
        embeddings = model(input_tensor)
    return embeddings

# Check if A100 GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the ViT model
model = vit_b_16(pretrained=True).to(device)
model.eval()  # Set the model to evaluation mode

# Define image transformations (resize to 224x224 for ViT, normalize with ImageNet stats)
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize image to 224x224 as required by ViT
    transforms.ToTensor(),         # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize with ImageNet stats
])

# Load image paths
image_paths = load_image_paths('image_paths.txt')

# Batch processing
batch_size = 16  # Adjust as necessary
num_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size != 0 else 0)

# Create the "embeddings_output" directory if it doesn't exist
output_dir = "/pscratch/sd/k/kn13/embeddings_out_1"
os.makedirs(output_dir, exist_ok=True)
    
# Process images in batches with progress monitoring
for i in range(num_batches):
    batch_paths = image_paths[i*batch_size:(i+1)*batch_size]

    # Filter out any None values resulting from image loading errors
    batch_images = [load_and_preprocess_image(p, transform) for p in batch_paths]
    batch_images = [img for img in batch_images if img is not None]

    if not batch_images:  # Skip the batch if all images failed to load
        print(f"Skipping batch {i+1}/{num_batches} due to image loading errors.")
        continue

    batch_images = torch.stack(batch_images).to(device)

    # Get embeddings
    embeddings = get_embeddings(model, batch_images)
    
    # Convert embeddings to CPU and save
    embeddings_cpu = embeddings.cpu().numpy()
    output_path = os.path.join(output_dir, f'embeddings_batch_{i}.npy')
    np.save(output_path, embeddings_cpu)

    print(f'Batch {i+1}/{num_batches} processed and saved.')  # Progress update


print('All batches processed.')
