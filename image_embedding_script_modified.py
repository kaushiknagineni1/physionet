
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import timm  # Huggingface timm library to access the requested models

# Function to load image paths from a file with error handling
def load_image_paths(file_path):
    try:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []

# Function to load and preprocess an image with error handling
def load_and_preprocess_image(image_path, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    except (IOError, OSError) as e:
        print(f"Error loading image '{image_path}': {e}")
        return None

# Function to get embeddings
def get_embeddings(model, input_tensor):
    with torch.no_grad():
        embeddings = model(input_tensor)
    return embeddings

# List of supported models
supported_models = {
    'swin': 'swin_base_patch4_window7_224',
    'deit': 'deit_base_patch16_224',
    'beit': 'beit_base_patch16_224',
    'cvt': 'cvt_21_384',
    'pit': 'pit_base',
    'cait': 'cait_s24_224',
    'coatnet': 'coatnet_1_224',
    'levit': 'levit_256',
    't2t_vit': 't2t_vit_14'
}

# Function to load the model
def load_model(model_name):
    if model_name not in supported_models:
        raise ValueError(f"Model '{model_name}' not supported.")
    return timm.create_model(supported_models[model_name], pretrained=True, num_classes=0).to(device)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Select the model you want to use (e.g., 'swin', 'deit', 'beit', etc.)
model_name = 'swin'  # Replace this with the desired model name
model = load_model(model_name)
model.eval()

# Define image transformations (resize to 224x224 and normalize with ImageNet stats)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load image paths
image_paths = load_image_paths('image_paths.txt')

# Batch processing
batch_size = 16
num_batches = len(image_paths) // batch_size + (1 if len(image_paths) % batch_size != 0 else 0)

# Create the output directory if it doesn't exist
output_dir = "./embeddings_output"
os.makedirs(output_dir, exist_ok=True)

# Process images in batches
for i in range(num_batches):
    batch_paths = image_paths[i*batch_size:(i+1)*batch_size]

    # Filter out any None values from image loading errors
    batch_images = [load_and_preprocess_image(p, transform) for p in batch_paths]
    batch_images = [img for img in batch_images if img is not None]

    if not batch_images:
        print(f"Skipping batch {i+1}/{num_batches} due to image loading errors.")
        continue

    batch_images = torch.stack(batch_images).to(device)

    # Get embeddings
    embeddings = get_embeddings(model, batch_images)
    
    # Convert embeddings to CPU and save in .pt format
    embeddings_cpu = embeddings.cpu()
    output_path = os.path.join(output_dir, f'embeddings_batch_{i}.pt')
    torch.save(embeddings_cpu, output_path)

    print(f'Batch {i+1}/{num_batches} processed and saved.')

print('All batches processed.')
