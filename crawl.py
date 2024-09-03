import os
import glob

def collect_image_paths(base_dir, file_extensions=['.png', '.jpg', '.jpeg']):
    """
    Recursively collect all image paths in the given directory.

    Parameters:
    base_dir (str): The base directory to start searching for images.
    file_extensions (list): List of file extensions to consider as images.

    Returns:
    list: List of paths to image files.
    """
    image_paths = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in file_extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Base directory where the PhysioNet database is stored
base_dir = '/global/cfs/cdirs/m1532/Projects_MVP/_datasets/mimic_cxr_jpg/'

# Collect all image paths
image_paths = collect_image_paths(base_dir)

# Save the image paths to a text file
with open('image_paths.txt', 'w') as f:
    for path in image_paths:
        f.write(f"{path}\n")

print(f'Collected {len(image_paths)} image paths.')
