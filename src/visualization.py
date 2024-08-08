import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from models.unet3d import PlainUNet3D


def crop_to_original(segmentation, original_shape):
    return segmentation[:original_shape[0], :original_shape[1], :original_shape[2]]

# Load the image
def load_nifti(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

# Save the segmentation output
def save_nifti(data, affine, file_path):
    nii_image = nib.Nifti1Image(data, affine)
    nib.save(nii_image, file_path)

def visualize_segmentation(image, segmentation):

    # Map segmentation to RGBA colors for overlay
    colors = np.array([
        [0, 0, 0, 0],       # Background - transparent
        [255, 0, 0, 128],   # Kidney - semi-transparent red
        [0, 0, 255, 128]    # Tumor - semi-transparent blue
    ])  # RGBA for background, kidney, tumor

    # Ensure the segmentation data is of integer type
    segmentation = segmentation.astype(int)

    # Check if the segmentation values are within the valid range for indexing colors
    if np.any(segmentation >= len(colors)):
        raise ValueError("Segmentation contains values out of range for color mapping.")

    # Map segmentation to RGB colors
    rgba_image = colors[segmentation]
    
    # Overlay the RGBA segmentation map on the original grayscale image
    plt.imshow(image, cmap='gray')  # Show original image in grayscale
    plt.imshow(rgba_image, interpolation='nearest')  # Overlay segmentation
    plt.axis('off')
    plt.savefig('overlay_one_case.png')
    plt.show()

if __name__ == '__main__':
    case_id = 211
    image_data, affine = load_nifti(f'../kits19_data/data/case_{case_id:05}/imaging.nii.gz')
    segmentation_data, _ = load_nifti(f'../kits19_data/data/segmentation/prediction_{case_id:05}.nii.gz')
    segmentation_data = crop_to_original(segmentation_data, image_data.shape)
    print(segmentation_data.shape)
    slice = image_data.shape[0] // 2
    # Assuming the middle slice is interesting
    original_slice = image_data[slice, :, :]
    segmentation_slice = segmentation_data[slice, :, :]

    visualize_segmentation(original_slice, segmentation_slice)