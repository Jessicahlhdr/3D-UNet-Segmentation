import torch
import numpy as np

import nibabel as nib
import SimpleITK as sitk
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from models.unet3d import ResidualUNet3D
from resample_one_case import resample_and_visualize, upscale_image
from configs import load_config
from pathlib import Path
from utils.ensemble import ensemble_predict

def load_nifti(file_path):
    """ Load a NIfTI file as numpy array and affine matrix. """
    nii = nib.load(file_path)
    return nii.get_fdata(), nii.affine

def save_nifti(data, affine, file_path):
    """ Save a numpy array as a NIfTI file. """
    ensure_dir(file_path)
    nii_image = nib.Nifti1Image(data, affine)
    nib.save(nii_image, file_path)

def numpy_to_sitk(image_array, affine):
    """
    Convert a numpy array to a SimpleITK Image with the given affine transformation parameters.

    Args:
        image_array (numpy.ndarray): The image data as a numpy array.
        affine (tuple): A tuple containing the direction, origin, and spacing of the image.

    Returns:
        SimpleITK.Image: The image as a SimpleITK Image object.
    """
    sitk_image = sitk.GetImageFromArray(image_array)
    sitk_image.SetDirection(affine[0])
    sitk_image.SetOrigin(affine[1])
    sitk_image.SetSpacing(affine[2])
    return sitk_image

def pad_image_to_fit_windows(image, window_size, overlap, config):
    """ Pad the image so that it can be evenly divided into windows of the specified size with the given overlap. """
    target_sizes = []
    intensity_min = config.TRANSFORM.INTENSITY_MIN
    intensity_max = config.TRANSFORM.INTENSITY_MAX
    intensity_mean = config.TRANSFORM.INTENSITY_MEAN
    intensity_std = config.TRANSFORM.INTENSITY_STD
    
    image = np.clip(image, intensity_min, intensity_max)
    image = (image - intensity_mean) / intensity_std
    for dim, size in enumerate(image.shape):
        if size < window_size[dim]:
            target_size = window_size[dim]
        else:
            step = window_size[dim] * (1 - overlap)
            num_steps = np.ceil((size - window_size[dim]) / step) + 1
            target_size = int(window_size[dim] + (num_steps - 1) * step)
        target_sizes.append(target_size)

    padding = [(0, max(0, target - image.shape[dim])) for dim, target in enumerate(target_sizes)]
    image_padded = np.pad(image, padding, mode='constant', constant_values=config.TRANSFORM.IMAGE_PAD_VALUE)

    return image_padded

def calculate_windows(image_shape, window_size, overlap=0.5):
    """ Calculate how many windows fit into the image and their sizes.
    Returns:
        num_windows: A list containing the number of windows along each dimension.
        step_size: A list of tuples, where each tuple contains (step, window_size) for each dimension.
    """
    # Calculate step size based on window size and overlap
    step_size = [(int(ws * (1 - overlap)), ws) for ws in window_size]
    
    # Initialize num_windows list
    num_windows = []
    
    # Calculate the number of windows for each dimension
    for idx, (s, (step, ws)) in enumerate(zip(image_shape, step_size)):
        # Calculate the number of steps that can be taken
        steps = (s - ws) / step
        
        # If there is a remainder, add an additional window
        if (s - ws) % step != 0:
            num = int(np.ceil(steps)) + 1
        else:
            num = int(steps) + 1
        
        # Ensure there is at least one window in each dimension
        num_windows.append(max(1, num))
    """
    print("Number of windows:", num_windows)
    print("Step sizes:", step_size)
    """
    return num_windows, step_size

def extract_windows(image, num_windows, window_size, steps):
    """ Extract windows from the padded image. """
    windows = []
    for z in range(num_windows[0]):
        for y in range(num_windows[1]):
            for x in range(num_windows[2]):
                start_z = z * steps[0][0]
                start_y = y * steps[1][0]
                start_x = x * steps[2][0]
                end_z = min(start_z + window_size[0], image.shape[0])
                end_y = min(start_y + window_size[1], image.shape[1])
                end_x = min(start_x + window_size[2], image.shape[2])
                window = image[start_z:end_z, start_y:end_y, start_x:end_x]
                pad_z = window_size[0] - (end_z - start_z)
                pad_y = window_size[1] - (end_y - start_y)
                pad_x = window_size[2] - (end_x - start_x)
                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    window = np.pad(window, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

                """
                plt.figure(figsize=(8,6))
                plt.imshow(window[:,:,window.shape[2] // 2].transpose(1,0), cmap='gray')
                plt.savefig(f'windowing/window_{z}_{y}_{x}.png')
                plt.close()
                """

                windows.append((window, (slice(start_z, end_z), slice(start_y, end_y), slice(start_x, end_x))))

    return windows

def preprocess(image):
    """ Convert it to a tensor on GPU. """
    return torch.tensor(image).float().unsqueeze(0).unsqueeze(0).to('cuda') 

def infer_windows(model, windows):
    """ Perform inference on extracted windows using GPU. """
    segmented = []
    for window, slices in tqdm(windows):
        input_tensor = preprocess(window)
        # input_tensor: (N, C, X, Y, Z)
        input_tensor = input_tensor
        with torch.no_grad():
            output = ensemble_predict(model, input_tensor) 
            logits = output.squeeze(0).cpu().numpy()
            # logits: (C, X, Y, Z) -> (X, Y, Z, C)
            logits = logits.transpose(1, 2, 3, 0)
            segmented.append((logits, slices))
    return segmented

def softmax(x):
    """Compute softmax values for each sets of scores in x along the last dimension."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))  # subtract max for numerical stability
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def reconstruct_volume(segmented, image_shape):
    """ Reconstruct the full volume from segmented windows. """
    full_volume = np.zeros(image_shape + (3,), dtype=np.float32)
    for logits, slices in segmented:
        full_volume[slices] += logits
    return np.argmax(softmax(full_volume), axis=-1).astype(np.uint8)

def crop_to_original(segmentation, original_shape):
    """ Crop the padded segmentation back to the original image size. """
    return segmentation[:original_shape[0], :original_shape[1], :original_shape[2]]

def ensure_dir(file_path):
    """ Ensure that the directory of the file path exists. """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_case(case_id, model, window_size, overlap, config):
    # Load the original image and its metadata
    kits19_dir = Path(config.DATA.KITS19_DIR)
    path = kits19_dir / f'case_{case_id:05}/imaging.nii.gz'
    sitk_image = sitk.ReadImage(str(path))
    affine = (sitk_image.GetDirection(), sitk_image.GetOrigin(), sitk_image.GetSpacing())

    # Resample and process the image
    original_array, image_data = resample_and_visualize(config, case_id, is_label=False)
    original_shape = image_data.shape

    image_data = pad_image_to_fit_windows(image_data, window_size, overlap, config)

    num_windows, steps = calculate_windows(image_data.shape, window_size, overlap)
    windows = extract_windows(image_data, num_windows, window_size, steps)
    segmented_windows = infer_windows(model, windows)
    output = reconstruct_volume(segmented_windows, image_data.shape)

    output = crop_to_original(output, original_shape)
    """
    plt.figure(figsize=(8,6))
    img = plt.imshow(output[:,:, output.shape[2] // 2])
    plt.colorbar(img)
    plt.savefig('output.png')
    plt.close()
    """

    # Convert the output numpy array to SimpleITK image
    sitk_output = numpy_to_sitk(output, affine)

    # Resample back to original size
    original_sitk_image = sitk.ReadImage(str(path))  # Reload the original image if necessary
    resampled_output = upscale_image(sitk_output, original_array.transpose(2,1,0).shape)
    
    """
    output_array = sitk.GetArrayFromImage(resampled_output).astype(int)
    plt.figure(figsize=(8,6))
    img = plt.imshow(output_array[:,:, output_array.shape[2] // 2])
    plt.colorbar(img)
    plt.savefig('output_test.png')
    plt.close()
    """
    
    # Save the resampled output
    output_path = f'../kits19_data/data/segmentation/prediction_{case_id:05}.nii.gz'
    ensure_dir(output_path)

    sitk.WriteImage(resampled_output, output_path)

models = [] 
for i in range(0, 4):
    model = PlainUNet3D()
    state_dict = torch.load(f"./best_model/outputs_residual_unet_0.03/fold_{i}/model_best.pth")
    model.load_state_dict(state_dict, strict=False)
    model = model.to('cuda') 
    model.eval()
    models.append(model)

window_size = (160, 160, 80)
overlap = 0.5
config = load_config()

# Process cases from case_00210 to case_00299
for case_number in tqdm(range(210, 300)):
    case_id = case_number
    print(f"Processing {'case_00{case_number}'}")
    process_case(case_id, models, window_size, overlap, config)
    print(f"Completed {'case_00{case_number}'}")
