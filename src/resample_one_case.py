import timeit
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm

from visualization import visualize_segmentation
from configs import load_config

def resample_spacing(config, itk_image, resample_method):
    """Apply resampling to an image with a fixed spacing.

    Args:
        config (YACS CfgNode): config.
        itk_image (SimpleITK Image): SimpleITK Image object to resample.
        resample_method (int): SimpleITK resampler flag.

    Returns:
        SimpleITK Image: SimpleITK Image object after resampling.
    """
    size = np.array(itk_image.GetSize())
    spacing = np.array(itk_image.GetSpacing())
    new_spacing = config.DATA.SPACING
    new_size = (size * (spacing / new_spacing)).astype(int)

    return sitk.Resample(itk_image, new_size.tolist(), sitk.Transform(), resample_method, itk_image.GetOrigin(),
                         new_spacing, itk_image.GetDirection(), 0.0, itk_image.GetPixelID())


def resample_and_visualize(config, case_id, is_label):
    """Load an image, apply resampling with a given spacing, and return both original and resampled arrays.

    Args:
        config (YACS CfgNode): config.
        case_id (int): case id of the target image.
        is_label (bool): True if target is labels.
    """
    filename = 'segmentation' if is_label else 'imaging'
    dtype = int if is_label else float
    resample_method = sitk.sitkNearestNeighbor if is_label else sitk.sitkBSplineResamplerOrder3

    kits19_dir = Path(config.DATA.KITS19_DIR)
    path = kits19_dir / f'case_{case_id:05}/{filename}.nii.gz'
    itk_image = sitk.ReadImage(str(path))
    original_array = sitk.GetArrayFromImage(itk_image).astype(dtype)

    # Resample the image
    resampled_image = resample_spacing(config, itk_image, resample_method)
    resampled_array = sitk.GetArrayFromImage(resampled_image).astype(dtype)
    
    return original_array, resampled_array


def upscale_image(original_segmentation, new_size):
    """
    Upscale an image to a new size using nearest neighbor interpolation.
    
    Args:
        original_segmentation (SimpleITK.Image): The original segmentation image.
        new_size (tuple): The desired new size (width, height).
        
    Returns:
        SimpleITK.Image: The upscaled image.
    """
    original_size = original_segmentation.GetSize()
    original_spacing = original_segmentation.GetSpacing()
    
    # Calculate the new spacing based on the original size and new size
    new_spacing = (original_size[0] * original_spacing[0] / new_size[0], 
                   original_size[1] * original_spacing[1] / new_size[1],
                   original_size[2] * original_spacing[2] / new_size[2])
    
    # Setup the resample image filter
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_filter.SetOutputPixelType(original_segmentation.GetPixelID())
    
    # Use the original image properties
    resample_filter.SetOutputOrigin(original_segmentation.GetOrigin())
    resample_filter.SetOutputDirection(original_segmentation.GetDirection())
    
    return resample_filter.Execute(original_segmentation)

def main():
    t0 = timeit.default_timer()
    config = load_config()
    print('successfully loaded config:')

    # Specify the case to process
    case_id = 1  # Example case ID
    print(f"Processing case: {case_id}")

    # Process and visualize image and labels
    original_image, resampled_image = resample_and_visualize(config, case_id, is_label=False)
    original_segmentation, resampled_segmentation = resample_and_visualize(config, case_id, is_label=True)
    
    visualize_resampled_case(original_image, resampled_image, original_segmentation, resampled_segmentation)

    elapsed = timeit.default_timer() - t0
    print('time: {:.3f} min'.format(elapsed / 60.0))

if __name__ == '__main__':
    def visualize_resampled_case(original_image, resampled_image, original_segmentation, resampled_segmentation):
        # Visualize central slice
        central_slice_idx = original_image.shape[2] // 2
        central_slice_idx_resampled = resampled_image.shape[2] // 2
        orig_img = original_image[:, :, central_slice_idx]
        resamp_img = resampled_image[:, :, central_slice_idx_resampled]
        orig_seg = original_segmentation[:, :, central_slice_idx]
        resamp_seg = resampled_segmentation[:, :, central_slice_idx_resampled]

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 2, 1)
        plt.imshow(orig_img.transpose(1,0), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(resamp_img.transpose(1,0), cmap='gray')
        plt.title('Resampled Image')
        plt.axis('off')

        plt.subplot(2, 2, 3)
        visualize_segmentation(orig_img.transpose(1,0), orig_seg.transpose(1,0))

        plt.subplot(2, 2, 4)
        visualize_segmentation(resamp_img.transpose(1,0), resamp_seg.transpose(1,0))

        plt.show()

    main()