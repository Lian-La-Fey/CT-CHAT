import os
import torch
from transformer_maskgit import CTViT
import numpy as np
import nibabel as nib
import argparse
import torch.nn.functional as F

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from glob import glob


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

def nii_img_to_tensor(path, slope, intercept, xy_spacing, z_spacing, device='cpu'):
    nii_img = nib.load(str(path))
    img_data = nii_img.get_fdata()

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5
    
    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)

    img_data = slope * img_data + intercept
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = np.transpose(img_data, (1, 2, 0))

    img_data = (((img_data) / 1000)).astype(np.float32)
    slices = []

    tensor = torch.tensor(img_data)
    # Get the dimensions of the input tensor
    target_shape = (480, 480, 240)

    # Extract dimensions
    h, w, d = tensor.shape

    # Calculate cropping/padding values for height, width, and depth
    dh, dw, dd = target_shape
    h_start = max((h - dh) // 2, 0)
    h_end = min(h_start + dh, h)
    w_start = max((w - dw) // 2, 0)
    w_end = min(w_start + dw, w)
    d_start = max((d - dd) // 2, 0)
    d_end = min(d_start + dd, d)

    # Crop or pad the tensor
    tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

    pad_h_before = (dh - tensor.size(0)) // 2
    pad_h_after = dh - tensor.size(0) - pad_h_before

    pad_w_before = (dw - tensor.size(1)) // 2
    pad_w_after = dw - tensor.size(1) - pad_w_before

    pad_d_before = (dd - tensor.size(2)) // 2
    pad_d_after = dd - tensor.size(2) - pad_d_before

    tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

    tensor = tensor.permute(2, 0, 1)

    tensor = tensor.unsqueeze(0)

    return tensor.to(device=device)

def _mp_fn(index, args, file_list):
    device = xm.xla_device()
    world_size = xr.world_size()
    
    print(f"--> Starting process {index+1}/{world_size} on device: {device}")
    
    image_encoder = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    ).to(device).eval()
    
    ct_clip_weights = torch.load("/home/raspuntinov/gcs/report_gen_models/ct_clip/CT-CLIP_v2.pt", map_location="cpu")
    visual_weights = {}
    for key, value in ct_clip_weights.items():
        if key.startswith('visual_transformer.'):
            new_key = key[len('visual_transformer.'):]
            visual_weights[new_key] = value
    
    image_encoder.load_state_dict(visual_weights, strict=False)
    
    # --- File Processing Loop ---
    # Each process iterates over a unique slice of the file list.
    # For example, with 4 TPUs:
    # Process 0 gets files 0, 4, 8, ...
    # Process 1 gets files 1, 5, 9, ...
    for file_path in file_list[index::world_size]:
        try:
            print(f"Process {index+1}: Processing {os.path.basename(file_path)}")
            
            image = nii_img_to_tensor(
                path=file_path,
                slope=args.slope,
                intercept=args.intercept,
                xy_spacing=args.xy_spacing,
                z_spacing=args.z_spacing,
                device=device
            )

            image_encoded = image_encoder(image.unsqueeze(0), return_encoded_tokens=True)

            image_name = os.path.basename(file_path).split(".")[0]
            output_path = os.path.join(args.output_dir, f'{image_name}.npz')
            
            np.savez(output_path, arr=image_encoded.cpu().detach().numpy())

        except Exception as e:
            print(f"Process {index+1}: Failed to process {file_path}. Error: {e}")

    # A barrier ensures all processes finish before the master process continues.
    xm.rendezvous("all_processes_done")
    print(f"<-- Process {index+1}/{world_size} finished.")

def main():
    parser = argparse.ArgumentParser(description='Process a folder of NIfTI images in parallel on multiple TPUs.')

    # CHANGED: Argument now takes a folder path
    parser.add_argument('--folder_path', type=str, default="/home/raspuntinov/gcs/ct_rate/dataset/valid_fixed")
    parser.add_argument('--output_dir', type=str, default='/home/raspuntinov/gcs/ct_rate/dataset/embeddings', help='Directory to save the output embeddings.')
    parser.add_argument('--slope', type=float, default=1, help='Slope for rescaling the image.')
    parser.add_argument('--intercept', type=float, default=0, help='Intercept for rescaling the image.')
    parser.add_argument('--xy_spacing', type=float, default=1, help='XY spacing of the image.')
    parser.add_argument('--z_spacing', type=float, default=1, help='Z spacing of the image.')

    args = parser.parse_args()
    
    nii_gz_files = glob(f"{args.folder_path}/**/*.nii.gz", recursive=True)
    all_files = sorted(nii_gz_files)

    if not all_files:
        print(f"Error: No .nii or .nii.gz files found in '{args.folder_path}'.")
        return

    print(f"Found {len(all_files)} files to process.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        
    print("Spawning processes for all available TPU devices...")
    # xs.xla_spawn(_mp_fn, args=(args, all_files))
    torch_xla.launch(_mp_fn, args=(args, all_files))
    print("All processing complete.")

if __name__ == '__main__':
    main()