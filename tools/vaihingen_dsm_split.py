import glob
import os
import numpy as np
import cv2
from PIL import Image
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
import argparse
import torch
import albumentations as albu
from torchvision.transforms import (Pad, Resize, RandomHorizontalFlip, RandomVerticalFlip)
import random

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_dsm_padded(dsm_array, patch_size, mode):
    """Pad DSM to be divisible by patch_size"""
    oh, ow = dsm_array.shape[0], dsm_array.shape[1]
    rh, rw = oh % patch_size, ow % patch_size
    width_pad = 0 if rw == 0 else patch_size - rw
    height_pad = 0 if rh == 0 else patch_size - rh

    h, w = oh + height_pad, ow + width_pad
    # Pad with 0 using numpy (albumentations PadIfNeeded doesn't support value parameter for non-RGB)
    if width_pad > 0 or height_pad > 0:
        dsm_pad = np.zeros((h, w), dtype=dsm_array.dtype)
        dsm_pad[:oh, :ow] = dsm_array
    else:
        dsm_pad = dsm_array
    return dsm_pad


def dsm_augment(dsm_array, patch_size, mode='train', val_scale=1.0):
    """Apply same augmentations as RGB images to keep consistency"""
    dsm_list = []
    image_height, image_width = dsm_array.shape[0], dsm_array.shape[1]

    if mode == 'train':
        # Apply same augmentations as RGB: original, horizontal flip, vertical flip
        h_flip = RandomHorizontalFlip(p=1.0)
        v_flip = RandomVerticalFlip(p=1.0)
        
        # Convert to PIL Image for transforms (treat as grayscale)
        dsm_pil = Image.fromarray(dsm_array.astype(np.float32), mode='F')
        dsm_h_flip = h_flip(dsm_pil.copy())
        dsm_v_flip = v_flip(dsm_pil.copy())
        
        dsm_list_train = [dsm_pil, dsm_h_flip, dsm_v_flip]
        
        for i in range(len(dsm_list_train)):
            dsm_tmp = np.array(dsm_list_train[i])
            dsm_tmp = get_dsm_padded(dsm_tmp, patch_size, mode)
            dsm_list.append(dsm_tmp)
    else:
        # For test/val mode, apply rescale if needed
        rescale = Resize(size=(int(image_height * val_scale), int(image_width * val_scale)))
        dsm_pil = Image.fromarray(dsm_array.astype(np.float32), mode='F')
        dsm_rescaled = rescale(dsm_pil.copy())
        dsm_tmp = np.array(dsm_rescaled)
        dsm_tmp = get_dsm_padded(dsm_tmp, patch_size, mode)
        dsm_list.append(dsm_tmp)
    
    return dsm_list


def vaihingen_dsm_format(inp):
    """Process a single DSM area file"""
    (dsm_path, output_dir, mode, val_scale, split_size, stride) = inp
    
    # Extract area number from filename (e.g., dsm_09cm_matching_area11.tif -> area11)
    dsm_filename = os.path.splitext(os.path.basename(dsm_path))[0]
    # Convert dsm_09cm_matching_area11 -> top_mosaic_09cm_area11 to match RGB patch naming
    # This ensures DSM patches have the same base name as RGB patches
    area_part = dsm_filename.replace('dsm_09cm_matching_', '')  # area11
    rgb_base_name = f"top_mosaic_09cm_{area_part}"  # top_mosaic_09cm_area11
    
    # Load DSM (single channel float)
    dsm_img = Image.open(dsm_path)
    dsm_array = np.array(dsm_img)
    
    # Apply augmentations (same as RGB images)
    dsm_list = dsm_augment(dsm_array, split_size, mode=mode, val_scale=val_scale)
    
    for m in range(len(dsm_list)):
        k = 0
        dsm = dsm_list[m]
        
        for y in range(0, dsm.shape[0], stride):
            for x in range(0, dsm.shape[1], stride):
                dsm_tile = dsm[y:y + split_size, x:x + split_size]
                
                if dsm_tile.shape[0] == split_size and dsm_tile.shape[1] == split_size:
                    # Use same naming as RGB patches: top_mosaic_09cm_area11_m_k.tif
                    output_filename = "{}_{}_{}.tif".format(rgb_base_name, m, k)
                    out_dsm_path = os.path.join(output_dir, output_filename)
                    
                    # Save as float32 TIFF
                    dsm_tile_float = dsm_tile.astype(np.float32)
                    # Use PIL to save as TIFF with float support
                    dsm_tile_img = Image.fromarray(dsm_tile_float, mode='F')
                    dsm_tile_img.save(out_dsm_path)
                
                k += 1


def parse_args():
    parser = argparse.ArgumentParser(description='Split Vaihingen DSM images into patches')
    parser.add_argument("--dsm-dir", type=str, required=True, help="Directory containing DSM area files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for DSM patches")
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'test', 'val'], help="Mode: train or test")
    parser.add_argument("--val-scale", type=float, default=1.0, help="Scale factor for validation/test")
    parser.add_argument("--split-size", type=int, default=1024, help="Patch size")
    parser.add_argument("--stride", type=int, default=512, help="Stride for sliding window (512 for train, 1024 for test)")
    return parser.parse_args()


if __name__ == "__main__":
    seed_everything(SEED)
    args = parse_args()
    dsm_dir = args.dsm_dir
    output_dir = args.output_dir
    mode = args.mode
    val_scale = args.val_scale
    split_size = args.split_size
    stride = args.stride
    
    # Find all DSM area files
    dsm_paths = glob.glob(os.path.join(dsm_dir, "dsm_09cm_matching_area*.tif"))
    dsm_paths.sort()
    
    if len(dsm_paths) == 0:
        print(f"Warning: No DSM files found in {dsm_dir}")
        exit(1)
    
    print(f"Found {len(dsm_paths)} DSM files to process")
    print(f"Mode: {mode}, Split size: {split_size}, Stride: {stride}")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    inp = [(dsm_path, output_dir, mode, val_scale, split_size, stride) for dsm_path in dsm_paths]
    
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(vaihingen_dsm_format, inp)
    t1 = time.time()
    split_time = t1 - t0
    print(f'DSM splitting completed in {split_time:.2f} seconds')
    print(f'Output directory: {output_dir}')
