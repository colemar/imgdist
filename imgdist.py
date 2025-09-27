#!/usr/bin/env python3

import sys
from PIL import Image
import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim
import argparse

def compute_diffs(picture1_path, picture2_path, compute_ssim=False, compute_ms_ssim=False, use_gpu=False):
    try:
        picture1 = Image.open(picture1_path).convert('RGB')
        picture2 = Image.open(picture2_path).convert('RGB')

        if picture1.size != picture2.size:
            return "Error: pictures must have the same width and height."

        width, height = picture1.size
        pixel_count = width * height

        # The dictionary is initialized empty and populated gradually
        results = {}
        results["size"] = f"{width} x {height}"

        # --- CALCULATION OF NORMALIZED EUCLIDEAN NORM FOR YCbCr ---

        # Convert the images to YCbCr and calculate the difference on the entire 3D array
        diff = np.array(picture1.convert('YCbCr'), dtype=np.float64) - np.array(picture2.convert('YCbCr'), dtype=np.float64)

        # Calculate the 3 norms at once by collapsing axes 0 and 1
        norms = np.linalg.norm(diff, axis=(0, 1))

        # Define the maximum differences per channel and calculate the maximum norms
        max_diffs = np.array([235.0 - 16.0, 240.0 - 16.0, 240.0 - 16.0], dtype=np.float64) # Y, Cb, Cr
        max_norms = np.sqrt(pixel_count) * max_diffs

        # Normalize the norms with a vector division
        results["ycbcr"] = norms / max_norms

        # --- CALCULATION OF NORMALIZED EUCLIDEAN NORM FOR HSV ---

        # Convert the images to NumPy arrays. Pillow uses uint8 [0, 255] for HSV.
        # Calculate the absolute difference for all channels at once.
        diff = np.abs(np.array(picture1.convert('HSV'), dtype=np.float64) - np.array(picture2.convert('HSV'), dtype=np.float64))

        # Specific correction for the H channel (Hue, index 0), which is circular.
        # Replace the H difference with its minimum distance on the ring.
        diff[..., 0] = np.minimum(diff[..., 0], 255 - diff[..., 0])

        # Calculate the 3 norms (H, S, V) at once by collapsing the image axes (0 and 1).
        norms = np.linalg.norm(diff, axis=(0, 1))

        # Define the maximum differences per channel (H is ~127.5, S and V are 255).
        max_diffs = np.array([127.5, 255.0, 255.0], dtype=np.float64)
        max_norms = np.sqrt(pixel_count) * max_diffs

        # Normalize the norms with a vector division
        results["hsv"] = norms / max_norms

        # --- CALCULATION OF NORMALIZED EUCLIDEAN NORM FOR LAB ---

        # Convert the images to LAB and calculate the difference on the entire 3D array
        diff = np.array(picture1.convert('LAB'), dtype=np.float64) - np.array(picture2.convert('LAB'), dtype=np.float64)

        # Calculate the 3 norms at once by collapsing axes 0 and 1
        norms = np.linalg.norm(diff, axis=(0, 1))

        # Define the maximum differences per channel and calculate the maximum norms.
        # For LAB, all channels use the full 0-255 range.
        max_diffs = np.array([255.0, 255.0, 255.0], dtype=np.float64) # L, a, b
        max_norms = np.sqrt(pixel_count) * max_diffs

        # Normalize the norms with a vector division
        results["lab"] = norms / max_norms

        # --- CALCULATION OF NORMALIZED MEAN DELTA E (MDE) ---

        # Calculate the Euclidean norm (Delta E) for each pixel along the channel axis (axis=2)
        pixel_delta_e = np.linalg.norm(diff, axis=2) # use LAB diff

        # Calculate the mean of all Delta E values to get the MDE.
        mean_delta_e = np.mean(pixel_delta_e)

        # Define the maximum theoretical Delta E possible.
        max_delta_e = 255.0 * np.sqrt(3)

        # Normalize the MDE to bring it into the 0-1 range.
        results["mde"] = mean_delta_e / max_delta_e


        # Convert to grayscale only if necessary for SSIM or MS-SSIM
        if compute_ssim or compute_ms_ssim:
            # Convert the images to float arrays [0, 1], the format required by pytorch-msssim
            gray1_float = np.array(picture1.convert('L'), dtype=np.float32) / 255.0
            gray2_float = np.array(picture2.convert('L'), dtype=np.float32) / 255.0

            # Determine the device to use based on the --gpu flag
            if use_gpu and torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            # Convert to PyTorch tensors (adding the Batch and Channel dimensions)
            t1 = torch.from_numpy(gray1_float).unsqueeze(0).unsqueeze(0).to(device)
            t2 = torch.from_numpy(gray2_float).unsqueeze(0).unsqueeze(0).to(device)

            # Conditional SSIM calculation
            if compute_ssim:
                # Calculate SSIM score (the result is a tensor, we use .item() to extract the value)
                results["ssim"] = ssim(t1, t2, data_range=1.0).item()

            # Conditional MS-SSIM calculation
            if compute_ms_ssim:
                # Calculate MS-SSIM score (the result is a tensor, we use .item() to extract the value)
                results["ms_ssim"] = ms_ssim(t1, t2, data_range=1.0).item()

        # Return the dictionary
        return results

    except FileNotFoundError:
        return "Error: One or both files could not be found."
    except Exception as e:
        return f"Error: {e}"

# --- Argument handling and conditional printing ---
if __name__ == "__main__":
    # argparse setup for parameter handling
    parser = argparse.ArgumentParser(description="Calculate various distance metrics between two images.")
    parser.add_argument("picture1", help="Path to the first image.")
    parser.add_argument("picture2", help="Path to the second image.")
    parser.add_argument("-s", "--ssim", action="store_true", help="Also calculate the SSIM index (requires torch).")
    parser.add_argument("-m", "--ms_ssim", action="store_true", help="Also calculate the MS-SSIM index (requires torch).")
    parser.add_argument("-g", "--gpu", action="store_true", help="Enable GPU for PyTorch calculations, if available.")    
    
    args = parser.parse_args()

    # Run the main function, passing the flags
    results = compute_diffs(args.picture1, args.picture2, compute_ssim=args.ssim, compute_ms_ssim=args.ms_ssim, use_gpu=args.gpu)

    # Check if the result is a dictionary (success)
    if isinstance(results, dict):
        print(f"\nImages size: {results['size']}")

        # Print YcbCr Euclidean norm results
        norms = results["ycbcr"]
        print( "\n----------- YCbCr ------------")
        print(f"Euclidean distance Y: {norms[0]:.6f}")
        print(f"                  Cb: {norms[1]:.6f}")
        print(f"                  Cr: {norms[2]:.6f}")
        print(f"    Average distance: {sum(norms)/3:.6f}")

        # Print HSV Euclidean norm results
        norms = results["hsv"]
        print( "\n------------ HSV -------------")
        print(f"Euclidean distance V: {norms[2]:.6f}")
        print(f"                   S: {norms[1]:.6f}")
        print(f"                   H: {norms[0]:.6f}")
        print(f"    Average distance: {sum(norms)/3:.6f}")

        # Print LAB Euclidean norm results
        print( "\n------------ LAB -------------")
        norms = results["lab"]
        print(f"Euclidean distance L: {norms[0]:.6f}")
        print(f"                   a: {norms[1]:.6f}")
        print(f"                   b: {norms[2]:.6f}")
        print(f"    Average distance: {sum(norms)/3:.6f}")

        # Print normalized MDE results
        mde = results["mde"]
        print(f"\n      Normalized MDE: {mde:.6f}")

        # Conditional print of optional results
        if "ssim" in results or "ms_ssim" in results:
            print("\n----------------------- SSIM ------------------------")
        if "ssim" in results:
            print(f"               Structural Similarity (SSIM): {results['ssim']:.6f}")
        if "ms_ssim" in results:
            print(f"Multi-Scale Structural Similarity (MS-SSIM): {results['ms_ssim']:.6f}")

    else:
        # Print error
        print(results)
