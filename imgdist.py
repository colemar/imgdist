#!/usr/bin/env python3

import traceback
import sys
import argparse
import os
from PIL import Image
from pillow_heif import HeifImagePlugin
import pillow_jxl
import numpy as np
import torch
from pytorch_msssim import ssim, ms_ssim

def preprocess_image(image_path, needs_tensors=False, device=torch.device('cpu')):
    """
    Loads an image and converts it into all necessary formats (NumPy arrays and PyTorch tensors).
    This avoids redundant conversions.
    """
    img_pil = Image.open(image_path).convert('RGB')

    data_pack = {
        "path": image_path,
        "filename": os.path.basename(image_path), # Store just the filename
        "size": img_pil.size,
        "ycbcr_arr": np.array(img_pil.convert('YCbCr'), dtype=np.float64),
        "hsv_arr": np.array(img_pil.convert('HSV'), dtype=np.float64),
        "lab_arr": np.array(img_pil.convert('LAB'), dtype=np.float64),
    }

    if needs_tensors:
        gray_float = np.array(img_pil.convert('L'), dtype=np.float32) / 255.0
        data_pack["tensor"] = torch.from_numpy(gray_float).unsqueeze(0).unsqueeze(0).to(device)

    return data_pack

def compare_images(pack1, pack2, compute_ssim=False, compute_ms_ssim=False):
    """
    Calculates difference metrics between two preprocessed image data packs.
    """
    width, height = pack1["size"]
    pixel_count = width * height
    results = {"size": f"{width} x {height}"}

    # --- YCbCr NORM ---
    diff = pack1["ycbcr_arr"] - pack2["ycbcr_arr"]
    norms = np.linalg.norm(diff, axis=(0, 1))
    max_diffs = np.array([219.0, 224.0, 224.0], dtype=np.float64)
    max_norms = np.sqrt(pixel_count) * max_diffs
    results["ycbcr"] = norms / max_norms

    # --- HSV NORM ---
    diff = np.abs(pack1["hsv_arr"] - pack2["hsv_arr"])
    # Specific correction for the H channel (Hue, index 0), which is circular.
    diff[..., 0] = np.minimum(diff[..., 0], 255 - diff[..., 0])
    norms = np.linalg.norm(diff, axis=(0, 1))
    max_diffs = np.array([127.5, 255.0, 255.0], dtype=np.float64)
    max_norms = np.sqrt(pixel_count) * max_diffs
    results["hsv"] = norms / max_norms

    # --- LAB NORM & MDE ---
    diff_lab = pack1["lab_arr"] - pack2["lab_arr"]
    norms = np.linalg.norm(diff_lab, axis=(0, 1))
    max_diffs = np.array([255.0, 255.0, 255.0], dtype=np.float64)
    max_norms = np.sqrt(pixel_count) * max_diffs
    results["lab"] = norms / max_norms

    pixel_delta_e = np.linalg.norm(diff_lab, axis=2)
    mean_delta_e = np.mean(pixel_delta_e)
    max_delta_e = 255.0 * np.sqrt(3)
    results["mde"] = mean_delta_e / max_delta_e

    # --- SSIM & MS-SSIM ---
    if compute_ssim:
        results["ssim"] = ssim(pack1["tensor"], pack2["tensor"], data_range=1.0).item()
    if compute_ms_ssim:
        results["ms_ssim"] = ms_ssim(pack1["tensor"], pack2["tensor"], data_range=1.0).item()

    return results


def print_comparison_columns(r1, r2, headers):
    """Prints two result dictionaries in a side-by-side column format."""
    COL_WIDTH = 30
    SEPARATOR = " | "
    TOTAL_WIDTH = COL_WIDTH * 2 + len(SEPARATOR)

    y1, h1, l1, m1, s1, ms1 = r1["ycbcr"], r1["hsv"], r1["lab"], r1["mde"], r1.get("ssim"), r1.get("ms_ssim")
    ay1, ah1, al1 = [sum(d)/3 for d in [y1, h1, l1]]
    if isinstance(r2, dict):
        y2, h2, l2, m2, s2, ms2 = r2["ycbcr"], r2["hsv"], r2["lab"], r2["mde"], r2.get("ssim"), r2.get("ms_ssim")
        ay2, ah2, al2 = [sum(d)/3 for d in [y2, h2, l2]]
    else:
        SEPARATOR = ""
        TOTAL_WIDTH = COL_WIDTH
        y2, h2, l2, m2, s2, ms2, ay2, ah2, al2 = ([None,None,None],) * 3 + (None,) * 6

    def separator_line(label):
        label = f" {label} "
        left, r = divmod(TOTAL_WIDTH - len(label), 2)
        right = left + r
        print("-"*left + label + "-"*right)

    def data_line(prefix, num1, num2):
        GREEN = "\033[92m" # light green
        RESET = "\033[0m"  # color reset

        str1 = f"{num1:.6f}" if isinstance(num1, (int, float)) else " " * 8
        str2 = f"{num2:.6f}" if isinstance(num2, (int, float)) else " " * 8

        if isinstance(num1, (int, float)) and isinstance(num2, (int, float)):
            if (num1 > num2) == ('SSIM' in prefix):
                str1 = f"{GREEN}{str1}{RESET}"
            else:
                str2 = f"{GREEN}{str2}{RESET}"

        print(f"{prefix}{str1}{SEPARATOR}{str2}")

    ref_header, hd1, hd2 = headers
    print("\n" + f"Ref: {ref_header} {r1['size']}".center(TOTAL_WIDTH))
    print("="*TOTAL_WIDTH)
    print(f"{hd1.center(COL_WIDTH)}{SEPARATOR}{hd2.center(COL_WIDTH)}")

    separator_line("YCbCr")
    data_line("Euclidean distance Y: ", y1[0], y2[0])
    data_line("                  Cb: ", y1[1], y2[1])
    data_line("                  Cr: ", y1[2], y2[2])
    data_line("    Average distance: ", ay1  , ay2  )

    separator_line("HSV")
    data_line("Euclidean distance V: ", h1[2], h2[2])
    data_line("                   S: ", h1[1], h2[1])
    data_line("                   H: ", h1[0], h2[0])
    data_line("    Average distance: ", ah1  , ah2  )

    separator_line("LAB")
    data_line("Euclidean distance L: ", l1[0], l2[0])
    data_line("                   a: ", l1[1], l2[1])
    data_line("                   b: ", l1[2], l2[2])
    data_line("    Average distance: ", al1  , al2  )

    data_line("                      ", None, None)
    data_line("      Normalized MDE: ", m1   , m2   )

    if "ssim" in r1 or "ms_ssim" in r1:
        separator_line("SSIM")
    if "ssim" in r1:
        data_line("                SSIM: ", s1 , s2 )
    if "ms_ssim" in r1:
        data_line("             MS-SSIM: ", ms1, ms2)

    print("="*TOTAL_WIDTH)

def run_analysis(device, args):
    """ Funzione che esegue l'intera analisi con un device specifico. """
    num_images = len(args.image)
    needs_tensors = args.ssim or args.ms_ssim

    if num_images in (2, 3):
        # --- MODIFICA #1: Il messaggio sul device viene stampato solo se necessario. ---
        if needs_tensors:
            print(f"Using device: {device}", file=sys.stderr)

        ref_pack = preprocess_image(args.image[0], needs_tensors, device)
        comp1_pack = preprocess_image(args.image[1], needs_tensors, device)
        if ref_pack["size"] != comp1_pack["size"]: raise ValueError("Pictures must have the same size.")
        results1 = compare_images(ref_pack, comp1_pack, args.ssim, args.ms_ssim)
    else:
        print(f"Error: Requires 2 or 3 image paths, but {num_images} were provided.", file=sys.stderr)
        sys.exit(1)

    if num_images == 2:
        headers = (f"'{ref_pack['filename']}'", f"vs '{comp1_pack['filename']}'", "")
        print_comparison_columns(results1, None, headers)
    elif num_images == 3:
        comp2_pack = preprocess_image(args.image[2], needs_tensors, device)
        if ref_pack["size"] != comp2_pack["size"]: raise ValueError("Pictures must have the same size.")
        results2 = compare_images(ref_pack, comp2_pack, args.ssim, args.ms_ssim)
        headers = (f"'{ref_pack['filename']}'", f"vs '{comp1_pack['filename']}'", f"vs '{comp2_pack['filename']}'")
        print_comparison_columns(results1, results2, headers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate various distance metrics between images.")
    parser.add_argument("image", nargs='+', help=f"Path to image. 1st: reference, 2nd and 3rd: compared against 1st. 3rd is optional.")
    parser.add_argument("-s", "--ssim", action="store_true", help="Also calculate the SSIM index.")
    parser.add_argument("-m", "--ms_ssim", action="store_true", help="Also calculate the MS-SSIM index.")
    parser.add_argument("-g", "--gpu", action="store_true", help="Enable GPU for PyTorch calculations.")
    args = parser.parse_args()

    needs_tensors = args.ssim or args.ms_ssim

    initial_device = torch.device('cpu')
    if args.gpu:
        if torch.cuda.is_available():
            initial_device = torch.device('cuda')
        else:
            # --- MODIFICA #2: Il warning viene stampato solo se necessario. ---
            if needs_tensors:
                print("Warning: GPU requested (-g), but no CUDA device is available. Using CPU.", file=sys.stderr)

    try:
        run_analysis(initial_device, args)
    except Exception as e:
        if 'cuda' in str(e).lower() and initial_device.type == 'cuda':
            print("\nWarning: An error occurred while using the GPU. Falling back to CPU.", file=sys.stderr)
            print("\n--- Retrying on CPU ---", file=sys.stderr)
            cpu_device = torch.device('cpu')
            try:
                run_analysis(cpu_device, args)
            except Exception as final_e:
                print("\nError: The analysis failed even after falling back to CPU.", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                sys.exit(1)
        else:
            print("\nUnexpected error occurred:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)
