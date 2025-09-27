# Image Distance Metrics (imgdist)

A command-line tool to calculate and compare various perceptual distance metrics between two images.

## Features

Calculates normalized Euclidean distances in the following color spaces:
- YCbCr
- HSV
- LAB

It also computes:
- Normalized Mean Delta E (MDE)
- Structural Similarity Index (SSIM)
- Multi-Scale Structural Similarity Index (MS-SSIM)

## Usage

```bash
./imgdist.py [image1] [image2] [options]
```

### Options
* `-s`, `--ssim`: Calculate SSIM score.
* `-m`, `--ms_ssim`: Calculate MS-SSIM score.
* `-g`, `--gpu`: Use GPU for SSIM/MS-SSIM calculations (requires PyTorch with CUDA).

