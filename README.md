# LunaBotics Image Labeling

Automated image segmentation using Grounding DINO + SAM (Segment Anything Model) for detecting and segmenting objects in images.

## Features

- 🚀 Automated batch processing of image directories
- 🎯 Zero-shot object detection with custom labels
- 🖼️ Segmentation masks with bounding boxes
- 💾 JSON metadata export
- 🍎 Apple Silicon (MPS) support for Mac
- 🔥 CUDA support for NVIDIA GPUs

## Requirements

**For Mac (Apple Silicon M1/M2/M3):**
- macOS 12.3+
- Python 3.8+

**For CUDA/Other Systems:**
- Python 3.8+
- NVIDIA GPU (optional, for CUDA acceleration)

## Installation

### Mac (Apple Silicon)
```bash
pip install -r requirements_mac.txt
```

### CUDA/Other Systems
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Arguments

| Argument | Short | Required | Description |
|----------|-------|----------|-------------|
| `--input_dir` | `-i` | Yes | Input directory containing images |
| `--output_dir` | `-o` | Yes | Output directory for results |
| `--labels` | `-l` | Yes | Space-separated labels to detect |

**⚠️ Important:** Create input and output directories before running!

### Mac Usage

**Single label:**
```bash
python auto_g-sam_mac.py \
    -i ./sample_images \
    -o ./sample_results \
    -l rock
```

**Multiple labels:**
```bash
python auto_g-sam_mac.py \
    -i ./my_images \
    -o ./results \
    -l rock crater
```

### CUDA/Other Systems Usage

**Single label:**
```bash
python automated_grounded_sam.py \
    --input_dir ./sample_images \
    --output_dir ./sample_results \
    --labels rock
```

**Multiple labels:**
```bash
python automated_grounded_sam.py \
    --input_dir ./my_images \
    --output_dir ./results \
    --labels rock crater
```

## Quick Start

```bash
# 1. Create directories
mkdir -p input_images output_results

# 2. Add your images to input_images/

# 3. Run segmentation (Mac)
python auto_g-sam_mac.py -i ./input_images -o ./output_results -l rock crater

# 4. Check results in output_results/
```

## Output

For each input image, the script generates:

1. **Annotated image** - Original image with bounding boxes and segmentation masks
2. **JSON metadata** - Detection results with labels, confidence scores, and coordinates

### Example Output Structure
```
output_results/
├── annotated_image1.png
├── annotated_image1.json
├── annotated_image2.png
└── annotated_image2.json
```

### Example JSON
```json
[
  {
    "label": "rock",
    "score": 0.87,
    "box": {
      "xmin": 100,
      "ymin": 150,
      "xmax": 300,
      "ymax": 400
    },
    "polygon": [[100, 150], [120, 145], ...]
  }
]
```

## Examples

### Detect rocks
```bash
python auto_g-sam_mac.py -i ./images -o ./results -l rock
```

### Detect multiple objects
```bash
python auto_g-sam_mac.py -i ./images -o ./results -l rock crater boulder
```

### Custom threshold
```bash
python auto_g-sam_mac.py -i ./images -o ./results -l rock --threshold 0.4
```

## Processing Video Frames

Extract frames from video first:

```bash
# Install FFmpeg (Mac)
brew install ffmpeg

# Extract frames
mkdir video_frames
ffmpeg -i video.mp4 video_frames/frame_%04d.png

# Run segmentation
python auto_g-sam_mac.py -i ./video_frames -o ./video_results -l rock crater
```

## Troubleshooting

### Mac: "Device: cpu" instead of "Device: mps"

Check MPS availability:
```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Reinstall PyTorch with MPS support:
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Out of Memory

Process smaller batches or reduce image resolution before processing.

### No Detections Found

- Lower the threshold: `--threshold 0.2`
- Verify labels match objects in images
- Check that images loaded correctly

## Performance

**Mac (M1/M2/M3 with MPS):**
- ~3-5 seconds per image

**NVIDIA GPU (CUDA):**
- ~2-4 seconds per image

**CPU:**
- ~10-20 seconds per image

## Supported Image Formats

- PNG
- JPG/JPEG
- BMP
- TIFF

## License

This project uses:
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

## Contributing

Issues and pull requests welcome!

## Citation

If you use this tool, please cite the original papers:

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```
