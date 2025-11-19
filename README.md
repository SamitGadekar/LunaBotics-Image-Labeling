# LunaBotics Image Labeling

Automated image segmentation using Grounding DINO + SAM (Segment Anything Model) for detecting and segmenting objects in images. Supports both single directory and batch multi-directory processing.

## Features

- Automated batch processing of image directories
- Multi-directory processing for large datasets
- Zero-shot object detection with custom labels
- Segmentation masks with bounding boxes
- JSON metadata export with polygon coordinates
- Apple Silicon (MPS) support for Mac
- CUDA support for NVIDIA GPUs

## Requirements

- Python 3.8+

## Installation

```bash
# For CUDA support (Linux/Windows with NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
```bash
# Install dependencies
pip install -r requirements.txt

# Verify GPU access with this command:
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')"
```

## Usage

### Processing Multiple Directories

Process all subdirectories containing images:

```bash
python automated_grounded_sam.py \
    -i ./input_main_directory \
    -o ./output_main_directory \
    -l rock puddle ...
```

**Directory Structure:**
```
input_main_directory/
|–– rosbag_001/
|   |–– frame_000001.png
|   |–– frame_000002.png
|   |–– ...
|–– rosbag_002/
|   |–– frame_000001.png
|   |–– ...
|–– rosbag_003/
    |–– ...

output_main_directory/
|–– annotated_rosbag_001/
|   |–– annotated_frame_000001.png
|   |–– annotated_frame_000001.json
|   |–– ...
|–– annotated_rosbag_002/
|–– annotated_rosbag_003/
```

### Processing Single Directory

For processing images in a single directory, it's the same code!

```bash
python automated_grounded_sam.py \
    -i ./single_image_folder \
    -o ./single_output_folder \
    -l rock crater
```

### Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input_dir` | `-i` | Yes | - | Input directory (single folder or parent of multiple folders) |
| `--output_dir` | `-o` | Yes | - | Output directory for results |
| `--labels` | `-l` | Yes | - | Space-separated labels to detect |
| `--threshold` | `-t` | No | 0.3 | Detection confidence threshold (0.0-1.0) |
| `--no_polygon_refinement` | - | No | False | Disable polygon refinement |
| `--detector_id` | - | No | grounding-dino-tiny | Grounding DINO model variant |
| `--segmenter_id` | - | No | sam-vit-base | SAM model variant |
| `--no_annotations` | - | No | False | Don't save annotated images |
| `--no_metadata` | - | No | False | Don't save JSON metadata |
| `--no_progress` | - | No | False | Hide progress bars |

## Examples

### Base Usage
```bash
# Process multiple rosbag directories
python automated_grounded_sam.py \
    -i ./rosbags \
    -o ./annotated_rosbags \
    -l rock crater
```

### Custom Threshold + Metadata Only + Hide Progress Bars
```bash
# Lower threshold to detect more objects (may include false positives)
python automated_grounded_sam.py \
    -i ./images \
    -o ./results \
    -l rock \
    -t 0.2 \
    --no_annotations \
    --no_progress
```

## Output Format

### Annotated Images (.png files)
- Bounding boxes with labels and confidence scores
- Segmentation mask contours
- Random colors for each detection

### JSON Metadata
Each annotated image has a corresponding JSON file:

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
    "polygon": [
      [100, 150],
      [120, 145],
      [135, 148],
      ...
    ]
  }
]
```

## Output Example

```
======================================================================
FOUND 1 DIRECTORIES TO PROCESS
FOUND 2 IMAGES HERE TO PROCESS
Labels: ['rock']
Device: mps
======================================================================
Processing: sample_images (2 images)
 Processing sample_images:   0%|                                            | 0/2 [00:00<?, ?it/s]
 Processing sample_images:  50%|██████████████████                  | 1/2 [00:09<00:09,  9.57s/it]
 V COMPLETED: sample_images (2 images processed)

[Sub-directory 1/1]
Processing: Group_1 (2 images)
 Processing Group_1:   0%|                                                  | 0/2 [00:00<?, ?it/s]
 Processing Group_1:  50%|█████████████████████                     | 1/2 [00:06<00:06,  6.34s/it]
 V COMPLETED: Group_1 (2 images processed)

======================================================================
ALL PROCESSING COMPLETE
Input location: ./sample_images
Total sub-directories processed: 1
Total images processed in  sub-directories: 2
Total images processed in   this directory: 2
Output location: ./sample_results
======================================================================
```

## Processing Video Frames

// TODO

## Troubleshooting

### GPU Not Being Used

**Check GPU availability:**
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**For Mac:**
```bash
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
```

**Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Out of Memory

- Process fewer images at once
- Use smaller model: `--detector_id "IDEA-Research/grounding-dino-tiny"`
- Reduce image resolution before processing
- Clear GPU cache: `python -c "import torch; torch.cuda.empty_cache()"`

### No Detections Found

- Lower threshold: `-t 0.2` or `-t 0.15`
- Check labels match objects in images
- Try different label variations (e.g., "rock" vs "stone")
- Verify images are loading correctly

### Slow Processing

- Verify GPU is active: `nvidia-smi` (for CUDA) or Activity Monitor -> GPU (for Mac)
- Use tiny model variant: `--detector_id "IDEA-Research/grounding-dino-tiny"`
- Check for background processes using GPU

## Supported Image Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Model Information

### Grounding DINO
- **Tiny**: Fast, less accurate (`IDEA-Research/grounding-dino-tiny`)
- **Base**: Balanced (default, `IDEA-Research/grounding-dino-base`)

### Segment Anything (SAM)
- **ViT-Base**: Balanced (default, `facebook/sam-vit-base`)
- **ViT-Large**: More accurate, slower (`facebook/sam-vit-large`)
- **ViT-Huge**: Best accuracy, slowest (`facebook/sam-vit-huge`)

## License

This project uses:
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [Segment Anything (SAM)](https://github.com/facebookresearch/segment-anything)

## Contributing

Issues and pull requests welcome!

## Citation

If you use this tool in research, please cite:

```bibtex
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}

@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}
```

## Contact

For issues or questions, please open an issue on GitHub.
