#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Video Segmentation with Grounded SAM
Processes videos by extracting frames and running segmentation
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(
            score=detection_dict['score'],
            label=detection_dict['label'],
            box=BoundingBox(
                xmin=detection_dict['box']['xmin'],
                ymin=detection_dict['box']['ymin'],
                xmax=detection_dict['box']['xmax'],
                ymax=detection_dict['box']['ymax']
            )
        )


def get_device():
    """Get the best available device (CUDA for NVIDIA GPU, CPU otherwise)"""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Convert mask to polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return []
    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    return polygon


def polygon_to_mask(polygon: List[Tuple[int, int]], 
                   image_shape: Tuple[int, int]) -> np.ndarray:
    """Convert polygon to mask"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask


def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    """Extract bounding boxes from detection results"""
    boxes = [result.box.xyxy for result in results]
    return [boxes]


def refine_masks(masks: torch.BoolTensor, 
                polygon_refinement: bool = False) -> List[np.ndarray]:
    """Refine and convert masks to numpy arrays"""
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask)
            if polygon:
                mask = polygon_to_mask(polygon, shape)
                masks[idx] = mask

    return masks


def detect(image: Image.Image,
          labels: List[str],
          threshold: float = 0.3,
          detector_id: Optional[str] = None) -> List[DetectionResult]:
    """Detect objects using Grounding DINO"""
    device = get_device()
    detector_id = detector_id or "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, 
                              task="zero-shot-object-detection", 
                              device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(image: Image.Image,
           detection_results: List[DetectionResult],
           polygon_refinement: bool = False,
           segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    """Segment objects using SAM"""
    device = get_device()
    segmenter_id = segmenter_id or "facebook/sam-vit-base"

    segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
    processor = AutoProcessor.from_pretrained(segmenter_id)

    boxes = get_boxes(detection_results)
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

    outputs = segmentator(**inputs)
    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    return detection_results


def grounded_segmentation(image: Union[Image.Image, str],
                         labels: List[str],
                         threshold: float = 0.3,
                         polygon_refinement: bool = False,
                         detector_id: Optional[str] = None,
                         segmenter_id: Optional[str] = None) -> Tuple[np.ndarray, List[DetectionResult]]:
    """Complete grounded segmentation pipeline"""
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")

    detections = detect(image, labels, threshold, detector_id)
    
    if len(detections) == 0:
        return np.array(image), []
    
    detections = segment(image, detections, polygon_refinement, segmenter_id)
    return np.array(image), detections


def extract_frames(video_path: str, output_dir: str) -> int:
    """
    Extract frames from video using FFmpeg
    Returns the number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)
    
    output_pattern = os.path.join(output_dir, "frame_%06d.png")
    
    # Use FFmpeg to extract frames
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',  # High quality
        output_pattern,
        '-hide_banner',
        '-loglevel', 'error'
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        # Count extracted frames
        frame_count = len([f for f in os.listdir(output_dir) if f.endswith('.png')])
        return frame_count
    except subprocess.CalledProcessError as e:
        print(f"Error extracting frames: {e.stderr.decode()}")
        return 0
    except FileNotFoundError:
        print("Error: FFmpeg not found. Please install FFmpeg:")
        print("  Mac: brew install ffmpeg")
        print("  Ubuntu: sudo apt install ffmpeg")
        return 0


def save_annotation_json(detections: List[DetectionResult], output_path: str):
    """Save detection results as JSON"""
    metadata = []
    for det in detections:
        det_dict = {
            'label': det.label,
            'score': float(det.score),
            'box': {
                'xmin': int(det.box.xmin),
                'ymin': int(det.box.ymin),
                'xmax': int(det.box.xmax),
                'ymax': int(det.box.ymax)
            }
        }
        if det.mask is not None:
            polygon = mask_to_polygon(det.mask)
            if polygon:
                det_dict['polygon'] = polygon
        metadata.append(det_dict)
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def process_video(video_path: str,
                 output_base_dir: str,
                 labels: List[str],
                 threshold: float = 0.3,
                 polygon_refinement: bool = True,
                 detector_id: str = "IDEA-Research/grounding-dino-tiny",
                 segmenter_id: str = "facebook/sam-vit-base"):
    """Process a single video"""
    
    # Get video name without extension
    video_name = Path(video_path).stem
    
    # Create directory structure
    video_output_dir = os.path.join(output_base_dir, video_name)
    frames_dir = os.path.join(video_output_dir, f"{video_name}_frames")
    annotated_dir = os.path.join(video_output_dir, f"{video_name}_frames_annotated")
    
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(annotated_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Processing video: {video_name}")
    print(f"{'='*60}")
    
    # Step 1: Extract frames
    print(f"Extracting frames...")
    frame_count = extract_frames(video_path, frames_dir)
    
    if frame_count == 0:
        print(f"✗ No frames extracted from {video_name}")
        return
    
    print(f"✓ Extracted {frame_count} frames")
    
    # Step 2: Process each frame
    print(f"Running segmentation on {frame_count} frames...")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    for frame_file in tqdm(frame_files, desc=f"Processing {video_name}"):
        try:
            frame_path = os.path.join(frames_dir, frame_file)
            
            # Run segmentation
            _, detections = grounded_segmentation(
                image=frame_path,
                labels=labels,
                threshold=threshold,
                polygon_refinement=polygon_refinement,
                detector_id=detector_id,
                segmenter_id=segmenter_id
            )
            
            # Save JSON annotation
            json_filename = frame_file.replace('.png', '.json')
            json_path = os.path.join(annotated_dir, json_filename)
            save_annotation_json(detections, json_path)
            
        except Exception as e:
            print(f"✗ Error processing {frame_file}: {str(e)}")
            # Save empty JSON on error
            json_filename = frame_file.replace('.png', '.json')
            json_path = os.path.join(annotated_dir, json_filename)
            save_annotation_json([], json_path)
            continue
    
    print(f"✓ Completed processing {video_name}")
    print(f"  Frames: {frames_dir}")
    print(f"  Annotations: {annotated_dir}")


def process_videos_directory(input_videos_dir: str,
                            output_annotations_dir: str,
                            labels: List[str],
                            threshold: float = 0.3,
                            polygon_refinement: bool = True,
                            detector_id: str = "IDEA-Research/grounding-dino-tiny",
                            segmenter_id: str = "facebook/sam-vit-base"):
    """Process all videos in input directory"""
    
    # Get all video files
    input_path = Path(input_videos_dir)
    video_files = sorted([f for f in input_path.iterdir() 
                         if f.suffix.lower() == '.mp4'])
    
    if len(video_files) == 0:
        print(f"No .mp4 files found in {input_videos_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) to process")
    print(f"Labels: {labels}")
    device = get_device()
    print(f"Device: {device}")
    if device == "mps":
        print("Using Apple Metal Performance Shaders (MPS)")
    
    # Create output directory
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    # Process each video
    for video_file in video_files:
        process_video(
            video_path=str(video_file),
            output_base_dir=output_annotations_dir,
            labels=labels,
            threshold=threshold,
            polygon_refinement=polygon_refinement,
            detector_id=detector_id,
            segmenter_id=segmenter_id
        )
    
    print(f"\n{'='*60}")
    print(f"All videos processed successfully!")
    print(f"Output directory: {output_annotations_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Automated video segmentation with Grounded SAM"
    )
    parser.add_argument(
        "--input_videos", "-i",
        type=str,
        required=True,
        help="Input directory containing .mp4 video files"
    )
    parser.add_argument(
        "--output_annotations", "-o",
        type=str,
        required=True,
        help="Output directory for annotations"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        nargs="+",
        required=True,
        help="Labels to detect (e.g., rock crater)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.3,
        help="Detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--no_polygon_refinement",
        action="store_true",
        help="Disable polygon refinement"
    )
    parser.add_argument(
        "--detector_id",
        type=str,
        default="IDEA-Research/grounding-dino-tiny",
        help="Grounding DINO model ID"
    )
    parser.add_argument(
        "--segmenter_id",
        type=str,
        default="facebook/sam-vit-base",
        help="SAM model ID"
    )
    
    args = parser.parse_args()
    
    process_videos_directory(
        input_videos_dir=args.input_videos,
        output_annotations_dir=args.output_annotations,
        labels=args.labels,
        threshold=args.threshold,
        polygon_refinement=not args.no_polygon_refinement,
        detector_id=args.detector_id,
        segmenter_id=args.segmenter_id
    )


if __name__ == "__main__":
    main()
