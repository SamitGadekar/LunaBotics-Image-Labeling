#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Grounded SAM for Directory Processing
Processes all images in a directory using Grounding DINO + SAM
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
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


def annotate(image: Union[Image.Image, np.ndarray], 
             detection_results: List[DetectionResult]) -> np.ndarray:
    """Draw bounding boxes and masks on image"""
    image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
    image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

    for detection in detection_results:
        label = detection.label
        score = detection.score
        box = detection.box
        mask = detection.mask

        color = np.random.randint(0, 256, size=3)

        # Draw bounding box
        cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
        cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

        # Draw mask contours
        if mask is not None:
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)

    return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """Convert mask to polygon coordinates"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            mask = polygon_to_mask(polygon, shape)
            masks[idx] = mask

    return masks

def get_device():
    """Get the best available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def detect(image: Image.Image,
          labels: List[str],
          threshold: float = 0.3,
          detector_id: Optional[str] = None) -> List[DetectionResult]:
    """Detect objects using Grounding DINO"""
    device = get_device()
    detector_id = detector_id or "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

    labels = [label if label.endswith(".") else label + "." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    return results


def segment(image: Image.Image,
           detection_results: List[DetectionResult],
           polygon_refinement: bool = False,
           segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    """Segment objects using SAM"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
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


def save_results(image_array: np.ndarray,
                detections: List[DetectionResult],
                output_path: str,
                save_annotations: bool = True,
                save_metadata: bool = True):
    """Save annotated image and metadata"""
    # Save annotated image
    if save_annotations:
        annotated = annotate(image_array, detections)
        annotated_pil = Image.fromarray(annotated)
        annotated_pil.save(output_path)
    
    # Save metadata as JSON
    if save_metadata:
        metadata_path = output_path.replace('.png', '.json').replace('.jpg', '.json')
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
                det_dict['polygon'] = mask_to_polygon(det.mask)
            metadata.append(det_dict)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def process_directory(input_dir: str,
                     output_dir: str,
                     labels: List[str],
                     threshold: float = 0.3,
                     polygon_refinement: bool = True,
                     detector_id: str = "IDEA-Research/grounding-dino-tiny",
                     segmenter_id: str = "facebook/sam-vit-base",
                     save_annotations: bool = True,
                     save_metadata: bool = True,
                     image_extensions: List[str] = None,
                     show_progress: bool = False):
    """Process all images in a directory"""
    
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    input_path = Path(input_dir)
    image_files = sorted([f for f in input_path.iterdir() if f.suffix.lower() in image_extensions])

    print(f"Processing: {Path(input_dir).name} ({len(image_files)} images)")

    # Process each image
    for image_file in tqdm(image_files, desc=f" Processing {Path(input_dir).name}",
                           disable=not show_progress, leave=False):
        try:
            # Run segmentation
            image_array, detections = grounded_segmentation(
                image=str(image_file),
                labels=labels,
                threshold=threshold,
                polygon_refinement=polygon_refinement,
                detector_id=detector_id,
                segmenter_id=segmenter_id
            )
            
            # Save results
            output_path = os.path.join(output_dir, f"annotated_{image_file.name}")
            save_results(image_array, detections, output_path, 
                        save_annotations, save_metadata)

        except Exception as e:
            print(f" x Error processing {image_file.name}: {str(e)}")
            continue

    print(f" V COMPLETED: {Path(input_dir).name} ({len(image_files)} images processed)\n")
    
    return len(image_files)

def process_all_directories(input_main_dir: str,
                            output_main_dir: str,
                            labels: List[str],
                            threshold: float = 0.3,
                            polygon_refinement: bool = True,
                            detector_id: str = "IDEA-Research/grounding-dino-tiny",
                            segmenter_id: str = "facebook/sam-vit-base",
                            save_annotations: bool = True,
                            save_metadata: bool = True,
                            image_extensions: List[str] = None,
                            show_progress=True):
    """
    Process all rosbag directories.
    Takes the path to an input directory which contains
    all the directories of frames to process.
    """

    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    # Create output directory
    os.makedirs(output_main_dir, exist_ok=True)

    # Get all subdirectories
    input_main_path = Path(input_main_dir)
    image_dirs = sorted([d for d in input_main_path.iterdir() if d.is_dir()])
    total_images_here = len([f for f in input_main_path.iterdir() if f.suffix.lower() in image_extensions])

    print(f"\n{'='*70}")
    if (len(image_dirs) > 0):
        print(f"FOUND {len(image_dirs)} DIRECTORIES TO PROCESS")
    if (total_images_here > 0):
        print(f"FOUND {total_images_here} IMAGES HERE TO PROCESS")
    if (len(image_dirs) <= 0 & total_images_here <= 0):
        print(f"NOTHING TO PROCESS")
    print(f"Labels: {labels}")
    print(f"Device: {str(get_device())}")
    print(f"{'='*70}")

    try:
        # Process current working directory
        processed = process_directory(
            input_dir=input_main_dir,
            output_dir=output_main_dir,
            labels=labels,
            threshold=threshold,
            polygon_refinement=polygon_refinement,
            detector_id=detector_id,
            segmenter_id=segmenter_id,
            save_annotations=save_annotations,
            save_metadata=save_metadata,
            show_progress=show_progress
        )
        total_images_here = processed
    except Exception as e:
        print(f" x ERROR processing directory {Path(input_main_dir).name}: {str(e)}")


    total_images = 0
    # Process each directory
    for idx, image_dir in enumerate(image_dirs, 1):
        try:
            input_dir = str(image_dir)
            output_dir = os.path.join(output_main_dir, f"annotated_{image_dir.name}")

            print(f"[Sub-directory {idx}/{len(image_dirs)}]")
            # Process the directory
            processed = process_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                labels=labels,
                threshold=threshold,
                polygon_refinement=polygon_refinement,
                detector_id=detector_id,
                segmenter_id=segmenter_id,
                save_annotations=save_annotations,
                save_metadata=save_metadata,
                image_extensions=image_extensions,
                show_progress=show_progress
            )

            total_images += processed
            
        except Exception as e:
            print(f" x ERROR processing directory {image_dir.name}: {str(e)}")
            continue

    print(f"{'='*70}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"Input location: {input_main_dir}")
    print(f"Total sub-directories processed: {len(image_dirs)}")
    print(f"Total images processed in  sub-directories: {total_images}")
    print(f"Total images processed in   this directory: {total_images_here}")
    print(f"Output location: {output_main_dir}")
    print(f"{'='*70}\n")
    

def main():
    parser = argparse.ArgumentParser(
        description="Automated Grounded SAM for directory processing"
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="Input directory containing directories of images"
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--labels", "-l",
        type=str,
        nargs="+",
        required=True,
        help="Labels to detect (e.g., cat dog person)"
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
    parser.add_argument(
        "--no_annotations",
        action="store_true",
        help="Don't save annotated images"
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Don't save JSON metadata"
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Don't show progress bars"
    )
    
    args = parser.parse_args()
    
    process_all_directories(
        input_main_dir=args.input_dir,
        output_main_dir=args.output_dir,
        labels=args.labels,
        threshold=args.threshold,
        polygon_refinement=not args.no_polygon_refinement,
        detector_id=args.detector_id,
        segmenter_id=args.segmenter_id,
        save_annotations=not args.no_annotations,
        save_metadata=not args.no_metadata,
        show_progress=not args.no_progress
    )


if __name__ == "__main__":
    main()
