#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Grounded SAM for Directory Processing

This script processes images using Grounding DINO for zero-shot object detection
and SAM (Segment Anything Model) for instance segmentation. It can process both
individual directories and batch process multiple subdirectories.

Features:
- Zero-shot detection with custom labels
- Automatic segmentation mask generation
- Batch processing with progress tracking
- Robust error handling - continues on failures
- GPU acceleration (CUDA/MPS) support
- JSON metadata export with polygon coordinates

Author: Purdue Lunabotics Software Team
"""

import os
import json
import argparse
import traceback
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
    """
    Draws bounding boxes and segmentation masks on image.
    
    Annotates the input image with:
    - Colored bounding boxes around each detection
    - Class label and confidence score text
    - Segmentation mask contours (if available)
    
    Args:
        image: Input image as PIL Image or numpy array
        detection_results: List of DetectionResult objects to visualize
        
    Returns:
        Annotated image as RGB numpy array
        
    Note:
        Uses random colors for each detection to distinguish overlapping objects.
        Returns original image if annotation fails.
    """
    try:
        # Convert PIL Image to OpenCV format (BGR)
        image_cv2 = np.array(image) if isinstance(image, Image.Image) else image
        image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)

        # Draw each detection
        for detection in detection_results:
            try:
                label = detection.label
                score = detection.score
                box = detection.box
                mask = detection.mask

                # Generate random color for this detection
                color = np.random.randint(0, 256, size=3)

                # Draw bounding box rectangle
                cv2.rectangle(image_cv2, (box.xmin, box.ymin), (box.xmax, box.ymax), color.tolist(), 2)
                
                # Add label text above bounding box
                cv2.putText(image_cv2, f'{label}: {score:.2f}', (box.xmin, box.ymin - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

                # Draw segmentation mask contours if available
                if mask is not None:
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(image_cv2, contours, -1, color.tolist(), 2)
                    
            except Exception as e:
                # Skip this detection but continue with others
                print(f"    Warning: Failed to annotate detection {label}: {str(e)}")
                continue

        # Convert back to RGB for PIL/saving
        return cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"    Warning: Failed to annotate image: {str(e)}")
        # Return original image on failure
        return np.array(image) if isinstance(image, Image.Image) else image


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Converts binary segmentation mask to polygon coordinates.
    
    Finds contours in the mask and returns the largest one as a list
    of [x, y] coordinate pairs suitable for JSON serialization.
    
    Args:
        mask: Binary mask as 2D numpy array
        
    Returns:
        List of [x, y] coordinate pairs representing the polygon.
        Returns empty list if conversion fails or no contours found.
    """
    try:
        # Find all contours in the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return []
            
        # Use the largest contour (by area)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Reshape to list of [x, y] points
        polygon = largest_contour.reshape(-1, 2).tolist()
        return polygon
        
    except Exception as e:
        print(f"    Warning: Failed to convert mask to polygon: {str(e)}")
        return []


def polygon_to_mask(polygon: List[Tuple[int, int]], 
                   image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Converts polygon coordinates back to binary segmentation mask.
    
    Used for polygon refinement to create cleaner masks.
    
    Args:
        polygon: List of (x, y) coordinate tuples
        image_shape: Target mask shape as (height, width)
        
    Returns:
        Binary mask as 2D uint8 numpy array.
        Returns zeros array if conversion fails.
    """
    try:
        # Create empty mask
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Convert polygon to numpy array format
        pts = np.array(polygon, dtype=np.int32)
        
        # Fill polygon with white (255)
        cv2.fillPoly(mask, [pts], color=(255,))
        return mask
        
    except Exception as e:
        print(f"    Warning: Failed to convert polygon to mask: {str(e)}")
        return np.zeros(image_shape, dtype=np.uint8)


def get_boxes(results: List[DetectionResult]) -> List[List[List[float]]]:
    boxes = [result.box.xyxy for result in results]
    return [boxes]


def refine_masks(masks: torch.BoolTensor, 
                polygon_refinement: bool = False) -> List[np.ndarray]:
    """
    Converts SAM output masks to numpy arrays and optionally refines them.
    
    Post-processes the raw mask predictions from SAM by:
    1. Converting to CPU and numpy format
    2. Binarizing the masks
    3. Optionally refining by converting to polygon and back (smooths edges)
    
    Args:
        masks: Raw mask predictions from SAM as torch tensor
        polygon_refinement: If True, refine masks via polygon conversion
        
    Returns:
        List of binary masks as uint8 numpy arrays.
        Returns empty list if processing fails.
    """
    try:
        # Move to CPU and convert to float
        masks = masks.cpu().float()
        
        # Rearrange dimensions: (batch, channels, height, width) -> (batch, height, width, channels)
        masks = masks.permute(0, 2, 3, 1)
        
        # Average across channels and binarize
        masks = masks.mean(axis=-1)
        masks = (masks > 0).int()
        
        # Convert to numpy uint8 format
        masks = masks.numpy().astype(np.uint8)
        masks = list(masks)

        # Optional: refine masks by converting to polygon and back
        # This can smooth jagged edges but may lose fine details
        if polygon_refinement:
            for idx, mask in enumerate(masks):
                try:
                    shape = mask.shape
                    polygon = mask_to_polygon(mask)
                    if polygon:
                        mask = polygon_to_mask(polygon, shape)
                        masks[idx] = mask
                except Exception as e:
                    print(f"    Warning: Failed to refine mask {idx}: {str(e)}")
                    continue

        return masks
        
    except Exception as e:
        print(f"    Warning: Failed to refine masks: {str(e)}")
        return []


def get_device():
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
    """
    Performs zero-shot object detection using Grounding DINO.
    
    Grounding DINO can detect objects based on text descriptions without
    requiring training on those specific classes.
    
    Args:
        image: PIL Image to detect objects in
        labels: List of text labels to detect (e.g., ["rock", "crater"])
        threshold: Confidence threshold (0.0-1.0). Lower = more detections
        detector_id: HuggingFace model ID (default: grounding-dino-tiny)
        
    Returns:
        List of DetectionResult objects with bounding boxes and scores.
        Returns empty list if detection fails.
    """
    try:
        device = get_device()
        detector_id = detector_id or "IDEA-Research/grounding-dino-tiny"
        
        # Load Grounding DINO model via HuggingFace pipeline
        object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)

        # Ensure labels end with period (required by Grounding DINO)
        labels = [label if label.endswith(".") else label + "." for label in labels]
        
        # Run detection
        results = object_detector(image, candidate_labels=labels, threshold=threshold)
        
        # Convert to DetectionResult objects
        results = [DetectionResult.from_dict(result) for result in results]

        return results
        
    except Exception as e:
        print(f"    Warning: Detection failed: {str(e)}")
        return []


def segment(image: Image.Image,
           detection_results: List[DetectionResult],
           polygon_refinement: bool = False,
           segmenter_id: Optional[str] = None) -> List[DetectionResult]:
    """
    Generates segmentation masks using SAM (Segment Anything Model).
    
    Takes bounding boxes from Grounding DINO and uses them to prompt SAM
    to generate precise segmentation masks for each detected object.
    
    Args:
        image: PIL Image to segment
        detection_results: List of DetectionResult objects with bounding boxes
        polygon_refinement: If True, smooth masks via polygon conversion
        segmenter_id: HuggingFace model ID (default: sam-vit-base)
        
    Returns:
        Updated DetectionResult objects with mask field populated.
        Returns original results (without masks) if segmentation fails.
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter_id = segmenter_id or "facebook/sam-vit-base"

        # Load SAM model
        segmentator = AutoModelForMaskGeneration.from_pretrained(segmenter_id).to(device)
        processor = AutoProcessor.from_pretrained(segmenter_id)

        # Prepare bounding boxes as input prompts for SAM
        boxes = get_boxes(detection_results)
        inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(device)

        # Generate segmentation masks
        outputs = segmentator(**inputs)
        
        # Post-process masks to original image size
        masks = processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]

        # Convert masks to numpy and optionally refine
        masks = refine_masks(masks, polygon_refinement)

        # Attach masks to detection results
        for detection_result, mask in zip(detection_results, masks):
            detection_result.mask = mask

        return detection_results
        
    except Exception as e:
        print(f"    Warning: Segmentation failed: {str(e)}")
        # Return detections without masks rather than failing completely
        return detection_results


def grounded_segmentation(image: Union[Image.Image, str],
                         labels: List[str],
                         threshold: float = 0.3,
                         polygon_refinement: bool = False,
                         detector_id: Optional[str] = None,
                         segmenter_id: Optional[str] = None) -> Tuple[np.ndarray, List[DetectionResult]]:
    """
    Complete pipeline: detection + segmentation.
    
    Combines Grounding DINO (detection) and SAM (segmentation) to:
    1. Detect objects matching the given text labels
    2. Generate precise segmentation masks for each detection
    
    This is the main processing function called for each image.
    
    Args:
        image: PIL Image or path to image file
        labels: List of object classes to detect
        threshold: Detection confidence threshold (0.0-1.0)
        polygon_refinement: If True, smooth mask edges
        detector_id: Override default Grounding DINO model
        segmenter_id: Override default SAM model
        
    Returns:
        Tuple of (image_array, detections):
        - image_array: Original image as numpy array
        - detections: List of DetectionResult with boxes and masks
        
    Note:
        Returns empty detections list on failure but always returns valid image array.
    """
    try:
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        # Step 1: Detect objects using Grounding DINO
        detections = detect(image, labels, threshold, detector_id)
        
        # If no detections, return early (no need to run SAM)
        if len(detections) == 0:
            return np.array(image), []
        
        # Step 2: Generate segmentation masks using SAM
        detections = segment(image, detections, polygon_refinement, segmenter_id)
        
        return np.array(image), detections
        
    except Exception as e:
        print(f"    Warning: Grounded segmentation failed: {str(e)}")
        # Try to return original image at minimum
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            return np.array(image), []
        except:
            # Last resort: return blank image
            return np.zeros((100, 100, 3), dtype=np.uint8), []


def save_results(image_array: np.ndarray,
                detections: List[DetectionResult],
                output_path: str,
                save_annotations: bool = True,
                save_metadata: bool = True):
    """
    Saves detection results to disk.
    
    Saves two files per image:
    1. Annotated image (PNG) with bounding boxes and masks drawn
    2. Metadata (JSON) with detection details and polygon coordinates
    
    Args:
        image_array: Original image as numpy array
        detections: List of detection results
        output_path: Path where annotated image will be saved
        save_annotations: If True, save the annotated image
        save_metadata: If True, save the JSON metadata
        
    Note:
        Failures in saving are logged but don't raise exceptions.
        JSON filename is derived from output_path by changing extension.
    """
    # Save annotated image with visualizations
    if save_annotations:
        try:
            annotated = annotate(image_array, detections)
            annotated_pil = Image.fromarray(annotated)
            annotated_pil.save(output_path)
        except Exception as e:
            print(f"    Warning: Failed to save annotated image to {output_path}: {str(e)}")
    
    # Save detection metadata as JSON
    if save_metadata:
        try:
            # Generate JSON filename from image path
            metadata_path = output_path.replace('.png', '.json').replace('.jpg', '.json')
            metadata = []
            
            # Convert each detection to dictionary format
            for det in detections:
                try:
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
                    
                    # Add polygon coordinates if mask exists
                    if det.mask is not None:
                        polygon = mask_to_polygon(det.mask)
                        if polygon:
                            det_dict['polygon'] = polygon
                            
                    metadata.append(det_dict)
                    
                except Exception as e:
                    print(f"    Warning: Failed to process detection metadata: {str(e)}")
                    continue
            
            # Write JSON file
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            print(f"    Warning: Failed to save metadata to {metadata_path}: {str(e)}")


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
                     show_progress: bool = False,
                     ignore_patterns: List[str] = None):
    """
    Processes all images in a single directory.
    
    Iterates through all image files in the directory and runs detection
    and segmentation on each one. Continues processing even if individual
    images fail.
    
    Args:
        input_dir: Path to directory containing images
        output_dir: Path where results will be saved
        labels: List of object labels to detect
        threshold: Detection confidence threshold
        polygon_refinement: Whether to refine mask polygons
        detector_id: Grounding DINO model identifier
        segmenter_id: SAM model identifier
        save_annotations: Save annotated images
        save_metadata: Save JSON metadata
        image_extensions: List of valid image file extensions
        show_progress: Display progress bar
        ignore_patterns: List of filenames/patterns to ignore (e.g., [".DS_Store", "thumbs.db"])
        
    Returns:
        Number of successfully processed images
        
    Note:
        - Creates output directory if it doesn't exist
        - Skips non-image files automatically
        - Failed images are logged with error message
        - For failed images, saves error JSON if save_metadata=True
        - Keyboard interrupt (Ctrl+C) exits gracefully with statistics
        - Files matching ignore_patterns are silently skipped
    """
    
    if image_extensions is None:
        # Default supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    if ignore_patterns is None:
        ignore_patterns = []
    
    # Create output directory (no error if exists)
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f" x ERROR: Failed to create output directory {output_dir}: {str(e)}")
        return 0
    
    # Get all valid image files in directory
    input_path = Path(input_dir)
    try:
        all_files = [f for f in input_path.iterdir() if f.is_file()]
        
        # Filter out ignored files
        image_files = []
        for f in all_files:
            # Skip if file extension not in allowed list
            if f.suffix.lower() not in image_extensions:
                continue
            
            # Skip if filename matches any ignore pattern
            should_ignore = False
            for pattern in ignore_patterns:
                if pattern.lower() in f.name.lower():
                    should_ignore = True
                    break
            
            if not should_ignore:
                image_files.append(f)
        
        image_files = sorted(image_files)
        
    except Exception as e:
        print(f" x ERROR: Failed to list files in {input_dir}: {str(e)}")
        return 0

    # Skip if no images found
    if len(image_files) == 0:
        return 0

    print(f"Processing: {Path(input_dir).name} ({len(image_files)} images)")

    successful = 0  # Counter for successful processing
    failed = 0      # Counter for failed images
    
    # Process each image file
    for image_file in tqdm(image_files, desc=f" Processing {Path(input_dir).name}",
                           disable=not show_progress, leave=False):
        try:
            # Run detection and segmentation pipeline
            image_array, detections = grounded_segmentation(
                image=str(image_file),
                labels=labels,
                threshold=threshold,
                polygon_refinement=polygon_refinement,
                detector_id=detector_id,
                segmenter_id=segmenter_id
            )
            
            # Save annotated image and/or metadata
            output_path = os.path.join(output_dir, f"annotated_{image_file.name}")
            save_results(image_array, detections, output_path, 
                        save_annotations, save_metadata)
            
            successful += 1

        except KeyboardInterrupt:
            # User pressed Ctrl+C - exit gracefully
            print("\n\n x Process interrupted by user (Ctrl+C)")
            print(f"   Processed: {successful}/{len(image_files)} images")
            raise  # Re-raise to propagate up
            
        except Exception as e:
            # Image processing failed - log error and continue
            failed += 1
            print(f" x Error processing {image_file.name}: {str(e)}")
            
            # Save error information to JSON for tracking
            if save_metadata:
                try:
                    output_path = os.path.join(output_dir, f"annotated_{image_file.name}")
                    metadata_path = output_path.replace('.png', '.json').replace('.jpg', '.json')
                    with open(metadata_path, 'w') as f:
                        json.dump({"error": str(e), "file": str(image_file.name)}, f, indent=2)
                except:
                    pass  # Ignore errors in error logging
            continue

    # Print summary statistics
    status = f" V COMPLETED: {Path(input_dir).name}"
    if failed > 0:
        status += f" ({successful} successful, {failed} failed)"
    else:
        status += f" ({successful} images processed)"
    print(status + "\n")
    
    return successful


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
                            show_progress=True,
                            ignore_patterns: List[str] = None):
    """
    Processes all subdirectories and images in the main input directory.
    
    This function handles batch processing by:
    1. Processing images directly in the input directory
    2. Processing all subdirectories recursively
    3. Skipping files/folders matching ignore patterns
    
    Args:
        input_main_dir: Path to main directory containing subdirectories
        output_main_dir: Path where all results will be saved
        labels: List of object labels to detect
        threshold: Detection confidence threshold (0.0-1.0)
        polygon_refinement: Whether to refine mask polygons
        detector_id: Grounding DINO model identifier
        segmenter_id: SAM model identifier
        save_annotations: Save annotated images
        save_metadata: Save JSON metadata files
        image_extensions: List of valid image file extensions
        show_progress: Display progress bars
        ignore_patterns: List of names to ignore (files/folders containing these strings)
        
    Returns:
        None (prints summary statistics)
        
    Note:
        - Processes both root-level images and subdirectories
        - Ignores directories matching ignore_patterns
        - Continues processing even if individual directories fail
        - Provides detailed statistics at completion
        - Keyboard interrupt (Ctrl+C) exits gracefully with partial statistics
    """

    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

    if ignore_patterns is None:
        ignore_patterns = []

    # Create output directory
    try:
        os.makedirs(output_main_dir, exist_ok=True)
    except Exception as e:
        print(f"ERROR: Failed to create output directory {output_main_dir}: {str(e)}")
        return

    # Get all subdirectories and count root-level images
    input_main_path = Path(input_main_dir)
    try:
        # Get all subdirectories, filtering out ignored ones
        all_dirs = [d for d in input_main_path.iterdir() if d.is_dir()]
        image_dirs = []
        
        for d in all_dirs:
            # Skip if directory name matches any ignore pattern
            should_ignore = False
            for pattern in ignore_patterns:
                if pattern.lower() in d.name.lower():
                    should_ignore = True
                    break
            
            if not should_ignore:
                image_dirs.append(d)
        
        image_dirs = sorted(image_dirs)
        
        # Count images in root directory (excluding ignored files)
        all_root_files = [f for f in input_main_path.iterdir() 
                         if f.is_file() and f.suffix.lower() in image_extensions]
        
        root_images = []
        for f in all_root_files:
            should_ignore = False
            for pattern in ignore_patterns:
                if pattern.lower() in f.name.lower():
                    should_ignore = True
                    break
            
            if not should_ignore:
                root_images.append(f)
        
        total_images_here = len(root_images)
        
    except Exception as e:
        print(f"ERROR: Failed to list directories in {input_main_dir}: {str(e)}")
        return

    # Print header with discovery information
    print(f"\n{'='*70}")
    if len(image_dirs) > 0:
        print(f"FOUND {len(image_dirs)} DIRECTORIES TO PROCESS")
    if total_images_here > 0:
        print(f"FOUND {total_images_here} IMAGES HERE TO PROCESS")
    if ignore_patterns:
        print(f"IGNORING: {', '.join(ignore_patterns)}")
    if len(image_dirs) <= 0 and total_images_here <= 0:
        print(f"NOTHING TO PROCESS")
        print(f"{'='*70}\n")
        return
    print(f"Labels: {labels}")
    print(f"Device: {str(get_device())}")
    print(f"{'='*70}")

    processed_here = 0
    # Process images in current/root directory
    if total_images_here > 0:
        try:
            processed_here = process_directory(
                input_dir=input_main_dir,
                output_dir=output_main_dir,
                labels=labels,
                threshold=threshold,
                polygon_refinement=polygon_refinement,
                detector_id=detector_id,
                segmenter_id=segmenter_id,
                save_annotations=save_annotations,
                save_metadata=save_metadata,
                show_progress=show_progress,
                ignore_patterns=ignore_patterns
            )
        except KeyboardInterrupt:
            print("\n x Process interrupted by user")
            return
        except Exception as e:
            print(f" x ERROR processing directory {Path(input_main_dir).name}: {str(e)}")

    total_images = 0
    failed_dirs = 0
    
    # Process each subdirectory
    for idx, image_dir in enumerate(image_dirs, 1):
        try:
            input_dir = str(image_dir)
            output_dir = os.path.join(output_main_dir, f"annotated_{image_dir.name}")

            print(f"[Sub-directory {idx}/{len(image_dirs)}]")
            
            # Process the subdirectory
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
                show_progress=show_progress,
                ignore_patterns=ignore_patterns
            )

            total_images += processed
            
        except KeyboardInterrupt:
            print("\n x Process interrupted by user")
            print(f"   Completed {idx-1}/{len(image_dirs)} directories")
            return
        except Exception as e:
            failed_dirs += 1
            print(f" x ERROR processing directory {image_dir.name}: {str(e)}")
            print(f"   Stacktrace: {traceback.format_exc()}")
            continue

    # Print final summary
    print(f"{'='*70}")
    print(f"ALL PROCESSING COMPLETE")
    print(f"Input location: {input_main_dir}")
    if failed_dirs > 0:
        print(f"Sub-directories: {len(image_dirs) - failed_dirs} successful, {failed_dirs} failed")
    else:
        print(f"Total sub-directories processed: {len(image_dirs)}")
    print(f"Total images processed in  sub-directories: {total_images}")
    print(f"Total images processed in   this directory: {processed_here}")
    print(f"Output location: {output_main_dir}")
    print(f"{'='*70}\n")
    

def main():
    """
    Main entry point for the script.
    
    Parses command-line arguments and initiates batch processing
    of images using Grounding DINO and SAM models.
    
    Command-line Arguments:
        --input_dir, -i: Input directory (required)
        --output_dir, -o: Output directory (required)
        --labels, -l: Detection labels (required)
        --threshold, -t: Confidence threshold (default: 0.3)
        --ignore: Files/folders to ignore (optional, can specify multiple)
        --no_polygon_refinement: Disable polygon refinement
        --detector_id: Override default Grounding DINO model
        --segmenter_id: Override default SAM model
        --no_annotations: Skip saving annotated images
        --no_metadata: Skip saving JSON metadata
        --no_progress: Hide progress bars
    
    Examples:
        Basic usage:
            python script.py -i ./input -o ./output -l rock crater
        
        With ignore patterns:
            python script.py -i ./input -o ./output -l rock --ignore .DS_Store --ignore thumb
        
        Advanced:
            python script.py -i ./data -o ./results -l rock crater boulder \\
                --threshold 0.4 --ignore annotated --no_progress
    """
    parser = argparse.ArgumentParser(
        description="Automated Grounded SAM for directory processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  %(prog)s -i ./images -o ./results -l rock crater
  
  # Ignore specific files/folders
  %(prog)s -i ./data -o ./output -l rock --ignore .DS_Store --ignore thumbnail
  
  # Advanced configuration
  %(prog)s -i ./input -o ./output -l rock crater boulder \\
      --threshold 0.4 --ignore annotated --no_progress
        """
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
        "--ignore",
        type=str,
        action="append",
        default=[],
        help="Ignore files/folders containing this string (can use multiple times, e.g., --ignore .DS_Store --ignore thumb)"
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
    
    try:
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
            show_progress=not args.no_progress,
            ignore_patterns=args.ignore
        )
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting gracefully...")
    except Exception as e:
        print(f"\n\nFATAL ERROR: {str(e)}")
        print(f"Stacktrace:\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()
