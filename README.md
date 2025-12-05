# Object Tracking and Face Detection System

## Overview
This project implements a computer vision system for tracking and recognizing moving objects and human faces. The system processes video footage to detect, track, and identify multiple objects across frames.

The system has been tested using video footage containing various objects recognizable by COCO classes (Lin et al., 2014). The implementation uses computer vision libraries including OpenCV CV2 (OpenCV Team, 2023) and PyTorch (PyTorch Team, 2023).

The methodology involves using detection libraries with optimized configurations, iteratively refined to improve detection and tracking performance. The system generates annotated videos that visualize object detection and tracking results with bounding boxes and track identifiers.

## Features

The system provides two main capabilities:

### 1. Object Detection and Tracking
- Uses Mask R-CNN model (He et al., 2017) with a ResNet-50-FPN backbone
- Pretrained convolutional networks for robust object detection
- Tracks multiple objects across video frames
- Generates annotated output video with tracked objects

### 2. Face Detection and Tracking
- Implements Haar Cascades algorithm (Viola & Jones, 2001)
- Uses multiple trained cascade models (OpenCV, 2023) for improved accuracy:
  - Frontal face default
  - Portrait face
  - Frontal face alt
- Tracks detected faces across frames with unique identifiers
- Generates annotated output video with tracked faces

## Key Algorithms

The system implements several key computer vision algorithms:

- **Mask R-CNN**: Deep learning-based object detection (He et al., 2017)
- **Haar Cascades**: Classical face detection algorithm (Viola & Jones, 2001)
- **Intersection over Union (IoU)**: For tracking association between frames (Rezatofighi et al., 2019)
- **Hungarian Algorithm**: Optimal assignment for multi-object tracking
- **Non-Maximum Suppression (NMS)**: Removes duplicate detections

## Evaluation Metrics (Optional)

The system can optionally evaluate performance against ground truth annotations using standard computer vision metrics:

- **Intersection over Union (IoU)**: Measures bounding box detection accuracy (Rezatofighi et al., 2019)
- **Precision Score**: Evaluates detection accuracy using scikit-learn
- **Recall Score**: Measures detection completeness
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy Score**: Overall detection accuracy

To use evaluation features, create ground truth annotations using tools like [makesense.ai](https://www.makesense.ai/) and place them in the appropriate directories.

## Repository Contents

### Video Data
- **Input Video**: Stock footage from [Artlist](https://artlist.io/stock-footage/clip/experimenting-colleagues-friends-flambe/321802) - freely licensed video containing multiple objects including people and various items (COCO classes)
- **Object Detection Output**: Annotated video showing tracked objects with bounding boxes
- **Face Detection Output**: Annotated video showing tracked faces with identifiers

### Jupyter Notebook
**Main file**: `computer_vision_colab.ipynb`

This notebook contains the complete implementation:
- Python code for object detection and tracking
- Face detection and tracking algorithms
- Video processing and annotation generation
- Visualization of tracked objects and faces
- Optional evaluation against ground truth annotations

## Technical Details

### Tracking Methods
The system supports two frame-to-frame tracking approaches:
- **Intersection over Union (IoU)**: Measures overlap between bounding boxes
- **Point-to-Point Distance**: Euclidean distance between detection centroids (Dokmanic, Perotin, Ranieri, & Vetterli, 2015)

### Object Detection Parameters (Mask R-CNN)
- `score_threshold`: Filters detections based on confidence score per class
- Configurable threshold for balancing precision and recall

### Face Detection Parameters (Haar Cascades)
- `scaleFactor`: Controls multi-scale detection by specifying scale increment at each pyramid level
- `minNeighbors`: Minimum number of nearby detections required to confirm a detection
- `minSize` and `maxSize`: Filter unreliable results by defining bounding box dimension constraints

### Tracking Configuration
- **Minimum Track Length**: Filters out noise by requiring tracks to appear in multiple consecutive frames
- **Association Method**: Choose between IoU-based or centroid distance-based tracking
- **Detection Threshold**: Configurable confidence threshold for object/face detection

## Usage

To run the computer vision system:

1. Open `computer_vision_colab.ipynb` in Google Colab or Jupyter Notebook
2. Install required dependencies:
   ```bash
   pip install opencv-python scikit-learn torch torchvision
   ```
3. Run all cells to:
   - Load and process input video
   - Perform object detection and tracking
   - Perform face detection and tracking
   - Generate annotated output videos with bounding boxes and track IDs
   - (Optional) Evaluate against ground truth annotations if provided

## References

- He et al. (2017). Mask R-CNN
- Lin et al. (2014). COCO Dataset
- Viola & Jones (2001). Haar Cascades
- Rezatofighi et al. (2019). Intersection over Union
- Dokmanic, Perotin, Ranieri, & Vetterli (2015). Point-to-Point Distance
- OpenCV Team (2023). OpenCV Library
- PyTorch Team (2023). PyTorch Framework
- Scikit-learn developers (2024). Scikit-learn Library
- Skalski (2019). makesense.ai - Free online annotation tool
- Artlist. Stock footage: [Experimenting Colleagues Friends Flambe](https://artlist.io/stock-footage/clip/experimenting-colleagues-friends-flambe/321802)