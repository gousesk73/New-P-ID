# P&ID Line Detection Project

This project contains three methods for processing Piping and Instrumentation Diagram (P&ID) images to detect lines or contours. Each method is implemented in a separate Python script located in its respective folder: `method1/line.py`, `method2/line.py`, and `method3/line.py`. This README provides an overview of their features and instructions to run them.

## Folder Structure
```
root/
├── method1/
│   └── line.py
├── method2/
│   └── line.py
├── method3/
│   └── line.py
└── README.md
```

## Method 1: Line Detection with Hough Transform and DBSCAN
**File**: `method1/line.py`

### Features
- **Objective**: Detects and connects straight lines (including dashed lines) in P&ID diagrams, focusing on piping.
- **Cropping**: Automatically crops the image to focus on the main diagram area using contour detection.
- **Symbol/Text Removal**: Uses bounding box data from CSV files to mask and inpaint symbols and text, preserving lines.
- **Line Detection**: Employs Canny edge detection, Hough Line Transform, and DBSCAN clustering to detect and connect dashed line segments.
- **Filtering**: Filters lines by length, angle (horizontal/vertical), and proximity to symbols/text to reduce noise.
- **Output**: 
  - Visualizes symbol masks and detected lines.
  - Saves line coordinates to `detected_lines.csv`.
  - Optionally saves the cropped image.

### Dependencies
- `opencv-python` (cv2)
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

## Method 2: Contour Detection
**File**: `method2/line.py`

### Features
- **Objective**: Detects contours of larger structures (e.g., equipment) in P&ID diagrams.
- **Cropping**: Same as Method 1, crops the image to isolate the main diagram.
- **Preprocessing**: Applies Gaussian blur to reduce noise.
- **Contour Detection**: Uses Canny edge detection and contour filtering (area > 500) to identify outlines of significant objects.
- **Output**: Visualizes the cropped image with detected contours overlaid in green.
- **Simplicity**: No explicit symbol/text removal or line connection, making it lightweight but less focused on fine lines.

### Dependencies
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`

## Method 3: Line Detection with LSD (Line Segment Detector)
**File**: `method3/line.py`

### Features
- **Objective**: Detects line segments in P&ID diagrams using the Line Segment Detector (LSD) algorithm.
- **Preprocessing**: Converts the image to grayscale and enhances contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
- **Line Detection**: Uses LSD to find line segments in the image.
- **Visualization**: Draws detected lines on the original image.
- **Output**: Displays the original and processed images side by side using Matplotlib.

### Dependencies
- `opencv-python` (cv2)
- `numpy`
- `matplotlib`

## Prerequisites
1. **Python 3.x**: Ensure Python is installed on your system.
2. **Install Dependencies**: Run the following command to install required libraries:
   ```bash
   pip install opencv-python numpy pandas matplotlib scikit-learn
   ```
3. **Input Files**:
   - For Method 1: Prepare an image (e.g., `0.jpg`) and two CSV files:
     - `predictions.csv` (symbol bounding boxes with columns: `X1`, `Y1`, `X2`, `Y2`)
     - `detected_text_boxes.csv` (text bounding boxes with columns: `x_min`, `y_min`, `x_max`, `y_max`)
   - For Method 2: Only an image (e.g., `0.jpg`) is required.
   - For Method 3: An image (e.g., `cropped_image.png`) is required.

## How to Run

### Running Method 1
1. **Navigate to the `method1` folder**:
   ```bash
   cd method1
   ```
2. **Place input files** in the `method1` folder (or adjust paths in the script):
   - Image: `0.jpg`
   - Symbol bounding boxes: `predictions.csv`
   - Text bounding boxes: `detected_text_boxes.csv`
3. **Run the script**:
   ```bash
   python line.py
   ```
4. **Results**:
   - A plot will display the symbol mask and detected lines.
   - `detected_lines.csv` will be saved with line coordinates.
   - If `save_cropped_image=True`, a cropped image will be saved as `cropped_image.png`.

### Running Method 2
1. **Navigate to the `method2` folder**:
   ```bash
   cd method2
   ```
2. **Place the input image** in the `method2` folder (or adjust the path in the script):
   - Image: `0.jpg`
3. **Run the script**:
   ```bash
   python line.py
   ```
4. **Results**:
   - A plot will display the cropped image with detected contours in green.
   - If `save_cropped_image=True`, a cropped image will be saved as `cropped_image.png`.

### Running Method 3
1. **Navigate to the `method3` folder**:
   ```bash
   cd method3
   ```
2. **Place the input image** in the `method3` folder (or adjust the path in the script):
   - Image: `cropped_image.png`
3. **Run the script**:
   ```bash
   python line.py
   ```
4. **Results**:
   - A plot will display the original image alongside the detected lines.

