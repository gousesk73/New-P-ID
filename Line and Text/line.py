# line.py
import cv2
import numpy as np
import pandas as pd
from crop import crop_pid_diagram
import matplotlib.pyplot as plt

def remove_symbols_and_detect_lines(image_path, output_csv_path='detected_lines.csv'):
    """Detect lines in a P&ID diagram after cropping and symbol removal."""
    # Load and crop the image to remove extra text
    cropped_image = crop_pid_diagram(image_path)
    
    # Convert cropped image to grayscale
    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Step 1: Adaptive Threshold for symbol detection
    symbol_mask = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Step 2: Morphological operations to refine the mask (Larger kernel)
    kernel = np.ones((5, 5), np.uint8)  # Increased kernel size to remove symbols better
    refined_mask = cv2.morphologyEx(symbol_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    refined_mask = cv2.erode(refined_mask, kernel, iterations=2)  # Reduce noise further
    
    # Step 3: Inpaint to remove symbols while preserving lines
    img_no_symbols = cv2.inpaint(img, refined_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    
    # Step 4: Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_no_symbols)
    
    # Step 5: Canny Edge Detection
    edges = cv2.Canny(img_enhanced, 50, 150)
    
    # Step 6: Hough Line Transform for line detection (better settings)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=140, minLineLength=40, maxLineGap=15)
    
    # Step 7: Filter out short or irrelevant lines
    line_data = []  # List to store detected line coordinates
    img_colored = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img_height, img_width = img.shape[:2]  # Get image size
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Remove very short lines (noise)
            if length < 30:
                continue
            
            # Filter lines by angle to avoid false detections from text
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -10 <= angle <= 10 or 80 <= abs(angle) <= 100:  # Keep mostly horizontal & vertical lines
                cv2.line(img_colored, (x1, y1), (x2, y2), (0, 0, 255), 2) 
                
                line_data.append([x1, y1, x2, y2])

    # Save detected line coordinates to a CSV file
    df_lines = pd.DataFrame(line_data, columns=['X1', 'Y1', 'X2', 'Y2'])
    df_lines.to_csv(output_csv_path, index=False)

    # Display Results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(refined_mask, cmap='gray')
    plt.title('Symbol Mask')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_colored)
    plt.title('Detected Lines (Filtered)')
    plt.axis('off')
    
    plt.show()

    return cropped_image
