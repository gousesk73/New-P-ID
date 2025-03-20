import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

def create_masks_from_bboxes(image, symbol_bbox_file, text_bbox_file):
    """
    Create masks for symbols and text areas based on bounding boxes from two CSV files.
    Arguments:
        image: The image on which the masks should be created.
        symbol_bbox_file: Path to the CSV file containing symbol bounding box coordinates.
        text_bbox_file: Path to the CSV file containing text bounding box coordinates.
    Returns:
        symbol_mask: A binary mask where symbol areas are white (255).
        text_mask: A binary mask where text areas are white (255).
    """
    # Read bounding boxes from the two CSV files
    symbol_bboxes = pd.read_csv(symbol_bbox_file)
    text_bboxes = pd.read_csv(text_bbox_file)
    
    # Initialize empty masks for both symbol and text areas
    symbol_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    text_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Add symbol bounding boxes to symbol mask
    for _, row in symbol_bboxes.iterrows():
        x1, y1, x2, y2 = map(int, [row['X1'], row['Y1'], row['X2'], row['Y2']])
        cv2.rectangle(symbol_mask, (x1, y1), (x2, y2), 255, thickness=-1)
    
    # Add text bounding boxes to text mask
    for _, row in text_bboxes.iterrows():
        x1, y1, x2, y2 = map(int, [row['x_min'], row['y_min'], row['x_max'], row['y_max']])
        cv2.rectangle(text_mask, (x1, y1), (x2, y2), 255, thickness=-1)

    return symbol_mask, text_mask


def crop_pid_diagram(image_path, save_cropped_image=False, cropped_image_path='cropped_image.png'):
    """Enhanced cropping to remove extra text and focus on the main diagram."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]
    
    if not filtered_contours:
        raise ValueError("No suitable contour found for cropping.")
    
    max_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    if w > h:
        w = int(w * 0.77)
    
    cropped_img = img[y:y+h, x:x+w]
    
    if save_cropped_image:
        cv2.imwrite(cropped_image_path, cropped_img)
        print(f"Cropped image saved to {cropped_image_path}")
    
    return cropped_img


def connect_dashed_lines(lines, max_gap=30, angle_threshold=15):
    """
    Connect dashed line segments using DBSCAN clustering based on proximity and alignment.
    Arguments:
        lines: List of lines in the format [[x1, y1, x2, y2], ...]
        max_gap: Maximum gap between two line segments to cluster them.
        angle_threshold: Maximum angle difference (in degrees) to consider lines aligned.
    Returns:
        connected_lines: List of connected lines.
    """
    if not lines:
        return []

    # Prepare data for clustering: use the midpoint of each line segment and its angle
    midpoints = []
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line
        midpoint = [(x1 + x2) / 2, (y1 + y2) / 2]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        midpoints.append(midpoint)
        angles.append(angle)

    midpoints = np.array(midpoints)
    angles = np.array(angles)

    # Use DBSCAN to cluster line segments based on proximity
    clustering = DBSCAN(eps=max_gap, min_samples=1).fit(midpoints)
    labels = clustering.labels_

    # Group lines by cluster
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(lines[i])

    connected_lines = []
    for label in clusters:
        cluster_lines = clusters[label]
        if len(cluster_lines) == 1:
            connected_lines.append(cluster_lines[0])
            continue

        # Sort lines in the cluster by their dominant direction (horizontal or vertical)
        cluster_angles = [np.arctan2(line[3] - line[1], line[2] - line[0]) * 180 / np.pi for line in cluster_lines]
        avg_angle = np.mean(cluster_angles)
        if 45 <= abs(avg_angle) <= 135:  # Vertical lines
            cluster_lines.sort(key=lambda x: (x[1] + x[3]) / 2)  # Sort by y-coordinate
        else:  # Horizontal lines
            cluster_lines.sort(key=lambda x: (x[0] + x[2]) / 2)  # Sort by x-coordinate

        # Merge lines in the cluster if they are aligned
        current_line = list(cluster_lines[0])
        for i in range(1, len(cluster_lines)):
            x1, y1, x2, y2 = cluster_lines[i]
            prev_x1, prev_y1, prev_x2, prev_y2 = current_line

            prev_angle = np.arctan2(prev_y2 - prev_y1, prev_x2 - prev_x1) * 180 / np.pi
            curr_angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angle_diff = abs(prev_angle - curr_angle)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff

            # Calculate the gap between the end of the previous line and the start of the current line
            gap = np.sqrt((x1 - prev_x2) ** 2 + (y1 - prev_y2) ** 2)

            if angle_diff < angle_threshold and gap < max_gap:
                current_line[2] = x2
                current_line[3] = y2
            else:
                connected_lines.append(current_line)
                current_line = [x1, y1, x2, y2]

        connected_lines.append(current_line)

    return connected_lines


def remove_symbols_and_detect_lines(image_path, symbol_bbox_file, text_bbox_file, save_cropped_image=False, cropped_image_path='cropped_image.png'):
    # Load and crop the image to remove extra text
    cropped_image = crop_pid_diagram(image_path, save_cropped_image, cropped_image_path)
    
    # Create masks for both symbols and text areas
    symbol_mask, text_mask = create_masks_from_bboxes(cropped_image, symbol_bbox_file, text_bbox_file)
    
    # Convert cropped image to grayscale
    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    
    # Inpaint to remove symbols while preserving lines
    img_no_symbols = cv2.inpaint(img, symbol_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    
    # Step 1: Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_enhanced = clahe.apply(img_no_symbols)
    
    # Step 2: Canny Edge Detection with adjusted thresholds to reduce noise
    edges = cv2.Canny(img_enhanced, 50, 150)
    
    # Step 3: Morphological operation to connect small gaps in dashed lines
    kernel = np.ones((4, 4), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 4: Hough Line Transform with adjusted parameters
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=300, minLineLength=5, maxLineGap=16)
    
    # Step 5: Filter and connect lines
    line_data = []
    img_colored = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Filter out very short lines to reduce noise
            if length < 15:
                continue
            
            # Filter by angle to focus on horizontal and vertical lines
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if -15 <= angle <= 15 or 75 <= abs(angle) <= 105:
                # Additional filter: Check if the line is too close to a symbol or text
                mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                if symbol_mask[mid_y, mid_x] == 255 or text_mask[mid_y, mid_x] == 255:
                    continue
                line_data.append([x1, y1, x2, y2])

    # Step 6: Connect dashed lines using clustering
    connected_lines = connect_dashed_lines(line_data, max_gap=30, angle_threshold=15)

    # Draw the connected lines on the image
    for line in connected_lines:
        x1, y1, x2, y2 = line
        cv2.line(img_colored, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # Save detected line coordinates to a CSV file
    df_lines = pd.DataFrame(connected_lines, columns=['X1', 'Y1', 'X2', 'Y2'])
    df_lines.to_csv('detected_lines.csv', index=False)

    # Display Results
    plt.figure(figsize=(15, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(symbol_mask, cmap='gray')
    plt.title('Symbol Mask')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(img_colored)
    plt.title('Detected Lines (Filtered)')
    plt.axis('off')
    
    plt.show()

    return cropped_image

# Example usage
image_path = '0.jpg'  # Replace with the path to your image
symbol_bbox_file = 'predictions.csv'  # CSV file containing symbol bounding boxes
text_bbox_file = 'detected_text_boxes.csv'  # CSV file containing text bounding boxes
cropped_image = remove_symbols_and_detect_lines(image_path, symbol_bbox_file, text_bbox_file, save_cropped_image=True, cropped_image_path='cropped_image.png')
