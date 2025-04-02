import cv2
import pytesseract
from craft_text_detector import Craft
import numpy as np
import csv
from crop import crop_pid_diagram  # Import cropping function

# Initialize the CRAFT model
craft = Craft(output_dir='output')  # Set the output directory for detected text regions

# Function to detect text in an image
def detect_text_from_image(image):
    """Detect text regions using CRAFT."""
    prediction_result = craft.detect_text(image)
    text_boxes = prediction_result['boxes']
    return text_boxes

# Preprocess the image for better OCR performance
def preprocess_image(image):
    """Apply preprocessing steps to enhance text detection."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

# Function to handle vertical text by rotating it to horizontal
def handle_vertical_text(cropped_image, aspect_ratio_threshold=0.5):
    """Rotate text if it's vertical based on aspect ratio."""
    h, w = cropped_image.shape[:2]
    aspect_ratio = w / h
    if aspect_ratio <= aspect_ratio_threshold:  # Vertical text
        cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_90_CLOCKWISE)
    return cropped_image

# Function to recognize text from the detected boxes using Tesseract
def recognize_text_from_boxes(image, text_boxes):
    """Extract text from detected bounding boxes."""
    recognized_text = []
    for box in text_boxes:
        pts = box.astype(int)
        x_min, y_min = np.min(pts, axis=0)
        x_max, y_max = np.max(pts, axis=0)
        
        # Crop the detected text region
        cropped_image = image[y_min:y_max, x_min:x_max]
        
        # Preprocess the cropped image for better OCR
        processed_image = preprocess_image(cropped_image)
        
        # Handle vertical text (rotate if necessary)
        processed_image = handle_vertical_text(processed_image)
        
        # Use Tesseract to recognize text
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        recognized_text.append((text.strip(), x_min, y_min, x_max, y_max))  # Store text and its bounding box coordinates

    return recognized_text

# Function to save detected text along with its bounding boxes into a CSV file
def save_text_to_csv(recognized_text, output_csv_path):
    """Save recognized text and bounding boxes into a CSV file."""
    with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Detected Text', 'x_min', 'y_min', 'x_max', 'y_max'])  # Write header with bounding box columns
        for text, x_min, y_min, x_max, y_max in recognized_text:
            writer.writerow([text, x_min, y_min, x_max, y_max])  # Write each recognized text with bounding box

# Main function to run the detection, recognition, and saving to CSV
def process_image(image_path, output_csv_path):
    """Process an image: Crop → Detect Text → Recognize Text → Save to CSV"""
    # Step 1: Crop the image to remove extra text regions
    cropped_image = crop_pid_diagram(image_path)
    
    # Step 2: Perform text detection on the cropped image
    text_boxes = detect_text_from_image(cropped_image)
    
    # Step 3: Recognize text from detected text regions
    recognized_text = recognize_text_from_boxes(cropped_image, text_boxes)
    
    # Step 4: Draw bounding boxes on detected text (for visualization, optional)
    for box in text_boxes:
        pts = box.astype(int)
        cv2.polylines(cropped_image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    # Step 5: Save recognized text to a CSV file along with bounding boxes
    save_text_to_csv(recognized_text, output_csv_path)

    print(f"Detected text and bounding boxes have been saved to {output_csv_path}")
