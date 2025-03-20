import cv2
import matplotlib.pyplot as plt

def preprocess_image(image):
    """Convert to grayscale and apply CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE to enhance contrast in the image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)
    
    return enhanced_gray

def detect_lines_with_lsd(image):
    """Detect lines using LSD (Line Segment Detector)."""
    # Initialize the LSD detector
    lsd = cv2.createLineSegmentDetector(0)  # 0 for basic LSD detector
    
    # Detect lines (LSD returns a tuple containing the detected lines)
    result = lsd.detect(image)
    
    # If the result contains more than one value, unpack the lines only
    lines = result[0]  # We only need the lines
    
    return lines

def draw_lines(image, lines):
    """Draw detected lines on the image."""
    line_image = image.copy()
    
    # Draw each line detected by LSD
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]  # Extract the coordinates of each line segment
            cv2.line(line_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # Green color for lines
    
    return line_image

def process_image(image_path):
    """Process the image and detect lines using LSD."""
    # Step 1: Load the image
    img = cv2.imread(image_path)
    
    # Step 2: Preprocess the image to enhance contrast
    enhanced_image = preprocess_image(img)
    
    # Step 3: Detect lines using LSD
    lines = detect_lines_with_lsd(enhanced_image)
    
    # Step 4: Draw the detected lines on the original image
    line_image = draw_lines(img, lines)
    
    # Step 5: Display the results
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    # Lines Detected Image
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
    plt.title("Lines Detected with LSD")
    plt.axis('off')
    
    plt.show()

# Example usage:
image_path = "cropped_image.png"  # Replace with your image path
process_image(image_path)
