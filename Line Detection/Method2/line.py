import cv2
import numpy as np
import matplotlib.pyplot as plt

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


def preprocess_image(image):
    """Convert to grayscale and apply Gaussian blur."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return image, blurred

def edge_detection(blurred):
    """Detect edges using the Canny edge detection."""
    edges = cv2.Canny(blurred, 50, 150)
    return edges

def contour_detection(edges):
    """Find contours from the detected edges."""
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on area size (remove small contours like text/symbols)
    large_contours = [contour for contour in contours if cv2.contourArea(contour) > 500]

    return large_contours

def draw_contours(image, contours):
    """Draw contours on the original image."""
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    return contour_image

def process_pid_image(image_path, save_cropped_image=False, cropped_image_path='cropped_image.png'):
    """Complete pipeline for P&ID processing, including cropping and detection."""
    # Step 1: Crop the image
    cropped_img = crop_pid_diagram(image_path, save_cropped_image, cropped_image_path)
    
    # Step 2: Preprocess the cropped image
    image, blurred = preprocess_image(cropped_img)
    
    # Step 3: Detect edges
    edges = edge_detection(blurred)
    
    # Step 4: Detect contours
    contours = contour_detection(edges)
    
    # Step 5: Draw contours on the image
    contour_image = draw_contours(image, contours)
    
    # Step 6: Display results using matplotlib
    plt.figure(figsize=(10, 6))
       
    # Contour Detected Image
    plt.subplot(1, 1, 1)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title("Contours Detected")
    plt.axis('off')
    
    plt.show()

# Example Usage:
process_pid_image("0.jpg", save_cropped_image=False)
