# crop.py
import cv2

def crop_pid_diagram(image_path):
    """Enhanced cropping to remove extra text and focus on the main diagram."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 10000]  # Ignore small noise
    
    if not filtered_contours:
        raise ValueError("No suitable contour found for cropping.")
    
    max_contour = max(filtered_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    
    if w > h:  # Remove right side assuming text notes
        w = int(w * 0.77)
    
    return img[y:y+h, x:x+w]
