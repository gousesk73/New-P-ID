# main.py
from line import remove_symbols_and_detect_lines
from text import process_image  # Updated import path for text.py

def main():
    # Set the image path directly in the script
    image_path = '0.jpg'  # Replace with the path to your image
    
    # Choose the process type (uncomment the one you want to run)
    process_type = 'text'  # Choose between 'line' or 'text'
    
    if process_type == 'line':
        print("Starting Line Detection...")
        output_csv_path = 'detected_lines.csv'  # Path where the lines will be saved
        remove_symbols_and_detect_lines(image_path, output_csv_path)
    elif process_type == 'text':
        print("Starting Text Detection...")
        output_csv_path = 'detected_text.csv'  # Save path for recognized text
        process_image(image_path, output_csv_path)
    else:
        print("Invalid process type. Please choose 'line' or 'text'.")

if __name__ == "__main__":
    main()
