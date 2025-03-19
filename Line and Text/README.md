# **P&ID Diagram Text and Line Detection**

This project automates **text detection** and **line detection** in **Piping & Instrumentation Diagrams (P&IDs)**. It uses **CRAFT** for text detection, **Tesseract OCR** for text recognition, and **OpenCV** for line detection.

---

## **Features**
✅ **Automatic Cropping** → Removes unnecessary text regions (like notes) from the P&ID diagram.  
✅ **Text Detection & OCR** → Detects and extracts text using **CRAFT** and **Tesseract OCR**.  
✅ **Line Detection** → Detects and saves P&ID diagram lines using **OpenCV's Hough Transform**.  
✅ **CSV Output** → Extracted text and detected lines are saved as CSV files.  

---

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/gousesk73/New-P-ID.git
cd New-P-ID
```

### **2. Install Dependencies**
Install all required libraries using:
```bash
pip install -r requirements.txt
```
If you are facing compatibility issues, try using Python 3.10.0.

### **3. Install Tesseract OCR**
Make sure **Tesseract OCR** is installed on your system.

#### **Windows**
Download & Install Tesseract from:  
🔗 [Tesseract Download](https://github.com/UB-Mannheim/tesseract/wiki)

Then, add **Tesseract to your PATH** (update in your script if necessary).

#### **Linux (Ubuntu)**
```bash
sudo apt install tesseract-ocr
```

#### **Mac (Homebrew)**
```bash
brew install tesseract
```

---

## **Fix Import Error in CRAFT-Text-Detector**
If you encounter the following error:
```python
ImportError: cannot import name 'model_urls' from 'torchvision.models.vgg'
```
Follow these steps to fix it:

### **Option 1: Modify the CRAFT Code**
1. Open the following file in a text editor:
   ```
   site-packages/craft_text_detector/models/basenet/vgg16_bn.py
   ```
2. Find the line:
   ```python
   from torchvision.models.vgg import model_urls
   ```
3. Replace it with:
   ```python
   from torchvision.models import VGG16_BN_Weights
   ```
4. Find where `model_urls['vgg16_bn']` is used and replace it with:
   ```python
   weights = VGG16_BN_Weights.DEFAULT
   state_dict = weights.get_state_dict(progress=True)
   ```

### **Option 2: Downgrade `torchvision`**
If modifying the code is not feasible, downgrade `torchvision` to a compatible version:
```bash
pip install torchvision==0.13.1
```
After downgrading, restart your script and check if it works.

---

## **Usage**
### **Run Main Script**
```bash
python main.py
```
By default, it runs **text detection**. You can modify `main.py` to run **line detection**.

### **Modify Processing Type**
Inside `main.py`, change:
```python
process_type = 'text'  # Change to 'line' for line detection
```

#### **Run Line Detection**
```bash
python main.py
```
🔹 Saves detected **lines** in `detected_lines.csv`.

#### **Run Text Detection**
```bash
python main.py
```
🔹 Saves extracted **text** in `detected_text.csv`.

---

## **Output Files**
| File Name | Description |
|-----------|-------------|
| `detected_text.csv` | Extracted text from the P&ID diagram |
| `detected_lines.csv` | Coordinates of detected lines in the P&ID |

