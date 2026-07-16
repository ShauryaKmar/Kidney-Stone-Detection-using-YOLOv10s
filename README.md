# Kidney Stone Detection using YOLOv10s

A deep learning-based computer vision project that detects kidney stones in CT scan images using **YOLOv10s**. 
The system identifies the presence of kidney stones, localizes them with bounding boxes, estimates their size, and provides an interactive web interface for visualization.

---

## Features

- Detects kidney stones from CT scan images
- Localizes stones using bounding boxes
- Displays confidence score for each detection
- Estimates stone size from the detected bounding box
- User-friendly web interface for uploading and analyzing CT images
- Fast inference using YOLOv10s

---

## Tech Stack

- **Python**
- **YOLOv10s**
- **PyTorch**
- **OpenCV**
- **NumPy**
- **Ultralytics**
- **Streamlit**

---


## Working

1. Upload a CT scan image through the web interface.
2. The image is preprocessed and resized for inference.
3. YOLOv10s extracts hierarchical image features using its convolutional backbone.
4. The detection head predicts kidney stone locations, confidence scores, and class labels.
5. Non-Maximum Suppression (NMS) removes overlapping duplicate detections.
6. The bounding box dimensions are calculated in pixels.
7. If DICOM **Pixel Spacing** metadata is available, the pixel measurements are converted to millimeters for accurate stone size estimation.
8. The application displays:
   - Detected stone location
   - Bounding box
   - Confidence score
   - Estimated stone size (mm or pixels)

---

## Model

The project uses **YOLOv10s**, a real-time object detection model capable of accurately detecting small objects such as kidney stones.

The model predicts:

- Bounding Box Coordinates
- Confidence Score
- Object Class (Kidney Stone)

---

## Stone Size Estimation

The detected bounding box provides the width and height of the stone in pixels.

```
Width (px)  = x₂ − x₁

Height (px) = y₂ − y₁
```

If CT scan pixel spacing is available (DICOM metadata), the size can be converted into millimeters.

```
Width (mm)  = Width (px) × PixelSpacingX

Height (mm) = Height (px) × PixelSpacingY
```

The reported stone size corresponds to the largest diameter, which is the standard clinical measurement used for kidney stones.
```
Stone Size (mm) = max(Width (mm), Height (mm))
```

Otherwise, the size is reported in pixels.

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Kidney-Stone-Detection.git

cd Kidney-Stone-Detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Training

```bash
yolo detect train \
model=yolov10m.pt \
data=data.yaml \
epochs=100 \
imgsz=640
```

---

## Inference

```bash
python app.py
```  

or

```bash
python app.py
```

if using the web interface.

---
