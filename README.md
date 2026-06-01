# Construction Safety Object Detection (YOLOv8 + ONNX)

This repository contains a complete computer vision pipeline for training, evaluating, and deploying a real-time Personal Protective Equipment (PPE) detection model. Built using the Roboflow 100 (RF100) Construction Safety dataset, the final YOLOv8n model has been exported to ONNX format for highly optimized, framework-agnostic CPU deployment.

## Project Highlights
* **High Recall Safety Monitoring:** Achieved a recall of 0.837, prioritizing the reduction of false negatives (undetected safety violations).
* **Edge-Ready Deployment:** The exported ONNX model is highly compressed at **11.70 MB**.
* **Real-Time CPU Performance:** Achieves **29.21 FPS** inference speed on a local CPU (AMD Ryzen 7) without requiring a discrete GPU.

## 📁 Repository Structure
* `/dataset`: Dataset configuration (`data.yaml`) and metadata reports.
* `/notebooks`: Original training environments and execution logs.
* `/training`: Scripts for dataset preparation and YOLOv8 training loops.
* `/evaluation`: Precision-Recall curves, confusion matrices, and qualitative analysis of edge cases.
* `/deployment`: ONNX export scripts, validation tools, and hardware benchmarking.
* `/results`: Inference scripts and sample visual outputs (Image/Video).

## Installation

1. Clone the repository:
   ```bash
   git clone [INSERT_YOUR_GITHUB_LINK_HERE]
   cd construction-safety-yolo

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Usage & Inference

   The inference script utilizes `onnxruntime` for maximum efficiency on CPU. Run the following commands from the root directory to test the model:

   **Image Inference:**
   ```bash
   python results/inference.py --source results/sample_image.jpg

**Video Inference:**
   ```bash
   python results/inference.py --source results/sample_image.mp4

**Live Webcam Inference:**
   ```bash
   python results/inference.py --source 0