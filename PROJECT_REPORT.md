# Project Report: Construction Safety Object Detection

**Author:** Pavan S 

## Executive Summary
This project delivers a real-time computer vision pipeline for monitoring construction site safety. Utilizing the Roboflow 100 (RF100) benchmark dataset, multiple YOLOv8 architectures were trained to detect workers and their adherence to standard Personal Protective Equipment (PPE) protocols, specifically helmets and safety vests. The final model was exported to ONNX format for optimized, framework-agnostic deployment on local CPU hardware, achieving near real-time inference speeds.

## 1. Dataset & Methodology
* **Source:** RF100 Construction Safety Dataset (`construction-safety-gsnvb`).
* **Composition:** 2,424 in-the-wild images detailing 5 semantic classes (`person`, `helmet`, `no-helmet`, `vest`, `no-vest`).
* **Preprocessing:** Standardized to 640x640 resolution with auto-batching to maximize hardware utilization during training. 

## 2. Model Training & Selection
Two models, **YOLOv8n** (nano) and **YOLOv8s** (small), were trained for 100 epochs on a Tesla T4 GPU instance. 

While YOLOv8s achieved a marginally higher strict accuracy (mAP50-95: 0.491 vs 0.487), the **YOLOv8n** model was selected as the final deployment candidate. In a safety compliance context, minimizing False Negatives is critical to ensure worker safety. YOLOv8n demonstrated a significantly higher Recall (0.837 vs 0.741), reducing the likelihood of missed PPE violations. Furthermore, the nano architecture is roughly 4x smaller (6.3 MB) and executes nearly twice as fast, making it the superior choice for local hardware deployment.

*(For a detailed breakdown of failure cases, PR curves, and prediction samples, please reference `evaluation/evaluation_report.md`)*

## 3. ONNX Deployment & Benchmarking
To break the dependency on the PyTorch ecosystem and ensure cross-platform edge compatibility, the YOLOv8n model was exported to the `.onnx` format using `onnxslim` for graph optimization. 

**Local Deployment Benchmarks:**
* **Execution Environment:** AMD Ryzen 7 7435HS (CPUExecutionProvider)
* **ONNX Model Size:** 11.70 MB
* **Average Inference Time:** 34.23 ms per frame
* **Estimated Speed:** 29.21 FPS

## Conclusion
This pipeline successfully demonstrates the viability of utilizing lightweight object detection models for critical safety monitoring. By prioritizing high recall and ONNX optimization, the system achieves a strong balance of accuracy and real-time execution speed suitable for direct deployment on standard monitoring hardware.