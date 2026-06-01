import time
import os
from ultralytics import YOLO

def benchmark_onnx(model_path, test_image_path):
    print("--- Benchmarking ONNX Model on Local Laptop ---")
    
    # Model Size
    file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"ONNX Model Size: {file_size_mb:.2f} MB")
    
    # Load Model
    try:
        model = YOLO(model_path, task='detect')
        print("ONNX model loaded successfully via ONNXRuntime.")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # Warm-up runs 
    print("Warming up model...")
    for _ in range(3):
        _ = model(test_image_path, verbose=False)
        
    # Measure Inference Time
    print("Running inference benchmark...")
    iterations = 50
    start_time = time.time()
    
    for _ in range(iterations):
        results = model(test_image_path, verbose=False)
        
    end_time = time.time()
    
    avg_inference_time_ms = ((end_time - start_time) / iterations) * 1000
    fps = 1000 / avg_inference_time_ms
    
    print(f"Average Inference Time: {avg_inference_time_ms:.2f} ms")
    print(f"Estimated FPS: {fps:.2f}")
    
    # Verify Output 
    results[0].save(filename="deployment/onnx_sample_prediction.jpg")
    print("\nVisual verification saved to deployment/onnx_sample_prediction.jpg")

if __name__ == "__main__":
    onnx_model = r"deployment\best.onnx"
    test_image = r"dataset\test\images\ppe_0000_jpg.rf.12dba82119baadab649f2fb1fda2afff.jpg" 
    
    if os.path.exists(onnx_model):
        benchmark_onnx(onnx_model, test_image)
    else:
        print(f"Model not found at {onnx_model}. Please run export.py first.")