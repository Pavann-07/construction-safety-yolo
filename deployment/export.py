from ultralytics import YOLO

def export_model():
    print("--- Loading YOLOv8n PyTorch Model ---")
    model = YOLO('D:/Intern/Task_1/deployment/best.pt')
    print("--- Exporting to ONNX ---")
    export_path = model.export(format='onnx', dynamic=False)
    
    print(f"ONNX export successful! Saved to: {export_path}")

if __name__ == "__main__":
    export_model()