import cv2
import argparse
import os
from ultralytics import YOLO

def run_inference(source, model_path='deployment/best.onnx', output_dir='demo/'):
    print(f"Loading ONNX model from {model_path}...")
    
    model = YOLO(model_path, task='detect')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Handle Webcam vs File input
    if source == '0':
        source = 0
        is_video = True
        out_path = os.path.join(output_dir, 'sample_output_webcam.mp4')
    else:
        # Determine if the source is a video or image based on extension
        is_video = source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        filename = os.path.basename(source)
        name, ext = os.path.splitext(filename)
        # Force the output names requested by the task
        out_name = "sample_output.mp4" if is_video else "sample_output.jpg"
        out_path = os.path.join(output_dir, out_name)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Setup VideoWriter if processing a video file or webcam
    if is_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0 or fps > 60: 
            fps = 30 # Safe fallback for webcams or weird metadata
        
        # mp4v codec is standard for mp4 exports in OpenCV
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print("Starting inference... Press 'q' on the video window to stop early.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run Inference
        results = model(frame, verbose=False)
        
        # Plot the bounding boxes onto the frame
        annotated_frame = results[0].plot()
        
        # Display the frame on your screen
        cv2.imshow("Construction Safety Detection - Pavan S", annotated_frame)
        
        # Save output to disk
        if is_video:
            out.write(annotated_frame)
        else:
            cv2.imwrite(out_path, annotated_frame)
            print(f"Saved annotated image to {out_path}")
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    if is_video:
        out.release()
        print(f"Saved annotated video to {out_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construction Safety Inference Demo")
    parser.add_argument('--source', type=str, required=True, help="Path to image/video or '0' for webcam")
    args = parser.parse_args()
    
    # Ensure the model path is absolute to avoid relative path issues
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    onnx_path = os.path.join(project_root, 'deployment', 'best.onnx')
    
    run_inference(args.source, model_path=onnx_path)