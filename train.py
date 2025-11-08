from ultralytics import YOLO
import os

if __name__ == '__main__':
    # Check if a checkpoint exists to resume training
    checkpoint_path = 'runs/detect/yolo12m_surgical_tool_detection/weights/last.pt'
    
    if os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
        # When resuming, just call train() with resume=True
        results = model.train(resume=True)
    else:
        print("Starting new training from base model")
        # Load a pre-trained model
        model = YOLO('base_models/yolo12m.pt') ### l size requires 1.5 hours per epoch, however at m size it takes only 17 minutes

        # Train the model
        results = model.train(
            data='dataset/data.yaml',
            device='0',  # GPU ID (leave blank for CPU)
            epochs=300,
            imgsz=640,
            batch=16,
            name='yolo12m_surgical_tool_detection',
            patience=50,     # Stop training if mAP doesn't improve for 50 epochs
            save_period=10,   # Save a checkpoint every 10 epochs
            cache='ram'# or 'disk' (optional, caches images for faster loading if memory permits)
        )

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Export the model to ONNX format
    success = model.export(format='onnx')
