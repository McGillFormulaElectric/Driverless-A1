from ultralytics import YOLO
import torch
import os

SELECTED_MODEL = "yolo11s.yaml"  # Change this to train a different variant

DATA_CONFIG = "config.yaml"

EPOCHS = 1
BATCH_SIZE = 4
WORKERS = 4
USE_AMP = True
OUTPUT_DIR = "runs/cone_detection/"
DEVICE_SELECTION = None


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        # Apple MPS (Mac GPU)
        print("[INFO] Using Apple MPS for training.")
        return "mps"
    elif torch.cuda.is_available():
        # CUDA GPU
        device_count = torch.cuda.device_count()
        devices = list(range(device_count))
        device_names = [torch.cuda.get_device_name(i) for i in devices]
        print(f"[INFO] CUDA available. Using GPUs: {device_names}")
        return "0"  # returns list of GPU indices
    else:
        # CPU fallback
        print("[INFO] No GPU detected. Using CPU.")
        return "cpu"

def initialize_model(model_path):
    print(f"[INFO] Loading YOLOv8 model: {model_path}")
    model = YOLO(model_path)
    return model

def train_model(model, device):
    print("[INFO] Starting training...")
    results = model.train(
        data=DATA_CONFIG,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        workers=WORKERS,
        device=device,
        amp=USE_AMP,
        project=OUTPUT_DIR,
        name=SELECTED_MODEL
    )
    
    # Save best weights
    save_path = os.path.join(OUTPUT_DIR, SELECTED_MODEL, "weights", "best.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"[INFO] Training complete! Model saved at: {save_path}")
    
    # Validation metrics
    metrics = model.val()
    print("[INFO] Validation metrics:")
    print(metrics)
    
    return save_path

if __name__ == "__main__":
    device = get_device()                       # Step 1: Determine device(s)
    model = initialize_model(SELECTED_MODEL)   # Step 2: Load YOLOv8 model
    trained_model_path = train_model(model, device)  # Step 3: Train model
