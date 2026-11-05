# Experiment Log

| Model | Epochs | Batch Size | Workers | AMP | Training Time | Validation Accuracy | Notes |
| ------- | ------ | ---------- | ------- | ---- | ------------- | ------------------- | ------- |
| yolo11s.yaml | 1 | 4 | 4 | True | 0.037 h | mAP50 0.357, mAP50-95 0.184 | First run. GPU_mem 1.24G, GPU 39%, CPU 41% (GPU and CPU from a rerun, same scores, 0.038 h) |
| yolov8n.yaml | 1 | 4 | 4 | True | 0.031 h | mAP50 0.308, mAP50-95 0.156 | GPU_mem 0.607G, GPU 28%, CPU 44% (GPU and CPU from a rerun, same scores and time) |
| yolov8s.yaml | 1 | 4 | 4 | True | 0.031 h | mAP50 0.359, mAP50-95 0.186 | GPU_mem 1.13G, GPU 47%, CPU 40% |
| yolo11n.yaml | 1 | 4 | 4 | True | 0.037 h | mAP50 0.314, mAP50-95 0.161 | GPU_mem 0.689G, GPU 28%, CPU 42% (time, GPU and CPU from a rerun, same scores, first run 0.039 h) |
| yolo11s.yaml | 2 | 4 | 4 | True | 0.065 h | mAP50 0.438, mAP50-95 0.248 | Epochs test. GPU_mem 1.21G (read mid-run), GPU 37%, CPU 37% |
| yolo11s.yaml | 1 | 8 | 4 | True | 0.032 h | mAP50 0.327, mAP50-95 0.162 | Batch size test. GPU_mem 2.15G, GPU 48%, CPU 45% |
| yolo11s.yaml | 1 | 4 | 10 | True | 0.042 h | mAP50 0.341, mAP50-95 0.173 | Workers test. GPU_mem 1.21G, GPU 40%, CPU 39% |
| yolo11s.yaml | 1 | 4 | 4 | False | 0.034 h | mAP50 0.348, mAP50-95 0.179 | AMP test. GPU_mem 2.23G, GPU 60%, CPU 38% |