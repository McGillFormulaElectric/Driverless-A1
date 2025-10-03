import cv2
from ultralytics import YOLO
import os

if __name__ == "__main__":
    video_path = 'Chalmers_driverless.mp4'  # input video
    video_path_out = os.path.splitext(video_path)[0] + '_out.mp4'  # output video

    model_path = "runs/cone_detection/yolo11s.yaml6/weights/best.pt" # PATH OF BEST PT FILE
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(video_path_out,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)[0]

        # Draw all bounding boxes, including low-confidence predictions
        for box in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = box
            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Add class label + confidence
            cv2.putText(frame, f"{results.names[int(class_id)].upper()} {score:.2f}",
                        (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Write frame to output video
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved processed video to: {video_path_out}")
