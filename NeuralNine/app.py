from flask import Flask, Response, render_template
import cv2
import torch
import torchvision
from torchvision import transforms as T

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# COCO instance category names (list truncated for brevity)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Transformation for the input frame
transform = T.Compose([T.ToTensor()])
score_threshold = 0.5  # Adjust as needed

# Initialize camera (default is 0 for the primary webcam)
camera = cv2.VideoCapture(1)


def generate_frames():
    """Generate video frames with object detection."""
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame from the camera.")
            break

        # Convert frame to RGB (as the model expects RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_tensor = transform(frame_rgb)

        # Perform object detection
        with torch.no_grad():
            predictions = model([image_tensor])

        # Extract boxes, scores, and labels
        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        labels = predictions[0]['labels']

        for i, score in enumerate(scores):
            if score >= score_threshold:  # Filter by confidence score
                x1, y1, x2, y2 = boxes[i].int().tolist()
                label_id = labels[i].item()

                # Ensure label_id is valid
                if label_id < 0 or label_id >= len(COCO_INSTANCE_CATEGORY_NAMES):
                    print(f"Invalid label ID: {label_id}. Skipping detection.")
                    continue

                label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id]
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"{label_name} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )

        # Encode frame as JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    """Route to serve video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """Route to render the index page."""
    return render_template('index.html')


if __name__ == '__main__':
    try:
        # Start the Flask app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        # Release the camera resource
        camera.release()
        print("Camera released.")
