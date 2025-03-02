import cv2
import torch
import torchvision
from torchvision import transforms as T
from torchvision.ops import nms  # Non-Maximum Suppression

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Define image transform
transform = T.Compose([T.ToTensor()])

# COCO class names
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Set a lower score threshold to detect all objects
score_threshold = 0.3

# IoU threshold for NMS
nms_threshold = 0.5

# Open camera feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise Exception("Could not open video device")

print("Press 'q' to quit the video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting ...")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_tensor = transform(frame_rgb)

    with torch.no_grad():
        predictions = model([image_tensor])

    pred = predictions[0]
    boxes = pred['boxes']
    scores = pred['scores']
    labels = pred['labels']

    # Apply Non-Maximum Suppression (NMS)
    keep_indices = nms(boxes, scores, nms_threshold)
    boxes = boxes[keep_indices]
    scores = scores[keep_indices]
    labels = labels[keep_indices]

    for i, score in enumerate(scores):
        if score >= score_threshold:  # Include all objects with score >= 0.3
            x1, y1, x2, y2 = boxes[i].int().tolist()
            label_id = labels[i].item()
            label_name = COCO_INSTANCE_CATEGORY_NAMES[label_id] if label_id < len(
                COCO_INSTANCE_CATEGORY_NAMES) else str(label_id)

            # Draw bounding boxes and labels for all objects
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label_name} {score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detected objects
    cv2.imshow("Detected Objects", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
