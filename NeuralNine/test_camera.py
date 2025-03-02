import cv2

camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Error: Unable to access the camera.")
else:
    print("Camera is working!")

camera.release()
