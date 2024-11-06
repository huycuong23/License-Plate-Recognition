from PIL import Image
import cv2
import torch
import time
import function.utils_rotate as utils_rotate
import function.helper as helper

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Set up FPS calculation
prev_frame_time = 0

# Initialize video capture
vid = cv2.VideoCapture(0)  # Use 0 or another index if 1 doesn't work
if not vid.isOpened():
    print("Error: Could not open video source.")
    exit()

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        print("Error: Could not read frame.")
        break

    # Detect license plates
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    
    for plate in list_plates:
        x, y = int(plate[0]), int(plate[1])           # xmin, ymin
        w, h = int(plate[2] - plate[0]), int(plate[3] - plate[1])  # width, height
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 225), thickness=2)

        # Try reading the plate text
        lp = ""
        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    cv2.putText(frame, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    break
            if lp != "unknown":
                break

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)

    # Display the frame
    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()
