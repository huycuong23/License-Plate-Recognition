import os
from PIL import Image
import cv2
import torch
import time
import function.utils_rotate as utils_rotate
import function.helper as helper
from datetime import datetime
import logging

os.makedirs("database", exist_ok=True)

# Setup logging
logging.basicConfig(filename="system_logs.log", 
                    filemode="a", 
                    format="%(asctime)s - %(levelname)s - %(message)s", 
                    level=logging.INFO)

# Load YOLO models
yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector_nano_61.pt', force_reload=True, source='local')
yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr_nano_62.pt', force_reload=True, source='local')
yolo_license_plate.conf = 0.60

# Set up FPS calculation
prev_frame_time = 0

# Initialize video capture
vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
vid.set(cv2.CAP_PROP_FPS, 60)

if not vid.isOpened():
    logging.error("Could not open video source.")
    exit()

# Function to calculate fees based on time spent
def calculate_money(time_in, time_out):
    rate_per_hour = 10000
    time_in_dt = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")
    time_out_dt = datetime.strptime(time_out, "%Y-%m-%d %H:%M:%S")
    time_diff = time_out_dt - time_in_dt
    hours = time_diff.total_seconds() / 3600
    money = hours * rate_per_hour
    return round(money, 2)

# Read data from `database/data-in.txt`
def read_data_in():
    data_in = {}
    try:
        with open("database/data-in.txt", "r") as file:
            for line in file:
                lp, time_in = line.strip().split(" - ")
                data_in[lp] = time_in
    except FileNotFoundError:
        pass
    return data_in

# Write data to `database/data-in.txt`
def write_data_in(lp, time_in):
    with open("database/data-in.txt", "a") as file:
        file.write(f"{lp} - {time_in}\n")

# Delete data to `database/data-in.txt`
def remove_lp_from_data_in(lp):
    try:
        with open("database/data-in.txt", "r") as file:
            lines = file.readlines()
        with open("database/data-in.txt", "w") as file:
            for line in lines:
                if not line.startswith(lp):
                    file.write(line)
    except FileNotFoundError:
        pass

# Write to `database/data-out.txt`
def write_data_out(lp, time_in, time_out, money):
    with open("database/data-out.txt", "a") as file:
        file.write(f"{lp} - {time_in} - {time_out} - {money}VND\n")
    remove_lp_from_data_in(lp)

# Store the current time
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data_in = read_data_in()

while True:
    ret, frame = vid.read()
    if not ret or frame is None:
        logging.error("Could not read frame.")
        break

    # Detect license plates and log the YOLO output
    plates = yolo_LP_detect(frame, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()
    list_read_plates = set()
    total_plates_detected = len(list_plates)  # Total plates detected
    total_plates_read = 0  # Counter for plates successfully read

    # Log detection info for each license plate detected
    for plate in list_plates:
        x, y = int(plate[0]), int(plate[1])           # xmin, ymin
        w, h = int(plate[2] - plate[0]), int(plate[3] - plate[1])  # width, height
        confidence = plate[4]
        logging.info(f"License Plate Detected - Location: ({x}, {y}), Size: ({w}, {h}), Confidence: {confidence}")
        
        crop_img = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color=(0, 0, 225), thickness=2)

        # Try reading the plate text and log the recognized text
        lp = ""
        for cc in range(2):
            for ct in range(2):
                lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                if lp != "unknown":
                    list_read_plates.add(lp)
                    total_plates_read += 1  # Increment counter for successful reads
                    logging.info(f"Recognized License Plate Text: {lp}")
                    cv2.putText(frame, lp, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                    break
            if lp != "unknown":
                break

    # Calculate and display FPS
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 2)
    logging.info(f"FPS: {int(fps)}")
    
    # Log the totals for plates detected and read
    logging.info(f"Total Plates Detected: {total_plates_detected}, Total Plates Read: {total_plates_read}")

    # Handle license plate processing when Enter key is pressed
    if cv2.waitKey(1) & 0xFF == 13:  # Enter key
        data_in = read_data_in()
        for lp in list_read_plates:
            if lp not in data_in:
                time_in = get_current_time()
                write_data_in(lp, time_in)
                logging.info(f"License Plate {lp} entered at {time_in}")
            else:
                time_in = data_in[lp]
                time_out = get_current_time()
                money = calculate_money(time_in, time_out)
                write_data_out(lp, time_in, time_out, money)
                logging.info(f"License Plate {lp} exited at {time_out}, Fees: {money}đ")
                data_in.pop(lp)

    # Display the frame
    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()