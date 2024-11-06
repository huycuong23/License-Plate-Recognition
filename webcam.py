from PIL import Image
import cv2
import torch
import time
import function.utils_rotate as utils_rotate
import function.helper as helper
from datetime import datetime

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

# Helper function to calculate money based on time
def calculate_money(time_in, time_out):
    # Assuming a rate per hour, you can modify this based on your pricing logic
    rate_per_hour = 10000
    time_in_dt = datetime.strptime(time_in, "%Y-%m-%d %H:%M:%S")
    time_out_dt = datetime.strptime(time_out, "%Y-%m-%d %H:%M:%S")
    time_diff = time_out_dt - time_in_dt
    hours = time_diff.total_seconds() / 3600  # Convert to hours
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
        pass  # If the file doesn't exist, start with an empty dictionary
    return data_in

# Write data to `database/data-in.txt`
def write_data_in(lp, time_in):
    with open("database/data-in.txt", "a") as file:
        file.write(f"{lp} - {time_in}\n")

# Delete data to `database/data-in.txt`
def remove_lp_from_data_in(lp):
    try:
        # Open the "data-in.txt" file and read all lines
        with open("database/data-in.txt", "r") as file:
            lines = file.readlines()
        
        # Open "data-in.txt" again in write mode to overwrite the content
        with open("database/data-in.txt", "w") as file:
            for line in lines:
                # Write all lines except the one containing the given license plate
                if not line.startswith(lp):
                    file.write(line)
    except FileNotFoundError:
        pass  # If the file doesn't exist, there's nothing to remove

# Write to `database/data-out.txt`
def write_data_out(lp, time_in, time_out, money):
    with open("database/data-out.txt", "a") as file:
        file.write(f"{lp} - {time_in} - {time_out} - ${money}\n")
    remove_lp_from_data_in(lp)

# Store the current time in a string format
def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

data_in = read_data_in()

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

    # Check for Enter key press to handle license plate processing
    if cv2.waitKey(1) & 0xFF == 13:  # Enter key
        data_in = read_data_in()
        for lp in list_read_plates:
            if lp not in data_in:
                # If the license plate is not in database/data-in.txt, write it with the current time
                time_in = get_current_time()
                write_data_in(lp, time_in)
                print(f"License Plate {lp} added to data-in at {time_in}")
            else:
                # If the license plate exists, calculate the time and money
                time_in = data_in[lp]
                time_out = get_current_time()
                money = calculate_money(time_in, time_out)
                write_data_out(lp, time_in, time_out, money)
                print(f"License Plate {lp} processed: {money}đ due.")
                # Remove from data-in (or keep it depending on business logic)
                data_in.pop(lp)

    # Display the frame
    cv2.imshow('License Plate Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
vid.release()
cv2.destroyAllWindows()
