from ultralytics import YOLO
import cv2
import time
import os
import platform
import webbrowser
import urllib.parse
import serial
import time

arduino = serial.Serial('COM5', 9600)   # change COM port
time.sleep(2)

import json

def update_dashboard(status, location, timestamp, message, image=""):
    data = {
        "status": status,
        "location": location,
        "time": timestamp,
        "message": message,
        "image": image
    }

    with open("status.json", "w") as f:
        json.dump(data, f)

update_dashboard(
    "No Animal Detected",
    "",
    "",
    "Monitoring...",
    ""
)

import geocoder

def get_real_location():
    g = geocoder.ip("me")
    if g.ok:
        return g.latlng[0], g.latlng[1], g.city
    else:
        return None, None, "Unknown"
    
EMAIL_TO = "arg.39452161@gmail.com"

VIDEO_PATH = "test.mp4"
OUTPUT_VIDEO = "intrusion_output.mp4"
OUTPUT_IMAGE = "detected_frame.jpg"
LOG_FILE = "alerts/alert_log.txt"

LATITUDE, LONGITUDE, LOCATION_NAME = get_real_location()
def beep_alert():
    arduino.write(b'1')

model = YOLO("best.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print("Video loaded, intrusion monitoring started")

animal_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.15)

    for r in results:
        if len(r.boxes) > 0 and not animal_detected:
            animal_detected = True
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            update_dashboard(
                "Animal Detected",
                f"{LOCATION_NAME} ({LATITUDE}, {LONGITUDE})",
                timestamp,
                "Animal detected! Alert triggered.",
                OUTPUT_IMAGE
            )
            beep_alert()

            print("🚨 ANIMAL DETECTED!")
            print(f"📍 {LOCATION_NAME} | Lat: {LATITUDE}, Lon: {LONGITUDE}")

            frame = r.plot()

            cv2.imwrite(OUTPUT_IMAGE, frame)

            with open(LOG_FILE, "a") as f:
                f.write(
                    f"Animal detected at {timestamp} | "
                    f"{LOCATION_NAME} | "
                    f"Lat: {LATITUDE}, Lon: {LONGITUDE}\n"
                )

            subject = "Animal Intrusion Alert"
            body = f"""
ALERT: Animal Intrusion Detected

Location Name: {LOCATION_NAME}
Latitude: {LATITUDE}
Longitude: {LONGITUDE}

Google Maps:
https://www.google.com/maps?q={LATITUDE},{LONGITUDE}

Time: {timestamp}
"""

            mailto_link = (
                f"mailto:{EMAIL_TO}"
                f"?subject={urllib.parse.quote(subject)}"
                f"&body={urllib.parse.quote(body)}"
            )

            webbrowser.open(mailto_link)

            out.write(frame)
            cap.release()
            out.release()

            print("🎥 Video saved at:", OUTPUT_VIDEO)
            print("🖼 Image saved at:", OUTPUT_IMAGE)
            print("📧 Email sent succesfully")
            exit()

        frame = r.plot()

    out.write(frame)

cap.release()
out.release()

print("No animal detected in video")
update_dashboard(
    "No Animal Detected",
    "",
    "",
    "Monitoring..."
)