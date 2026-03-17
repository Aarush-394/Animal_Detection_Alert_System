from ultralytics import YOLO
import cv2
import time
import os
import platform
import webbrowser
import urllib.parse

# ---------------- LOCATION ----------------
import geocoder

def get_real_location():
    g = geocoder.ip("me")
    if g.ok:
        return g.latlng[0], g.latlng[1], g.city
    else:
        return None, None, "Unknown"
# ------------------------------------------

# ---------------- EMAIL ----------------
EMAIL_TO = "arg.39452161@gmail.com"
# ---------------------------------------

# ---------------- PATHS ----------------
VIDEO_PATH = "test.mp4"
OUTPUT_VIDEO = "intrusion_output.mp4"
OUTPUT_IMAGE = "detected_frame.jpg"
LOG_FILE = "alert_log.txt"

LATITUDE, LONGITUDE, LOCATION_NAME = get_real_location()
# ---------------------------------------

# 🔔 Beep function (works only on Windows local PC)
def beep_alert():
    if platform.system() == "Windows":
        import winsound
        for i in range(3):
            winsound.Beep(1200, 1000)
    else:
        print("🔔 Beep triggered (audio not supported here)")

# Load model
model = YOLO("best.pt")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(f"❌ Cannot open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

print("✅ Video loaded, intrusion monitoring started")

alert_sent = False  # ensures alert only once

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.15)

    for r in results:
        if len(r.boxes) > 0:
            # Draw boxes for accuracy visualization
            frame = r.plot()

            # Trigger alert only once
            if not alert_sent:
                alert_sent = True
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                beep_alert()
                print("🚨 ANIMAL DETECTED!")
                print(f"📍 {LOCATION_NAME} | Lat: {LATITUDE}, Lon: {LONGITUDE}")

                # Log alert
                with open(LOG_FILE, "a") as f:
                    f.write(
                        f"Animal detected at {timestamp} | "
                        f"{LOCATION_NAME} | "
                        f"Lat: {LATITUDE}, Lon: {LONGITUDE}\n"
                    )

                # Prepare email (manual send)
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
                exit()
    out.write(frame)
    # break
cap.release()
out.release()

print("✅ Video processing completed")
print("🎥 Output saved at:", OUTPUT_VIDEO)