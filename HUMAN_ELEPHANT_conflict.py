from ultralytics import YOLO
import cv2
import time
from pushbullet import Pushbullet
import os
from datetime import datetime
import winsound

# Load YOLO model
model = YOLO('yolov9c.pt')

# Elephant class ID
ELEPHANT_CLASS_ID = 20

# Setup camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Add as many Pushbullet API keys as you want
pushbullet_tokens = [
    "o.QOqMmNCqV3tZiHWfT43AIi4YHsTIelWk",
    "o.ryw7V7lcQKje7KgVR5RyXTUfCaw7XGyM",
]

pb_clients = []
for token in pushbullet_tokens:
    try:
        pb_clients.append(Pushbullet(token))
        print(f"Pushbullet connected: {token[:10]}...")
    except:
        print(f"Failed to connect: {token[:10]}...")

pushbullet_connected = len(pb_clients) > 0

# Cooldown to avoid too many alerts
last_alert_time = 0
alert_cooldown = 10  # seconds


def send_alert():
    global last_alert_time
    now = time.time()

    if now - last_alert_time < alert_cooldown:
        print("Alert cooldown. Not sending again.")
        return

    if not pushbullet_connected:
        print("No Pushbullet available.")
        return

    ALERT_MESSAGE = "⚠️ Elephant detected near your monitoring area! Stay alert."

    for pb in pb_clients:
        try:
            pb.push_note("Elephant Alert!", ALERT_MESSAGE)
            print("Pushbullet alert sent.")
        except Exception as e:
            print("Pushbullet Error:", e)

    last_alert_time = now


def save_screenshot(frame):
    """Store elephant image with timestamp"""
    alert_folder = "alerts"
    os.makedirs(alert_folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"elephant_{timestamp}.jpg"
    filepath = os.path.join(alert_folder, filename)

    cv2.imwrite(filepath, frame)
    print(f"Saved screenshot: {filepath}")


elephant_present = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error.")
        break

    # Run detection for elephant only
    results = model(frame, classes=[ELEPHANT_CLASS_ID])
    elephant_found = False

    for box in results[0].boxes:
        if int(box.cls) == ELEPHANT_CLASS_ID:
            elephant_found = True

    # If elephant detected
    if elephant_found:
        print("🐘 Elephant Detected!")

        # Save screenshot once per detection cycle
        if not elephant_present:
            save_screenshot(frame)
            send_alert()

        elephant_present = True

        # Continuous beep as long as elephant stays
        frequency = 2000  # Hz
        duration = 300  # ms per beep
        winsound.Beep(frequency, duration)

    else:
        elephant_present = False

    # Show annotated frame
    annotated = results[0].plot()
    cv2.imshow("Elephant Detection System", annotated)

    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
