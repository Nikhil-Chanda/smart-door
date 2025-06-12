import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import os
import pickle
import RPi.GPIO as GPIO
import threading

# === VERIFICATION ===
while not os.getenv('DISPLAY'):
    time.sleep(1)

# === CONFIGURATION ===
frh = os.path.dirname(__file__)
THRESHOLD = 0.4  # Face match threshold
AUTHORIZED_NAMES = os.listdir(os.path.join(frh, "dataset"))
cv_scaler = 4

# === LOAD ENCODINGS ===
print("[INFO] Loading encodings...")
with open(os.path.join(frh, "encodings.pickle"), "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# === INITIALIZE CAMERA ===
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# === GPIO SETUP ===
motor_channel = (8, 10, 38)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(motor_channel, GPIO.OUT)
GPIO.setup(37, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
pwm = GPIO.PWM(38, 1000)
pwm.start(80)   # Work at 80%

# === TRACKING VARIABLES ===
lock_state = 0  # 0 - closed, 1 - open
door_thread_running = False  # Prevent thread overlap

# === FUNCTIONS ===
def open_door():
    global lock_state
    while not lock_state:
        GPIO.output(16, GPIO.HIGH)
        time.sleep(0.4)     # Duration the motor spins
        GPIO.output(16, GPIO.LOW)
        lock_state = 1
    time.sleep(5)

def close_door():
    global lock_state
    while lock_state:
        inp = GPIO.input(37)
        if inp:
            time.sleep(3)
            GPIO.output(12, GPIO.HIGH)
            time.sleep(0.3)     # Duration the motor spins
            GPIO.output(12, GPIO.LOW)
            lock_state = 0
            break
        time.sleep(1)

def handle_door():
    global door_thread_running
    if door_thread_running:
        return  # Already processing
    door_thread_running = True
    try:
        print("[INFO] Starting door thread.")
        open_door()
        close_door()
        print("[INFO] Door thread finished.")
    finally:
        door_thread_running = False

def process_frame(frame):
    global lock_state

    resized_frame = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_resized_frame)
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations, model='large')

    authorized_detected = False  # Track if an authorized face is detected

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        min_distance = 1.0
        best_match_index = -1

        if known_face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            min_distance = np.min(face_distances)
            best_match_index = np.argmin(face_distances)

        name = "Unknown"
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        if min_distance < THRESHOLD and best_match_index != -1:
            name = known_face_names[best_match_index]
            if name in AUTHORIZED_NAMES:
                authorized_detected = True
                # ✅ Authorized person (green)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 210, 0), 2)
                cv2.putText(frame, f"Welcome to VGNet Lab, {name}", (left, top - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 210, 0), 2)

                if not lock_state and not door_thread_running:
                    threading.Thread(target=handle_door).start()
            else:
                # ❌ Unauthorized person (red)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unauthorized person detected", (left, top - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            # ❌ Unknown person (also red)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 210), 2)
            cv2.putText(frame, "Unauthorized person detected", (left, top - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 210), 2)

    # === SHOW MESSAGE IN BOTTOM-RIGHT CORNER ===
    if authorized_detected and not lock_state:
        msg = "Please Close The Door Properly"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.2
        thickness = 4
        color = (255, 0, 0)

        text_size, _ = cv2.getTextSize(msg, font, font_scale, thickness)
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = frame.shape[0] - 20

        cv2.putText(frame, msg, (text_x, text_y), font, font_scale, color, thickness)

    return frame

# === MAIN LOOP ===
try:
    while True:
        frame = picam2.capture_array()
        processed_frame = process_frame(frame)
        cv2.imshow("VGNet Lab Entry", processed_frame)
        cv2.setWindowProperty("VGNet Lab Entry", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        if cv2.waitKey(1) == ord("q"):
            break
except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user.")
finally:
    print("[INFO] Cleaning up...")
    cv2.destroyAllWindows()
    picam2.stop()
    GPIO.cleanup()
