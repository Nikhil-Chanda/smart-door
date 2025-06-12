import face_recognition
import cv2
import numpy as np
from picamera2 import Picamera2
import time
import pickle
from gpiozero import LED

# Load face encodings
print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = [name.lower() for name in data["names"]]  # Make case-insensitive

# Camera setup
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

# GPIO setup
output = LED(14)

# Settings
cv_scaler = 4  # Downscale factor
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

# Authorized names (case-insensitive)
authorized_names = ["john", "alice", "bob", "anil shaji"]  # <- add new names in lowercase
tolerance_threshold = 0.3  # Adjust as needed for match sensitivity


def process_frame(frame):
    global face_locations, face_encodings, face_names

    resized = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations, model='large')

    face_names = []
    authorized_detected = False

    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        best_distance = distances[best_match_index]

        if best_distance < tolerance_threshold:
            name = known_face_names[best_match_index]
            print(f"[MATCH] {name} (distance: {best_distance:.4f})")
            if name in authorized_names:
                authorized_detected = True
        else:
            name = "Unknown"
            print(f"[REJECTED] Face not recognized (distance: {best_distance:.4f})")

        face_names.append(name)

    output.on() if authorized_detected else output.off()
    return frame


def draw_results(frame):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler

        cv2.rectangle(frame, (left, top), (right, bottom), (244, 42, 3), 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), (244, 42, 3), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        label = name if name != "Unknown" else "Unrecognized"
        cv2.putText(frame, label, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)

        if name in authorized_names:
            cv2.putText(frame, "Authorized", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)

    return frame


def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    return fps


# Main loop
while True:
    frame = picam2.capture_array()
    processed = process_frame(frame)
    display = draw_results(processed)
    current_fps = calculate_fps()

    cv2.putText(display, f"FPS: {current_fps:.1f}", (display.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Video', display)
    if cv2.waitKey(1) == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()
output.off()
