import cv2
import face_recognition
import numpy as np
import threading
import time
import pickle

print("[INFO] Loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = [name.lower() for name in data["names"]]

# Settings - tuned for Raspberry Pi performance
tolerance_threshold = 0.5
process_every_n_frames = 5  # Increase skipping to reduce load
cv_scaler = 2.0  # Larger scale factor = smaller images, faster but less detail

# Globals for threading
current_frame = None
frame_count = 0
lock = threading.Lock()

face_locations = []
face_names = []

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # Slightly lighter CLAHE
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def detect_and_recognize(frame):
    global face_locations, face_names
    enhanced = apply_clahe(frame)
    small_frame = cv2.resize(enhanced, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Use 'hog' model instead of 'cnn' for speed (CNN is slow on Pi)
    detected_locations = face_recognition.face_locations(rgb_small, model='hog')
    face_encodings = face_recognition.face_encodings(rgb_small, detected_locations)

    detected_names = []
    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < tolerance_threshold:
                detected_names.append(known_face_names[best_match_index])
            else:
                detected_names.append("Unknown")
        else:
            detected_names.append("Unknown")

    with lock:
        face_locations[:] = detected_locations
        face_names[:] = detected_names

def face_thread_func():
    global frame_count
    while True:
        time.sleep(0.02)  # Slightly longer sleep reduces CPU usage
        with lock:
            if current_frame is None:
                continue
            if frame_count % process_every_n_frames != 0:
                continue
            frame_to_process = current_frame.copy()
        detect_and_recognize(frame_to_process)

def draw_results(frame):
    with lock:
        locations = face_locations.copy()
        names = face_names.copy()

    for (top, right, bottom, left), name in zip(locations, names):
        # Scale back up to original frame size
        top = int(top * cv_scaler)
        right = int(right * cv_scaler)
        bottom = int(bottom * cv_scaler)
        left = int(left * cv_scaler)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    return frame

def main():
    global current_frame, frame_count

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam")
        return

    threading.Thread(target=face_thread_func, daemon=True).start()

    fps = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame")
            break

        with lock:
            current_frame = frame.copy()
            frame_count += 1

        display_frame = draw_results(frame)

        fps += 1
        elapsed = time.time() - start_time
        if elapsed >= 1:
            cv2.putText(display_frame, f"FPS: {fps}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            fps = 0
            start_time = time.time()

        cv2.imshow("Face Recognition (Multi-face)", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

