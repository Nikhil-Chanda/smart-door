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

# Settings
tolerance_threshold = 0.5  # Adjust for your dataset & distance
process_every_n_frames = 3  # Process every nth frame to increase FPS
cv_scaler = 1.5  # Resize factor for faster processing but keeping detail

# Globals for threading
current_frame = None
frame_count = 0
lock = threading.Lock()

face_locations = []
face_names = []

def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def detect_and_recognize(frame):
    global face_locations, face_names

    enhanced = apply_clahe(frame)
    small_frame = cv2.resize(enhanced, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect multiple faces using CNN model
    face_locations = face_recognition.face_locations(rgb_small, model='cnn')
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    face_names = []
    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_face_encodings, encoding)
        if len(distances) > 0:
            best_match_index = np.argmin(distances)
            best_distance = distances[best_match_index]
            if best_distance < tolerance_threshold:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"
        else:
            name = "Unknown"
        face_names.append(name)

def face_thread_func():
    global frame_count
    while True:
        time.sleep(0.01)
        with lock:
            if current_frame is None:
                continue
            if frame_count % process_every_n_frames == 0:
                frame_to_process = current_frame.copy()
            else:
                continue
        detect_and_recognize(frame_to_process)

def draw_results(frame):
    # Draw rectangle & label for each detected face
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top = int(top * cv_scaler)
        right = int(right * cv_scaler)
        bottom = int(bottom * cv_scaler)
        left = int(left * cv_scaler)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 2)

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
