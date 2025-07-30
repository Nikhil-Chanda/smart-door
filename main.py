import face_recognition
import cv2
import numpy as np
import time
from datetime import datetime
import csv
import logging
import os
import pickle
import RPi.GPIO as GPIO
import threading
# import pyttsx3
from gtts import gTTS
import subprocess
import tempfile
import queue
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
# Supabase
from supabase import create_client, Client

# === SUPABASE CONFIG ===
supabase_url: str = "https://prfkhjuujnheztwhwmcd.supabase.co"
supabase_key: str = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InByZmtoanV1am5oZXp0d2h3bWNkIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE4MDk2NjIsImV4cCI6MjA1NzM4NTY2Mn0.j92nEtB5mUORV5VlCpLsTbJNinSykjnpaX0R1cnZQXc'

supabase: Client = create_client(supabase_url, supabase_key)

# === HELPER: Check and Insert History ===

def log_if_authorized (name: str, request_name: str = "Door Temporary Open via Pi"):
    profile_response = supabase.table("profiles").select("*").eq("name", name).execute()
    #print(supabase.table("profiles").select("*").execute())

    if profile_response.data and len(profile_response.data) > 0:
        user_profile = profile_response.data[0]
        user_id = user_profile["id"]

        # Insert into 'history'
        insert_response = supabase.table("history").insert({
            "user_id": user_id,
            "name": name,
            "request_name": request_name,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        if insert_response.data is not None:
            print(f"[INFO] Entry logged for {name}")
            return True
        else:
            print(f"[ERROR] Failed to insert history for {name}: {insert_response}")
            return False
    else:
        print(f"[WARN] Name '{name}' not found in profiles table.")
        return False


# === WAIT FOR DISPLAY TO BE READY ===
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
    data = pickle.load(f)
known_face_encodings = data["encodings"]
known_face_names = data["names"]


# === INITIALIZE USB CAMERA ===
print("[INFO] Initializing USB webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
if not cap.isOpened():
    raise Exception("[ERROR] Could not open USB webcam.")

# === GPIO SETUP ===
GPIO.cleanup()
motor_channel = (11, 16, 15)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
for pin in motor_channel:
    GPIO.setup(pin, GPIO.OUT)
GPIO.setup(37, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
pwm = GPIO.PWM(15, 1000)
pwm.start(80)
GPIO.output(11, GPIO.LOW)
GPIO.output(16, GPIO.LOW)

# === TEXT-TO-SPEECH SETUP ===
speech_queue = queue.Queue()
close_speaking_event = threading.Event()
running = True

def speech_worker():
    while running:
        try:
            message = speech_queue.get(timeout=1)
            if message:
                tts = gTTS(text=message, lang='en')
                with tempfile.NamedTemporaryFile(delete=True, suffix='.mp3') as fp:
                    tts.save(fp.name)
                    subprocess.run(['mpg123', '-q', fp.name], check=True)
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"[ERROR] TTS failed: {e}")

def speak_welcome(name):
    speech_queue.put(f"Welcome to VGNet Lab {name}")

def speak_close():
    while close_speaking_event.is_set():
        speech_queue.put("Please close the door properly. Thank you for keeping VGNet Lab secure.")
        time.sleep(5)

# === TRACKING VARIABLES ===
lock_state = 0
door_thread_running = False
last_greeted_time = {}
GREETING_COOLDOWN = 10
running = True
lock = threading.Lock()
current_frame = None
frame_count = 0
face_locations = []
face_names = []
seen_during_open = set()

def open_door():
    global lock_state
    if not lock_state:
        print("[INFO] Opening door...")
        GPIO.output(11, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(11, GPIO.LOW)
        lock_state = 1
        time.sleep(5)

def close_door():
    global lock_state
    if not close_speaking_event.is_set():
        close_speaking_event.set()
        threading.Thread(target=speak_close, daemon=True).start()

    while lock_state and running:
        try:
            if GPIO.input(37):
                print("[INFO] Closing door...")
                GPIO.output(16, GPIO.HIGH)
                time.sleep(0.4)
                GPIO.output(16, GPIO.LOW)
                lock_state = 0
                close_speaking_event.clear()
                seen_during_open.clear()
        except RuntimeError as e:
            print(f"[ERROR] GPIO input error: {e}")
            break
        time.sleep(1)

def handle_door():
    global door_thread_running
    if door_thread_running or not running:
        return
    door_thread_running = True
    try:
        print("[INFO] Starting door control thread.")
        open_door()
        close_door()
        print("[INFO] Door control thread finished.")
    except Exception as e:
        print(f"[ERROR] Door control thread error: {e}")
    finally:
        door_thread_running = False  # Crucial to reset to allow future openings!

def log_entry(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(frh, "entry_log.csv")

    try:
        file_exists = os.path.isfile(log_path)
        with open(log_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Name", "Timestamp"])
            writer.writerow([name, timestamp])
        logging.info(f"Logged entry: {name} at {timestamp}")
    except Exception as e:
        logging.error(f"Failed to log entry for {name}: {e}")

def process_frame(frame):
    global lock_state

    resized_frame = cv2.resize(frame, (0, 0), fx=1 / cv_scaler, fy=1 / cv_scaler)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(rgb_resized_frame)
    face_encs = face_recognition.face_encodings(rgb_resized_frame, face_locs, model='large')

    names = []

    for (top, right, bottom, left), face_encoding in zip(face_locs, face_encs):
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

            # Greeting and door control logic
            current_time = time.time()
            last_time = last_greeted_time.get(name, 0)
            if current_time - last_time > GREETING_COOLDOWN:
                threading.Thread(target=speak_welcome, args=(name,), daemon=True).start()
                last_greeted_time[name] = current_time
            if lock_state and name not in seen_during_open:
                seen_during_open.add(name)
                log_entry(name)
                log_if_authorized(name)

            if not lock_state and not door_thread_running:
                threading.Thread(target=handle_door, daemon=True).start()

        names.append(name)

    with lock:
        global face_locations, face_names
        face_locations = face_locs
        face_names = names

    if lock_state:
        msg = "Please Close The Door Properly\nThank You for Keeping VGNet Lab Secure!"
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_path = os.path.join(frh, "fonts", "batmfa__.ttf")
        font_size = 40
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        lines = msg.split('\n')
        y_text = pil_img.height - (font_size + 10) * len(lines) - 10
        for line in lines:
            try:
                bbox = draw.textbbox((0, 0), line, font=font)
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
            except AttributeError:
                w, h = font.getsize(line)
            x_text = pil_img.width - w - 20
            draw.text((x_text, y_text), line, font=font, fill=(0, 220, 220))
            y_text += h + 10

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    return frame

def draw_results(frame):
    with lock:
        locations = face_locations.copy()
        names = face_names.copy()

    for (top, right, bottom, left), name in zip(locations, names):
        top = int(top * cv_scaler)
        right = int(right * cv_scaler)
        bottom = int(bottom * cv_scaler)
        left = int(left * cv_scaler)

        color = (0, 210, 0) if name != "Unknown" else (0, 0, 210)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    return frame

# === Threaded face processing ===
frame_lock = threading.Lock()
processed_frame = None
processing_thread_running = True

def face_processing_thread():
    global processed_frame, current_frame, processing_thread_running
    while processing_thread_running:
        if current_frame is not None:
            with frame_lock:
                frame_copy = current_frame.copy()
            processed = process_frame(frame_copy)
            with frame_lock:
                processed_frame = processed
        else:
            time.sleep(0.01)

def main():
    global current_frame, frame_count, running, processing_thread_running

    threading.Thread(target=speech_worker, daemon=True).start()

    cv2.namedWindow("VGNet Lab Entrance", cv2.WINDOW_NORMAL)
    cv2.moveWindow("VGNet Lab Entrance", 100, 100)

    fps = 0
    start_time = time.time()

    # Start face processing thread
    thread = threading.Thread(target=face_processing_thread, daemon=True)
    thread.start()

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to grab frame from camera.")
                continue

            with frame_lock:
                current_frame = frame.copy()
            frame_count += 1

            with frame_lock:
                display_frame = processed_frame.copy() if processed_frame is not None else frame.copy()

            display_frame = draw_results(display_frame)

            fps += 1
            elapsed = time.time() - start_time
            if elapsed >= 1.0:
                print(f"[INFO] Approx. FPS: {fps}")
                fps = 0
                start_time = time.time()

            cv2.imshow("VGNet Lab Entrance", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                running = False

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        print("[INFO] Cleaning up...")
        running = False
        processing_thread_running = False
        thread.join(timeout=2)
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
