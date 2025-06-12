import cv2
import os
from datetime import datetime
from picamera2 import Picamera2
import time
import numpy as np
import random

# Change this to the name of the person you're photographing
PERSON_NAME = input("New User Name : ")
dataset_folder = os.path.dirname(__file__)

def create_folder(name):
    dataset_folder = os.path.join(dataset_folder, "dataset")
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder


def random_brightness(img):
    """ Slight random brightness adjustment """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = random.randint(-40, 40)
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], value)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def capture_photos(name):
    folder = create_folder(name)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    time.sleep(2)  # Camera warm-up

    photo_count = 0
    prompts = [
        "Look straight at the camera.",
        "Turn your head to the LEFT.",
        "Turn your head to the RIGHT.",
        "Smile slightly.",
        "Keep a neutral face.",
        "Lean a bit forward."
    ]

    print(f"Taking photos for '{name}'. Press SPACE to capture, 'q' to quit.")

    while True:
        frame = picam2.capture_array()
        display_frame = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Capture', display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Capture face when spacebar is pressed
            if len(faces) == 0:
                print("No face detected. Try again.")
                continue

            for (x, y, w, h) in faces:
                face_img = frame[y:y + h, x:x + w]
                face_img = cv2.resize(face_img, (224, 224))

                # Save original
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{name}_{timestamp}.jpg"
                filepath = os.path.join(folder, filename)
                cv2.imwrite(filepath, face_img)
                photo_count += 1
                print(f"[{photo_count}] Saved: {filepath}")

                # Save flipped version (data augmentation)
                flipped = cv2.flip(face_img, 1)
                flip_name = f"{name}_{timestamp}_flip.jpg"
                cv2.imwrite(os.path.join(folder, flip_name), flipped)

                # Save brightness adjusted version (data augmentation)
                bright = random_brightness(face_img)
                bright_name = f"{name}_{timestamp}_bright.jpg"
                cv2.imwrite(os.path.join(folder, bright_name), bright)

            # Prompt next variation
            if photo_count % 3 == 0 and photo_count < len(prompts) * 3:
                print(f"\n➡️ Next: {prompts[photo_count // 3 % len(prompts)]}\n")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    picam2.stop()
    print(f"\n✅ Photo capture completed. {photo_count} faces saved for '{name}'.")


if __name__ == "__main__":
    capture_photos(PERSON_NAME)
