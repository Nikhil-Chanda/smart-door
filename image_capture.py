import cv2
import os
import time

def main():
    # Step 1: Get user's name
    person_name = input("Enter the name of the person: ").strip()

    # Step 2: Create folder inside dataset/
    dataset_path = "dataset"
    person_path = os.path.join(dataset_path, person_name)
    os.makedirs(person_path, exist_ok=True)
    print(f"[INFO] Folder created at: {person_path}")

    # Step 3: Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise IOError("[ERROR] Cannot access webcam.")

    print("[INFO] Camera warming up...")
    time.sleep(2)

    # Step 4: Load face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Step 5: Define prompts and parameters
    prompts = [
        "Look straight (bright light)",
        "Turn head left (bright light)",
        "Turn head right (bright light)",
        "Look up (bright light)",
        "Look down (bright light)",
        "Smile slightly (bright light)",
        "Look straight (low light)",
        "Turn head left (low light)",
        "Turn head right (low light)",
        "Look up (low light)",
        "Look down (low light)",
        "Smile slightly (low light)",
    ]
    images_per_prompt = 5  # capture 5 images per prompt
    max_images = len(prompts) * images_per_prompt

    img_count = 0
    prompt_index = 0

    # Timer variables to control capture speed without freezing frames
    capture_interval = 1.0  # seconds between captures
    last_capture_time = 0

    print("[INFO] Starting capture.")
    print("[INSTRUCTION] Please capture all images under BRIGHT light first.")
    print("[INSTRUCTION] Then dim lights or move to low-light area when prompted.")
    print("[INSTRUCTION] Press 'q' anytime to quit.\n")

    try:
        while img_count < max_images:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Show current prompt on screen
            current_prompt = prompts[prompt_index]
            cv2.putText(frame, current_prompt, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            # Save face images if detected and interval passed
            if len(faces) > 0 and (time.time() - last_capture_time) > capture_interval:
                x, y, w, h = faces[0]  # take first detected face
                face_img = frame[y:y+h, x:x+w]

                img_count += 1
                img_file = os.path.join(person_path, f"{img_count}.jpg")
                cv2.imwrite(img_file, face_img)
                print(f"[INFO] ({img_count}/{max_images}) Saved: {img_file}")

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                last_capture_time = time.time()

                # Change prompt every 'images_per_prompt' images
                if img_count % images_per_prompt == 0:
                    prompt_index += 1
                    if prompt_index == 6:
                        print("\n[INSTRUCTION] Please dim the lights or move to a low-light area now.")
                        print("Waiting 10 seconds for lighting adjustment...\n")
                        time.sleep(10)  # give user time to adjust lighting

            cv2.imshow("Capturing Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Capture interrupted by user.")
                break

    except KeyboardInterrupt:
        print("[INFO] Capture interrupted by keyboard.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Done. {img_count} images saved in '{person_path}'")

if __name__ == "__main__":
    main()
