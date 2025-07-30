import os
from imutils import paths
import face_recognition
import pickle
import cv2
import numpy as np
from tqdm import tqdm

dataset_path = "dataset"
encodings_path = "encodings.pickle"

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equalized = clahe.apply(gray)
    return cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)

def load_data():
    if os.path.exists(encodings_path):
        print("[INFO] Loading existing encodings...")
        with open(encodings_path, "rb") as f:
            data = pickle.load(f)
            encodings = [np.array(e, dtype=np.float32) for e in data["encodings"]]
            names = data["names"]
            image_paths = data.get("image_paths", [])
            return encodings, names, image_paths
    return [], [], []

def save_data(encodings, names, image_paths):
    tmp_path = encodings_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({
            "encodings": encodings,
            "names": names,
            "image_paths": image_paths
        }, f)
    os.replace(tmp_path, encodings_path)

def process_images():
    knownEncodings, knownNames, knownImagePaths = load_data()

    current_imagePaths = set(paths.list_images(dataset_path))

    # Remove encodings for deleted images
    new_encodings = []
    new_names = []
    new_image_paths = []
    removed_count = 0

    for enc, name, img_path in zip(knownEncodings, knownNames, knownImagePaths):
        if img_path in current_imagePaths:
            new_encodings.append(enc)
            new_names.append(name)
            new_image_paths.append(img_path)
        else:
            removed_count += 1

    print(f"[INFO] Removed {removed_count} encodings for deleted images.")

    # Identify new images to process
    encoded_set = set(new_image_paths)
    images_to_process = list(current_imagePaths - encoded_set)

    print(f"[INFO] Processing {len(images_to_process)} new images.")

    for imagePath in tqdm(images_to_process):
        name = os.path.basename(os.path.dirname(imagePath))
        image = cv2.imread(imagePath)
        if image is None:
            print(f"[WARNING] Cannot read {imagePath}. Skipping.")
            continue

        rgb = preprocess_image(image)
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)

        if len(encodings) == 0:
            print(f"[WARNING] No faces found in {imagePath}. Skipping.")
            continue

        for encoding in encodings:
            new_encodings.append(np.array(encoding, dtype=np.float32))
            new_names.append(name)
            new_image_paths.append(imagePath)

    save_data(new_encodings, new_names, new_image_paths)

    print(f"\n✅ Update complete.")
    print(f"Total images encoded: {len(new_image_paths)}")
    print(f"Total unique people: {len(set(new_names))}")

if __name__ == "__main__":
    process_images()
