import os
from imutils import paths
import face_recognition
import pickle
import cv2
import hashlib

# Paths
dataset_path = "dataset"
encodings_path = "encodings.pickle"
processed_log = "processed_images.txt"

# Load previous encodings if they exist
if os.path.exists(encodings_path):
    print("[INFO] Loading existing encodings...")
    with open(encodings_path, "rb") as f:
        data = pickle.load(f)
        knownEncodings = data["encodings"]
        knownNames = data["names"]
else:
    knownEncodings = []
    knownNames = []

# Load list of already processed images
if os.path.exists(processed_log):
    with open(processed_log, "r") as f:
        processed_images = set(line.strip() for line in f.readlines())
else:
    processed_images = set()

# Get image paths
imagePaths = list(paths.list_images(dataset_path))
new_images_processed = 0

for (i, imagePath) in enumerate(imagePaths):
    # Skip if already processed
    if imagePath in processed_images:
        continue

    print(f"[INFO] Processing new image {i + 1}/{len(imagePaths)}: {imagePath}")
    name = os.path.basename(os.path.dirname(imagePath))
    image = cv2.imread(imagePath)
    if image is None:
        print(f"[WARNING] Skipping unreadable image: {imagePath}")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

    # Mark this image as processed
    with open(processed_log, "a") as f:
        f.write(f"{imagePath}\n")

    new_images_processed += 1

# Save updated encodings
with open(encodings_path, "wb") as f:
    pickle.dump({"encodings": knownEncodings, "names": knownNames}, f)

print(f"\n✅ Incremental update complete.")
print(f"New images processed: {new_images_processed}")
print(f"Total people: {len(set(knownNames))}, Total encodings: {len(knownEncodings)}")
