import os
import numpy as np

DATASET_PATH = "dataset/dynamic/raw"
OUTPUT_PATH = "dataset/dynamic/processed"

X = []
y = []

labels = sorted(os.listdir(DATASET_PATH))

for label in labels:
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    print(f"Processing: {label}")

    for file_name in os.listdir(label_path):

        if not file_name.endswith(".npy"):
            continue

        file_path = os.path.join(label_path, file_name)

        sequence = np.load(file_path)

        # Normalize landmarks per frame (wrist-relative shift and scale normalization)
        normalized_seq = []
        for frame in sequence:
            landmarks = np.array(frame).reshape(21, 3)
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            max_val = np.max(np.abs(landmarks))
            if max_val != 0:
                landmarks = landmarks / max_val
            normalized_seq.append(landmarks.flatten())

        X.append(normalized_seq)
        y.append(label)

X = np.array(X)
y = np.array(y)

os.makedirs(OUTPUT_PATH, exist_ok=True)

np.save(
    os.path.join(OUTPUT_PATH, "dynamic_landmarks.npy"),
    X
)

np.save(
    os.path.join(OUTPUT_PATH, "dynamic_labels.npy"),
    y
)

print("\nDataset Created!")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))