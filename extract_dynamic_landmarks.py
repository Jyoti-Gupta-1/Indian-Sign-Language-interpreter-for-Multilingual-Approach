import os
import cv2
import numpy as np
import mediapipe as mp

# ======================================
# PATHS
# ======================================

VIDEO_PATH = "dataset/dynamic/videos"
OUTPUT_PATH = "dataset/dynamic/raw"

SEQUENCE_LENGTH = 30

SUPPORTED_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mpeg",
    ".mpg"
)

# ======================================
# MEDIAPIPE
# ======================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# ======================================
# PROCESS LABELS
# ======================================

labels = sorted(os.listdir(VIDEO_PATH))

for label in labels:

    label_video_path = os.path.join(VIDEO_PATH, label)

    if not os.path.isdir(label_video_path):
        continue

    label_output_path = os.path.join(OUTPUT_PATH, label)

    os.makedirs(label_output_path, exist_ok=True)

    existing = [
        int(f.split(".")[0])
        for f in os.listdir(label_output_path)
        if f.endswith(".npy") and f.split(".")[0].isdigit()
    ]

    next_index = max(existing) + 1 if existing else 0

    print(f"\n========== {label} ==========")

    videos = [
        f for f in os.listdir(label_video_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print(f"Videos Found : {len(videos)}")

    for video_name in videos:

        video_path = os.path.join(
            label_video_path,
            video_name
        )

        print(f"Processing : {video_name}")

        cap = cv2.VideoCapture(video_path)

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if total_frames <= 0:
            print("Skipped (Empty Video)")
            cap.release()
            continue

        frame_indices = np.linspace(
            0,
            total_frames - 1,
            SEQUENCE_LENGTH,
            dtype=int
        )

        sequence = []

        current_frame = 0
        target_pointer = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            if target_pointer >= len(frame_indices):
                break

            if current_frame == frame_indices[target_pointer]:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb)

                if results.multi_hand_landmarks:

                    hand = results.multi_hand_landmarks[0]

                    row = []

                    for lm in hand.landmark:

                        row.extend([
                            lm.x,
                            lm.y,
                            lm.z
                        ])

                    sequence.append(row)

                else:

                    if len(sequence):

                        sequence.append(sequence[-1])

                    else:

                        sequence.append([0] * 63)

                target_pointer += 1

            current_frame += 1

        cap.release()

        while len(sequence) < SEQUENCE_LENGTH:

            sequence.append(sequence[-1])

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        save_path = os.path.join(
            label_output_path,
            f"{next_index}.npy"
        )

        np.save(
            save_path,
            sequence
        )

        print(f"Saved -> {save_path}")

        next_index += 1

hands.close()

print("\n======================================")
print("Dynamic Landmark Extraction Completed!")
print("======================================")