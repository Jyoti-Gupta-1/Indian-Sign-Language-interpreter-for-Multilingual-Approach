

import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# -----------------------------
# Load models
# -----------------------------
dynamic_model = load_model("gesture_lstm.h5")    # Sequence-based dynamic gestures
static_model = load_model("static_letters.h5")  # CNN static gestures

# -----------------------------
# Load static gesture classes
# -----------------------------
try:
    with open("static_gestures.txt", "r") as f:
        STATIC_GESTURES = [line.strip() for line in f.readlines()]
except FileNotFoundError:
    # fallback if file missing
    STATIC_GESTURES = ['1', '2', '3', '4', '5', '6', '7', '8', '9',
                       'O','P','Q','R','S','T','U','V','W','X','Y','Z']

DYNAMIC_GESTURES = ['bye', 'hello', 'namaste', 'thank_you', 'yes']  # replace with your dynamic gestures
SEQUENCE_LENGTH = 30
sequence = []

IMG_SIZE = 64  # must match static model training

# -----------------------------
# MediaPipe Hands setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # detect both hands
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture_text = "No gesture"
    static_texts = []

    if results.multi_hand_landmarks:
        # -----------------------------
        # Process each detected hand for static gestures
        # -----------------------------
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract hand bounding box
            h, w, _ = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w) - 10
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w) + 10
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h) - 10
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h) + 10

            x_min, y_min = max(x_min, 0), max(y_min, 0)
            x_max, y_max = min(x_max, w), min(y_max, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size != 0:
                hand_img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
                hand_img = img_to_array(hand_img) / 255.0
                hand_img = np.expand_dims(hand_img, axis=0)
                static_pred = static_model.predict(hand_img, verbose=0)[0]
                idx = np.argmax(static_pred)
                static_texts.append(STATIC_GESTURES[idx])

            # -----------------------------
            # Collect landmarks for dynamic gesture
            # -----------------------------
            frame_data = []
            for lm in hand_landmarks.landmark:
                frame_data.extend([lm.x, lm.y, lm.z])
            sequence.append(frame_data)
            sequence = sequence[-SEQUENCE_LENGTH:]

        # -----------------------------
        # Predict dynamic gesture
        # -----------------------------
        if len(sequence) == SEQUENCE_LENGTH:
            prediction = dynamic_model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            gesture_text = DYNAMIC_GESTURES[np.argmax(prediction)]
        else:
            gesture_text = "..."  # waiting for sequence to fill

    # -----------------------------
    # Display both static and dynamic gestures
    # -----------------------------
    cv2.putText(frame,
                "Static: " + ", ".join(static_texts) if static_texts else "Static: None",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(frame,
                "Dynamic: " + gesture_text,
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Live Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
