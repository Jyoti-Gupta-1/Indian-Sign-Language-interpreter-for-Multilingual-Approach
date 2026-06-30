

import cv2
import mediapipe as mp
import numpy as np
import joblib

from tensorflow.keras.models import load_model


class StaticPredictor:

    def __init__(self):

        # ==========================================
        # LOAD CNN MODEL
        # ==========================================

        self.model = load_model("static_model.keras")

        self.encoder = joblib.load(
            "static_label_encoder.pkl"
        )

        print("Static CNN Loaded Successfully!")

        # ==========================================
        # MEDIAPIPE
        # ==========================================

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

        self.hands = self.mp_hands.Hands(

            static_image_mode=False,

            max_num_hands=2,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5

        )

    # ==========================================
    # PREDICT STATIC GESTURE
    # ==========================================

    def predict(self, frame):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.hands.process(rgb)

        prediction = "No Hand Detected"

        confidence = 0

        if results.multi_hand_landmarks:

            row = []

            for hand_landmarks in results.multi_hand_landmarks:

                self.mp_draw.draw_landmarks(

                    frame,

                    hand_landmarks,

                    self.mp_hands.HAND_CONNECTIONS

                )

                for lm in hand_landmarks.landmark:

                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

            # ----------------------------------
            # PAD IF ONLY ONE HAND
            # ----------------------------------

            while len(row) < 126:

                row.extend([0, 0, 0])

            row = row[:126]

            # ----------------------------------
            # NORMALIZATION
            # ----------------------------------

            landmarks = np.array(row).reshape(42, 3)

            wrist = landmarks[0]

            landmarks = landmarks - wrist

            max_value = np.max(np.abs(landmarks))

            if max_value != 0:

                landmarks = landmarks / max_value

            normalized = landmarks.flatten()

            # CNN expects (batch,126,1)

            normalized = normalized.reshape(
                1,
                126,
                1
            )

            # ----------------------------------
            # CNN PREDICTION
            # ----------------------------------

            probs = self.model.predict(
                normalized,
                verbose=0
            )

            class_index = np.argmax(probs)

            confidence = float(
                probs[0][class_index] * 100
            )

            prediction = self.encoder.inverse_transform(
                [class_index]
            )[0]

        return prediction, confidence, frame

    # ==========================================
    # CLEANUP
    # ==========================================

    def close(self):

        self.hands.close()