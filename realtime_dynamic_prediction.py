import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import deque, Counter
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont
import time

# ======================================
# WORD MAP (Dataset Label -> Proper English)
# ======================================
WORD_MAP = {
    "thank_you": "thank you",
    "bye": "goodbye",
    "hello": "hello",
    "yes": "yes",
    "namaste": "namaste"
}

# ======================================
# CUSTOM TRANSLATIONS
# ======================================
CUSTOM_TRANSLATIONS = {
    "en": {
        "hello": "hello",
        "goodbye": "goodbye",
        "thank you": "thank you",
        "yes": "yes",
        "namaste": "namaste"
    },
    "hi": {
        "hello": "नमस्ते",
        "goodbye": "अलविदा",
        "thank you": "धन्यवाद",
        "yes": "हाँ",
        "namaste": "नमस्ते"
    },
    "mr": {
        "hello": "नमस्कार",
        "goodbye": "निरोप",
        "thank you": "धन्यवाद",
        "yes": "हो",
        "namaste": "नमस्कार"
    }
}

# ======================================
# TRANSLATION MANAGER (with caching)
# ======================================
class TranslationManager:
    def __init__(self):
        self.translator = Translator()
        self.cached_sentence_translation = ""
        self.last_sentence_key = None
        self.cached_prediction_translation = ""
        self.last_prediction_key = None

    def translate_sentence(self, sentence_list, lang_code):
        if not sentence_list:
            return ""
        
        key = (tuple(sentence_list), lang_code)
        if key == self.last_sentence_key:
            return self.cached_sentence_translation

        # Map words to proper English
        mapped_words = [WORD_MAP.get(w, w) for w in sentence_list]
        english_text = " ".join(mapped_words)

        if lang_code == "en":
            self.cached_sentence_translation = english_text
        else:
            english_lower = english_text.lower()
            if english_lower in CUSTOM_TRANSLATIONS.get(lang_code, {}):
                self.cached_sentence_translation = CUSTOM_TRANSLATIONS[lang_code][english_lower]
            else:
                try:
                    translated_obj = self.translator.translate(english_text, dest=lang_code)
                    self.cached_sentence_translation = translated_obj.text
                except Exception as e:
                    # Fallback to word-by-word custom translation or english
                    fallback_words = []
                    for w in mapped_words:
                        w_lower = w.lower()
                        if w_lower in CUSTOM_TRANSLATIONS.get(lang_code, {}):
                            fallback_words.append(CUSTOM_TRANSLATIONS[lang_code][w_lower])
                        else:
                            fallback_words.append(w)
                    self.cached_sentence_translation = " ".join(fallback_words)
                    print(f"Sentence Translation error (using fallback): {e}")
        
        self.last_sentence_key = key
        return self.cached_sentence_translation

    def translate_prediction(self, prediction, lang_code):
        if not prediction or prediction in ["Show Gesture", "Uncertain"]:
            return prediction
        
        key = (prediction, lang_code)
        if key == self.last_prediction_key:
            return self.cached_prediction_translation

        english_word = WORD_MAP.get(prediction, prediction)

        if lang_code == "en":
            self.cached_prediction_translation = english_word
        else:
            english_word_lower = english_word.lower()
            if english_word_lower in CUSTOM_TRANSLATIONS.get(lang_code, {}):
                self.cached_prediction_translation = CUSTOM_TRANSLATIONS[lang_code][english_word_lower]
            else:
                try:
                    translated_obj = self.translator.translate(english_word, dest=lang_code)
                    self.cached_prediction_translation = translated_obj.text
                except Exception as e:
                    self.cached_prediction_translation = english_word
                    print(f"Prediction Translation error (using fallback): {e}")

        self.last_prediction_key = key
        return self.cached_prediction_translation

# ======================================
# LOAD MODELS
# ======================================
print("Loading Dynamic Model...")
dynamic_model = load_model("dynamic_model.keras")
dynamic_label_encoder = joblib.load("dynamic_label_encoder.pkl")
print("Dynamic model loaded successfully!")
print("Dynamic Classes:", dynamic_label_encoder.classes_)

print("Loading Static Model...")
static_model = joblib.load("static_landmark_model.pkl")
print("Static model loaded successfully!")

# ======================================
# INITIALIZE TRANSLATION & STATE
# ======================================
translation_manager = TranslationManager()
sentence = []
current_prediction = ""
display_prediction = "Show Gesture"
display_confidence = None
consecutive_missed_frames = 0

# Mode settings: "dynamic" or "static"
mode = "dynamic"

# Language settings
current_language_name = "English"
current_language_code = "en"

LANGUAGES = {
    ord('1'): ("English", "en"),
    ord('2'): ("Hindi", "hi"),
    ord('3'): ("Marathi", "mr")
}

# ======================================
# MEDIAPIPE SETUP
# ======================================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Use max_num_hands=2 to support two-hand static gestures
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================================
# PREDICTION BUFFERS
# ======================================
sequence = deque(maxlen=30)
prediction_history = deque(maxlen=10)

# ======================================
# START WEBCAM
# ======================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

# Setup variables for FPS calculation
fps = 0
fps_frame_counter = 0
fps_timer = time.time()

print("\nStarting Hybrid Prediction System...")
print("Hotkeys:")
print("SPACE      -> Add current word/letter")
print("BACKSPACE  -> Delete last word/letter")
print("C          -> Clear sentence")
print("M          -> Toggle between STATIC and DYNAMIC modes")
print("1          -> English")
print("2          -> Hindi")
print("3          -> Marathi")
print("ESC        -> Exit")

# ======================================
# MAIN LOOP
# ======================================
while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read webcam frame.")
        break

    # Calculate FPS (running average/update every 0.5s for stability)
    fps_frame_counter += 1
    time_now = time.time()
    if time_now - fps_timer >= 0.5:
        fps = int(fps_frame_counter / (time_now - fps_timer))
        fps_frame_counter = 0
        fps_timer = time_now

    # Mirror effect
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    results = hands.process(rgb)

    # ======================================
    # HAND DETECTION & PREDICTION
    # ======================================
    if results.multi_hand_landmarks:
        consecutive_missed_frames = 0
        
        # 1. Handle Dynamic Prediction Mode (uses exactly 1 hand)
        if mode == "dynamic":
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            row = []
            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])
            
            # Normalize frame landmarks (wrist-relative shift and scale normalization)
            landmarks = np.array(row).reshape(21, 3)
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            max_val = np.max(np.abs(landmarks))
            if max_val != 0:
                landmarks = landmarks / max_val
            normalized_row = landmarks.flatten().tolist()
            
            sequence.append(normalized_row)

            if len(sequence) == 30:
                input_data = np.array(sequence)
                input_data = np.expand_dims(input_data, axis=0)

                prediction = dynamic_model.predict(input_data, verbose=0)[0]
                confidence = np.max(prediction)
                predicted_index = np.argmax(prediction)
                predicted_label = dynamic_label_encoder.inverse_transform([predicted_index])[0]

                if confidence > 0.70:
                    prediction_history.append(predicted_label)
                    stable_prediction = Counter(prediction_history).most_common(1)[0][0]
                    current_prediction = stable_prediction
                    display_prediction = stable_prediction
                    display_confidence = confidence
                else:
                    # Do NOT clear prediction history to prevent flickering; just display Uncertain
                    display_prediction = "Uncertain"
                    display_confidence = None

        # 2. Handle Static Prediction Mode (uses up to 2 hands)
        else:
            row = []
            # Draw landmarks and collect for all visible hands
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y, lm.z])

            # Pad to 126 features if only one hand is detected
            while len(row) < 126:
                row.extend([0, 0, 0])
            row = row[:126]

            # Normalize static landmarks (relative to wrist of first hand)
            landmarks = np.array(row).reshape(42, 3)
            wrist = landmarks[0]
            landmarks = landmarks - wrist
            max_value = np.max(np.abs(landmarks))
            if max_value != 0:
                landmarks = landmarks / max_value
            normalized_row = landmarks.flatten().reshape(1, -1)

            # Predict using Random Forest probabilities
            probabilities = static_model.predict_proba(normalized_row)[0]
            confidence = np.max(probabilities)
            predicted_label = static_model.classes_[np.argmax(probabilities)]

            if confidence > 0.75:
                prediction_history.append(predicted_label)
                stable_prediction = Counter(prediction_history).most_common(1)[0][0]
                current_prediction = stable_prediction
                display_prediction = stable_prediction
                display_confidence = confidence
            else:
                # Do NOT clear prediction history to prevent flickering; just display Uncertain
                display_prediction = "Uncertain"
                display_confidence = None

    else:
        # No hands detected: allow a 15-frame grace period for dynamic mode
        consecutive_missed_frames += 1
        if mode == "dynamic" and consecutive_missed_frames <= 15:
            # Skip updating sequence for this frame but do not clear it
            pass
        else:
            # Clear sequence and buffers after grace period expires, or immediately for static mode
            sequence.clear()
            prediction_history.clear()
            current_prediction = ""
            display_prediction = "Show Gesture"
            display_confidence = None

    # ======================================
    # TRANSLATION
    # ======================================
    translated_prediction = translation_manager.translate_prediction(display_prediction, current_language_code)
    translated_sentence = translation_manager.translate_sentence(sentence, current_language_code)

    # ======================================
    # UI OVERLAY RENDERING
    # ======================================
    # Top banner (y = 0 to 110) & Bottom banner (y = height - 70 to height)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 110), (30, 30, 30), -1)
    cv2.rectangle(overlay, (0, height - 70), (width, height), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Draw texts using PIL
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font_path = r"C:\Windows\Fonts\Nirmala.ttc"
    font_large = ImageFont.truetype(font_path, 24)
    font_medium = ImageFont.truetype(font_path, 18)
    font_small = ImageFont.truetype(font_path, 14)

    # Top Banner Info
    mode_color = (0, 255, 0) if mode == "dynamic" else (255, 215, 0)
    draw.text((20, 15), f"MODE: {mode.upper()}", font=font_medium, fill=mode_color)
    draw.text((250, 15), f"LANG: {current_language_name} ({current_language_code})", font=font_medium, fill=(0, 255, 255))
    draw.text((width - 120, 15), f"FPS: {fps}", font=font_medium, fill=(255, 255, 255))

    if display_prediction in ["Show Gesture", "Uncertain"] or display_confidence is None:
        pred_text = f"Prediction: {translated_prediction}"
    else:
        pred_text = f"Prediction: {translated_prediction} ({int(display_confidence * 100)}%)"
    draw.text((20, 45), pred_text, font=font_large, fill=(0, 255, 0))

    draw.text((20, 75), f"Sentence: {translated_sentence}", font=font_medium, fill=(255, 255, 255))

    # Bottom Banner Info
    row1_text = "SPACE: Add Word/Letter  |  BKSP: Delete Last  |  C: Clear  |  M: Toggle Mode"
    draw.text((20, height - 60), row1_text, font=font_small, fill=(240, 240, 240))
    row2_text = "1: English  |  2: Hindi  |  3: Marathi  |  ESC: Exit"
    draw.text((20, height - 35), row2_text, font=font_small, fill=(240, 240, 240))

    frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    # ======================================
    # SHOW WINDOW
    # ======================================
    cv2.imshow("Hybrid Sign Language System", frame)

    # ======================================
    # HOTKEYS HANDLING
    # ======================================
    key = cv2.waitKey(1) & 0xFF

    # ESC -> Exit
    if key == 27:
        break

    # SPACE -> Add current prediction
    elif key == 32:
        if current_prediction != "":
            sentence.append(current_prediction)

    # BACKSPACE -> Delete last item
    elif key == 8:
        if sentence:
            sentence.pop()

    # C -> Clear sentence
    elif key == ord('c') or key == ord('C'):
        sentence.clear()

    # M -> Toggle mode
    elif key == ord('m') or key == ord('M'):
        mode = "static" if mode == "dynamic" else "dynamic"
        sequence.clear()
        prediction_history.clear()
        current_prediction = ""
        display_prediction = "Show Gesture"
        display_confidence = None

    # Language Switch
    elif key in LANGUAGES:
        current_language_name, current_language_code = LANGUAGES[key]

# ======================================
# CLEANUP
# ======================================
cap.release()
hands.close()
cv2.destroyAllWindows()