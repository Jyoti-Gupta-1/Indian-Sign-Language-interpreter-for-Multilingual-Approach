

# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# from tensorflow.keras.models import load_model
# from collections import deque, Counter
# from googletrans import Translator
# from PIL import Image, ImageDraw, ImageFont


# def draw_unicode_text(frame, text, position, font_size=30, color=(0, 255, 255)):
#     img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     draw = ImageDraw.Draw(img_pil)

#     def draw_unicode_text(frame, text, position,
#                        font_size=30,
#                        color=(0, 255, 255)):

#         img_pil = Image.fromarray(
#         cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         )

#     draw = ImageDraw.Draw(img_pil)


#     font = ImageFont.truetype(
#     r"C:\Windows\Fonts\Nirmala.ttc",
#     font_size
# )

#     draw.text(
#         position,
#         text,
#         font=font,
#         fill=(color[2], color[1], color[0])
#     )

#     return cv2.cvtColor(
#         np.array(img_pil),
#         cv2.COLOR_RGB2BGR
#     )

#     draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))

#     return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# # ==============================
# # LOAD MODEL
# # ==============================

# model = load_model("dynamic_model.keras")
# label_encoder = joblib.load("dynamic_label_encoder.pkl")

# print("Dynamic model loaded successfully!")
# print("Classes:", label_encoder.classes_)

# # ==============================
# # TRANSLATION SETUP
# # ==============================

# translator = Translator()

# sentence = []

# # Convert dataset labels into natural English
# WORD_MAP = {
#     "thank_you": "thank you",
#     "bye": "goodbye",
#     "hello": "hello",
#     "yes": "yes",
#     "namaste": "namaste"
# }

# CUSTOM_TRANSLATIONS = {
#     "en": {
#         "hello": "hello",
#         "bye": "goodbye",
#         "thank you": "thank you",
#         "yes": "yes",
#         "namaste": "namaste"
#     },

#     "hi": {
#         "hello": "नमस्ते",
#         "goodbye": "अलविदा",
#         "thank you": "धन्यवाद",
#         "yes": "हाँ",
#         "namaste": "नमस्ते"
#     },

#     "mr": {
#         "hello": "नमस्कार",
#         "goodbye": "निरोप",
#         "thank you": "धन्यवाद",
#         "yes": "हो",
#         "namaste": "नमस्कार"
#     }
# }

# # Default Language
# current_language_name = "English"
# current_language_code = "en"

# # Language Hotkeys
# LANGUAGES = {
#     ord('1'): ("English", "en"),
#     ord('2'): ("Hindi", "hi"),
#     ord('3'): ("Marathi", "mr")
# }

# # Current Stable Prediction
# current_prediction = ""

# # ==============================
# # MEDIAPIPE SETUP
# # ==============================

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils

# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# # ==============================
# # BUFFERS
# # ==============================

# # Stores 30-frame sequence
# sequence = deque(maxlen=30)

# # Stores previous predictions
# prediction_history = deque(maxlen=10)

# # ==============================
# # START WEBCAM
# # ==============================

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Failed to open webcam.")
#     exit()

# print("\nStarting Dynamic Prediction...")
# print("Hotkeys:")
# print("SPACE      -> Add word")
# print("BACKSPACE  -> Delete last word")
# print("C          -> Clear sentence")
# print("1          -> English")
# print("2          -> Hindi")
# print("3          -> Marathi")
# print("ESC        -> Exit")
# # ==============================
# # MAIN LOOP
# # ==============================

# while True:

#     success, frame = cap.read()

#     if not success:
#         print("Failed to read webcam frame.")
#         break

#     # Mirror effect
#     frame = cv2.flip(frame, 1)

#     # Convert to RGB
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process hand landmarks
#     results = hands.process(rgb)

#     display_text = "Show Gesture"

#     # ==============================
#     # HAND DETECTION
#     # ==============================

#     if results.multi_hand_landmarks:

#         hand_landmarks = results.multi_hand_landmarks[0]

#         # Draw hand landmarks
#         mp_draw.draw_landmarks(
#             frame,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS
#         )

#         row = []

#         # Extract 21×3 landmarks
#         for lm in hand_landmarks.landmark:
#             row.extend([lm.x, lm.y, lm.z])

#         # Store frame
#         sequence.append(row)

#         # ==============================
#         # DYNAMIC PREDICTION
#         # ==============================

#         if len(sequence) == 30:

#             input_data = np.array(sequence)
#             input_data = np.expand_dims(input_data, axis=0)

#             prediction = model.predict(
#                 input_data,
#                 verbose=0
#             )[0]

#             confidence = np.max(prediction)

#             predicted_index = np.argmax(prediction)

#             predicted_label = label_encoder.inverse_transform(
#                 [predicted_index]
#             )[0]

#             # Confidence threshold
#             if confidence > 0.70:

#                 prediction_history.append(
#                     predicted_label
#                 )

#                 stable_prediction = Counter(
#                     prediction_history
#                 ).most_common(1)[0][0]

#                 current_prediction = stable_prediction

#                 display_text = (
#                     f"{stable_prediction}"
#                     f" ({confidence:.2f})"
#                 )

#             else:
#                 display_text = (
#                     f"Uncertain ({confidence:.2f})"
#                 )

#     else:

#         # No hand detected
#         sequence.clear()

#     # ==============================
#     # SENTENCE TRANSLATION
#     # ==============================

#   mapped_sentence = []

# for word in sentence:
#     mapped_sentence.append(
#         WORD_MAP.get(word, word)
#     )

# translated_words = []

# for word in mapped_sentence:

#     # Use our custom translations first
#     if word in CUSTOM_TRANSLATIONS[current_language_code]:

#         translated_words.append(
#             CUSTOM_TRANSLATIONS[current_language_code][word]
#         )

#     # Fallback to Google Translate
#     else:
#         try:
#             translated_words.append(
#                 translator.translate(
#                     word,
#                     dest=current_language_code
#                 ).text
#             )

#         except Exception:
#             translated_words.append(word)

#     translated_sentence = " ".join(translated_words)

#     if english_sentence != "":

#         try:

#             translated_sentence = (
#                 translator.translate(
#                     english_sentence,
#                     dest=current_language_code
#                 ).text
#             )

#         except Exception:

#             translated_sentence = (
#                 english_sentence
#             )
#                 # ==============================
#     # DISPLAY UI
#     # ==============================

#     # Top Information
#     cv2.putText(
#         frame,
#         f"Prediction : {display_text}",
#         (20, 35),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.8,
#         (0, 255, 0),       # Green
#         2
#     )

#     cv2.putText(
#         frame,
#         f"Language : {current_language_name}",
#         (20, 70),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.7,
#         (255, 255, 0),     # Cyan
#         2
#     )


#     frame = draw_unicode_text(
#     frame,
#     f"Sentence : {translated_sentence}",
#     (20, 80),
#     font_size=28,
#     color=(0, 255, 255)
#     )

#     # ==============================
#     # LEFT SIDE CONTROLS
#     # ==============================

#     cv2.rectangle(
#         frame,
#         (5, 130),
#         (170, 260),
#         (50, 50, 50),
#         -1
#     )

#     cv2.putText(
#         frame,
#         "CONTROLS",
#         (20, 155),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (255, 255, 255),
#         2
#     )

#     cv2.putText(
#         frame,
#         "SPACE : Add",
#         (20, 185),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 255),
#         1
#     )

#     cv2.putText(
#         frame,
#         "BKSP  : Delete",
#         (20, 210),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 255),
#         1
#     )

#     cv2.putText(
#         frame,
#         "C     : Clear",
#         (20, 235),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 255),
#         1
#     )

#     cv2.putText(
#         frame,
#         "ESC   : Exit",
#         (20, 260),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (0, 255, 255),
#         1
#     )

#     # ==============================
#     # RIGHT SIDE LANGUAGES
#     # ==============================

#     width = frame.shape[1]

#     cv2.rectangle(
#         frame,
#         (width - 180, 130),
#         (width - 5, 240),
#         (50, 50, 50),
#         -1
#     )

#     cv2.putText(
#         frame,
#         "LANGUAGES",
#         (width - 165, 155),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.6,
#         (255, 255, 255),
#         2
#     )

#     cv2.putText(
#         frame,
#         "1 : English",
#         (width - 165, 185),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (255, 0, 255),
#         1
#     )

#     cv2.putText(
#         frame,
#         "2 : Hindi",
#         (width - 165, 210),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (255, 0, 255),
#         1
#     )

#     cv2.putText(
#         frame,
#         "3 : Marathi",
#         (width - 165, 235),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         0.5,
#         (255, 0, 255),
#         1
#     )

#     # ==============================
#     # SHOW WINDOW
#     # ==============================

#     cv2.imshow(
#         "Dynamic Sign Prediction",
#         frame
#     )

#     # ==============================
#     # HOTKEYS
#     # ==============================

#     key = cv2.waitKey(1) & 0xFF

#     # ESC
#     if key == 27:
#         break

#     # SPACE -> Add word
#     elif key == 32:
#         if current_prediction != "":
#             sentence.append(
#                 current_prediction
#             )

#     # BACKSPACE -> Delete
#     elif key == 8:
#         if sentence:
#             sentence.pop()

#     # C -> Clear
#     elif key == ord('c'):
#         sentence.clear()

#     # Language Switching
#     elif key in LANGUAGES:
#         current_language_name, current_language_code = (
#             LANGUAGES[key]
#         )


# # ==============================
# # CLEANUP
# # ==============================

# cap.release()
# hands.close()
# cv2.destroyAllWindows()






import cv2
import mediapipe as mp
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from collections import deque, Counter
from googletrans import Translator
from PIL import Image, ImageDraw, ImageFont

# ======================================
# UNICODE TEXT DRAWING FUNCTION
# ======================================

def draw_unicode_text(frame, text, position,
                      font_size=28,
                      color=(0, 255, 255)):

    img_pil = Image.fromarray(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    )

    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(
        r"C:\Windows\Fonts\Nirmala.ttc",
        font_size
    )

    draw.text(
        position,
        text,
        font=font,
        fill=(color[2], color[1], color[0])
    )

    return cv2.cvtColor(
        np.array(img_pil),
        cv2.COLOR_RGB2BGR
    )


# ======================================
# LOAD DYNAMIC MODEL
# ======================================

print("Loading Dynamic Model...")

model = load_model("dynamic_model.keras")

label_encoder = joblib.load(
    "dynamic_label_encoder.pkl"
)

print("Dynamic model loaded successfully!")
print("Classes:", label_encoder.classes_)


# ======================================
# GOOGLE TRANSLATE
# ======================================

translator = Translator()


# ======================================
# SENTENCE BUFFER
# ======================================

sentence = []

current_prediction = ""


# ======================================
# WORD MAP
# Dataset Label → Proper English
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
# LANGUAGE SETTINGS
# ======================================

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

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======================================
# PREDICTION BUFFERS
# ======================================

# Stores 30 frames for LSTM input
sequence = deque(maxlen=30)

# Used for prediction stabilization
prediction_history = deque(maxlen=10)

# ======================================
# START WEBCAM
# ======================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

print("\nStarting Dynamic Prediction...")
print("Hotkeys:")
print("SPACE      -> Add current word")
print("BACKSPACE  -> Delete last word")
print("C          -> Clear sentence")
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

    # Mirror effect
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Process hand landmarks
    results = hands.process(rgb)

    display_text = "Show Gesture"

    # ======================================
    # HAND DETECTION
    # ======================================

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        # Draw hand skeleton
        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        row = []

        # Extract 21 landmarks × 3 values
        for lm in hand_landmarks.landmark:

            row.extend([
                lm.x,
                lm.y,
                lm.z
            ])

        # Store frame in sequence
        sequence.append(row)

        # ======================================
        # DYNAMIC PREDICTION
        # ======================================

        if len(sequence) == 30:

            input_data = np.array(sequence)

            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            prediction = model.predict(
                input_data,
                verbose=0
            )[0]

            confidence = np.max(prediction)

            predicted_index = np.argmax(
                prediction
            )

            predicted_label = (
                label_encoder.inverse_transform(
                    [predicted_index]
                )[0]
            )

            # ======================================
            # STABILIZATION
            # ======================================

            if confidence > 0.70:

                prediction_history.append(
                    predicted_label
                )

                stable_prediction = Counter(
                    prediction_history
                ).most_common(1)[0][0]

                current_prediction = (
                    stable_prediction
                )

                display_text = (
                    f"{stable_prediction} "
                    f"({confidence:.2f})"
                )

            else:

                display_text = (
                    f"Uncertain "
                    f"({confidence:.2f})"
                )

    else:

        # No hand detected
        sequence.clear()

        prediction_history.clear()

        # ======================================
    # TRANSLATE SENTENCE
    # ======================================

    mapped_sentence = []

    for word in sentence:
        mapped_sentence.append(
            WORD_MAP.get(word, word)
        )

    translated_words = []

    for word in mapped_sentence:

        # Use custom translations first
        if word in CUSTOM_TRANSLATIONS[current_language_code]:

            translated_words.append(
                CUSTOM_TRANSLATIONS[current_language_code][word]
            )

        # Fallback to Google Translate
        else:

            try:
                translated_words.append(
                    translator.translate(
                        word,
                        dest=current_language_code
                    ).text
                )

            except Exception:
                translated_words.append(word)

    translated_sentence = " ".join(
        translated_words
    )

    # ======================================
    # TRANSLATE LIVE PREDICTION
    # ======================================

    translated_prediction = display_text

    if current_prediction != "":

        english_prediction = WORD_MAP.get(
            current_prediction,
            current_prediction
        )

        if "(" in display_text:
            confidence_text = display_text[
                display_text.find("("):
            ]
        else:
            confidence_text = ""

        if english_prediction in CUSTOM_TRANSLATIONS[current_language_code]:

            translated_prediction = (
                CUSTOM_TRANSLATIONS[current_language_code][
                    english_prediction
                ]
                + " "
                + confidence_text
            )

        else:

            try:
                translated_prediction = (
                    translator.translate(
                        english_prediction,
                        dest=current_language_code
                    ).text
                    + " "
                    + confidence_text
                )

            except Exception:

                translated_prediction = (
                    display_text
                )

    # ======================================
    # DISPLAY UI
    # ======================================

    # Prediction
    frame = draw_unicode_text(
        frame,
        f"Prediction : {translated_prediction}",
        (20, 20),
        font_size=28,
        color=(0, 255, 0)
    )

    # Language
    cv2.putText(
        frame,
        f"Language : {current_language_name}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    # Sentence
    frame = draw_unicode_text(
        frame,
        f"Sentence : {translated_sentence}",
        (20, 90),
        font_size=28,
        color=(0, 255, 255)
    )

    # ======================================
    # LEFT CONTROLS PANEL
    # ======================================

    cv2.rectangle(
        frame,
        (5, 130),
        (180, 270),
        (50, 50, 50),
        -1
    )

    cv2.putText(
        frame,
        "CONTROLS",
        (20, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE : Add",
        (20, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "BKSP  : Delete",
        (20, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "C     : Clear",
        (20, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    cv2.putText(
        frame,
        "ESC   : Exit",
        (20, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1
    )

    # ======================================
    # RIGHT LANGUAGE PANEL
    # ======================================

    width = frame.shape[1]

    cv2.rectangle(
        frame,
        (width - 180, 130),
        (width - 5, 240),
        (50, 50, 50),
        -1
    )

    cv2.putText(
        frame,
        "LANGUAGES",
        (width - 165, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "1 : English",
        (width - 165, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1
    )

    cv2.putText(
        frame,
        "2 : Hindi",
        (width - 165, 210),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1
    )

    cv2.putText(
        frame,
        "3 : Marathi",
        (width - 165, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 0, 255),
        1
    )

    # ======================================
    # SHOW WINDOW
    # ======================================

    cv2.imshow(
        "Dynamic Sign Prediction",
        frame
    )

    # ======================================
    # HOTKEYS
    # ======================================

    key = cv2.waitKey(1) & 0xFF

    # ESC
    if key == 27:
        break

    # SPACE → Add current prediction
    elif key == 32:

        if current_prediction != "":

            sentence.append(
                current_prediction
            )

    # BACKSPACE → Delete last word
    elif key == 8:

        if sentence:
            sentence.pop()

    # C → Clear sentence
    elif key == ord('c'):

        sentence.clear()

    # Language Switching
    elif key in LANGUAGES:

        current_language_name, current_language_code = (
            LANGUAGES[key]
        )


# ======================================
# CLEANUP
# ======================================

cap.release()
hands.close()
cv2.destroyAllWindows()