# # # import cv2
# # # import mediapipe as mp
# # # import numpy as np
# # # import os

# # # # CHANGE THIS for each gesture when collecting
# # # GESTURE = "yes"

# # # SAVE_DIR = f"dataset/dynamic/{GESTURE}"
# # # os.makedirs(SAVE_DIR, exist_ok=True)

# # # mp_hands = mp.solutions.hands
# # # hands = mp_hands.Hands(
# # #     max_num_hands=1,
# # #     min_detection_confidence=0.6,
# # #     min_tracking_confidence=0.5
# # # )
# # # mp_draw = mp.solutions.drawing_utils

# # # SEQUENCE_LENGTH = 30   # 30 frame sequences
# # # sequence = []
# # # sample_no = 0

# # # cap = cv2.VideoCapture(0)
# # # print("Press 'r' to start recording a 30-frame sequence.")
# # # print("Press 'q' to quit.")

# # # while True:
# # #     ret, frame = cap.read()
# # #     if not ret:
# # #         break

# # #     frame = cv2.flip(frame, 1)
# # #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # #     results = hands.process(rgb)

# # #     if results.multi_hand_landmarks:
# # #         lm = results.multi_hand_landmarks[0]
# # #         mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

# # #         frame_data = []
# # #         for p in lm.landmark:
# # #             frame_data.extend([p.x, p.y, p.z])

# # #         if len(sequence) < SEQUENCE_LENGTH:
# # #             sequence.append(frame_data)

# # #         # Save when 30 frames (1 sequence) completed
# # #         if len(sequence) == SEQUENCE_LENGTH:
# # #             np.save(f"{SAVE_DIR}/{sample_no}.npy", np.array(sequence))
# # #             print(f"Saved sample: {sample_no}")
# # #             sample_no += 1
# # #             sequence = []

# # #     cv2.imshow("Dynamic Dataset Collector", frame)

# # #     key = cv2.waitKey(1) & 0xFF
    
# # #     if key == ord('r'):   # restart sequence capture
# # #         sequence = []

# # #     elif key == ord('q'):
# # #         break

# # # cap.release()
# # # cv2.destroyAllWindows()









# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import os
# # import time

# # # ==========================================================
# # # ENTER GESTURE NAME HERE
# # # ==========================================================
# # GESTURE = "happy"

# # SAVE_DIR = os.path.join("dataset", "dynamic", "raw", GESTURE)
# # os.makedirs(SAVE_DIR, exist_ok=True)

# # # ==========================================================
# # # FIND NEXT SAMPLE NUMBER
# # # ==========================================================
# # existing = [
# #     int(f.split(".")[0])
# #     for f in os.listdir(SAVE_DIR)
# #     if f.endswith(".npy") and f.split(".")[0].isdigit()
# # ]

# # sample_no = max(existing) + 1 if existing else 0

# # # ==========================================================
# # # MEDIAPIPE
# # # ==========================================================
# # mp_hands = mp.solutions.hands
# # mp_draw = mp.solutions.drawing_utils

# # hands = mp_hands.Hands(
# #     static_image_mode=False,
# #     max_num_hands=1,
# #     min_detection_confidence=0.6,
# #     min_tracking_confidence=0.5
# # )

# # # ==========================================================
# # # SETTINGS
# # # ==========================================================
# # SEQUENCE_LENGTH = 30

# # recording = False
# # countdown = False
# # countdown_start = 0

# # sequence = []

# # status = "READY"

# # # ==========================================================
# # # NORMALIZATION
# # # ==========================================================
# # def normalize_frame(frame_data):

# #     landmarks = np.array(frame_data).reshape(21, 3)

# #     wrist = landmarks[0]

# #     landmarks = landmarks - wrist

# #     max_val = np.max(np.abs(landmarks))

# #     if max_val != 0:
# #         landmarks = landmarks / max_val

# #     return landmarks.flatten()

# # # ==========================================================
# # # CAMERA
# # # ==========================================================
# # cap = cv2.VideoCapture(0)

# # print("=" * 50)
# # print("Dynamic Dataset Collector")
# # print("=" * 50)
# # print("Gesture :", GESTURE)
# # print()
# # print("R -> Record Sample")
# # print("D -> Delete Last Sample")
# # print("Q -> Quit")
# # print("=" * 50)

# # # ==========================================================
# # # LOOP
# # # ==========================================================
# # while True:

# #     ret, frame = cap.read()

# #     if not ret:
# #         break

# #     frame = cv2.flip(frame, 1)

# #     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# #     results = hands.process(rgb)

# #     if countdown:

# #         elapsed = time.time() - countdown_start

# #         remaining = 3 - int(elapsed)

# #         cv2.putText(
# #             frame,
# #             f"Recording Starts In : {remaining}",
# #             (120,220),
# #             cv2.FONT_HERSHEY_SIMPLEX,
# #             1,
# #             (0,255,255),
# #             3
# #         )

# #         if elapsed >= 3:

# #             countdown = False
# #             recording = True
# #             sequence = []
# #             status = "RECORDING"

# #     if results.multi_hand_landmarks:

# #         hand = results.multi_hand_landmarks[0]

# #         mp_draw.draw_landmarks(
# #             frame,
# #             hand,
# #             mp_hands.HAND_CONNECTIONS
# #         )

# #         if recording:

# #             row = []

# #             for lm in hand.landmark:
# #                 row.extend([lm.x, lm.y, lm.z])

# #             row = normalize_frame(row)

# #             sequence.append(row)

# #             cv2.rectangle(
# #                 frame,
# #                 (5,5),
# #                 (635,475),
# #                 (0,255,0),
# #                 4
# #             )

# #             progress = int((len(sequence)/SEQUENCE_LENGTH)*300)

# #             cv2.rectangle(frame,(170,440),(470,460),(255,255,255),2)
# #             cv2.rectangle(frame,(170,440),(170+progress,460),(0,255,0),-1)

# #             cv2.putText(
# #                 frame,
# #                 f"Frame : {len(sequence)}/{SEQUENCE_LENGTH}",
# #                 (190,430),
# #                 cv2.FONT_HERSHEY_SIMPLEX,
# #                 0.7,
# #                 (0,255,0),
# #                 2
# #             )

# #             if len(sequence) == SEQUENCE_LENGTH:

# #                 np.save(
# #                     os.path.join(SAVE_DIR,f"{sample_no}.npy"),
# #                     np.array(sequence)
# #                 )

# #                 print(f"Saved Sample {sample_no}")

# #                 sample_no += 1

# #                 recording = False

# #                 status = "SAVED"

# #                 sequence = []

# #     # ======================================================
# #     # UI
# #     # ======================================================

# #     overlay = frame.copy()

# #     cv2.rectangle(overlay,(0,0),(640,90),(0,0,0),-1)
# #     cv2.rectangle(overlay,(0,430),(640,480),(0,0,0),-1)

# #     cv2.addWeighted(
# #         overlay,
# #         0.5,
# #         frame,
# #         0.5,
# #         0,
# #         frame
# #     )

# #     cv2.putText(
# #         frame,
# #         f"Gesture : {GESTURE}",
# #         (20,30),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.8,
# #         (0,255,255),
# #         2
# #     )

# #     cv2.putText(
# #         frame,
# #         f"Samples : {sample_no}",
# #         (20,60),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.8,
# #         (255,255,255),
# #         2
# #     )

# #     if status == "READY":
# #         color = (0,255,255)
# #     elif status == "RECORDING":
# #         color = (0,255,0)
# #     else:
# #         color = (255,255,0)

# #     cv2.putText(
# #         frame,
# #         f"Status : {status}",
# #         (360,60),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.8,
# #         color,
# #         2
# #     )

# #     cv2.putText(
# #         frame,
# #         "R : Record",
# #         (20,465),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.65,
# #         (255,255,255),
# #         2
# #     )

# #     cv2.putText(
# #         frame,
# #         "D : Delete Last",
# #         (180,465),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.65,
# #         (255,255,255),
# #         2
# #     )

# #     cv2.putText(
# #         frame,
# #         "Q : Quit",
# #         (470,465),
# #         cv2.FONT_HERSHEY_SIMPLEX,
# #         0.65,
# #         (255,255,255),
# #         2
# #     )

# #     cv2.imshow("Dynamic Dataset Collector", frame)

# #     key = cv2.waitKey(1) & 0xFF

# #     if key == ord("r") and not recording and not countdown:

# #         countdown = True
# #         countdown_start = time.time()
# #         status = "COUNTDOWN"

# #     elif key == ord("d"):

# #         if sample_no > 0:

# #             last_file = os.path.join(
# #                 SAVE_DIR,
# #                 f"{sample_no-1}.npy"
# #             )

# #             if os.path.exists(last_file):

# #                 os.remove(last_file)

# #                 sample_no -= 1

# #                 print("Deleted Last Sample")

# #     elif key == ord("q"):

# #         break

# # cap.release()
# # cv2.destroyAllWindows()









# import cv2
# import mediapipe as mp
# import numpy as np
# import os
# import time

# # ==========================================================
# # SETTINGS
# # ==========================================================
# GESTURE = "kind"
# SEQUENCE_LENGTH = 30
# COOLDOWN_TIME = 2.0  # Time in seconds to wait between recordings

# SAVE_DIR = os.path.join("dataset", "dynamic", "raw", GESTURE)
# os.makedirs(SAVE_DIR, exist_ok=True)

# # Find next sample number
# existing = [
#     int(f.split(".")[0])
#     for f in os.listdir(SAVE_DIR)
#     if f.endswith(".npy") and f.split(".")[0].isdigit()
# ]
# sample_no = max(existing) + 1 if existing else 0

# # MediaPipe Setup
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5)

# sequence = []
# recording = False
# status = "READY"
# last_record_time = 0

# def normalize_frame(frame_data):
#     landmarks = np.array(frame_data).reshape(21, 3)
#     wrist = landmarks[0]
#     landmarks = landmarks - wrist
#     max_val = np.max(np.abs(landmarks))
#     return (landmarks / max_val if max_val != 0 else landmarks).flatten()

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret: break
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     current_time = time.time()

#     if results.multi_hand_landmarks:
#         hand = results.multi_hand_landmarks[0]
#         mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

#         # Start recording if READY and cooldown period has passed
#         if status == "READY" and (current_time - last_record_time) > COOLDOWN_TIME:
#             recording = True
#             status = "RECORDING"
#             sequence = []

#         if recording:
#             row = []
#             for lm in hand.landmark:
#                 row.extend([lm.x, lm.y, lm.z])
#             sequence.append(normalize_frame(row))

#             # Progress bar
#             progress = int((len(sequence)/SEQUENCE_LENGTH)*300)
#             cv2.rectangle(frame, (170, 440), (170 + progress, 460), (0, 255, 0), -1)

#             if len(sequence) == SEQUENCE_LENGTH:
#                 np.save(os.path.join(SAVE_DIR, f"{sample_no}.npy"), np.array(sequence))
#                 print(f"Saved Sample {sample_no}")
#                 sample_no += 1
#                 recording = False
#                 status = "READY"
#                 last_record_time = time.time() # Reset timer for cooldown

#     # UI Overlay
#     cv2.putText(frame, f"Gesture: {GESTURE} | Samples: {sample_no}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.putText(frame, f"Status: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if recording else (255, 255, 255), 2)
    
#     cv2.imshow("Dynamic Dataset Collector", frame)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# cap.release()
# cv2.destroyAllWindows()













import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==========================================================
# SETTINGS
# ==========================================================
GESTURE = "tired"
SEQUENCE_LENGTH = 30
COOLDOWN_TIME = 2.0  # Time in seconds to wait between recordings

SAVE_DIR = os.path.join("dataset", "dynamic", "raw", GESTURE)
os.makedirs(SAVE_DIR, exist_ok=True)

# Find next sample number
existing = [
    int(f.split(".")[0])
    for f in os.listdir(SAVE_DIR)
    if f.endswith(".npy") and f.split(".")[0].isdigit()
]
sample_no = max(existing) + 1 if existing else 0

# MediaPipe Setup (Updated for 2 hands)
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.6, 
    min_tracking_confidence=0.5
)

sequence = []
recording = False
status = "READY"
last_record_time = 0

def normalize_frame(frame_data):
    # Reshape for 21 landmarks x 3 coordinates
    landmarks = np.array(frame_data).reshape(21, 3)
    wrist = landmarks[0]
    landmarks = landmarks - wrist
    max_val = np.max(np.abs(landmarks))
    return (landmarks / max_val if max_val != 0 else landmarks).flatten()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    current_time = time.time()

    # Capture data if hands are present
    if results.multi_hand_landmarks:
        # Determine if we are recording the primary hand (index 0)
        hand = results.multi_hand_landmarks[0]
        
        # Draw all detected hands
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if status == "READY" and (current_time - last_record_time) > COOLDOWN_TIME:
            recording = True
            status = "RECORDING"
            sequence = []

        if recording:
            row = []
            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])
            sequence.append(normalize_frame(row))

            progress = int((len(sequence)/SEQUENCE_LENGTH)*300)
            cv2.rectangle(frame, (170, 440), (170 + progress, 460), (0, 255, 0), -1)

            if len(sequence) == SEQUENCE_LENGTH:
                np.save(os.path.join(SAVE_DIR, f"{sample_no}.npy"), np.array(sequence))
                print(f"Saved Sample {sample_no} (Hands detected: {len(results.multi_hand_landmarks)})")
                sample_no += 1
                recording = False
                status = "READY"
                last_record_time = time.time()

    # UI Overlay
    cv2.putText(frame, f"Gesture: {GESTURE} | Samples: {sample_no}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if recording else (255, 255, 255), 2)
    
    cv2.imshow("Dynamic Dataset Collector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()