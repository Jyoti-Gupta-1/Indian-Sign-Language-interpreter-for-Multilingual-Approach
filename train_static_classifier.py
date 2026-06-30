import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# ==============================
# LOAD NORMALIZED DATASET
# ==============================

DATASET_PATH = "dataset/static/processed/static_landmarks_normalized.csv"

# low_memory=False removes dtype warning
df = pd.read_csv(DATASET_PATH, low_memory=False)

print("Dataset loaded successfully!")
print(f"Total samples: {len(df)}")

# ==============================
# CREATE RESULTS DIRECTORY
# ==============================

os.makedirs("results/static", exist_ok=True)

# ==============================
# FEATURES AND LABELS
# ==============================

# All landmark columns
X = df.iloc[:, :-1]

# Convert ALL labels to string
# This fixes mixed int/string label issue
y = df.iloc[:, -1].astype(str)

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "static_label_encoder.pkl"
)

print("\nClasses:")
print(encoder.classes_)

y = to_categorical(y_encoded)



# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_train = X_train.values.reshape((-1,126,1))
X_test = X_test.values.reshape((-1,126,1))

print("\n========== DATA SPLIT ==========")
print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))

model = Sequential()

model.add(
    Conv1D(
        64,
        kernel_size=3,
        activation="relu",
        input_shape=(126,1)
    )
)

model.add(MaxPooling1D(2))

model.add(
    Conv1D(
        128,
        3,
        activation="relu"
    )
)

model.add(MaxPooling1D(2))

model.add(Flatten())

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(y.shape[1], activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    epochs=50,
    batch_size=32
)


# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy",
    linewidth=2
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy",
    linewidth=2
)

plt.title("Static CNN Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.grid(True)

plt.legend()

plt.savefig(
    "results/static/accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ==============================
# LOSS GRAPH
# ==============================

plt.figure(figsize=(8,5))

plt.plot(
    history.history["loss"],
    label="Training Loss",
    linewidth=2
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

plt.title("Static CNN Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.grid(True)

plt.legend()

plt.savefig(
    "results/static/loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



print("Training completed!")

# ==============================
# PREDICTIONS
# ==============================

print("\nMaking predictions...")

# y_pred = model.predict(X_test)

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

# ==============================
# EVALUATION
# ==============================

accuracy = accuracy_score(y_true, y_pred)

print("\n========== RESULTS ==========")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=encoder.classes_
    )
)


# ==============================
# CONFUSION MATRIX
# ==============================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.title("Static Model Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "results/static/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# METRICS GRAPH
# ==============================

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)

plt.figure(figsize=(6,5))

plt.bar(
    ["Precision","Recall","F1 Score"],
    [precision, recall, f1],
    color=["royalblue","seagreen","tomato"]
)

plt.ylim(0,1)

plt.title("Static Model Metrics")

plt.savefig(
    "results/static/metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# SAVE CLASSIFICATION REPORT
# ==============================

report = classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_
)

with open(
    "results/static/classification_report.txt",
    "w"
) as f:

    f.write(report)



# ==============================
# SAVE ACCURACY
# ==============================

with open(
    "results/static/accuracy.txt",
    "w"
) as f:

    f.write(
        f"Accuracy : {accuracy*100:.2f}%"
    )

# ==============================
# SAVE MODEL
# ==============================

model.save("static_model.keras")

print("\n========== MODEL SAVED ==========")
print("Model saved successfully!")
print("static_model.keras")
print("static_label_encoder.pkl")