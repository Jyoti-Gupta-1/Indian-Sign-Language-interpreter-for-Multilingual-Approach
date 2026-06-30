import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ==============================
# LOAD DATASET
# ==============================

X = np.load("dataset/dynamic/processed/dynamic_landmarks.npy")
y = np.load("dataset/dynamic/processed/dynamic_labels.npy")

print("Dataset Loaded")
print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ==============================
# CREATE RESULTS DIRECTORY
# ==============================

os.makedirs("results/dynamic", exist_ok=True)



# ==============================
# LABEL ENCODING
# ==============================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(y)

joblib.dump(
    encoder,
    "dynamic_label_encoder.pkl"
)

print("\nClasses:")
print(encoder.classes_)

y_categorical = to_categorical(y_encoded)

# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_categorical,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ==============================
# BUILD MODEL
# ==============================

model = Sequential()

model.add(
    LSTM(
        128,
        return_sequences=True,
        activation="tanh",
        input_shape=(30, 63)
    )
)

model.add(Dropout(0.3))

model.add(
    LSTM(
        64,
        activation="tanh"
    )
)

model.add(Dropout(0.3))

model.add(
    Dense(
        64,
        activation="relu"
    )
)

model.add(Dropout(0.2))

model.add(
    Dense(
        len(encoder.classes_),
        activation="softmax"
    )
)

# ==============================
# COMPILE
# ==============================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# TRAIN
# ==============================

history = model.fit(
    X_train,
    y_train,
    epochs=80,
    batch_size=16,
    validation_data=(X_test, y_test)
)

# ==============================
# ACCURACY GRAPH
# ==============================

plt.figure(figsize=(8,5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.title("Dynamic Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.grid(True)

plt.savefig(
    "results/dynamic/accuracy.png",
    dpi=300
)

plt.close()




# ==============================
# LOSS GRAPH
# ==============================

plt.figure(figsize=(8,5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.title("Dynamic Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "results/dynamic/loss.png",
    dpi=300
)

plt.close()

# ==============================
# EVALUATE
# ==============================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nDynamic Accuracy:", accuracy * 100)


# ==============================
# PREDICTIONS
# ==============================

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred, axis=1)

y_true = np.argmax(y_test, axis=1)

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

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    "results/dynamic/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==============================
# PRECISION RECALL F1
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
    [precision, recall, f1]
)

plt.ylim(0,1)

plt.title("Dynamic Model Metrics")

plt.savefig(
    "results/dynamic/metrics.png",
    dpi=300
)

plt.close()


# ==============================
# CLASSIFICATION REPORT
# ==============================

report = classification_report(
    y_true,
    y_pred,
    target_names=encoder.classes_
)

with open(
    "results/dynamic/classification_report.txt",
    "w"
) as f:

    f.write(report)

print(report)

# ==============================
# SAVE MODEL
# ==============================

model.save("dynamic_model.keras")

print("\nModel Saved Successfully!")
print("dynamic_model.keras")
print("dynamic_label_encoder.pkl")