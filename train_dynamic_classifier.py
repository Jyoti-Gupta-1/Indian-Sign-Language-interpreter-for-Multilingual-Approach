import numpy as np
import joblib

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
# EVALUATE
# ==============================

loss, accuracy = model.evaluate(
    X_test,
    y_test
)

print("\nDynamic Accuracy:", accuracy * 100)

# ==============================
# SAVE MODEL
# ==============================

model.save("dynamic_model.keras")

print("\nModel Saved Successfully!")
print("dynamic_model.keras")
print("dynamic_label_encoder.pkl")