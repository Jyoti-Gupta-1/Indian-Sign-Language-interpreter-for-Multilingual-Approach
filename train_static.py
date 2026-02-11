


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 15

# -----------------------------
# Data preparation
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    "dataset/static",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    "dataset/static",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# Save static gesture classes for live detection
STATIC_GESTURES = [k for k, v in sorted(train_gen.class_indices.items(), key=lambda item: item[1])]
with open("static_gestures.txt", "w") as f:
    for gesture in STATIC_GESTURES:
        f.write(f"{gesture}\n")

# -----------------------------
# Model definition
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_gen.num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# Train model
# -----------------------------
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# -----------------------------
# Save model
# -----------------------------
model.save("static_letters.h5")
print("Static gesture model saved as static_letters.h5")
print("Static gesture classes saved in static_gestures.txt")
