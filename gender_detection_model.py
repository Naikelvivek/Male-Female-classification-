import os
import cv2
import numpy as np
import random
import glob
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Hyperparameters
epochs = 200  # Train longer for better accuracy
lr = 1e-4     # Lower learning rate for stable training
batch_size = 64
img_dims = (224, 224, 3)  # Increased size for better feature extraction

data = []
labels = []

# Load images
image_files = [
    f for f in glob.glob(r'C:\Users\naike\OneDrive\Desktop\projects\hackathon\gender detection\Gender Detection\gender_dataset_face' + "/**/*", recursive=True) 
    if not os.path.isdir(f)
]
random.shuffle(image_files)

for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  
    label = 1 if label == "woman" else 0  
    labels.append(label)

# Convert to NumPy array
data = np.array(data, dtype="float32") / 255.0  
labels = np.array(labels).astype("int")  

# Train-test split
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = to_categorical(trainY, num_classes=2) 
testY = to_categorical(testY, num_classes=2)

# Data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# Load MobileNetV2 as base model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=img_dims)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)  
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation="softmax")(x)

# Create final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
opt = Adam(learning_rate=lr)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the model
H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
              validation_data=(testX, testY),
              epochs=epochs, verbose=1)

# Save the model
model.save('gender_detection.keras')
