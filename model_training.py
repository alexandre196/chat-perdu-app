import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Paramètres
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = 'data/train'  # Doit contenir exactement deux sous‑dossiers : 'cats' et 'other'

# Générateur avec découpe train/validation
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print("Classes détectées :", train_generator.class_indices)
# Ex : {'cats': 0, 'other': 1}

# Construction du modèle
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes : cats / other
])

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entraînement
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Sauvegarde
model.save('chat_recognition_model.h5')
print("✅ Modèle entraîné et sauvegardé sous 'chat_recognition_model.h5'")
