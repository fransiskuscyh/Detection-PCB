import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image

def check_images(directory):
    print(f"Checking images in directory: {directory}")
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            for img in os.listdir(folder_path):
                try:
                    img_path = os.path.join(folder_path, img)
                    with Image.open(img_path) as im:
                        im.verify()
                except Exception as e:
                    print(f"Corrupted file: {img_path}, error: {e}")

train_dir = os.path.join(os.getcwd(), 'datasets/train')
test_dir = os.path.join(os.getcwd(), 'datasets/test')

assert os.path.exists(train_dir), f"Train directory not found: {train_dir}"
assert os.path.exists(test_dir), f"Test directory not found: {test_dir}"

check_images(train_dir)
check_images(test_dir)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    classes=['Pass', 'Reject'] 
)

validation_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary',
    classes=['Pass', 'Reject'] 
)

print(f"Train samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
base_model.trainable = False  

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])

model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

try:
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        callbacks=[early_stopping, lr_scheduler]
    )
except Exception as e:
    print("Error during training:", e)

model.save('pcb_detection_cnn_modelv5.h5')
print("Model saved as 'pcb_detection_cnn_modelv5.h5'")

validation_generator.reset()
y_pred = (model.predict(validation_generator) > 0.5).astype("int32")
y_true = validation_generator.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Pass', 'Reject']))

conf_matrix = confusion_matrix(y_true, y_pred)
plt.imshow(conf_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(ticks=[0, 1], labels=['Pass', 'Reject'])
plt.yticks(ticks=[0, 1], labels=['Pass', 'Reject'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print("Class Indices:", train_generator.class_indices)
