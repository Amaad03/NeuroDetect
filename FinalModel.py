import tensorflow as tf
from tensorflow import keras
from keras import datasets, layers, models, regularizers
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.layers import Input
import os 
import numpy as np
import random
""
train_dir = 'dataset/Testing'
test_dir = 'dataset/Training'
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_aug = ImageDataGenerator(rescale=1./255)

train_generator = train_aug.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    subset='training',
    class_mode='categorical'
)

validation_generator = train_aug.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    subset='validation',
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_aug.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

#base_model.trainable = False
base_model.trainable = True
for layer in base_model.layers[:10]:  # Adjust this number to unfreeze more layers
    layer.trainable = False


optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model2.keras', monitor='val_loss', save_best_only=True)

""
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=11,
    callbacks=[early_stopping, model_checkpoint, lr_reduction],
    verbose=1
)

model.save('FinalModel.keras')

test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label = 'Validation Loss')
plt.show()