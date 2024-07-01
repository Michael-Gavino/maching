
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuraciones
IMG_SIZE = (800, 532)  # Tamaño de las imágenes
BATCH_SIZE = 32  # Tamaño del batch

# Directorio de las imágenes y CSV de etiquetas
image_dir = "train"
csv_file = 'csv'

# Leer el CSV
df = pd.read_csv(csv_file)

# Preparar el DataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalización
    validation_split=0.2  # Dividir en train/val
)

# Generadores de imágenes
train_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=df,
    directory=image_dir,
    x_col='filename',
    y_col='label',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)


from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Cargar el modelo base ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Añadir capas personalizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Definir el modelo
mimodelo = Model(inputs=base_model.input, outputs=predictions)

# Congelar capas base
for layer in base_model.layers:
    layer.trainable = False

# Compilar el modelo
mimodelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = mimodelo.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=10
)

# Descongelar capas y continuar el entrenamiento
for layer in base_model.layers:
    layer.trainable = True

mimodelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

history_fine = mimodelo.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=val_generator.samples // BATCH_SIZE,
    epochs=5
)

# Guardar el modelo
mimodelo.save('mimodelo_model.h5')


# Evaluar el modelo
loss, accuracy = mimodelo.evaluate(val_generator, steps=val_generator.samples // BATCH_SIZE)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Graficar precisión y pérdida
plt.figure(figsize=(12, 4))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.show()
