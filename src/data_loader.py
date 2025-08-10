# Stub - implement your dataset loading here for training model

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    val_generator = datagen.flow_from_directory(
        data_dir,
        target_size=(64,64),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator
