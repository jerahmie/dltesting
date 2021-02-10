#!/usr/bin/env python
"""Small covnet for dogs vs cats
    from "Deep Learning with Python" by Collet, Chapter 5.
"""
import os
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main(base_dir):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        r"/train_data",
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        r"/validation_data",
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50)

    model.save(os.path.join("/save_data", "cats_and_dogs_small_1.h5"))

if __name__ == "__main__":
    base_dir = os.path.join('/mnt','Data','kaggle','dogs-vs-cats-small')
    #train_dir = os.path.join(base_dir, 'train')
    #validation_dir = os.path.join(base_dir, 'validation')
    main(base_dir)

