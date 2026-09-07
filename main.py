import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from keras.datasets import mnist
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import keras

tf.get_logger().setLevel(logging.ERROR)

def show_images(images, titles, layout=(3,3)):
    total = layout[0] * layout[1]
    for i in range(total):
        plt.subplot(layout[0],layout[1],i+1)
        plt.title(f"hello {titles[i]}")
        plt.imshow(images[i],cmap="gray")
    plt.show()

def load_data():
    (raw_X_train, Y_train), (raw_X_test, Y_test) = mnist.load_data()
    X_train = raw_X_train.astype("float32") / 255
    X_test = raw_X_test.astype("float32") / 255
    # show_images(X_train,Y_train,(5,5))

    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)

    return X_train, X_test, Y_train, Y_test

def forward(X_train, Y_train):
    model = Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.RandomRotation(0.12),
        layers.RandomTranslation(0.08, 0.08),
        layers.RandomZoom(0.1),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        Y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1,
        callbacks=[early_stopping]
    )

    return model, history

X_train, X_test, Y_train, Y_test = load_data()
model, history = forward(X_train, Y_train)

model.save("mnist_model.keras")

print(model)
print(history)

test_loss, test_accuracy = model.evaluate(X_test, Y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)