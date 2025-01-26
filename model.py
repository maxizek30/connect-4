from tensorflow.python.keras import models, layers
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow as tf

def create_model(state_shape, num_actions):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*state_shape, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_actions)
    ])
    optimizer = adam_v2.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')
    return model