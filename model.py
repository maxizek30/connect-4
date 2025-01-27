from tensorflow.python.keras import models, layers
from tensorflow.python.keras.optimizers import adam_v2
import tensorflow as tf

def create_model(state_shape, num_actions):
    """
    Create a deep learning model for a Connect4 AI agent.

    Args:
    - state_shape (tuple): The shape of the input state (e.g., the board dimensions, typically (6, 7)).
    - num_actions (int): The number of possible actions (e.g., 7 columns in Connect4).

    Returns:
    - model (tf.keras.Model): The compiled neural network model ready for training.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*state_shape, 1)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # Output layer:
        # - Number of neurons equals the number of actions (e.g., 7 for 7 columns)
        # - No activation function (linear output for Q-value predictions)
        layers.Dense(num_actions)
    ])
    # Define the optimizer for training
    # Adam optimizer is used with a learning rate of 0.001
    optimizer = adam_v2.Adam(learning_rate=0.001)
    # Compile the model with Mean Squared Error (MSE) loss
    # - MSE is commonly used for regression tasks like Q-value prediction
    model.compile(optimizer=optimizer, loss='mse')
    # Return the compiled model
    return model