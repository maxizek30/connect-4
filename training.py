import numpy as np
def train_step(model, target_model, replay_buffer, batch_size, gamma):
    """
    Perform a single training step for the Q-learning agent.

    Args:
    - model (tf.keras.Model): The current Q-network being trained.
    - target_model (tf.keras.Model): The target Q-network for more stable updates.
    - replay_buffer (ReplayBuffer): Replay buffer containing past experiences.
    - batch_size (int): The number of experiences to sample for training.
    - gamma (float): The discount factor for future rewards (0 <= gamma <= 1).
    """
    # Ensure the replay buffer has enough samples to create a batch
    if replay_buffer.size() < batch_size:
        return # Exit if there aren't enough samples for a full batch

    # Sample a batch of experiences from the replay buffer
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    # Predict Q-values for the current states using the Q-network
    q_values = model.predict(states)

    # Predict Q-values for the next states using the target network
    next_q_values = target_model.predict(next_states)

    # Create a copy of the Q-values to update the targets
    target_q_values = q_values.copy()

    # Update the Q-value targets for each sampled experience
    for i in range(batch_size):
        if dones[i]: # If the episode ended with this experience
            target_q_values[i, actions[i]] = rewards[i] # Target is the immediate reward
        else: # If the episode is not done
            # Target includes the discounted maximum Q-value from the next state
            target_q_values[i, actions[i]] = rewards[i] + gamma * np.max(next_q_values[i])

    # Perform a single gradient descent step on the Q-network
    # Train the model to minimize the difference between predicted and target Q-values
    model.train_on_batch(states, target_q_values)
