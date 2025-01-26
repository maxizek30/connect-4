from environment import Connect4Env
from replay_buffer import ReplayBuffer
from model import create_model
import training as train_step
import numpy as np
import tensorflow as tf
print("GPUs available:", len(tf.config.list_physical_devices('GPU')))

env = Connect4Env()
replay_buffer = ReplayBuffer()
state_shape = (6, 7)
num_actions = 7
model = create_model(state_shape, num_actions)
target_model = create_model(state_shape, num_actions)
target_model.set_weights(model.get_weights())

# Hyperparameters
num_episodes = 1000
batch_size = 128
gamma = 0.95  # Discount factor
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.997  # Slower decay
target_update_freq = 10

def evaluate_agent(env, model, num_games=10):
    wins = 0
    for _ in range(num_games):
        state = env.reset()
        state = state[..., np.newaxis]  # Add channel dimension
        done = False
        while not done:
            q_values = model.predict(state[np.newaxis, ...])
            action = np.argmax(q_values[0])
            state, _, done = env.step(action)
            state = state[..., np.newaxis]  # Add channel dimension
        # Check if player 1 wins
        if env.current_player == -1:  # Opponent lost
            wins += 1
    print(f"Win Rate: {wins / num_games:.2f}")

# Training loop
for episode in range(num_episodes):
    if episode % 50 == 0:
        model.save(f"models/connect4_model_episode_{episode}.h5")
    if episode % 10 == 0:
        evaluate_agent(env, model)
    if episode % 200 == 0:
        replay_buffer = ReplayBuffer(max_size=100_000)

    state = env.reset()
    state = state[..., np.newaxis]  # Add channel dimension

    done = False
    total_reward = 0

    while not done:
        if np.random.random() < epsilon:
            action = np.random.choice(env.valid_actions())
        else:
            q_values = model.predict(state[np.newaxis, ...])   # Exploit
            action = np.argmax(q_values[0])
        next_state, reward, done = env.step(action)
        next_state = next_state[..., np.newaxis]  # Add channel dimension
        total_reward += reward
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state

        train_step(model, target_model, replay_buffer, batch_size, gamma)

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % target_update_freq == 0:
        target_model.set_weights(model.get_weights())

    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")






